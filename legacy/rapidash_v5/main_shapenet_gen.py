import os
import argparse

import torch
import numpy as np
import pytorch_lightning as pl
from chamferdist import ChamferDistance
from torch.utils.data import DataLoader

import wandb
from datasets.pointflow_dataset import ShapeNet15kPointClouds, collate_fn
from datasets.shapenet_evaluation_metrics import compute_all_metrics
from models.rapidash import Rapidash
from utils import RandomSOd, TimerCallback, subtract_mean, CosineWarmupScheduler

# Performance optimization
torch.set_float32_matmul_precision('medium')

NPOINTS = 2048

class EDMPrecond(torch.nn.Module):
    def __init__(self, model, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, hparams=None):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model
        self.hparams = hparams

    def forward(self, x, pos, edge_index, batch, sigma, rot=None):
        sigma = sigma.reshape(-1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        pos_in = c_in * pos

        # Prepare for Rapidash
        scalars = []
        vecs = []
        if self.hparams.scalar_features:
            if "coords" in self.hparams.scalar_features:
                scalars.append(pos_in)
            if "pose" in self.hparams.scalar_features and rot is not None:
                scalars.append(rot.transpose(-2,-1).unsqueeze(0).expand(pos.shape[0], -1, -1).flatten(-2, -1))
                
        if self.hparams.vector_features:
            if "coords" in self.hparams.vector_features:
                vecs.append(pos_in[:,None,:])
            if "pose" in self.hparams.vector_features and rot is not None:
                vecs.append(rot.transpose(-2,-1).unsqueeze(0).expand(pos.shape[0], -1, -1))

        scalars = torch.cat(scalars, dim=-1) if scalars else None
        vecs = torch.cat(vecs, dim=-2) if vecs else None
        
        # Ensure at least one input feature
        if scalars is None and vecs is None:
            scalars = torch.ones(pos.size(0), 1).type_as(pos)

        # Add noise conditioning
        if scalars is None:
            scalars = torch.ones_like(vecs[:,0,:1]) * c_noise
        else:
            scalars = torch.cat([scalars, c_noise.expand(scalars.shape[0], 1)], dim=-1)

        # Forward pass
        scalars_out, vecs_out = self.model(scalars, pos_in, edge_index, batch, vec=vecs)
        dpos = vecs_out[:,0,:] if self.hparams.orientations != 0 else scalars_out
        F_pos = pos_in - dpos
        
        # Apply noise-dependent skip connection
        D_pos = c_skip * pos + c_out * F_pos.to(torch.float32)
        return x, D_pos

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, normalize_x_factor=4., hparams=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.chamferdistance = ChamferDistance()
        self.hparams = hparams

    def __call__(self, net, inputs):
        pos, x, edge_index, batch, rot = inputs['pos'], inputs['x'], inputs['edge_index'], inputs['batch'], inputs['rot']
        
        if self.hparams.force_zero_mean:
            pos = subtract_mean(pos, batch)

        # Generate random noise level per point cloud
        rnd_normal = torch.randn([batch.max() + 1, 1], device=pos.device)
        rnd_normal = rnd_normal[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Add noise to positions
        if self.hparams.force_zero_mean:
            pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma
        else:
            pos_noisy = pos + torch.randn_like(pos) * sigma

        # Denoise
        _, D_pos = net(x, pos_noisy, edge_index, batch, sigma, rot)
        
        # Compute loss
        error_pos = (D_pos - pos) ** 2
        loss = (weight * error_pos).mean()
        return loss, (x, D_pos)

def edm_sampler(
    net,
    pos_0,
    x_0,
    edge_index,
    batch,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=20,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    return_intermediate=False,
    random_rot=False,
    force_zero_mean=True
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=pos_0.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    if force_zero_mean:
        pos_0 = subtract_mean(pos_0, batch)
    # Main sampling loop.
    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]
    steps = [(x_next.cpu(), pos_next.cpu())]
    
    if random_rot:
        rotation_generator = RandomSOd(3)
        rot = rotation_generator().type_as(pos_0)
    else:  
        rot = torch.eye(3, device=pos_0.device)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur, pos_cur = x_next, pos_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        pos_hat = pos_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(pos_cur)

        # Euler step.
        x_hat = torch.ones_like(x_hat)  # Shapenet fix: node features should always be ones
        x_denoised, pos_denoised = net(x_hat, pos_hat, edge_index, batch, t_hat, rot)
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next = torch.ones_like(x_next)  # Shapenet fix: node features should always be ones
            x_denoised, pos_denoised = net(x_next, pos_next, edge_index, batch, t_next, rot)
            # x_denoised = torch.ones_like(x_denoised)  # Shapenet fix: node features should always be ones
            # dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            # x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            # x_next = torch.ones_like(x_next)  # Shapenet fix: node features should always be ones
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        if force_zero_mean:
            pos_next = subtract_mean(pos_next, batch)
        steps.append((x_next.cpu(),pos_next.cpu()))

    if return_intermediate:
        return steps
    
    return x_next, pos_next


class ShapeNetGenModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.rotation_generator = RandomSOd(3)
        self.register_buffer('rescale_factor', torch.ones((1, 3), dtype=torch.float32))
        
        # Calculate input dimensions
        in_channels_scalar = (
            3 * ("coords" in self.hparams.scalar_features) +
            9 * ("pose" in self.hparams.scalar_features)
        )
        in_channels_vec = (
            1 * ("coords" in self.hparams.vector_features) +
            3 * ("pose" in self.hparams.vector_features)
        )
        
        # Ensure at least one input channel
        if in_channels_scalar + in_channels_vec == 0:
            in_channels_scalar = 1

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels_scalar + in_channels_vec,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=3 if self.hparams.orientations == 0 else 0,
            output_dim_vec=0 if self.hparams.orientations == 0 else 1,
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            dim=3,
            num_ori=self.hparams.orientations,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='node',
            last_feature_conditioning=True,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )

        # Initialize EDM components
        self.model = EDMPrecond(self.net, sigma_data=self.hparams.sigma_data, hparams=self.hparams)
        self.criterion = EDMLoss(sigma_data=self.hparams.sigma_data, 
                               normalize_x_factor=self.hparams.normalize_x_factor,
                               hparams=self.hparams)

        # For evaluation
        self.pos_ref_list = []
        self.pos_gen_list = []

    def training_step(self, batch, batch_idx):
        if self.hparams.train_augm:
            rot = self.rotation_generator().type_as(batch['pos'])
            batch['pos'] = torch.einsum('ij, bj->bi', rot, batch['pos'])
            batch['rot'] = rot
        else:
            batch['rot'] = torch.eye(3, device=batch['pos'].device)
        loss, _ = self.criterion(self.model, batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch['rot'] = torch.eye(3, device=batch['pos'].device)
        loss, _ = self.criterion(self.model, batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        samples = self.sample(self.hparams.batch_size)
        rand_idx = torch.randint(0, len(samples), (1,)).item()
        shape = samples[rand_idx][0].cpu().numpy()
        wandb.log({"val_gen_point_cloud": wandb.Object3D(shape)})

    def test_step(self, batch, batch_idx):
        pos_ref = batch['pos'].reshape(-1, NPOINTS, 3)
        samples = self.sample(pos_ref.shape[0])
        pos_generated = torch.stack([s[0] for s in samples], dim=0)
        self.pos_ref_list.append(pos_ref)
        self.pos_gen_list.append(pos_generated)

    def on_test_epoch_end(self):
        pos_ref = torch.cat(self.pos_ref_list, dim=0)
        pos_ref = pos_ref * self.rescale_factor
        pos_gen = torch.cat(self.pos_gen_list, dim=0)

        if self.hparams.force_zero_mean:
            pos_gen = pos_gen - pos_gen.mean(dim=1, keepdim=True)
            pos_ref = pos_ref - pos_ref.mean(dim=1, keepdim=True)

        results = compute_all_metrics(pos_gen, pos_ref, self.hparams.batch_size)
        self.log("1-NNA-CD", results['1-NN-CD-acc'])

        for pos in pos_gen:
            wandb.log({"test_point_cloud": wandb.Object3D(pos.cpu().numpy())})

        self.pos_ref_list.clear()
        self.pos_gen_list.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def sample(self, num_samples=100, random_rot=False):
        self.eval()
        with torch.no_grad():
            num_atoms = torch.full((num_samples,), NPOINTS, dtype=torch.long, device=self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)
            
            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            # if self.hparams.force_zero_mean:  # TODO: needed?
                # pos_0 = subtract_mean(pos_0, batch_idx)
                
            x_0 = torch.ones([len(batch_idx), 1], device=self.device)
            
            samples = edm_sampler(
                self.model, pos_0, x_0, None, batch_idx,
                num_steps=self.hparams.num_steps,
                sigma_max=self.hparams.sigma_max,
                S_churn=self.hparams.S_churn,
                random_rot=random_rot,
                force_zero_mean=self.hparams.force_zero_mean
            )
            
        return [(samples[1][batch_idx==i] * self.rescale_factor, 
                 samples[0][batch_idx==i]) for i in range(batch_idx.max()+1)]

    def set_rescale_factor(self, train_loader):
        self.rescale_factor.copy_(torch.tensor(
            train_loader.dataset.all_points_std[0] / train_loader.dataset.all_points_std_global[0],
            device=self.device
        ))

def load_data(args):
    """Load and preprocess ShapeNet dataset."""
    train_set = ShapeNet15kPointClouds(
        root=args.data_dir,
        split='train',
        categories=args.categories,
        tr_sample_size=NPOINTS,
        normalize_std_per_axis=False,
        normalize_global=True,
        random_subsample=True
    )
    
    val_set = ShapeNet15kPointClouds(
        root=args.data_dir,
        split='val',
        categories=args.categories,
        tr_sample_size=NPOINTS,
        all_points_mean=train_set[0]['mean'],
        all_points_std=train_set[0]['std'],
        all_points_std_global=train_set[0]['std_global'],
        random_subsample=True
    )
    
    if args.dataset_fraction < 1.0:
        train_set = torch.utils.data.Subset(
            train_set,
            torch.randperm(len(train_set))[:int(len(train_set)*args.dataset_fraction)]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    train_loader, val_loader = load_data(args)

    # Setup hardware configuration
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
        
    # Configure logging
    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project="Rapidash-ShapeNet-Gen",
            name=f"{args.model}-EDM",
            config=args,
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [TimerCallback()]
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=args.enable_progress_bar,
        check_val_every_n_epoch=max(1, int(1/args.dataset_fraction))
    )

    # Train or test
    if args.test_ckpt is None:
        model = ShapeNetGenModel(args)
        model.set_rescale_factor(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, val_loader)  # Following LION paper, use validation set for testing
    else:
        model = ShapeNetGenModel.load_from_checkpoint(args.test_ckpt)
        model.set_rescale_factor(train_loader)
        model.hparams.S_churn = args.S_churn
        model.hparams.sigma_max = args.sigma_max
        model.hparams.num_steps = args.num_steps
        model.hparams.batch_size = args.batch_size
        trainer.test(model, val_loader)  # Following LION paper, use validation set for testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShapeNet Point Cloud Generation Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Model architecture
    parser.add_argument('--model', type=str, default="rapidash", help='Model type')
    parser.add_argument('--hidden_dim', type=eval, default=[256,256,256,256,256], help='Hidden dimensions')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--orientations', type=int, default=0, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--layers', type=eval, default=[0,1,1,1,1], help='Layers per scale')
    parser.add_argument('--edge_types', type=eval, default=["knn-8","knn-8","knn-8","knn-8","fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[0.25,0.25,0.25,0.25], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="Tn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=True, help='Use skip connections')
    
    # Model features
    parser.add_argument('--scalar_features', type=eval, default=["coords"], help='Features to use as scalars')
    parser.add_argument('--vector_features', type=eval, default=[], help='Features to use as vectors')
    parser.add_argument('--force_zero_mean', type=eval, default=True, help='Force zero mean for point clouds')
    
    # Diffusion parameters
    parser.add_argument('--S_churn', type=float, default=0, help='Noise level')
    parser.add_argument('--sigma_max', type=float, default=20, help='Maximum sigma')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--sigma_data', type=float, default=1, help='Data sigma')
    parser.add_argument('--normalize_x_factor', type=float, default=4.0, help='Feature normalization factor')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=False, help='Use rotation augmentation')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/shapenet", help='Data directory')
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--categories', type=eval, default=['airplane', 'chair', 'car', 'lamp', 'table', 'sofa', 'cabinet', 'bench', 'telephone', 'speaker', 'monitor', 'vessel', 'rifle'], help='Categories to train on')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID for configuration labeling')
    
    args = parser.parse_args()
    
    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    # Adjust epochs based on dataset fraction
    args.epochs = int(args.epochs / args.dataset_fraction)
    
    main(args)