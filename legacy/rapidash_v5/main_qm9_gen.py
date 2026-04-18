import os
import argparse

import torch
import numpy as np
from tqdm import trange
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.qm9 import QM9Dataset, collate_fn
from datasets.qm9_bond_analyze import check_stability
from datasets.qm9_rdkit_utils import BasicMolecularMetrics
from models.rapidash import Rapidash
from utils import (CosineWarmupScheduler, RandomSOd, TimerCallback,
                   fully_connected_edge_index, subtract_mean)

# Performance optimization
torch.set_float32_matmul_precision('medium')

class EDMPrecond(torch.nn.Module):
    """Preconditioning module for EDM (Equivariant Diffusion Model)."""
    def __init__(
        self,
        model: torch.nn.Module,
        sigma_min: float = 0,
        sigma_max: float = float('inf'),
        sigma_data: float = 0.5,
        hparams = None
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model
        self.hparams = hparams

    def forward(self, x, pos, edge_index, batch, sigma, rot=None):
        sigma = sigma.reshape(-1, 1)

        # Calculate skip connection and scaling factors
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        pos_in = c_in * pos

        # Prepare features
        scalars = []
        vecs = []
        
        if "coords" in self.hparams.scalar_features:
            scalars.append(pos_in)
        if "coords" in self.hparams.vector_features:
            vecs.append(pos_in[:,None,:])
        # The current scalar features that need to be denoised are always added
        scalars.append(x_in)  
            
        scalars = torch.cat(scalars, dim=-1) if scalars else torch.ones(pos_in.size(0), 1).type_as(pos_in)
        vecs = torch.cat(vecs, dim=1) if vecs else None

        # Add noise conditioning
        scalars = torch.cat([scalars, c_noise.expand(scalars.shape[0], 1)], dim=-1)

        # Model forward pass
        scalars_out, vecs_out = self.model(scalars, pos_in, edge_index, batch, vec=vecs)
        if self.hparams.orientations > 0:
            dpos = vecs_out[:,0,:]
            dx = scalars_out
        else:
            dpos = scalars_out[:,:3]
            dx = scalars_out[:,3:]

        # Apply denoising and skip connection
        F_x = x_in - dx
        F_pos = pos_in - dpos
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_pos = c_skip * pos + c_out * F_pos.to(torch.float32)
        
        return D_x, D_pos
    
class EDMLoss:
    """Loss function for EDM training."""
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5, 
        normalize_x_factor=4.,
        normalize_charge_factor=8.,
        hparams=None
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.normalize_x_factor = normalize_x_factor
        self.normalize_charge_factor = normalize_charge_factor
        self.hparams = hparams
        self.use_charges = hparams.use_charges

    def __call__(self, net, inputs):
        pos, x, edge_index, batch = inputs['pos'], inputs['x'], inputs['edge_index'], inputs['batch']
        pos = subtract_mean(pos, batch)
        
        # Normalize features
        if self.use_charges:
            x[:,:-1] = x[:,:-1] / self.normalize_x_factor
            x[:,-1] = x[:,-1] / self.normalize_charge_factor
        else:
            x = x / self.normalize_x_factor

        # Random noise level
        rnd_normal = torch.randn([batch.max() + 1, 1], device=pos.device, dtype=torch.float32)
        rnd_normal = rnd_normal[batch]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Add noise to inputs
        x_noisy = x + torch.randn_like(x) * sigma
        pos_noisy = pos + subtract_mean(torch.randn_like(pos), batch) * sigma

        # Compute denoised predictions and loss
        D_x, D_pos = net(x_noisy, pos_noisy, edge_index, batch, sigma)
        error_pos = (D_pos - pos) ** 2
        error_x = (D_x - x) ** 2
        loss = (weight * error_x).mean() + (weight * error_pos).mean()

        return loss, (D_x, D_pos)

def sampler(
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
):
    """EDM sampling function with Euler-Maruyama SDE solver."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=pos_0.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # Main sampling loop
    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]
    steps = [(x_next.cpu(), pos_next.cpu())]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, pos_cur = x_next, pos_next

        # Increase noise temporarily
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        pos_hat = pos_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(pos_cur)

        # Euler step
        x_denoised, pos_denoised = net(x_hat, pos_hat, edge_index, batch, t_hat)
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        # Second-order correction
        if i < num_steps - 1:
            x_denoised, pos_denoised = net(x_next, pos_next, edge_index, batch, t_next)
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        steps.append((x_next.cpu(), pos_next.cpu()))

    if return_intermediate:
        return steps
    
    pos_next = subtract_mean(pos_next, batch)
    return x_next, pos_next

class QM9Model(pl.LightningModule):
    """PyTorch Lightning module for QM9 molecule generation."""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.validation_frequency = getattr(args, 'validation_frequency', 20)  # Default to 20 if not specified
        
        # Calculate input channels
        in_channels = (
            5 + 3 * ("coords" in self.hparams.scalar_features) + 
            1 * ("coords" in self.hparams.vector_features) + 
            1 * self.hparams.use_charges
        )

        # Calculate output channels
        out_channels_scalar = 5 + 1 * self.hparams.use_charges + (3 if self.hparams.orientations == 0 else 0)
        out_channels_vec = 1 if self.hparams.orientations > 0 else 0


        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=out_channels_scalar,
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            output_dim_vec=out_channels_vec,
            dim=3,
            num_ori=self.hparams.orientations,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='node',
            last_feature_conditioning=True,
            skip_connections=self.hparams.skip_connections,
        )
        
        # Setup EDM components
        self.model = EDMPrecond(self.net, sigma_data=self.hparams.sigma_data, hparams=self.hparams)
        self.criterion = EDMLoss(sigma_data=self.hparams.sigma_data, hparams=self.hparams)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        self.sampler = sampler

    def set_num_atoms_sampler(self, num_atoms_sampler):
        self.num_atoms_sampler = num_atoms_sampler

    def init_molecule_analyzer(self, dataset_info, smiles_list):
        self.molecule_analyzer = BasicMolecularMetrics(dataset_info, smiles_list)
    
    def _evaluate_batch_metrics(self, molecules):
        """Evaluate stability metrics for a batch of molecules."""
        count_mol_stable = count_atm_stable = count_mol_total = count_atm_total = 0
        
        for mol in molecules:
            is_stable, nr_stable, total = check_stability(*mol)
            count_atm_stable += nr_stable
            count_atm_total += total
            count_mol_stable += int(is_stable)
            count_mol_total += 1
            
        return {
            "atom_stability": 100. * count_atm_stable/count_atm_total,
            "molecule_stability": 100. * count_mol_stable/count_mol_total
        }

    def _log_metrics(self, results, prefix=""):
        """Log metrics with optional prefix."""
        for key, value in results.items():
            self.log(f"{prefix}{key}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        if self.hparams.train_augm:
            rot = self.rotation_generator().type_as(batch['pos'])
            batch['pos'] = torch.einsum('ij, bj->bi', rot, batch['pos'])
        loss, _ = self.criterion(self.model, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.criterion(self.model, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch['batch'].max()+1)
        return loss
    
    def on_validation_epoch_end(self):
        full_validation = (self.current_epoch + 1) % self.validation_frequency == 0
        
        results = self.validate(
            num_molecules=10000 if full_validation else self.hparams.batch_size,
            batch_size=self.hparams.batch_size,
            rdkit_metrics=full_validation
        )
        
        # Add estimate suffix for quick validations
        if not full_validation:
            results = {f"{k} (estimate)": v for k, v in results.items()}
            
        self._log_metrics(results)
        return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        return None
    
    def on_test_epoch_end(self):
        results = self.validate(
            num_molecules=10000, 
            batch_size=self.hparams.batch_size, 
            rdkit_metrics=True
        )
        self._log_metrics(results, prefix="final/")
        return super().on_test_epoch_end()
  
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def sample(self, num_molecules=100):
        """Generate molecular samples."""
        self.eval()
        with torch.no_grad():
            num_atoms = self.num_atoms_sampler(num_molecules).to(self.device)
            batch_indices = torch.arange(len(num_atoms), device=self.device)
            batch_idx = torch.repeat_interleave(batch_indices, num_atoms)
            
            # Initialize random positions and features
            pos_0 = torch.randn([len(batch_idx), 3], device=self.device)
            pos_0 = subtract_mean(pos_0, batch_idx)
            x_0 = torch.randn([len(batch_idx), 5 + 1 * self.hparams.use_charges], device=self.device)
            edge_index = fully_connected_edge_index(batch_idx)
            
            # Sample using EDM
            samples = self.sampler(
                self.model, 
                pos_0, 
                x_0, 
                edge_index, 
                batch_idx, 
                S_churn=self.hparams.S_churn, 
                num_steps=self.hparams.num_steps, 
                sigma_max=self.hparams.sigma_max
            )
            
        # Convert to list of molecules
        sample_list = []
        for i in range(batch_idx.max()+1):
            positions = samples[1][batch_idx==i]
            if self.hparams.use_charges:
                atom_types = samples[0][batch_idx==i,:-1].argmax(dim=-1)
                charges = (samples[0][batch_idx==i,-1] * self.hparams.normalize_charge_factor).round().long()
                sample_list.append((positions, atom_types, charges))
            else:
                atom_types = samples[0][batch_idx==i].argmax(dim=-1)
                sample_list.append((positions, atom_types))
        return sample_list
    
    def validate(self, num_molecules=10000, batch_size=100, rdkit_metrics=True):
        """Validate generated molecules using stability and RDKit metrics."""
        # Generate molecules in batches
        steps = num_molecules // batch_size
        molecules = []
        for _ in trange(steps):
            molecules += self.sample(batch_size)

        # Get stability metrics
        results_dict = self._evaluate_batch_metrics(molecules)

        # Add RDKit metrics if requested
        if rdkit_metrics:
            [validity, uniqueness, novelty], _ = self.molecule_analyzer.evaluate(molecules)
            results_dict.update({
                "validity": validity,
                "uniqueness": uniqueness,
                "novelty": novelty,
                "discovery": validity * uniqueness * novelty
            })

        return results_dict

def load_data(args):
    """Load and preprocess QM9 dataset."""
    train_set = QM9Dataset(
        split='train', 
        root=args.data_dir, 
        use_charges=args.use_charges
    )
    val_set = QM9Dataset(
        split='val', 
        root=args.data_dir, 
        use_charges=args.use_charges
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
    
    num_atoms_sampler = train_set.NumAtomsSampler()
    smiles_list = [train_set[i]['smiles'] for i in range(len(train_set))]
    dataset_info = train_set.dataset_info
    
    return train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info

def main(args):
    """Main training/testing function."""
    # Set random seed
    pl.seed_everything(42)

    # Load data
    train_loader, val_loader, num_atoms_sampler, smiles_list, dataset_info = load_data(args)

    # Configure hardware settings
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
        
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # Configure logging
    if args.log:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            project="Rapidash-QM9-Gen",
            name=None,
            config=args, 
            save_dir=save_dir
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='molecule_stability',
            mode='max',
            every_n_epochs=args.validation_frequency,
            save_last=True
        ),
        TimerCallback()
    ]
    
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
        enable_progress_bar=args.enable_progress_bar
    )

    # Train or test model
    if args.test_ckpt is None:
        # Training mode
        model = QM9Model(args)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, val_loader)
        # Test again with best checkpoint
        trainer.test(model, val_loader, ckpt_path=callbacks[0].best_model_path)
    else:
        # Testing mode
        model = QM9Model.load_from_checkpoint(args.test_ckpt)
        model.set_num_atoms_sampler(num_atoms_sampler)
        model.init_molecule_analyzer(dataset_info, smiles_list)
        
        # Update sampling parameters
        model.hparams.S_churn = args.S_churn
        model.hparams.sigma_max = args.sigma_max
        model.hparams.num_steps = args.num_steps
        model.hparams.batch_size = args.batch_size
        
        trainer.test(model, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QM9 Molecular Geometry Generation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--seed', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=256, help='Hidden dimension(s)')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--layers', type=eval, default=9, help='Layers per scale')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use U-Net style skip connections')
    
    # EDM parameters
    parser.add_argument('--S_churn', type=float, default=10, help='Noise level for EDM sampling')
    parser.add_argument('--sigma_max', type=float, default=1, help='Maximum sigma for EDM')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--sigma_data', type=float, default=1, help='Data sigma for EDM')
    
    # Feature configuration
    parser.add_argument('--use_charges', type=eval, default=True, help='Use atomic charges')
    parser.add_argument('--scalar_features', type=eval, default=[], help='Features to use as scalars: ["coords"]')
    parser.add_argument('--vector_features', type=eval, default=[], help='Features to use as vectors: ["coords"]')
    parser.add_argument('--normalize_x_factor', type=float, default=4.0, help='Feature normalization factor')
    parser.add_argument('--normalize_charge_factor', type=float, default=8.0, help='Charge normalization factor')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--validation_frequency', type=int, default=20, help='Validation frequency in epochs')
    
    # System and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/qm9", help='Data directory')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--enable_progress_bar', type=eval, default=True)
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    
    # Configuration overrides
    parser.add_argument('--config', type=eval, default=None, help='Dictionary of parameter overrides')
    
    args = parser.parse_args()

    # Apply configuration overrides
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)