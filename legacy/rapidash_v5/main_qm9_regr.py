import os
import argparse

import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.rapidash import Rapidash
from utils import CosineWarmupScheduler, RandomSOd, TimerCallback

# Performance optimizations
torch.set_float32_matmul_precision('medium')


class QM9Model(pl.LightningModule):
    """Lightning module for QM9 molecular property prediction."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels = (
            11 +  # base atom features
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            1 * ("coords" in self.hparams.vector_features)    # position as vector
        )

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=1,  # Single target property prediction
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            output_dim_vec=0,
            dim=3,
            num_ori=self.hparams.orientations,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='graph',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )
        
        # Initialize normalization parameters
        self.shift = 0.
        self.scale = 1.
        
        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, graph):
        # Prepare input features
        x = []
        vec = []
        
        # Add base atomic features
        x.append(graph.x)
        
        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(graph.pos)
            
        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(graph.pos[:,None,:])
            
        # Combine features
        x = torch.cat(x, dim=-1) if x else torch.ones(graph.pos.size(0), 1).type_as(graph.pos)
        vec = torch.cat(vec, dim=1) if vec else None
            
        # Forward pass
        pred, _ = self.net(x, graph.pos, graph.edge_index, graph.batch, vec=vec)
        return pred.squeeze(-1)
    
    def set_dataset_statistics(self, dataloader):
        """Compute mean and standard deviation of target property."""
        print('Computing dataset statistics...')
        ys = []
        for data in dataloader:
            ys.append(data.y)
        ys = np.concatenate(ys)
        self.shift = np.mean(ys)
        self.scale = np.std(ys)
        print(f'Target statistics - Mean: {self.shift:.4f}, Std: {self.scale:.4f}')

    def training_step(self, graph, batch_idx):
        # Apply rotation augmentation if enabled
        if self.hparams.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            
        # Forward pass and loss computation
        pred = self(graph)
        loss = torch.mean(torch.abs(pred - (graph.y - self.shift) / self.scale))
        self.train_metric(pred * self.scale + self.shift, graph.y)
        return loss

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred * self.scale + self.shift, graph.y)

    def test_step(self, graph, batch_idx):
        pred = self(graph)
        self.test_metric(pred * self.scale + self.shift, graph.y)

    def on_train_epoch_end(self):
        self.log("train MAE", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid MAE", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test MAE", self.test_metric, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay and learning rate schedule."""
        # Separate parameters into decay and no-decay groups
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias') or pn.endswith('layer_scale'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate parameter grouping
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not grouped!"

        # Create optimizer and scheduler
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.lr)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def load_data(args):
    """Load and preprocess QM9 dataset."""
    # Load dataset
    dataset = QM9(root=args.data_dir)
    
    # Create train/val/test split (same as DimeNet)
    random_state = np.random.RandomState(seed=42)
    perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
    train_idx, val_idx, test_idx = perm[:110000], perm[110000:120000], perm[120000:]
    datasets = {'train': dataset[train_idx], 'val': dataset[val_idx], 'test': dataset[test_idx]}
    
    # Select target property
    targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
               'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11, 12, 13, 14, 15])
    dataset.data.y = dataset.data.y[:, idx]
    dataset.data.y = dataset.data.y[:, targets.index(args.target)]

    # Create dataloaders
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, 
                         shuffle=(split == 'train'), 
                         num_workers=args.num_workers)
        for split, dataset in datasets.items()
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    train_loader, val_loader, test_loader = load_data(args)

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
            project="Rapidash-QM9-Regr",
            name=None,  # Let wandb auto-generate the run name
            config=args, 
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='valid MAE', mode='min', 
                                   every_n_epochs=1, save_last=True),
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

    # Train or test
    if args.test_ckpt is None:
        model = QM9Model(args)
        model.set_dataset_statistics(train_loader)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, test_loader, ckpt_path=callbacks[0].best_model_path)
    else:
        model = QM9Model.load_from_checkpoint(args.test_ckpt)
        model.set_dataset_statistics(train_loader)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QM9 Property Prediction Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--basis_dim', type=int, default=None, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--layers', type=int, default=7, help='Number of layers')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use U-Net style skip connections')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=[], help='Features to use as scalars: ["coords"]')
    parser.add_argument('--vector_features', type=eval, default=[], help='Features to use as vectors: ["coords"]')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/qm9", help='Data directory')
    parser.add_argument('--target', type=str, default="homo", help='Target property to predict')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')

    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()

    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)

    main(args)