import os
import argparse

import torch
import torchmetrics
import pytorch_lightning as pl
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

from models.rapidash import Rapidash
from utils import (CosineWarmupScheduler, NormalizeCoord, RandomJitter,
                   RandomRotatePerturbation, RandomShift, RandomSOd,
                   SamplePoints, TimerCallback)

# Performance optimization
torch.set_float32_matmul_precision('medium')

# Some augmentation functions
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, RandomScale


class ModelNet40Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels = (
            3 * ("coords" in self.hparams.scalar_features) +  # x,y,z coordinates as scalars
            3 * ("normals" in self.hparams.scalar_features) +  # normal components as scalars
            1 * ("coords" in self.hparams.vector_features) +  # position as vector
            1 * ("normals" in self.hparams.vector_features) +  # normal as vector
            3 * ("pose" in self.hparams.vector_features)      # pose matrix (3 vectors)
        )

        # Ensure at least one input channel if none are specified
        if in_channels == 0:
            in_channels = 1  # will use constant ones as input

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=40,  # ModelNet40 has 40 classes
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
        
        # Setup metrics
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=40)

    def forward(self, data):
        # Apply rotation augmentation if enabled (during training)
        if self.training and self.hparams.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
            if hasattr(data, 'normal'):
                data.normal = torch.einsum('ij,bj->bi', rot, data.normal)
        else:
            rot = torch.eye(3, device=data.pos.device)

        # Prepare input features
        x = []  # scalar features
        vec = []  # vector features

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "normals" in self.hparams.scalar_features and hasattr(data, 'normal'):
            x.append(data.normal)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "normals" in self.hparams.vector_features and hasattr(data, 'normal'):
            vec.append(data.normal[:,None,:])
        if "pose" in self.hparams.vector_features:
            vec.append(rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1))

        # Combine features
        if not x and not vec:  # Only add constant ones if both x and vec are empty
            x = torch.ones(data.pos.size(0), 1).type_as(data.pos)
        else:
            x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None

        # Forward pass
        pred, _ = self.net(x, data.pos, data.edge_index, data.batch, vec=vec)
        return pred

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = torch.nn.functional.cross_entropy(pred, data.y)
        self.train_metric(pred, data.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        self.valid_metric(pred, data.y)

    def test_step(self, data, batch_idx):
        pred = self(data)
        self.test_metric(pred, data.y)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_data(args):
    """Load and preprocess ModelNet40 dataset using PyG."""
    
    # Define transforms
    train_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
        RandomRotatePerturbation(angle_sigma=0.06, angle_clip=0.18),
        RandomScale((0.8, 1.25)),
        RandomShift(shift_range=0.1),
        RandomJitter(sigma=0.01, clip=0.05),
    ])

    test_transform = T.Compose([
        NormalizeCoord(),
        SamplePoints(num=args.num_points, remove_faces=True, include_normals=True),
    ])
    
    # Create datasets
    train_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=True,
        transform=train_transform,
    )
    
    test_dataset = ModelNet(
        args.data_dir,
        name='40',
        train=False,
        transform=test_transform,
    )
   
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    train_loader, test_loader = load_data(args)

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
            project="Rapidash-ModelNet40",
            config=args,
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='valid_acc',
            mode='max',
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

    # Train or test
    if args.test_ckpt is None:
        model = ModelNet40Model(args)
        trainer.fit(model, train_loader, test_loader)
        # Use last model (not the best) because we cannot use the test set for model selection
        trainer.test(model, test_loader, ckpt_path=callbacks[0].last_model_path)  
    else:
        model = ModelNet40Model.load_from_checkpoint(args.test_ckpt)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ModelNet40 Classification Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=[128,128,128,128,128], help='Hidden dimension(s)')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--layers', type=eval, default=[1,1,1,1,1], help='Layers per scale')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["knn-8","knn-8","knn-8","knn-8","fc" ], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[0.25,0.25,0.25,0.25], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use U-Net style skip connections')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=[], help='Features to use as scalars: ["coords", "normals"]')
    parser.add_argument('--vector_features', type=eval, default=["normal","pose"], help='Features to use as vectors: ["coords", "normals", "pose"]')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample')

    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/modelnet", help='Data directory')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    
    # System and checkpointing
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    args = parser.parse_args()

    # Overwrite default settings with values from config if provided
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)