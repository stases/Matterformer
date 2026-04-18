import os
import argparse

import torch
import torchmetrics
import pytorch_lightning as pl

from datasets.motion import MotionDataset
from models.rapidash import Rapidash
from utils import (CosineWarmupScheduler, RandomSO2AroundAxis, RandomSOd,
                   TimerCallback)

# Performance optimization
torch.set_float32_matmul_precision('medium')


class CMUMotionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        self.z_rotation_generator = RandomSO2AroundAxis(axis=1, degrees=15)  # axis 1 is  the correct axis not the axis z
        # self.n_joints = 31
        
        # Calculate input dimensions
        in_channels = (
            3 * ("coords" in self.hparams.scalar_features) +
            3 * ("velocity" in self.hparams.scalar_features) +
            3 * ("normals" in self.hparams.scalar_features) +
            9 * ("pose" in self.hparams.scalar_features) +
            1 * ("coords" in self.hparams.vector_features) +
            1 * ("velocity" in self.hparams.vector_features) +
            1 * ("normals" in self.hparams.vector_features) +
            3 * ("pose" in self.hparams.vector_features)
        )
        
        # Ensure at least one input channel
        if in_channels == 0:
            in_channels = 1
            
        # Calculate output dimensions
        out_channels_scalar = 3 if self.hparams.orientations == 0 else 0
        out_channels_vec = 0 if self.hparams.orientations == 0 else 1

        # Initialize model
        self.net = Rapidash(
            input_dim=in_channels,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=out_channels_scalar,
            output_dim_vec=out_channels_vec,
            num_layers=self.hparams.layers,
            edge_types=self.hparams.edge_types,
            equivariance=self.hparams.equivariance,
            ratios=self.hparams.ratios,
            dim=3,
            num_ori=self.hparams.orientations,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim,
            degree=self.hparams.degree,
            widening_factor=self.hparams.widening,
            layer_scale=self.hparams.layer_scale,
            task_level='node',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections
        )

        # Setup metrics
        self.train_metric = torchmetrics.MeanSquaredError()
        self.valid_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()

    def forward(self, data):
        # Prepare features
        x = []  # scalar features
        vec = []  # vector features
        
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "velocity" in self.hparams.scalar_features:
            x.append(data.vel)
        if "normals" in self.hparams.scalar_features:
            x.append(data.normals)
        if "pose" in self.hparams.scalar_features:
            x.append(data.rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1).flatten(-2, -1))
            
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "velocity" in self.hparams.vector_features:
            vec.append(data.vel[:,None,:])
        if "normals" in self.hparams.vector_features:
            vec.append(data.normals[:,None,:])
        if "pose" in self.hparams.vector_features:
            vec.append(data.rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1))
            # vec.append(data.rot.unsqueeze(0).expand(data.pos.shape[0], -1, -1))

        # Combine features
        x = torch.cat(x, dim=-1) if x else None
        vec = torch.cat(vec, dim=1) if vec else None
        x = torch.ones(data.pos.size(0), 1).type_as(data.pos) if (x is None and vec is None) else x

        # Forward pass
        pred_scalar, pred_vec = self.net(x, data.pos, data.edge_index, data.batch, vec=vec)
        
        # Process output
        if self.hparams.orientations == 0:
            delta = pred_scalar
        else:
            delta = pred_vec[:,0,:]
            
        return data.pos + delta

    def training_step(self, batch, batch_idx):
        # Apply rotation augmentation if enabled
        if self.hparams.train_augm:
            if self.hparams.z_augm:
                rot = self.z_rotation_generator().type_as(batch.pos)
            else:
                rot = self.rotation_generator().type_as(batch.pos)
            batch.pos = torch.einsum('ij,bj->bi', rot, batch.pos)
            batch.vel = torch.einsum('ij,bj->bi', rot, batch.vel)
            batch.normals = torch.einsum('ij,bj->bi', rot, batch.normals)
            batch.y = torch.einsum('ij,bj->bi', rot, batch.y)
            batch.rot = rot
        else:
            batch.rot = torch.eye(3, device=batch.pos.device)

        y_pred = self(batch)
        loss = torch.nn.functional.mse_loss(y_pred, batch.y)
        self.train_metric(y_pred.contiguous(), batch.y.contiguous())
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.test_augm:
            if self.hparams.z_augm:
                rot = self.z_rotation_generator().type_as(batch.pos)
            else:
                rot = self.rotation_generator().type_as(batch.pos)
            batch.pos = torch.einsum('ij,bj->bi', rot, batch.pos)
            batch.vel = torch.einsum('ij,bj->bi', rot, batch.vel)
            batch.normals = torch.einsum('ij,bj->bi', rot, batch.normals)
            batch.y = torch.einsum('ij,bj->bi', rot, batch.y)
            batch.rot = rot
        else:
            batch.rot = torch.eye(3, device=batch.pos.device)

        y_pred = self(batch)
        self.valid_metric(y_pred.contiguous(), batch.y.contiguous())

    def test_step(self, batch, batch_idx):
        if self.hparams.test_augm:
            if self.hparams.z_augm:
                rot = self.z_rotation_generator().type_as(batch.pos)
            else:
                rot = self.rotation_generator().type_as(batch.pos)
            batch.pos = torch.einsum('ij,bj->bi', rot, batch.pos)
            batch.vel = torch.einsum('ij,bj->bi', rot, batch.vel)
            batch.normals = torch.einsum('ij,bj->bi', rot, batch.normals)
            batch.y = torch.einsum('ij,bj->bi', rot, batch.y)
            batch.rot = rot
        else:
            batch.rot = torch.eye(3, device=batch.pos.device)
        y_pred = self(batch)
        self.test_metric(y_pred.contiguous(), batch.y.contiguous())

    def on_train_epoch_end(self):
        self.log("train_mse", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_mse", self.valid_metric, prog_bar=True)

    def on_test_epoch_end(self):
        if self.hparams.test_augm:
            if self.hparams.z_augm:
                prefix = "test_z_rot"
            else:
                prefix = "test_full_rot"
        else:
            prefix = "test"
        self.log(f"{prefix}_mse", self.test_metric, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=self.hparams.lr, 
                                    weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, 
                                        self.hparams.warmup, 
                                        self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main(args):
    # Set random seed
    pl.seed_everything(args.seed)

    # Load data
    dataset = MotionDataset(
        batch_size=args.batch_size,
        all_joint_normals=args.all_joint_normals,
        num_training_samples=args.max_train_sample
    )
    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    # Configure hardware
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
            project="Rapidash-CMU-Motion",
            name=None,
            config=args,
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val_mse',
            mode='min',
            every_n_epochs=1,
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
        model = CMUMotionModel(args)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        # Test without augmentation
        trainer.test(model, test_loader, ckpt_path=callbacks[0].best_model_path)
        # Test with z-axis rotation
        model.hparams.test_augm = True
        model.hparams.z_augm = True
        trainer.test(model, test_loader, ckpt_path=callbacks[0].best_model_path)
        # Test with full rotation
        model.hparams.z_augm = False
        trainer.test(model, test_loader, ckpt_path=callbacks[0].best_model_path)
    else:
        model = CMUMotionModel.load_from_checkpoint(args.test_ckpt)
        # Test without augmentation
        trainer.test(model, test_loader)
        # Test with z-axis rotation
        model.hparams.test_augm = True
        model.hparams.z_augm = True
        trainer.test(model, test_loader)
        # Test with full rotation
        model.hparams.z_augm = False
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMU Motion Prediction Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-12)
    parser.add_argument('--seed', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=256, help='Hidden dimension(s)')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of basis MLP')
    parser.add_argument('--layers', type=eval, default=7, help='Layers per scale')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use skip connections')
    
    # Feature configuration
    parser.add_argument('--scalar_features', type=eval, default=[], 
                       help='Features to use as scalars: ["coords", "velocity", "normals", "pose"]')
    parser.add_argument('--vector_features', type=eval, default=["velocity","pose"], 
                       help='Features to use as vectors: ["coords", "velocity", "normals", "pose"]')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance: "Tn" or "SEn"')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--test_augm', type=eval, default=False, help='Use rotation augmentation during testing')
    parser.add_argument('--z_augm', type=eval, default=False, help='Use only z-axis rotation for augmentations')
    
    # Dataset configuration
    parser.add_argument('--root', type=str, default="./datasets/motion", help='Data directory')
    parser.add_argument('--max_train_sample', type=int, default=5000, help='Maximum number of training samples')
    parser.add_argument('--all_joint_normals', type=eval, default=False, help='Use normals for all joints')
    
    # System and logging
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--enable_progress_bar', type=eval, default=True, help='Show progress bar')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    
    # Checkpointing
    parser.add_argument('--test_ckpt', type=str, default=None, help='Checkpoint for testing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Checkpoint to resume from')
    
    # Configuration overrides
    parser.add_argument('--config', type=eval, default=None, help='Dictionary of parameter overrides')
    
    args = parser.parse_args()

    # Apply configuration overrides
    if args.config is not None:
        for key, value in args.config.items():
            setattr(args, key, value)
    
    main(args)