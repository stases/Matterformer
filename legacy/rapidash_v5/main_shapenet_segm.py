import os
import argparse

import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
import torch_geometric.transforms as T

from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torchmetrics.functional import jaccard_index

from models.rapidash import Rapidash
from utils import CosineWarmupScheduler, RandomSOd, TimerCallback

# Performance optimization
torch.set_float32_matmul_precision('medium')

class RandomRotateWithNormals(torch.nn.Module):
    """Rotates node positions and normal vectors randomly."""
    def __init__(self, degrees, axis=0):
        super().__init__()
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-abs(degrees), abs(degrees))
        self.axis = axis

    def forward(self, data):
        degree = np.pi * np.random.uniform(*self.degrees) / 180.0
        sin, cos = np.sin(degree), np.cos(degree)

        if self.axis == 0:
            matrix = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        elif self.axis == 1:
            matrix = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        else:
            matrix = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

        data.pos = torch.matmul(data.pos, matrix.to(data.pos.dtype).to(data.pos.device))
        data.x = torch.matmul(data.x, matrix.to(data.x.dtype).to(data.x.device))
        return data

class ShapeNetModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        self.n_parts = 50  # Total number of part categories across all shape types
        
        # Calculate total input channels
        in_channels = (
            16 * self.hparams.use_category +  # one-hot encoded category
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
            output_dim=self.n_parts,
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
            task_level='node',
            last_feature_conditioning=False,
            skip_connections=self.hparams.skip_connections,
            basis_dim=self.hparams.basis_dim,
            basis_hidden_dim=self.hparams.basis_hidden_dim
        )
        
        # Setup metrics
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_parts)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_parts)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_parts)
        
        # For IoU tracking
        self.ious = []
        self.categories = []

    def forward(self, data):
        # Apply rotation augmentation if enabled (during training or testing)
        if (self.training and self.hparams.train_augm) or (not self.training and self.hparams.test_augm):
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
            data.x = torch.einsum('ij,bj->bi', rot, data.x)
        else:
            rot = torch.eye(3, device=data.pos.device)

        # Prepare input features
        x = []  # scalar features
        vec = []  # vector features

        # Add category information
        if self.hparams.use_category:
            category = torch.nn.functional.one_hot(data.category, num_classes=16).float()
            x.append(category[data.batch])

        # Add scalar features
        if "coords" in self.hparams.scalar_features:
            x.append(data.pos)
        if "normals" in self.hparams.scalar_features:
            x.append(data.x)

        # Add vector features
        if "coords" in self.hparams.vector_features:
            vec.append(data.pos[:,None,:])
        if "normals" in self.hparams.vector_features:
            vec.append(data.x[:,None,:])
        if "pose" in self.hparams.vector_features:  # Add pose vector feature support
            vec.append(rot.transpose(-2,-1).unsqueeze(0).expand(data.pos.shape[0], -1, -1))

        # Combine features
        x = torch.cat(x, dim=-1) if x else torch.ones(data.pos.size(0), 1).type_as(data.pos)
        vec = torch.cat(vec, dim=1) if vec else None

        # Forward pass
        pred, _ = self.net(x, data.pos, data.edge_index, data.batch, vec=vec)
        return pred

    def training_step(self, graph, batch_idx):
        pred = self(graph)
        loss = torch.nn.functional.cross_entropy(pred, graph.y)
        self.train_metric(pred, graph.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred, graph.y)
        
        # Compute per-shape IoU
        sizes = (graph.ptr[1:] - graph.ptr[:-1]).tolist()
        y_map = torch.empty(50, device=pred.device).long()
        
        for y_pred, y, category_idx in zip(pred.split(sizes), graph.y.split(sizes), graph.category.tolist()):
            category_str = list(ShapeNet.seg_classes.keys())[category_idx]
            part = torch.tensor(ShapeNet.seg_classes[category_str], device=pred.device)
            y_map[part] = torch.arange(part.size(0), device=pred.device)
            
            # Compute per-part IoUs
            iou_per_part = jaccard_index(
                y_pred[:, part].argmax(dim=-1), 
                y_map[y], 
                task="multiclass",
                num_classes=part.size(0),
                average="none"
            )
            
            # Set IoU=1 for parts that are neither in ground truth nor predicted
            absent_parts = list(set(range(part.size(0))) - 
                            (set(y_map[y].cpu().numpy()) | 
                            set(y_pred[:,part].argmax(dim=-1).cpu().numpy())))
            iou_per_part[absent_parts] = 1.0
            
            # Average over parts
            iou = iou_per_part.mean()
            self.ious.append(iou)
            
        self.categories.append(graph.category)

    def test_step(self, graph, batch_idx):
        return self.validation_step(graph, batch_idx)

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_metric, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_metric, prog_bar=True)
        
        iou = torch.stack(self.ious)
        categories = torch.cat(self.categories)
        class_ious = scatter(iou, categories, reduce='mean')
        
        self.log("valid_iou_class", class_ious.mean(), prog_bar=True)
        self.log("valid_iou_instance", iou.mean(), prog_bar=True)
        
        for i, category in enumerate(ShapeNet.seg_classes.keys()):
            self.log(f"valid_iou_{category}", class_ious[i])
            
        self.ious.clear()
        self.categories.clear()

    def on_test_epoch_end(self):
        # Add suffix based on whether test augmentation is enabled
        suffix = "_rotated" if self.hparams.test_augm else ""
            
        self.log(f"test_acc{suffix}", self.test_metric, prog_bar=True)
        
        iou = torch.stack(self.ious)
        categories = torch.cat(self.categories)
        class_ious = scatter(iou, categories, reduce='mean')
        
        self.log(f"test_iou_class{suffix}", class_ious.mean(), prog_bar=True)
        self.log(f"test_iou_instance{suffix}", iou.mean(), prog_bar=True)
        
        for i, category in enumerate(ShapeNet.seg_classes.keys()):
            self.log(f"test_iou_{category}{suffix}", class_ious[i])
            
        self.ious.clear()
        self.categories.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.hparams.warmup, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def load_data(args):
    """Load and preprocess ShapeNet dataset."""
    ShapeNet.url = "https://huggingface.co/CYYSC/files/resolve/9cefffb2c1d1d60e09b0654da44e3917ada1d999/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip?download=true"
    
    # Load dataset
    category = None  # Train on all categories
    
    # Data augmentation transforms
    transform_train = T.Compose([
        T.NormalizeScale(),
        T.RandomScale((0.8, 1.2)),
        T.RandomJitter(0.01),
        RandomRotateWithNormals(15, axis=0),
        RandomRotateWithNormals(15, axis=1),
        RandomRotateWithNormals(15, axis=2)
    ])
    transform_test = T.Compose([
        T.NormalizeScale(),
    ])
    
    train_dataset = ShapeNet(
        args.data_dir, category, split='train', 
        transform=transform_train,
        include_normals=True
    )
    val_dataset = ShapeNet(
        args.data_dir, category, split='val',
        transform=transform_test,
        include_normals=True
    )
    test_dataset = ShapeNet(
        args.data_dir, category, split='test',
        transform=transform_test,
        include_normals=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader, test_loader

def main(args, checkpoint_interval=1, check_val_every_n_epoch=1):
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
            project="Rapidash-ShapeNet-Segm",
            config=args,
            save_dir=save_dir
        )
    else:
        logger = None

    # Setup callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='valid_iou_instance',
            mode='max',
            every_n_epochs=checkpoint_interval,  # Changed from hardcoded 1 to using interval
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
        enable_progress_bar=args.enable_progress_bar,
        check_val_every_n_epoch=check_val_every_n_epoch  # Added validation interval parameter
    )

    # Train or test
    if args.test_ckpt is None:
        model = ShapeNetModel(args)
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        # Test without augmentation
        trainer.test(model, test_loader, ckpt_path = callbacks[0].best_model_path)
        # Test with augmentation
        model.hparams.test_augm = True
        trainer.test(model, test_loader, ckpt_path = callbacks[0].best_model_path)
    else:
        model = ShapeNetModel.load_from_checkpoint(args.test_ckpt)
        # Test without augmentation
        trainer.test(model, test_loader)
        # Test with augmentation
        model.hparams.test_augm = True
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShapeNet Part Segmentation Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='Weight decay')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=eval, default=[256,256,256,256,256], help='Hidden dimension(s)')
    parser.add_argument('--basis_dim', type=int, default=256, help='Basis dimension')
    parser.add_argument('--basis_hidden_dim', type=int, default=128, help='Hidden dimension of the basis function MLP')
    parser.add_argument('--layers', type=eval, default=[0, 1, 1, 1, 1], help='Layers per scale')
    parser.add_argument('--orientations', type=int, default=8, help='Number of orientations')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    parser.add_argument('--edge_types', type=eval, default=["knn-8","knn-8", "knn-8", "knn-8", "fc"], help='Edge types')
    parser.add_argument('--ratios', type=eval, default=[0.25, 0.25, 0.25, 0.25], help='Pooling ratios')
    parser.add_argument('--widening', type=int, default=4, help='Network widening factor')
    parser.add_argument('--layer_scale', type=eval, default=None, help='Layer scaling factor')
    parser.add_argument('--equivariance', type=str, default="SEn", help='Type of equivariance')
    parser.add_argument('--skip_connections', type=eval, default=False, help='Use U-Net style skip connections')
    
    # Training features
    parser.add_argument('--train_augm', type=eval, default=True, help='Use rotation augmentation during training')
    parser.add_argument('--test_augm', type=eval, default=False, help='Use rotation augmentation during testing')
    
    # Input features
    parser.add_argument('--scalar_features', type=eval, default=[None], help='Features to use as scalars: ["coords", "normals"]')
    parser.add_argument('--vector_features', type=eval, default=["normals", "pose"], help='Features to use as vectors: ["coords", "normals", "pose"]')
    parser.add_argument('--use_category', type=eval, default=True, help='Use shape category information')
    
    # Sweep configuration
    parser.add_argument('--config', type=eval, default=None, help='Sweep configuration dictionary')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID in case you would want to label the configuration')
    
    # Data and logging
    parser.add_argument('--data_dir', type=str, default="./datasets/shapenet", help='Data directory')
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--log', type=eval, default=True, help='Enable logging')
    
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
    
    args.epochs = int(args.epochs / args.dataset_fraction)
    
    main(
        args,
        checkpoint_interval=int(1/args.dataset_fraction),
        check_val_every_n_epoch=int(1/args.dataset_fraction),
    )