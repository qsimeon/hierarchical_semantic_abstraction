import os
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from models import AbstractionPipeline
from utils.losses import SemanticLoss, AbstractionLoss, ConsistencyLoss

class AbstractionTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for the Hierarchical Semantic Abstraction model.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[AbstractionPipeline] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model: Optional pre-initialized model
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Initialize model if not provided
        if model is None:
            self.model = AbstractionPipeline(
                clip_model=config['model']['clip']['model_name'],
                feature_dim=config['model']['hierarchical_transformer']['hidden_dim'],
                transformer_depth=config['model']['hierarchical_transformer']['num_layers'],
                num_heads=config['model']['hierarchical_transformer']['num_heads'],
                num_levels=config['model']['hierarchical_transformer']['num_abstraction_levels'],
                ff_dim=config['model']['hierarchical_transformer']['ff_dim'],
                dropout=config['model']['hierarchical_transformer']['dropout'],
                use_semantic_anchoring=config['model']['hierarchical_transformer']['semantic_anchoring'],
                decoder_channels=config['model']['feature_decoder']['channels'],
                initial_size=config['model']['feature_decoder']['initial_size'],
                device=self.device
            )
        else:
            self.model = model
            
        # Initialize loss functions
        self.semantic_loss = SemanticLoss()
        self.abstraction_loss = AbstractionLoss()
        self.consistency_loss = ConsistencyLoss()
        
        # Loss weights
        self.loss_weights = config['training']['loss_weights']
        
        # Metrics for tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer parameters
        opt_config = self.config['training']['optimizer']
        
        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config['weight_decay'],
            betas=(opt_config['beta1'], opt_config['beta2'])
        )
        
        # Create scheduler
        scheduler_config = self.config['training']['scheduler']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step logic.
        
        Args:
            batch: Dictionary containing the batch data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary containing the loss values
        """
        # Get input images
        images = batch['original']
        
        # Generate abstractions
        outputs = self.model(
            images,
            return_all_levels=True,
            return_features=True
        )
        
        abstractions = outputs['abstractions']
        features = outputs['features']
        
        # Calculate losses
        semantic_loss = self.semantic_loss(
            abstractions,
            images,
            self.model.clip_encoder
        )
        
        abstraction_loss = self.abstraction_loss(
            abstractions,
            features
        )
        
        consistency_loss = self.consistency_loss(
            abstractions,
            features
        )
        
        # Combine losses
        total_loss = (
            self.loss_weights['semantic'] * semantic_loss +
            self.loss_weights['abstraction'] * abstraction_loss +
            self.loss_weights['consistency'] * consistency_loss
        )
        
        # Log losses
        self.log('train_loss', total_loss)
        self.log('train_semantic_loss', semantic_loss)
        self.log('train_abstraction_loss', abstraction_loss)
        self.log('train_consistency_loss', consistency_loss)
        
        # Store outputs for epoch end
        self.training_step_outputs.append({
            'loss': total_loss,
            'semantic_loss': semantic_loss,
            'abstraction_loss': abstraction_loss,
            'consistency_loss': consistency_loss
        })
        
        return {
            'loss': total_loss,
            'abstractions': abstractions,
            'features': features
        }
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step logic.
        
        Args:
            batch: Dictionary containing the batch data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary containing the validation metrics
        """
        # Get input images
        images = batch['original']
        
        # Generate abstractions
        outputs = self.model(
            images,
            return_all_levels=True,
            return_features=True
        )
        
        abstractions = outputs['abstractions']
        features = outputs['features']
        
        # Calculate losses
        semantic_loss = self.semantic_loss(
            abstractions,
            images,
            self.model.clip_encoder
        )
        
        abstraction_loss = self.abstraction_loss(
            abstractions,
            features
        )
        
        consistency_loss = self.consistency_loss(
            abstractions,
            features
        )
        
        # Combine losses
        total_loss = (
            self.loss_weights['semantic'] * semantic_loss +
            self.loss_weights['abstraction'] * abstraction_loss +
            self.loss_weights['consistency'] * consistency_loss
        )
        
        # Log losses
        self.log('val_loss', total_loss)
        self.log('val_semantic_loss', semantic_loss)
        self.log('val_abstraction_loss', abstraction_loss)
        self.log('val_consistency_loss', consistency_loss)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'loss': total_loss,
            'semantic_loss': semantic_loss,
            'abstraction_loss': abstraction_loss,
            'consistency_loss': consistency_loss,
            'abstractions': abstractions,
            'images': images
        })
        
        return {
            'loss': total_loss,
            'abstractions': abstractions,
            'features': features
        }
    
    def on_train_epoch_end(self) -> None:
        """Logic at training epoch end."""
        # Calculate mean losses
        avg_loss = torch.stack(
            [x['loss'] for x in self.training_step_outputs]
        ).mean()
        
        avg_semantic_loss = torch.stack(
            [x['semantic_loss'] for x in self.training_step_outputs]
        ).mean()
        
        avg_abstraction_loss = torch.stack(
            [x['abstraction_loss'] for x in self.training_step_outputs]
        ).mean()
        
        avg_consistency_loss = torch.stack(
            [x['consistency_loss'] for x in self.training_step_outputs]
        ).mean()
        
        # Log epoch metrics
        self.log('train_epoch_loss', avg_loss)
        self.log('train_epoch_semantic_loss', avg_semantic_loss)
        self.log('train_epoch_abstraction_loss', avg_abstraction_loss)
        self.log('train_epoch_consistency_loss', avg_consistency_loss)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Logic at validation epoch end."""
        # Calculate mean losses
        avg_loss = torch.stack(
            [x['loss'] for x in self.validation_step_outputs]
        ).mean()
        
        avg_semantic_loss = torch.stack(
            [x['semantic_loss'] for x in self.validation_step_outputs]
        ).mean()
        
        avg_abstraction_loss = torch.stack(
            [x['abstraction_loss'] for x in self.validation_step_outputs]
        ).mean()
        
        avg_consistency_loss = torch.stack(
            [x['consistency_loss'] for x in self.validation_step_outputs]
        ).mean()
        
        # Log epoch metrics
        self.log('val_epoch_loss', avg_loss)
        self.log('val_epoch_semantic_loss', avg_semantic_loss)
        self.log('val_epoch_abstraction_loss', avg_abstraction_loss)
        self.log('val_epoch_consistency_loss', avg_consistency_loss)
        
        # Log validation images if using WandB
        if isinstance(self.logger, WandbLogger):
            # Get first batch of images
            batch_data = self.validation_step_outputs[0]
            images = batch_data['images'][:8]  # Log first 8 images
            abstractions = [abs[:8] for abs in batch_data['abstractions']]
            
            # Create image grid
            grid_images = []
            for i in range(len(images)):
                row = [images[i]]
                for abs_level in abstractions:
                    row.append(abs_level[i])
                grid_images.extend(row)
            
            # Log to WandB
            self.logger.experiment.log({
                "validation_examples": [
                    wandb.Image(img) for img in grid_images
                ]
            })
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments for command line usage."""
        parser = parent_parser.add_argument_group("AbstractionTrainer")
        
        # Training params
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=32)
        
        # Model params
        parser.add_argument("--feature_dim", type=int, default=768)
        parser.add_argument("--num_levels", type=int, default=4)
        
        return parent_parser