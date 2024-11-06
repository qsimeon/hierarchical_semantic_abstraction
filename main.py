import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models import AbstractionPipeline
from data import AbstractionDataset
from training import AbstractionTrainer, AbstractionEvaluator
from utils.visualization import (
    create_abstraction_grid,
    save_visualization,
    plot_training_progress
)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(config: Dict[str, Any]) -> WandbLogger:
    """Initialize Weights & Biases logger."""
    wandb_config = config['wandb']
    logger = WandbLogger(
        project=wandb_config['project'],
        name=wandb_config.get('name', None),
        entity=wandb_config.get('entity', None),
        tags=wandb_config.get('tags', None)
    )
    return logger

def create_callbacks(config: Dict[str, Any], output_dir: Path) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks

def train(args):
    """Training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    output_dir = Path(config['project']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize data loaders
    train_dataset = AbstractionDataset(
        data_dir=config['data']['dataset']['train_path'],
        image_size=config['data']['dataset']['image_size'],
        num_abstraction_levels=config['model']['hierarchical_transformer']['num_abstraction_levels'],
        train=True
    )
    
    val_dataset = AbstractionDataset(
        data_dir=config['data']['dataset']['val_path'],
        image_size=config['data']['dataset']['image_size'],
        num_abstraction_levels=config['model']['hierarchical_transformer']['num_abstraction_levels'],
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['data']['dataset']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['data']['dataset']['num_workers']
    )
    
    # Initialize model
    model = AbstractionPipeline(
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
        device=device
    ).to(device)
    
    # Initialize trainer module
    trainer_module = AbstractionTrainer(config, model)
    
    # Set up W&B logging
    logger = setup_wandb(config) if config['wandb']['enable'] else None
    
    # Create callbacks
    callbacks = create_callbacks(config, output_dir)
    
    # Initialize PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=config['training']['training_params']['num_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        precision=config['training']['training_params']['precision'],
        gradient_clip_val=config['training']['training_params']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['training_params']['accumulate_grad_batches'],
        check_val_every_n_epoch=config['training']['training_params']['check_val_every_n_epoch']
    )
    
    # Train model
    trainer.fit(trainer_module, train_loader, val_loader)
    
    return trainer_module, model

def evaluate(args, model=None):
    """Evaluation function."""
    # Load configuration
    config = load_config(args.config)
    
    # Load model if not provided
    if model is None:
        model = AbstractionPipeline.load_from_checkpoint(args.checkpoint)
    
    # Initialize evaluator
    evaluator = AbstractionEvaluator(
        model=model,
        config=config
    )
    
    # Load test dataset
    test_dataset = AbstractionDataset(
        data_dir=config['data']['dataset']['test_path'],
        image_size=config['data']['dataset']['image_size'],
        num_abstraction_levels=config['model']['hierarchical_transformer']['num_abstraction_levels'],
        train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['data']['dataset']['num_workers']
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        test_loader,
        save_results=True,
        output_dir=Path(config['project']['output_dir']) / "evaluation"
    )
    
    # Generate sample visualizations
    evaluator.generate_evaluation_samples(
        test_loader,
        num_samples=16,
        output_dir=Path(config['project']['output_dir']) / "samples"
    )
    
    return metrics

def inference(args):
    """Inference function for a single image."""
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = AbstractionPipeline.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Load and preprocess image
    from PIL import Image
    from torchvision import transforms
    
    image = Image.open(args.image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config['data']['dataset']['image_size'],) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Generate abstractions
    with torch.no_grad():
        outputs = model(
            image_tensor,
            return_all_levels=True
        )
    
    # Create visualization
    grid = create_abstraction_grid(
        image_tensor,
        outputs if isinstance(outputs, list) else outputs['abstractions']
    )
    
    # Save result
    output_dir = Path(config['project']['output_dir']) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(
        grid,
        output_dir / f"abstraction_grid_{Path(args.image_path).stem}.png"
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hierarchical Semantic Abstraction')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'inference'])
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, help='Path to input image for inference')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        trainer_module, model = train(args)
        if args.evaluate_after_training:
            evaluate(args, model)
            
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for evaluation mode")
        evaluate(args)
        
    elif args.mode == 'inference':
        if args.checkpoint is None or args.image_path is None:
            raise ValueError("Checkpoint and image path required for inference mode")
        inference(args)

if __name__ == '__main__':
    main()