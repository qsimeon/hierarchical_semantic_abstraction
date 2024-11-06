import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from training import AbstractionTrainer
from utils.losses import SemanticLoss, AbstractionLoss, ConsistencyLoss, CombinedLoss
from training.evaluation import AbstractionEvaluator

def test_trainer_initialization(config, mock_model):
    """Test trainer initialization."""
    trainer = AbstractionTrainer(config, mock_model)
    assert isinstance(trainer, pl.LightningModule)
    assert trainer.model is not None
    assert hasattr(trainer, 'semantic_loss')
    assert hasattr(trainer, 'abstraction_loss')
    assert hasattr(trainer, 'consistency_loss')

def test_loss_computation(mock_batch, device):
    """Test all loss computations."""
    # Test Semantic Loss
    semantic_loss = SemanticLoss()
    semantic_value = semantic_loss(
        mock_batch['abstractions'],
        mock_batch['original'],
        mock_model.clip_encoder
    )
    assert isinstance(semantic_value, torch.Tensor)
    assert semantic_value.requires_grad
    
    # Test Abstraction Loss
    abstraction_loss = AbstractionLoss()
    abstraction_value = abstraction_loss(
        mock_batch['abstractions'],
        mock_batch['abstractions']  # Using as feature maps for test
    )
    assert isinstance(abstraction_value, torch.Tensor)
    assert abstraction_value.requires_grad
    
    # Test Consistency Loss
    consistency_loss = ConsistencyLoss()
    consistency_value = consistency_loss(
        mock_batch['abstractions'],
        mock_batch['abstractions']  # Using as feature maps for test
    )
    assert isinstance(consistency_value, torch.Tensor)
    assert consistency_value.requires_grad
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    combined_value = combined_loss(
        mock_batch['abstractions'],
        mock_batch['original'],
        mock_batch['abstractions'],  # Using as feature maps for test
        mock_model.clip_encoder
    )
    assert isinstance(combined_value, torch.Tensor)
    assert combined_value.requires_grad

def test_training_step(config, mock_model, mock_batch):
    """Test training step execution."""
    trainer = AbstractionTrainer(config, mock_model)
    
    output = trainer.training_step(mock_batch, 0)
    assert 'loss' in output
    assert isinstance(output['loss'], torch.Tensor)
    assert output['loss'].requires_grad

def test_validation_step(config, mock_model, mock_batch):
    """Test validation step execution."""
    trainer = AbstractionTrainer(config, mock_model)
    
    output = trainer.validation_step(mock_batch, 0)
    assert 'loss' in output
    assert isinstance(output['loss'], torch.Tensor)

def test_optimizer_configuration(config, mock_model):
    """Test optimizer configuration."""
    trainer = AbstractionTrainer(config, mock_model)
    optimizer_config = trainer.configure_optimizers()
    
    assert 'optimizer' in optimizer_config
    assert 'lr_scheduler' in optimizer_config

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_loop(config, mock_model, mock_dataset):
    """Test complete training loop."""
    # Create trainer
    trainer = AbstractionTrainer(config, mock_model)
    
    # Create dataloaders
    train_loader = DataLoader(
        mock_dataset,
        batch_size=2,
        shuffle=True
    )
    val_loader = DataLoader(
        mock_dataset,
        batch_size=2,
        shuffle=False
    )
    
    # Create PyTorch Lightning trainer
    pl_trainer = Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        enable_progress_bar=False,
        logger=False
    )
    
    # Run training
    pl_trainer.fit(
        trainer,
        train_loader,
        val_loader
    )

def test_model_saving(config, mock_model, tmp_path):
    """Test model checkpoint saving."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path,
        filename='test-{epoch:02d}',
        save_top_k=1,
        monitor='val_loss'
    )
    
    trainer = AbstractionTrainer(config, mock_model)
    pl_trainer = Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        logger=False
    )
    
    # Create dummy data
    train_loader = DataLoader(
        mock_dataset,
        batch_size=2,
        shuffle=True
    )
    
    pl_trainer.fit(trainer, train_loader)
    assert checkpoint_callback.best_model_path != ""

def test_evaluator(config, mock_model, mock_dataset):
    """Test evaluator functionality."""
    evaluator = AbstractionEvaluator(
        model=mock_model,
        config=config
    )
    
    # Create dataloader
    dataloader = DataLoader(
        mock_dataset,
        batch_size=2,
        shuffle=False
    )
    
    # Test batch evaluation
    batch = next(iter(dataloader))
    metrics = evaluator.evaluate_batch(batch)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    
    # Test full dataset evaluation
    metrics = evaluator.evaluate_dataset(
        dataloader,
        save_results=False
    )
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_size_handling(config, mock_model, mock_dataset, batch_size):
    """Test handling of different batch sizes during training."""
    trainer = AbstractionTrainer(config, mock_model)
    
    dataloader = DataLoader(
        mock_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    output = trainer.training_step(batch, 0)
    assert output['loss'].shape == torch.Size([])