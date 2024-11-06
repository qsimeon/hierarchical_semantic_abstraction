import pytest
import torch
import torch.nn as nn

from models import (
    CLIPEncoder,
    HierarchicalTransformer,
    FeatureDecoder,
    AbstractionPipeline
)

def test_clip_encoder(mock_images, device, mock_text):
    """Test CLIP encoder functionality."""
    encoder = CLIPEncoder(device=device)
    
    # Test image encoding
    image_features = encoder.encode_image(mock_images)
    assert image_features.shape[1] == encoder.feature_dim
    
    # Test text encoding
    text_features = encoder.encode_text(mock_text)
    assert text_features.shape[1] == encoder.feature_dim
    
    # Test intermediate features
    _, intermediates = encoder(mock_images, return_intermediate=True)
    assert len(intermediates) > 0

def test_hierarchical_transformer(mock_images, device):
    """Test hierarchical transformer functionality."""
    batch_size = mock_images.shape[0]
    feature_dim = 768
    num_levels = 4
    
    # Create input features
    features = torch.randn(batch_size, 16, feature_dim).to(device)
    
    transformer = HierarchicalTransformer(
        dim=feature_dim,
        depth=2,
        num_heads=8,
        num_levels=num_levels
    ).to(device)
    
    # Test single level transformation
    output = transformer(features, level=0)
    assert output.shape == features.shape
    
    # Test multi-level abstraction
    abstractions = transformer.generate_abstraction_levels(features)
    assert len(abstractions) == num_levels
    assert all(abs.shape == features.shape for abs in abstractions)

def test_feature_decoder(device):
    """Test feature decoder functionality."""
    batch_size = 2
    feature_dim = 768
    initial_size = 8
    
    decoder = FeatureDecoder(
        feature_dim=feature_dim,
        initial_size=initial_size
    ).to(device)
    
    # Create input features
    features = torch.randn(batch_size, 16, feature_dim).to(device)
    
    # Test forward pass
    output = decoder(features)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 3  # RGB output
    
    # Test with intermediates
    output, intermediates = decoder(features, return_intermediates=True)
    assert len(intermediates) > 0

def test_full_pipeline(mock_images, device, config):
    """Test complete abstraction pipeline."""
    pipeline = AbstractionPipeline(
        clip_model=config["model"]["clip"]["model_name"],
        feature_dim=config["model"]["hierarchical_transformer"]["hidden_dim"],
        transformer_depth=2,
        num_heads=config["model"]["hierarchical_transformer"]["num_heads"],
        num_levels=config["model"]["hierarchical_transformer"]["num_abstraction_levels"],
        device=device
    ).to(device)
    
    # Test single level abstraction
    output = pipeline(mock_images, target_level=0)
    assert output.shape == mock_images.shape
    
    # Test multi-level abstraction
    outputs = pipeline(mock_images, return_all_levels=True)
    assert len(outputs) == pipeline.num_levels
    
    # Test with feature return
    outputs = pipeline(
        mock_images,
        return_all_levels=True,
        return_features=True
    )
    assert 'abstractions' in outputs
    assert 'features' in outputs

def test_model_training(mock_model, mock_batch, mock_optimizer):
    """Test model training step."""
    mock_model.train()
    
    # Forward pass
    outputs = mock_model(
        mock_batch['original'],
        return_all_levels=True,
        return_features=True
    )
    
    # Check outputs
    assert 'abstractions' in outputs
    assert 'features' in outputs
    
    # Compute dummy loss and backprop
    loss = sum(abs.mean() for abs in outputs['abstractions'])
    loss.backward()
    
    # Check gradients
    for name, param in mock_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    # Test optimizer step
    mock_optimizer.step()
    mock_optimizer.zero_grad()

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_model_batch_handling(mock_model, device, batch_size):
    """Test model handling of different batch sizes."""
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Test forward pass
    outputs = mock_model(images, return_all_levels=True)
    assert len(outputs) == mock_model.num_levels
    assert all(abs.shape[0] == batch_size for abs in outputs)

def test_model_device_transfer(mock_model, mock_batch):
    """Test model transfer between devices."""
    if torch.cuda.is_available():
        # Move to CPU
        mock_model.to('cpu')
        outputs_cpu = mock_model(mock_batch['original'].cpu())
        
        # Move back to CUDA
        mock_model.to('cuda')
        outputs_gpu = mock_model(mock_batch['original'].cuda())
        
        assert outputs_cpu.device.type == 'cpu'
        assert outputs_gpu.device.type == 'cuda'