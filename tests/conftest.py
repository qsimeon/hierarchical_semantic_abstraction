import pytest
import torch
import yaml
from pathlib import Path
import numpy as np

from models import AbstractionPipeline
from data import AbstractionDataset

@pytest.fixture
def device():
    """Return device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path("config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override certain settings for testing
    config["data"]["batch_size"] = 2
    config["model"]["hierarchical_transformer"]["num_layers"] = 2
    config["training"]["training_params"]["num_epochs"] = 2
    
    return config

@pytest.fixture
def mock_images(device):
    """Generate mock images for testing."""
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    images = torch.randn(batch_size, channels, height, width).to(device)
    return images

@pytest.fixture
def mock_dataset(tmp_path):
    """Create a mock dataset for testing."""
    # Create temporary image files
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create random images
    num_images = 4
    for i in range(num_images):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image_path = data_dir / f"image_{i}.jpg"
        Image.fromarray(image).save(image_path)
    
    # Create dataset
    dataset = AbstractionDataset(
        data_dir=data_dir,
        image_size=64,
        num_abstraction_levels=2,
        train=True
    )
    
    return dataset

@pytest.fixture
def mock_model(device, config):
    """Create a mock model for testing."""
    model = AbstractionPipeline(
        clip_model=config["model"]["clip"]["model_name"],
        feature_dim=config["model"]["hierarchical_transformer"]["hidden_dim"],
        transformer_depth=2,  # Reduced for testing
        num_heads=config["model"]["hierarchical_transformer"]["num_heads"],
        num_levels=config["model"]["hierarchical_transformer"]["num_abstraction_levels"],
        ff_dim=config["model"]["hierarchical_transformer"]["ff_dim"],
        dropout=config["model"]["hierarchical_transformer"]["dropout"],
        use_semantic_anchoring=config["model"]["hierarchical_transformer"]["semantic_anchoring"],
        decoder_channels=config["model"]["feature_decoder"]["channels"],
        initial_size=config["model"]["feature_decoder"]["initial_size"],
        device=device
    ).to(device)
    
    return model

@pytest.fixture
def mock_batch(mock_images):
    """Create a mock batch for testing."""
    return {
        "original": mock_images,
        "clip_input": mock_images,
        "abstractions": [mock_images for _ in range(4)]
    }

@pytest.fixture
def mock_optimizer(mock_model):
    """Create optimizer for testing."""
    return torch.optim.AdamW(
        mock_model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

@pytest.fixture
def mock_text():
    """Generate mock text data for testing."""
    return [
        "A beautiful landscape",
        "Abstract art composition"
    ]