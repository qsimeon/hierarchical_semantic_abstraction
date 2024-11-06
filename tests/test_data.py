import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from data import AbstractionDataset
from data.data_utils import (
    load_image,
    prepare_clip_input,
    create_abstraction_levels,
    get_augmentation_pipeline
)

def test_load_image(tmp_path):
    """Test image loading functionality."""
    # Create test image
    image_path = tmp_path / "test.jpg"
    test_image = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    test_image.save(image_path)
    
    # Test loading
    loaded_image = load_image(image_path)
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == 'RGB'
    
    # Test error handling
    with pytest.raises(Exception):
        load_image(tmp_path / "nonexistent.jpg")

def test_prepare_clip_input():
    """Test CLIP input preparation."""
    # Test with PIL image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    clip_input = prepare_clip_input(test_image, image_size=32)
    assert isinstance(clip_input, torch.Tensor)
    assert clip_input.shape == (3, 32, 32)
    
    # Test with tensor
    tensor_image = torch.randn(3, 64, 64)
    clip_input = prepare_clip_input(tensor_image, image_size=32)
    assert clip_input.shape == (3, 32, 32)

def test_create_abstraction_levels():
    """Test abstraction level creation."""
    image = torch.randn(3, 64, 64)
    num_levels = 3
    
    abstractions = create_abstraction_levels(
        image,
        num_levels=num_levels
    )
    
    assert len(abstractions) == num_levels
    assert all(abs.shape == image.shape for abs in abstractions)

def test_augmentation_pipeline():
    """Test data augmentation pipeline."""
    # Test training augmentations
    train_transform = get_augmentation_pipeline(
        image_size=64,
        train=True
    )
    
    # Test validation augmentations
    val_transform = get_augmentation_pipeline(
        image_size=64,
        train=False
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Apply transformations
    train_result = train_transform(image=test_image)
    val_result = val_transform(image=test_image)
    
    assert isinstance(train_result['image'], torch.Tensor)
    assert isinstance(val_result['image'], torch.Tensor)
    assert train_result['image'].shape == (3, 64, 64)
    assert val_result['image'].shape == (3, 64, 64)

def test_dataset_initialization(mock_dataset):
    """Test dataset initialization."""
    assert len(mock_dataset) > 0
    assert mock_dataset.image_size == 64
    assert mock_dataset.num_abstraction_levels == 2

def test_dataset_getitem(mock_dataset):
    """Test dataset item retrieval."""
    item = mock_dataset[0]
    
    assert 'original' in item
    assert 'clip_input' in item
    assert 'abstractions' in item
    assert isinstance(item['original'], torch.Tensor)
    assert item['original'].shape == (3, 64, 64)

def test_dataloader(mock_dataset):
    """Test dataloader functionality."""
    batch_size = 2
    dataloader = DataLoader(
        mock_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    assert batch['original'].shape[0] == batch_size
    assert batch['clip_input'].shape[0] == batch_size
    assert all(abs.shape[0] == batch_size for abs in batch['abstractions'])

@pytest.mark.parametrize("train", [True, False])
def test_dataset_modes(tmp_path, train):
    """Test dataset in different modes."""
    # Create test data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create test images
    for i in range(4):
        image = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        image.save(data_dir / f"test_{i}.jpg")
    
    # Create dataset
    dataset = AbstractionDataset(
        data_dir=data_dir,
        image_size=64,
        num_abstraction_levels=2,
        train=train
    )
    
    # Test dataset properties
    assert len(dataset) == 4
    item = dataset[0]
    assert isinstance(item, dict)
    assert all(k in item for k in ['original', 'clip_input', 'abstractions'])

def test_dataset_caching(tmp_path):
    """Test dataset caching functionality."""
    # Create test data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    
    # Create test images
    for i in range(2):
        image = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        image.save(data_dir / f"test_{i}.jpg")
    
    # Create dataset with caching
    dataset = AbstractionDataset(
        data_dir=data_dir,
        image_size=64,
        num_abstraction_levels=2,
        cache_dir=cache_dir,
        preload=True
    )
    
    # Verify cache creation
    assert cache_dir.exists()
    assert len(dataset.cache) == len(dataset)

def test_dataset_transforms(mock_dataset):
    """Test custom transform application."""
    custom_transform = get_augmentation_pipeline(
        image_size=64,
        train=True
    )
    
    dataset = AbstractionDataset(
        data_dir=mock_dataset.data_dir,
        image_size=64,
        transform=custom_transform
    )
    
    item = dataset[0]
    assert item['original'].shape == (3, 64, 64)