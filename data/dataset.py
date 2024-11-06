import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
from tqdm import tqdm

from .data_utils import (
    load_image,
    prepare_clip_input,
    create_abstraction_levels,
    get_augmentation_pipeline
)

class AbstractionDataset(Dataset):
    """
    Dataset class for Hierarchical Semantic Abstraction.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        image_size: int = 256,
        num_abstraction_levels: int = 4,
        transform=None,
        train: bool = True,
        preload: bool = False,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the image data
            image_size: Size to resize images to
            num_abstraction_levels: Number of abstraction levels to generate
            transform: Optional transform to apply to images
            train: Whether this is a training dataset
            preload: Whether to preload data into memory
            cache_dir: Directory to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_abstraction_levels = num_abstraction_levels
        self.train = train
        self.preload = preload
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Set up transforms
        self.transform = transform if transform is not None else \
            get_augmentation_pipeline(image_size, train)
        
        # Get image paths
        self.image_paths = self._get_image_paths()
        
        # Initialize cache
        self.cache = {}
        if self.preload:
            self._preload_data()
    
    def _get_image_paths(self) -> List[Path]:
        """Get all image paths from the data directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.data_dir.glob(f"*{ext}"))
        
        return sorted(image_paths)
    
    def _preload_data(self) -> None:
        """Preload and process all images into memory."""
        cache_file = self.cache_dir / f"cache_{self.train}_sz{self.image_size}.pt"
        
        if self.cache_dir and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            self.cache = torch.load(cache_file)
            return
        
        print("Preloading dataset into memory...")
        for idx in tqdm(range(len(self))):
            path = self.image_paths[idx]
            self.cache[idx] = self._load_and_process_image(path)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.cache, cache_file)
            print(f"Cached processed data to {cache_file}")
    
    def _load_and_process_image(
        self,
        image_path: Path
    ) -> Dict[str, torch.Tensor]:
        """
        Load and process a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing processed image data
        """
        # Load image
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Apply transforms
        transformed = self.transform(image=np.array(image))
        image_tensor = transformed['image']
        
        # Prepare CLIP input
        clip_input = prepare_clip_input(image_tensor, self.image_size)
        
        # Create abstraction levels
        abstractions = create_abstraction_levels(
            image_tensor,
            self.num_abstraction_levels
        )
        
        return {
            'original': image_tensor,
            'clip_input': clip_input,
            'abstractions': abstractions
        }
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(
        self,
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the processed image data
        """
        if idx in self.cache:
            return self.cache[idx]
        
        image_path = self.image_paths[idx]
        return self._load_and_process_image(image_path)
    
    @staticmethod
    def get_dataloader(
        dataset: 'AbstractionDataset',
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            dataset: AbstractionDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )