import os
from pathlib import Path
from typing import List, Tuple, Union, Dict

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from path and convert to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image in RGB format
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def prepare_clip_input(
    image: Union[Image.Image, torch.Tensor],
    image_size: int = 256,
    normalize: bool = True
) -> torch.Tensor:
    """
    Prepare image for CLIP model input.
    
    Args:
        image: Input image (PIL Image or tensor)
        image_size: Target size for resizing
        normalize: Whether to normalize the image
        
    Returns:
        Preprocessed tensor ready for CLIP
    """
    if isinstance(image, Image.Image):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        image = transform(image)
    
    if normalize:
        # CLIP normalization values
        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        image = normalize(image)
    
    return image

def create_abstraction_levels(
    image: torch.Tensor,
    num_levels: int = 4,
    methods: List[str] = ['gaussian', 'quantize', 'edge']
) -> List[torch.Tensor]:
    """
    Create different abstraction levels of the input image.
    
    Args:
        image: Input image tensor
        num_levels: Number of abstraction levels
        methods: List of abstraction methods to use
        
    Returns:
        List of tensors representing different abstraction levels
    """
    abstractions = []
    
    for level in range(num_levels):
        # Apply different abstraction methods based on level
        if 'gaussian' in methods:
            sigma = (level + 1) * 2
            kernel_size = 2 * int(2 * sigma) + 1
            gaussian = T.GaussianBlur(kernel_size, sigma)
            abstracted = gaussian(image)
            abstractions.append(abstracted)
            
        if 'quantize' in methods:
            num_colors = 256 // (2 ** level)
            quantized = torch.floor(image * num_colors) / num_colors
            abstractions.append(quantized)
            
        if 'edge' in methods:
            # Simple edge detection using Sobel filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float()
            
            edges_x = torch.nn.functional.conv2d(
                image.unsqueeze(0),
                sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1),
                padding=1,
                groups=3
            )
            edges_y = torch.nn.functional.conv2d(
                image.unsqueeze(0),
                sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1),
                padding=1,
                groups=3
            )
            edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2))
            abstractions.append(edges.squeeze(0))
    
    return abstractions

def save_processed_data(
    data: Dict[str, torch.Tensor],
    save_dir: Union[str, Path],
    filename: str
) -> None:
    """
    Save processed data to disk.
    
    Args:
        data: Dictionary containing processed tensors
        save_dir: Directory to save the data
        filename: Name of the file to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"{filename}.pt"
    torch.save(data, save_path)

def get_augmentation_pipeline(
    image_size: int = 256,
    train: bool = True
) -> A.Compose:
    """
    Create augmentation pipeline using Albumentations.
    
    Args:
        image_size: Target image size
        train: Whether to include training augmentations
        
    Returns:
        Albumentations transformation pipeline
    """
    if train:
        transform = A.Compose([
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
            ToTensorV2()
        ])
    
    return transform