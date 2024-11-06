import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union
import seaborn as sns
from PIL import Image

def plot_attention_maps(
    attention_maps: torch.Tensor,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 8)
) -> None:
    """
    Plot attention maps from transformer layers.
    
    Args:
        attention_maps: Attention weights [B, H, N, N]
        save_path: Optional path to save visualization
        figsize: Figure size for the plot
    """
    # Convert to numpy
    if torch.is_tensor(attention_maps):
        attention_maps = attention_maps.detach().cpu().numpy()
    
    num_heads = attention_maps.shape[1]
    fig, axes = plt.subplots(
        2, num_heads // 2,
        figsize=figsize
    )
    axes = axes.flatten()
    
    # Plot each attention head
    for idx, ax in enumerate(axes):
        sns.heatmap(
            attention_maps[0, idx],
            ax=ax,
            cmap='viridis',
            cbar=False
        )
        ax.set_title(f'Head {idx + 1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_abstraction_grid(
    original: torch.Tensor,
    abstractions: List[torch.Tensor],
    num_samples: int = 8,
    normalize: bool = True
) -> torch.Tensor:
    """
    Create a grid visualization of abstractions.
    
    Args:
        original: Original images [B, C, H, W]
        abstractions: List of abstracted images
        num_samples: Number of samples to include
        normalize: Whether to normalize pixel values
        
    Returns:
        Grid tensor [C, H, W]
    """
    # Take subset of samples
    original = original[:num_samples]
    abstractions = [abs[:num_samples] for abs in abstractions]
    
    # Create rows for each sample
    rows = []
    for i in range(num_samples):
        row = [original[i]]
        for abs_level in abstractions:
            row.append(abs_level[i])
        rows.append(torch.cat(row, dim=2))
    
    # Combine rows
    grid = torch.cat(rows, dim=1)
    
    if normalize:
        grid = (grid + 1) / 2  # Scale from [-1, 1] to [0, 1]
    
    return grid

def save_visualization(
    tensor: torch.Tensor,
    save_path: Path,
    normalize: bool = True,
    quality: int = 95
) -> None:
    """
    Save tensor as image.
    
    Args:
        tensor: Image tensor [C, H, W]
        save_path: Path to save image
        normalize: Whether to normalize pixel values
        quality: JPEG quality for saving
    """
    # Ensure save directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Transpose to [H, W, C]
    if tensor.shape[0] in [1, 3]:
        tensor = tensor.transpose(1, 2, 0)
    
    # Normalize if needed
    if normalize:
        tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    
    # Save image
    Image.fromarray(tensor).save(str(save_path), quality=quality)

def visualize_feature_maps(
    features: torch.Tensor,
    num_channels: int = 16,
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize feature maps from model.
    
    Args:
        features: Feature tensor [B, C, H, W]
        num_channels: Number of channels to visualize
        figsize: Figure size
        save_path: Optional path to save visualization
    """
    # Select subset of channels
    features = features[0, :num_channels]  # Take first batch
    
    # Create grid
    num_cols = 4
    num_rows = (num_channels + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (feature_map, ax) in enumerate(zip(features, axes)):
        feature_map = feature_map.detach().cpu().numpy()
        im = ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {idx + 1}')
    
    # Remove empty subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_progress(
    metrics: dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot training progress metrics.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Optional path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for name, values in metrics.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_comparison_figure(
    images: List[torch.Tensor],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> None:
    """
    Create comparison figure of multiple images.
    
    Args:
        images: List of image tensors
        titles: List of titles for each image
        figsize: Figure size
        save_path: Optional path to save figure
    """
    assert len(images) == len(titles), "Number of images and titles must match"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    
    for img, title, ax in zip(images, titles, axes):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
            if img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
        
        # Normalize if needed
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
            
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()