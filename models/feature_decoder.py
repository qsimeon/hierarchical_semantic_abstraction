import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class ResidualBlock(nn.Module):
    """
    Residual block for feature decoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection if channels don't match
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.residual(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        x = x + residual
        x = F.relu(x)
        
        return x

class FeatureDecoder(nn.Module):
    """
    Decoder network to transform abstract features back to image space.
    """
    def __init__(
        self,
        feature_dim: int = 768,
        initial_size: int = 8,
        channels: List[int] = [512, 256, 128, 64, 3],
        use_residual: bool = True
    ):
        """
        Initialize feature decoder.
        
        Args:
            feature_dim: Input feature dimension
            initial_size: Initial spatial size after reshaping
            channels: List of channel numbers for each stage
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.initial_size = initial_size
        self.use_residual = use_residual
        
        # Initial projection and reshaping
        initial_channels = channels[0]
        self.initial_proj = nn.Linear(
            feature_dim,
            initial_channels * initial_size * initial_size
        )
        
        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            
            if use_residual:
                block = nn.Sequential(
                    ResidualBlock(in_channels, in_channels),
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
                
            self.decoder_blocks.append(block)
            
        # Remove ReLU from last block
        if isinstance(self.decoder_blocks[-1][-1], nn.ReLU):
            self.decoder_blocks[-1] = self.decoder_blocks[-1][:-1]
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through decoder.
        
        Args:
            x: Input features [B, N, D]
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Decoded image or tuple of (image, intermediate_features)
        """
        B = x.shape[0]
        intermediates = []
        
        # Initial projection and reshaping
        x = self.initial_proj(x)
        x = x.view(B, -1, self.initial_size, self.initial_size)
        
        if return_intermediates:
            intermediates.append(x)
        
        # Decode through blocks
        for block in self.decoder_blocks:
            x = block(x)
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def decode_abstractions(
        self,
        abstractions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Decode multiple abstraction levels.
        
        Args:
            abstractions: List of abstraction features
            
        Returns:
            List of decoded images at different abstraction levels
        """
        decoded = []
        
        for features in abstractions:
            decoded_image = self.forward(features)
            decoded.append(decoded_image)
            
        return decoded