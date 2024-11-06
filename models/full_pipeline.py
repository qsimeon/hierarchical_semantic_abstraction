import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple

from .clip_encoder import CLIPEncoder
from .hierarchical_transformer import HierarchicalTransformer
from .feature_decoder import FeatureDecoder

class AbstractionPipeline(nn.Module):
    """
    Complete pipeline for hierarchical semantic abstraction.
    """
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        feature_dim: int = 768,
        transformer_depth: int = 6,
        num_heads: int = 8,
        num_levels: int = 4,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        use_semantic_anchoring: bool = True,
        decoder_channels: List[int] = [512, 256, 128, 64, 3],
        initial_size: int = 8,
        device: str = "cuda"
    ):
        """
        Initialize the complete abstraction pipeline.
        
        Args:
            clip_model: CLIP model variant
            feature_dim: Feature dimension
            transformer_depth: Number of transformer layers
            num_heads: Number of attention heads
            num_levels: Number of abstraction levels
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            use_semantic_anchoring: Whether to use semantic anchoring
            decoder_channels: Channel numbers for decoder stages
            initial_size: Initial spatial size for decoder
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.num_levels = num_levels
        
        # Initialize components
        self.clip_encoder = CLIPEncoder(
            model_name=clip_model,
            device=device,
            freeze=True
        )
        
        self.transformer = HierarchicalTransformer(
            dim=feature_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            num_levels=num_levels,
            ff_dim=ff_dim,
            dropout=dropout,
            use_semantic_anchoring=use_semantic_anchoring
        )
        
        self.decoder = FeatureDecoder(
            feature_dim=feature_dim,
            initial_size=initial_size,
            channels=decoder_channels,
            use_residual=True
        )
        
    def encode_image(
        self,
        image: torch.Tensor,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Encode image using CLIP encoder.
        
        Args:
            image: Input image tensor
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Encoded features or tuple of (features, intermediates)
        """
        return self.clip_encoder(
            image,
            return_intermediate=return_intermediates
        )
    
    def generate_abstractions(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Generate abstractions using transformer.
        
        Args:
            features: Input features from encoder
            mask: Optional attention mask
            
        Returns:
            List of features at different abstraction levels
        """
        return self.transformer.generate_abstraction_levels(features, mask)
    
    def decode_abstractions(
        self,
        abstractions: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Decode abstraction features to images.
        
        Args:
            abstractions: List of abstraction features
            
        Returns:
            List of decoded images at different levels
        """
        return self.decoder.decode_abstractions(abstractions)
    
    def forward(
        self,
        image: torch.Tensor,
        target_level: Optional[int] = None,
        return_all_levels: bool = False,
        return_features: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
    ]:
        """
        Forward pass through complete pipeline.
        
        Args:
            image: Input image tensor
            target_level: Specific abstraction level to generate
            return_all_levels: Whether to return all abstraction levels
            return_features: Whether to return intermediate features
            
        Returns:
            Generated abstraction(s) and optionally features
        """
        # Encode image
        features = self.encode_image(image)
        
        # Generate abstractions
        if target_level is not None:
            # Generate specific level
            abstract_features = self.transformer(features, target_level)
            decoded = self.decoder(abstract_features)
            
            if return_features:
                return {
                    'abstraction': decoded,
                    'features': abstract_features
                }
            return decoded
            
        else:
            # Generate all levels
            abstract_features = self.generate_abstractions(features)
            decoded = self.decode_abstractions(abstract_features)
            
            if return_features:
                return {
                    'abstractions': decoded,
                    'features': abstract_features
                }
            
            if return_all_levels:
                return decoded
            
            return decoded[-1]  # Return highest abstraction level
        
    def get_clip_similarity(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CLIP similarity between two images.
        
        Args:
            image1: First image tensor
            image2: Second image tensor
            
        Returns:
            Cosine similarity score
        """
        # Encode both images
        features1 = self.encode_image(image1)
        features2 = self.encode_image(image2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(features1, features2)
        
        return similarity
    
    @torch.no_grad()
    def evaluate_abstractions(
        self,
        original: torch.Tensor,
        abstractions: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate abstraction quality using CLIP similarity.
        
        Args:
            original: Original image tensor
            abstractions: List of abstracted images
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Get original features
        orig_features = self.encode_image(original)
        
        # Compute similarities for each abstraction level
        for i, abstracted in enumerate(abstractions):
            # CLIP similarity
            abst_features = self.encode_image(abstracted)
            similarity = F.cosine_similarity(orig_features, abst_features)
            
            metrics[f'level_{i}_similarity'] = similarity
            
            # Structural similarity (optional)
            if hasattr(self, 'compute_ssim'):
                ssim = self.compute_ssim(original, abstracted)
                metrics[f'level_{i}_ssim'] = ssim
        
        return metrics