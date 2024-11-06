import torch
import torch.nn as nn
import clip
from typing import Tuple, Optional

class CLIPEncoder(nn.Module):
    """
    Wrapper for CLIP model to extract image and text features.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        freeze: bool = True
    ):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to load the model on
            freeze: Whether to freeze the CLIP weights
        """
        super().__init__()
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        
        # Get feature dimensions
        self.feature_dim = self.model.visual.output_dim
        
        if freeze:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze all parameters of the CLIP model."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image using CLIP.
        
        Args:
            image: Image tensor [B, C, H, W]
            normalize: Whether to normalize features
            
        Returns:
            Image features [B, feature_dim]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            image_features = self.model.encode_image(image)
            
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
        return image_features
    
    def encode_text(
        self,
        text: list,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text using CLIP.
        
        Args:
            text: List of strings to encode
            normalize: Whether to normalize features
            
        Returns:
            Text features [B, feature_dim]
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            text = clip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(text)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
        return text_features
    
    def get_visual_backbone(self) -> nn.Module:
        """Get the visual backbone of CLIP for feature extraction."""
        return self.model.visual
    
    def get_intermediate_features(
        self,
        image: torch.Tensor,
        layers: Optional[list] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get intermediate features from CLIP's visual backbone.
        
        Args:
            image: Input image tensor
            layers: List of layer indices to extract features from
            
        Returns:
            Tuple of feature tensors
        """
        # Default to last 4 layers if none specified
        if layers is None:
            layers = [-4, -3, -2, -1]
            
        features = []
        x = image
        
        # Extract features from visual backbone
        with torch.no_grad() if not self.training else torch.enable_grad():
            for i, block in enumerate(self.model.visual.transformer.resblocks):
                x = block(x)
                if i in layers:
                    features.append(x)
        
        return tuple(features)
    
    def forward(
        self,
        image: torch.Tensor,
        text: Optional[list] = None,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through CLIP encoder.
        
        Args:
            image: Image tensor
            text: Optional list of text strings
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Tuple of (image_features, text_features) if text is provided,
            else just image_features
        """
        image_features = self.encode_image(image)
        
        if return_intermediate:
            intermediate_features = self.get_intermediate_features(image)
            outputs = (image_features, intermediate_features)
        else:
            outputs = (image_features,)
            
        if text is not None:
            text_features = self.encode_text(text)
            outputs += (text_features,)
            
        return outputs if len(outputs) > 1 else outputs[0]