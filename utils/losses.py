import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class SemanticLoss(nn.Module):
    """
    Loss for semantic preservation using CLIP features.
    """
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        
    def forward(
        self,
        abstractions: List[torch.Tensor],
        original: torch.Tensor,
        clip_encoder: nn.Module
    ) -> torch.Tensor:
        """
        Compute semantic preservation loss.
        
        Args:
            abstractions: List of abstracted images
            original: Original images
            clip_encoder: CLIP encoder model
            
        Returns:
            Semantic preservation loss
        """
        # Get original features
        with torch.no_grad():
            orig_features = clip_encoder.encode_image(original)
            if self.normalize:
                orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
        
        total_loss = 0
        
        # Compute loss for each abstraction level
        for abstracted in abstractions:
            # Get abstracted features
            abst_features = clip_encoder.encode_image(abstracted)
            if self.normalize:
                abst_features = abst_features / abst_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity loss
            similarity = F.cosine_similarity(orig_features, abst_features)
            level_loss = 1 - similarity.mean()
            
            total_loss += level_loss
            
        return total_loss / len(abstractions)

class AbstractionLoss(nn.Module):
    """
    Loss for enforcing abstraction quality.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        abstractions: List[torch.Tensor],
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute abstraction quality loss.
        
        Args:
            abstractions: List of abstracted images
            features: List of feature maps
            
        Returns:
            Abstraction quality loss
        """
        total_loss = 0
        
        for level, (curr_abs, next_abs) in enumerate(zip(abstractions[:-1], abstractions[1:])):
            # Compute structural difference
            struct_diff = F.mse_loss(
                curr_abs,
                next_abs,
                reduction='none'
            )
            
            # Weight loss by level
            level_weight = (level + 1) / len(abstractions)
            level_loss = struct_diff.mean() * level_weight
            
            total_loss += level_loss
            
        # Add feature diversity loss
        feature_loss = 0
        for level_features in features:
            # Compute feature statistics
            mean = level_features.mean(dim=[2, 3])
            var = level_features.var(dim=[2, 3])
            
            # Encourage diverse features
            feature_loss += -torch.log(var + 1e-6).mean()
            
        total_loss += 0.1 * feature_loss
            
        return total_loss

class ConsistencyLoss(nn.Module):
    """
    Loss for maintaining hierarchical consistency.
    """
    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(
        self,
        abstractions: List[torch.Tensor],
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute hierarchical consistency loss.
        
        Args:
            abstractions: List of abstracted images
            features: List of feature maps
            
        Returns:
            Consistency loss
        """
        total_loss = 0
        
        # Progressive consistency loss
        for level, (curr_abs, next_abs) in enumerate(zip(abstractions[:-1], abstractions[1:])):
            # Compute consistency between adjacent levels
            consistency_loss = F.l1_loss(
                curr_abs,
                next_abs,
                reduction='mean'
            )
            
            # Add feature consistency
            curr_feat = features[level]
            next_feat = features[level + 1]
            
            feat_consistency = F.mse_loss(
                F.adaptive_avg_pool2d(curr_feat, next_feat.shape[-2:]),
                next_feat,
                reduction='mean'
            )
            
            total_loss += consistency_loss + self.lambda_reg * feat_consistency
            
        return total_loss / (len(abstractions) - 1)

class CombinedLoss(nn.Module):
    """
    Combined loss function for training.
    """
    def __init__(
        self,
        semantic_weight: float = 1.0,
        abstraction_weight: float = 0.5,
        consistency_weight: float = 0.3
    ):
        super().__init__()
        
        self.semantic_loss = SemanticLoss()
        self.abstraction_loss = AbstractionLoss()
        self.consistency_loss = ConsistencyLoss()
        
        self.semantic_weight = semantic_weight
        self.abstraction_weight = abstraction_weight
        self.consistency_weight = consistency_weight
        
    def forward(
        self,
        abstractions: List[torch.Tensor],
        original: torch.Tensor,
        features: List[torch.Tensor],
        clip_encoder: nn.Module
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            abstractions: List of abstracted images
            original: Original images
            features: List of feature maps
            clip_encoder: CLIP encoder model
            
        Returns:
            Combined loss
        """
        semantic_loss = self.semantic_loss(abstractions, original, clip_encoder)
        abstraction_loss = self.abstraction_loss(abstractions, features)
        consistency_loss = self.consistency_loss(abstractions, features)
        
        total_loss = (
            self.semantic_weight * semantic_loss +
            self.abstraction_weight * abstraction_loss +
            self.consistency_weight * consistency_loss
        )
        
        return total_loss