import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class MultiLevelAttention(nn.Module):
    """
    Multi-level attention module for hierarchical feature processing.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_levels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Level-specific queries, keys, and values
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        level: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-level attention.
        
        Args:
            x: Input tensor [B, N, D]
            level: Current abstraction level
            mask: Optional attention mask
            
        Returns:
            Attended features [B, N, D]
        """
        B, N, D = x.shape
        
        # Get level-specific projections
        q = self.q_projs[level](x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_projs[level](x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_projs[level](x).reshape(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, N, D/H]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.out_proj(x)
        
        return x

class HierarchicalTransformerBlock(nn.Module):
    """
    Transformer block with hierarchical attention and feed-forward layers.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_levels: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiLevelAttention(
            dim=dim,
            num_heads=num_heads,
            num_levels=num_levels,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        level: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, N, D]
            level: Current abstraction level
            mask: Optional attention mask
            
        Returns:
            Transformed features [B, N, D]
        """
        # Multi-level attention
        x = x + self.attn(self.norm1(x), level, mask)
        
        # Feed-forward network
        x = x + self.ff(self.norm2(x))
        
        return x

class SemanticAnchor(nn.Module):
    """
    Semantic anchoring module to maintain semantic consistency.
    """
    def __init__(
        self,
        dim: int,
        num_levels: int
    ):
        super().__init__()
        
        self.level_embeds = nn.Parameter(torch.randn(num_levels, dim))
        self.anchor_proj = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """
        Apply semantic anchoring.
        
        Args:
            x: Input features [B, N, D]
            level: Current abstraction level
            
        Returns:
            Anchored features [B, N, D]
        """
        # Get level-specific anchor
        anchor = self.level_embeds[level]
        
        # Project anchor
        anchor = self.anchor_proj(anchor)
        
        # Apply anchoring through addition
        x = x + anchor.unsqueeze(0).unsqueeze(0)
        
        return x

class HierarchicalTransformer(nn.Module):
    """
    Main hierarchical transformer for semantic abstraction.
    """
    def __init__(
        self,
        dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        num_levels: int = 4,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        use_semantic_anchoring: bool = True
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(dim, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_levels=num_levels,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Semantic anchoring
        self.use_semantic_anchoring = use_semantic_anchoring
        if use_semantic_anchoring:
            self.semantic_anchor = SemanticAnchor(dim, num_levels)
            
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
        # Level embedding
        self.level_embed = nn.Parameter(torch.randn(num_levels, dim))
        
        self.num_levels = num_levels
        
    def forward(
        self,
        x: torch.Tensor,
        level: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical transformer.
        
        Args:
            x: Input features [B, N, D]
            level: Current abstraction level
            mask: Optional attention mask
            
        Returns:
            Transformed features [B, N, D]
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Add level embedding
        x = x + self.level_embed[level].unsqueeze(0).unsqueeze(0)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, level, mask)
            
            # Apply semantic anchoring after each block if enabled
            if self.use_semantic_anchoring:
                x = self.semantic_anchor(x, level)
        
        # Final projection
        x = self.output_proj(x)
        
        return x
    
    def generate_abstraction_levels(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Generate all abstraction levels for input features.
        
        Args:
            x: Input features [B, N, D]
            mask: Optional attention mask
            
        Returns:
            List of features at different abstraction levels
        """
        abstractions = []
        
        for level in range(self.num_levels):
            level_features = self.forward(x, level, mask)
            abstractions.append(level_features)
            
        return abstractions