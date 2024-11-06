import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def compute_clip_score(
    original: torch.Tensor,
    generated: torch.Tensor,
    clip_encoder: torch.nn.Module
) -> torch.Tensor:
    """
    Compute CLIP score between original and generated images.
    
    Args:
        original: Original images
        generated: Generated images
        clip_encoder: CLIP encoder model
        
    Returns:
        CLIP similarity score
    """
    with torch.no_grad():
        # Get features
        orig_features = clip_encoder.encode_image(original)
        gen_features = clip_encoder.encode_image(generated)
        
        # Normalize features
        orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = F.cosine_similarity(orig_features, gen_features)
        
    return similarity.mean()

def compute_bertscore(
    original_text: List[str],
    generated_text: List[str]
) -> Dict[str, float]:
    """
    Compute BERTScore for text similarity.
    
    Args:
        original_text: List of original text strings
        generated_text: List of generated text strings
        
    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    from bert_score import score
    
    P, R, F1 = score(
        generated_text,
        original_text,
        lang="en",
        verbose=False
    )
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

def compute_perceptual_distance(
    original: torch.Tensor,
    generated: torch.Tensor,
    net_type: str = 'alex'
) -> torch.Tensor:
    """
    Compute perceptual distance using LPIPS.
    
    Args:
        original: Original images
        generated: Generated images
        net_type: Network type for LPIPS ('alex' or 'vgg')
        
    Returns:
        Perceptual distance score
    """
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=net_type
    ).to(original.device)
    
    with torch.no_grad():
        distance = lpips(original, generated)
    
    return distance.mean()

def compute_fid_score(
    real_features: torch.Tensor,
    generated_features: torch.Tensor
) -> float:
    """
    Compute Fréchet Inception Distance (FID) score.
    
    Args:
        real_features: Features from real images
        generated_features: Features from generated images
        
    Returns:
        FID score
    """
    # Calculate mean and covariance for real features
    mu1 = real_features.mean(dim=0)
    sigma1 = torch.cov(real_features.T)
    
    # Calculate mean and covariance for generated features
    mu2 = generated_features.mean(dim=0)
    sigma2 = torch.cov(generated_features.T)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = torch.matrix_power(
        torch.mm(sigma1, sigma2),
        0.5
    )
    
    fid = torch.real(
        diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    )
    
    return fid.item()

def compute_diversity_score(
    features: torch.Tensor,
    num_samples: int = 100
) -> float:
    """
    Compute diversity score for generated samples.
    
    Args:
        features: Feature vectors of samples
        num_samples: Number of random pairs to sample
        
    Returns:
        Diversity score
    """
    if len(features) < 2:
        return 0.0
        
    # Sample random pairs
    idx1 = torch.randint(0, len(features), (num_samples,))
    idx2 = torch.randint(0, len(features), (num_samples,))
    
    # Compute distances
    distances = F.pairwise_distance(
        features[idx1],
        features[idx2]
    )
    
    return distances.mean().item()

def compute_abstraction_metrics(
    abstractions: List[torch.Tensor],
    original: torch.Tensor,
    clip_encoder: torch.nn.Module
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for abstraction evaluation.
    
    Args:
        abstractions: List of abstracted images
        original: Original images
        clip_encoder: CLIP encoder model
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Initialize SSIM
    ssim = StructuralSimilarityIndexMeasure().to(original.device)
    
    for level, abstracted in enumerate(abstractions):
        # CLIP score
        clip_score = compute_clip_score(
            original,
            abstracted,
            clip_encoder
        )
        metrics[f'clip_score_level_{level}'] = clip_score.item()
        
        # SSIM
        ssim_score = ssim(original, abstracted)
        metrics[f'ssim_level_{level}'] = ssim_score.item()
        
        # Perceptual distance
        perceptual_dist = compute_perceptual_distance(
            original,
            abstracted
        )
        metrics[f'perceptual_distance_level_{level}'] = perceptual_dist.item()
        
    return metrics