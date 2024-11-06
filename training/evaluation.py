import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models import AbstractionPipeline
from utils.metrics import compute_clip_score, compute_bertscore

class AbstractionEvaluator:
    """
    Evaluator for the Hierarchical Semantic Abstraction model.
    """
    
    def __init__(
        self,
        model: AbstractionPipeline,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained abstraction model
            config: Configuration dictionary
            device: Device to use for evaluation
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize metrics
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex'
        ).to(device)
        
        # Results storage
        self.results = defaultdict(list)
        
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary of computed metrics
        """
        images = batch['original'].to(self.device)
        
        # Generate abstractions
        with torch.no_grad():
            outputs = self.model(
                images,
                return_all_levels=True,
                return_features=True
            )
        
        abstractions = outputs['abstractions']
        features = outputs['features']
        
        # Compute metrics for each abstraction level
        metrics = {}
        
        for level, abstracted in enumerate(abstractions):
            # CLIP Score
            if self.config['evaluation']['metrics']['clip_score']:
                clip_score = compute_clip_score(
                    images,
                    abstracted,
                    self.model.clip_encoder
                )
                metrics[f'clip_score_level_{level}'] = clip_score
            
            # SSIM
            if self.config['evaluation']['metrics']['ssim']:
                ssim_score = self.ssim(images, abstracted)
                metrics[f'ssim_level_{level}'] = ssim_score
            
            # LPIPS
            if self.config['evaluation']['metrics']['lpips']:
                lpips_score = self.lpips(images, abstracted)
                metrics[f'lpips_level_{level}'] = lpips_score
        
        return metrics
    
    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate complete dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            save_results: Whether to save evaluation results
            output_dir: Directory to save results
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.eval()
        
        # Storage for metrics
        all_metrics = defaultdict(list)
        
        # Evaluate batches
        for batch in tqdm(dataloader, desc="Evaluating"):
            metrics = self.evaluate_batch(batch)
            
            # Store metrics
            for name, value in metrics.items():
                all_metrics[name].append(value.cpu().numpy())
        
        # Compute means
        mean_metrics = {
            name: np.mean(values) for name, values in all_metrics.items()
        }
        
        # Save results if requested
        if save_results and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            np.save(
                output_dir / "evaluation_metrics.npy",
                mean_metrics
            )
            
            # Log to WandB if configured
            if self.config['wandb']['enable']:
                wandb.log(mean_metrics)
        
        return mean_metrics
    
    def generate_evaluation_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 16,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Generate and save evaluation samples.
        
        Args:
            dataloader: DataLoader for evaluation
            num_samples: Number of samples to generate
            output_dir: Directory to save samples
        """
        self.model.eval()
        
        # Get batch of images
        batch = next(iter(dataloader))
        images = batch['original'][:num_samples].to(self.device)
        
        # Generate abstractions
        with torch.no_grad():
            outputs = self.model(
                images,
                return_all_levels=True
            )
            
        abstractions = outputs['abstractions']
        
        # Save results if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_samples):
                # Create sample directory
                sample_dir = output_dir / f"sample_{i}"
                sample_dir.mkdir(exist_ok=True)
                
                # Save original
                self._save_image(
                    images[i],
                    sample_dir / "original.png"
                )
                
                # Save abstractions
                for level, abs_batch in enumerate(abstractions):
                    self._save_image(
                        abs_batch[i],
                        sample_dir / f"abstraction_level_{level}.png"
                    )
    
    def _save_image(
        self,
        image: torch.Tensor,
        path: Path
    ) -> None:
        """Helper function to save tensor as image."""
        # Ensure image is in correct format
        if image.dim() == 4:
            image = image.squeeze(0)
        
        # Convert to numpy and transpose
        image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Normalize to [0, 255]
        image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Save image
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
    def evaluate_human_samples(
        self,
        samples_dir: Path,
        results_file: Path
    ) -> Dict[str, float]:
        """
        Process human evaluation results.
        
        Args:
            samples_dir: Directory containing evaluation samples
            results_file: File containing human evaluation scores
            
        Returns:
            Dictionary of averaged human evaluation metrics
        """
        # Load human evaluation results
        import pandas as pd
        results = pd.read_csv(results_file)
        
        # Compute average scores for each aspect
        metrics = {}
        for aspect in self.config['evaluation']['human_evaluation']['aspects']:
            metrics[f'human_{aspect}'] = results[aspect].mean()
        
        return metrics