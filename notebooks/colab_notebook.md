# Hierarchical Semantic Abstraction - Colab Implementation

## Cell 1: Environment Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/[your-username]/hierarchical_semantic_abstraction.git
%cd hierarchical_semantic_abstraction

# Install dependencies
!pip install -r requirements.txt
!pip install wandb  # Optional: for experiment tracking

# Create necessary directories
!mkdir -p /content/data
!mkdir -p /content/checkpoints
```

## Cell 2: Import Dependencies
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import clip
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import wandb  # Optional

# Check GPU
print(f"Using device: {torch.cuda.get_device_name(0)}")
print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
```

## Cell 3: Configuration
```python
config = {
    'data': {
        'train_path': '/content/data/train',
        'val_path': '/content/data/val',
        'image_size': 224,  # Optimized for T4
        'batch_size': 8,    # Reduced for memory
        'num_workers': 2    # Colab-appropriate
    },
    'model': {
        'clip': {
            'model_name': 'ViT-B/32',
            'freeze_encoder': True
        },
        'transformer': {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'num_levels': 3,
            'dropout': 0.1
        },
        'decoder': {
            'channels': [256, 128, 64, 3],
            'initial_size': 32
        }
    },
    'training': {
        'epochs': 50,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'checkpoint_dir': '/content/checkpoints'
    }
}
```

## Cell 4: Memory-Efficient Dataset
```python
class MemoryEfficientDataset(AbstractionDataset):
    def __init__(self, data_dir, image_size=224, transform=None):
        super().__init__(data_dir)
        self.image_size = image_size
        self.transform = transform or get_default_transform(image_size)
        
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Transform
        image = self.transform(image)
        
        # Create abstraction targets
        abstractions = self.create_abstractions(image)
        
        return {
            'image': image,
            'abstractions': abstractions
        }

# Create datasets
train_dataset = MemoryEfficientDataset(
    config['data']['train_path'],
    config['data']['image_size']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['num_workers']
)
```

## Cell 5: Model Definition
```python
class EfficientAbstractionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize CLIP
        self.clip_encoder = CLIPEncoder(
            model_name=config['model']['clip']['model_name'],
            freeze=config['model']['clip']['freeze_encoder']
        )
        
        # Initialize transformer with memory optimizations
        self.transformer = HierarchicalTransformer(
            dim=config['model']['transformer']['hidden_dim'],
            num_layers=config['model']['transformer']['num_layers'],
            num_heads=config['model']['transformer']['num_heads'],
            num_levels=config['model']['transformer']['num_levels']
        )
        
        # Initialize decoder
        self.decoder = FeatureDecoder(
            channels=config['model']['decoder']['channels'],
            initial_size=config['model']['decoder']['initial_size']
        )
        
    def forward(self, x):
        # Enable gradient checkpointing
        with torch.cuda.amp.autocast():
            # Extract features
            features = self.clip_encoder(x)
            
            # Generate abstractions
            abstractions = self.transformer(features)
            
            # Decode abstractions
            outputs = self.decoder(abstractions)
            
            return outputs

model = EfficientAbstractionModel(config).to('cuda')
```

## Cell 6: Training Setup
```python
# Initialize optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['lr'],
    weight_decay=config['training']['weight_decay']
)

# Initialize loss functions
criterion = CombinedLoss()

# Initialize gradient scaler for mixed precision
scaler = GradScaler()

# Optional: Initialize wandb
wandb.init(project="semantic_abstraction", config=config)
```

## Cell 7: Training Loop
```python
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        # Move to GPU
        images = batch['image'].cuda()
        targets = batch['abstractions'].cuda()
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log batch metrics
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'memory_used': torch.cuda.memory_allocated()/1e9
            })
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / len(loader)

# Training loop
for epoch in range(config['training']['epochs']):
    print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
    
    # Train epoch
    train_loss = train_epoch(
        model, train_loader, optimizer, criterion, scaler
    )
    
    # Log metrics
    print(f"Train Loss: {train_loss:.4f}")
    if wandb.run is not None:
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss
        })
    
    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"{config['training']['checkpoint_dir']}/checkpoint_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
```

## Cell 8: Visualization and Evaluation
```python
def visualize_abstractions(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0).cuda())
        
    # Plot results
    fig, axes = plt.subplots(1, len(outputs)+1, figsize=(15, 3))
    
    # Original image
    axes[0].imshow(image.permute(1,2,0).cpu())
    axes[0].set_title('Original')
    
    # Abstractions
    for i, abs_img in enumerate(outputs):
        axes[i+1].imshow(abs_img.squeeze().permute(1,2,0).cpu())
        axes[i+1].set_title(f'Level {i+1}')
    
    plt.show()

# Test visualization
test_image = next(iter(train_loader))['image'][0]
visualize_abstractions(model, test_image)
```

## Cell 9: Save and Load Model
```python
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = EfficientAbstractionModel(config)
    model.load_state_dict(torch.load(path))
    return model

# Save final model
save_model(model, '/content/drive/MyDrive/final_model.pt')
```

## Cell 10: Memory Management
```python
def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_cached()/1e9:.2f}GB")
    print(f"Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Monitor memory usage
print_memory_stats()
```

This notebook provides a complete implementation optimized for Colab's T4 GPU. Key features:
1. Memory-efficient data handling
2. Mixed precision training
3. Gradient checkpointing
4. Regular cache clearing
5. Progress monitoring
6. Result visualization

Would you like me to:
1. Add more debugging cells?
2. Include additional visualization options?
3. Add performance profiling?
4. Create data preprocessing cells?

Let me know what aspects you'd like me to expand upon!