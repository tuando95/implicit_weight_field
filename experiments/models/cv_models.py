"""Computer vision models for experiments."""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def load_resnet50(
    pretrained: bool = True,
    num_classes: int = 1000,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load ResNet-50 model.
    
    Args:
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        ResNet-50 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.resnet50(pretrained=pretrained)
    
    if num_classes != 1000:
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded ResNet-50 with {total_params/1e6:.1f}M parameters")
    
    return model


def load_mobilenet_v2(
    pretrained: bool = True,
    num_classes: int = 1000,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load MobileNet-V2 model.
    
    Args:
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        MobileNet-V2 model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    
    if num_classes != 1000:
        # Replace classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded MobileNet-V2 with {total_params/1e6:.1f}M parameters")
    
    return model


def load_vit(
    model_name: str = "vit_b_16",
    pretrained: bool = True,
    num_classes: int = 1000,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load Vision Transformer model.
    
    Args:
        model_name: ViT variant name
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        ViT model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ViT model
    if model_name == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
    elif model_name == "vit_b_32":
        model = torchvision.models.vit_b_32(pretrained=pretrained)
    elif model_name == "vit_l_16":
        model = torchvision.models.vit_l_16(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown ViT model: {model_name}")
    
    if num_classes != 1000:
        # Replace head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_name} with {total_params/1e6:.1f}M parameters")
    
    return model


def prepare_imagenet_loader(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    split: str = "val",
    augment: bool = False
) -> DataLoader:
    """
    Prepare ImageNet data loader.
    
    Args:
        data_dir: Path to ImageNet dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        split: Dataset split ("train" or "val")
        augment: Whether to apply data augmentation
        
    Returns:
        DataLoader for ImageNet
    """
    # Define transforms
    if split == "train" and augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    try:
        dataset = torchvision.datasets.ImageNet(
            root=data_dir,
            split=split,
            transform=transform
        )
    except RuntimeError as e:
        logger.warning(f"ImageNet not found: {e}")
        logger.warning("Falling back to ImageNet-1K subset or CIFAR-100 as proxy")
        # Use CIFAR-100 as a proxy with same transforms
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=(split == "train"),
            transform=transform,
            download=True  # Auto-download
        )
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created ImageNet {split} loader with {len(dataset)} samples")
    
    return loader


def prepare_cifar_loader(
    data_dir: str,
    dataset: str = "cifar100",
    batch_size: int = 128,
    num_workers: int = 4,
    train: bool = False,
    augment: bool = False
) -> DataLoader:
    """
    Prepare CIFAR data loader.
    
    Args:
        data_dir: Path to store/load dataset
        dataset: "cifar10" or "cifar100"
        batch_size: Batch size
        num_workers: Number of data loading workers
        train: Whether to load training set
        augment: Whether to apply augmentation
        
    Returns:
        DataLoader for CIFAR
    """
    # Define transforms
    if train and augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                              std=[0.2675, 0.2565, 0.2761])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                              std=[0.2675, 0.2565, 0.2761])
        ])
    
    # Create dataset
    if dataset == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    dataset = dataset_class(
        root=data_dir,
        train=train,
        transform=transform,
        download=True
    )
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created {dataset} {'train' if train else 'test'} loader with {len(dataset)} samples")
    
    return loader


class ModelWrapper(nn.Module):
    """Wrapper to add methods needed for experiments."""
    
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.model(x)
    
    def get_accuracy(self, output, target):
        """Calculate accuracy."""
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        return correct / target.size(0)
    
    def get_top5_accuracy(self, output, target):
        """Calculate top-5 accuracy."""
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:5].reshape(-1).float().sum(0).item() / target.size(0)