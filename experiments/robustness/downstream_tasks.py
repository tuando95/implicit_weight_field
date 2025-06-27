"""Evaluate robustness on downstream tasks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import copy
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DownstreamResult:
    """Result from downstream task evaluation."""
    task_name: str
    original_accuracy: float
    compressed_accuracy: float
    accuracy_retention: float
    convergence_epochs: int
    final_loss: float
    gradient_norm_ratio: float  # Compressed vs original


class FineTuningEvaluator:
    """Evaluate fine-tuning stability of compressed models."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        early_stopping_patience: int = 3
    ):
        """
        Initialize fine-tuning evaluator.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
    
    def evaluate(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_name: str = "fine_tuning",
        device: Optional[torch.device] = None
    ) -> DownstreamResult:
        """
        Evaluate fine-tuning performance.
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model
            train_loader: Training data
            val_loader: Validation data
            task_name: Name of the task
            device: Device to use
            
        Returns:
            Downstream evaluation result
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clone models to avoid modifying originals
        orig_model = copy.deepcopy(original_model).to(device)
        comp_model = copy.deepcopy(compressed_model).to(device)
        
        # Fine-tune original model
        logger.info("Fine-tuning original model")
        orig_accuracy, orig_epochs, orig_gradients = self._fine_tune_model(
            orig_model, train_loader, val_loader, device
        )
        
        # Fine-tune compressed model
        logger.info("Fine-tuning compressed model")
        comp_accuracy, comp_epochs, comp_gradients = self._fine_tune_model(
            comp_model, train_loader, val_loader, device
        )
        
        # Calculate gradient norm ratio
        gradient_ratio = np.mean(comp_gradients) / (np.mean(orig_gradients) + 1e-8)
        
        return DownstreamResult(
            task_name=task_name,
            original_accuracy=orig_accuracy,
            compressed_accuracy=comp_accuracy,
            accuracy_retention=(comp_accuracy / orig_accuracy) * 100,
            convergence_epochs=comp_epochs,
            final_loss=0.0,  # Would need to track
            gradient_norm_ratio=gradient_ratio
        )
    
    def _fine_tune_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ) -> Tuple[float, int, List[float]]:
        """Fine-tune a model and return performance metrics."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        patience_counter = 0
        gradient_norms = []
        
        for epoch in range(self.num_epochs):
            # Training
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Track gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                optimizer.step()
            
            # Validation
            accuracy = self._evaluate_accuracy(model, val_loader, device)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                break
        
        return best_accuracy, epoch + 1, gradient_norms
    
    def _evaluate_accuracy(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total


class TransferLearningEvaluator:
    """Evaluate transfer learning capability."""
    
    def __init__(self, freeze_backbone: bool = True):
        """
        Initialize transfer learning evaluator.
        
        Args:
            freeze_backbone: Whether to freeze backbone during transfer
        """
        self.freeze_backbone = freeze_backbone
    
    def evaluate(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        source_task: str,
        target_task: str,
        target_train_loader: DataLoader,
        target_val_loader: DataLoader,
        num_classes: int,
        device: Optional[torch.device] = None
    ) -> DownstreamResult:
        """
        Evaluate transfer learning performance.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            source_task: Source task name
            target_task: Target task name
            target_train_loader: Target task training data
            target_val_loader: Target task validation data
            num_classes: Number of classes in target task
            device: Device to use
            
        Returns:
            Transfer learning result
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare models for transfer
        orig_transfer = self._prepare_for_transfer(
            copy.deepcopy(original_model), num_classes
        ).to(device)
        
        comp_transfer = self._prepare_for_transfer(
            copy.deepcopy(compressed_model), num_classes
        ).to(device)
        
        # Train on target task
        evaluator = FineTuningEvaluator()
        
        logger.info(f"Transfer learning: {source_task} -> {target_task}")
        
        result = evaluator.evaluate(
            orig_transfer,
            comp_transfer,
            target_train_loader,
            target_val_loader,
            task_name=f"transfer_{source_task}_to_{target_task}",
            device=device
        )
        
        return result
    
    def _prepare_for_transfer(
        self,
        model: nn.Module,
        num_classes: int
    ) -> nn.Module:
        """Prepare model for transfer learning."""
        # Freeze backbone if requested
        if self.freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
        
        # Replace final layer
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        
        return model


class FewShotEvaluator:
    """Evaluate few-shot learning performance."""
    
    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        num_episodes: int = 100
    ):
        """
        Initialize few-shot evaluator.
        
        Args:
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            num_episodes: Number of episodes to evaluate
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
    
    def evaluate(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        dataset: Any,
        device: Optional[torch.device] = None
    ) -> DownstreamResult:
        """
        Evaluate few-shot learning performance.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            dataset: Dataset for few-shot evaluation
            device: Device to use
            
        Returns:
            Few-shot evaluation result
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        original_model.eval()
        compressed_model.eval()
        
        # Extract features using models
        orig_accuracy = self._evaluate_few_shot(
            original_model, dataset, device
        )
        
        comp_accuracy = self._evaluate_few_shot(
            compressed_model, dataset, device
        )
        
        return DownstreamResult(
            task_name=f"{self.n_way}way_{self.k_shot}shot",
            original_accuracy=orig_accuracy,
            compressed_accuracy=comp_accuracy,
            accuracy_retention=(comp_accuracy / orig_accuracy) * 100,
            convergence_epochs=0,  # Not applicable
            final_loss=0.0,
            gradient_norm_ratio=1.0
        )
    
    def _evaluate_few_shot(
        self,
        model: nn.Module,
        dataset: Any,
        device: torch.device
    ) -> float:
        """Evaluate few-shot accuracy using prototypical networks approach."""
        accuracies = []
        
        for episode in range(self.num_episodes):
            # Sample episode
            support_data, support_labels, query_data, query_labels = \
                self._sample_episode(dataset, device)
            
            # Extract features
            with torch.no_grad():
                support_features = self._extract_features(model, support_data)
                query_features = self._extract_features(model, query_data)
            
            # Compute prototypes
            prototypes = []
            for c in range(self.n_way):
                class_features = support_features[support_labels == c]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            
            prototypes = torch.stack(prototypes)
            
            # Classify query examples
            distances = torch.cdist(query_features, prototypes)
            predictions = distances.argmin(dim=1)
            
            accuracy = (predictions == query_labels).float().mean().item()
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def _sample_episode(
        self,
        dataset: Any,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a few-shot episode."""
        # This is a simplified version - actual implementation would
        # properly sample from the dataset
        n_examples = self.k_shot + self.n_query
        
        # Dummy data for illustration
        support_data = torch.randn(
            self.n_way * self.k_shot, 3, 224, 224
        ).to(device)
        
        support_labels = torch.arange(self.n_way).repeat(self.k_shot).to(device)
        
        query_data = torch.randn(
            self.n_way * self.n_query, 3, 224, 224
        ).to(device)
        
        query_labels = torch.arange(self.n_way).repeat(self.n_query).to(device)
        
        return support_data, support_labels, query_data, query_labels
    
    def _extract_features(
        self,
        model: nn.Module,
        data: torch.Tensor
    ) -> torch.Tensor:
        """Extract features from model."""
        # Remove final classification layer
        if hasattr(model, 'fc'):
            features = model.features(data)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
        else:
            # Generic feature extraction
            features = model(data)
            if len(features.shape) > 2:
                features = features.mean(dim=[-2, -1])
        
        return features


def evaluate_downstream_robustness(
    original_model: nn.Module,
    compressed_model: nn.Module,
    tasks: List[str],
    data_loaders: Dict[str, Tuple[DataLoader, DataLoader]],
    device: Optional[torch.device] = None
) -> Dict[str, DownstreamResult]:
    """
    Evaluate robustness across multiple downstream tasks.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        tasks: List of tasks to evaluate
        data_loaders: Dictionary of task -> (train_loader, val_loader)
        device: Device to use
        
    Returns:
        Dictionary of task results
    """
    results = {}
    
    for task in tasks:
        logger.info(f"Evaluating downstream task: {task}")
        
        if task == "fine_tuning":
            evaluator = FineTuningEvaluator()
            train_loader, val_loader = data_loaders[task]
            
            result = evaluator.evaluate(
                original_model,
                compressed_model,
                train_loader,
                val_loader,
                task_name=task,
                device=device
            )
            
        elif task.startswith("transfer_"):
            evaluator = TransferLearningEvaluator()
            # Would need proper setup for transfer tasks
            result = None
            
        elif task.startswith("few_shot"):
            evaluator = FewShotEvaluator()
            # Would need proper dataset
            result = None
            
        else:
            logger.warning(f"Unknown task: {task}")
            continue
        
        if result:
            results[task] = result
    
    return results


def analyze_catastrophic_forgetting(
    model: nn.Module,
    original_task_loader: DataLoader,
    new_task_loader: DataLoader,
    num_epochs: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Analyze catastrophic forgetting after fine-tuning.
    
    Args:
        model: Model to analyze
        original_task_loader: Original task data
        new_task_loader: New task data
        num_epochs: Training epochs on new task
        device: Device to use
        
    Returns:
        Forgetting metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model = copy.deepcopy(model).to(device)
    
    # Evaluate on original task
    evaluator = FineTuningEvaluator()
    original_accuracy = evaluator._evaluate_accuracy(
        model, original_task_loader, device
    )
    
    # Fine-tune on new task
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        for data, target in new_task_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Re-evaluate on original task
    final_original_accuracy = evaluator._evaluate_accuracy(
        model, original_task_loader, device
    )
    
    # Evaluate on new task
    new_task_accuracy = evaluator._evaluate_accuracy(
        model, new_task_loader, device
    )
    
    forgetting = original_accuracy - final_original_accuracy
    
    return {
        'original_accuracy_before': original_accuracy,
        'original_accuracy_after': final_original_accuracy,
        'new_task_accuracy': new_task_accuracy,
        'forgetting': forgetting,
        'forgetting_ratio': forgetting / original_accuracy
    }