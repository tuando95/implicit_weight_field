"""Adversarial robustness evaluation for compressed models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Result from adversarial evaluation."""
    attack_type: str
    epsilon: float
    original_accuracy: float
    adversarial_accuracy: float
    original_robustness: float
    compressed_robustness: float
    robustness_retention: float
    attack_success_rate: float
    avg_perturbation_norm: float


class AdversarialEvaluator:
    """Evaluate adversarial robustness of models."""
    
    def __init__(
        self,
        attack_types: List[str] = ["fgsm", "pgd"],
        epsilons: List[float] = [0.01, 0.03, 0.1],
        pgd_steps: int = 20,
        pgd_alpha: Optional[float] = None
    ):
        """
        Initialize adversarial evaluator.
        
        Args:
            attack_types: Types of attacks to evaluate
            epsilons: Perturbation budgets
            pgd_steps: Number of PGD steps
            pgd_alpha: Step size for PGD (default: 2.5 * epsilon / steps)
        """
        self.attack_types = attack_types
        self.epsilons = epsilons
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
    
    def evaluate(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_loader: Any,
        device: Optional[torch.device] = None
    ) -> List[AdversarialResult]:
        """
        Evaluate adversarial robustness.
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_loader: Test data loader
            device: Device to use
            
        Returns:
            List of adversarial evaluation results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        original_model.eval()
        compressed_model.eval()
        
        results = []
        
        for attack_type in self.attack_types:
            for epsilon in self.epsilons:
                logger.info(f"Evaluating {attack_type} with epsilon={epsilon}")
                
                result = self._evaluate_attack(
                    original_model,
                    compressed_model,
                    test_loader,
                    attack_type,
                    epsilon,
                    device
                )
                
                results.append(result)
        
        return results
    
    def _evaluate_attack(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_loader: Any,
        attack_type: str,
        epsilon: float,
        device: torch.device
    ) -> AdversarialResult:
        """Evaluate a specific attack."""
        # Clean accuracy
        orig_clean_acc = self._evaluate_clean_accuracy(original_model, test_loader, device)
        comp_clean_acc = self._evaluate_clean_accuracy(compressed_model, test_loader, device)
        
        # Generate adversarial examples
        if attack_type == "fgsm":
            attack_fn = self._fgsm_attack
        elif attack_type == "pgd":
            attack_fn = self._pgd_attack
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Evaluate on adversarial examples
        orig_adv_acc, orig_success_rate, orig_pert_norm = self._evaluate_adversarial(
            original_model, test_loader, attack_fn, epsilon, device
        )
        
        comp_adv_acc, comp_success_rate, comp_pert_norm = self._evaluate_adversarial(
            compressed_model, test_loader, attack_fn, epsilon, device
        )
        
        # Calculate robustness (accuracy on adversarial examples)
        orig_robustness = orig_adv_acc / orig_clean_acc if orig_clean_acc > 0 else 0
        comp_robustness = comp_adv_acc / comp_clean_acc if comp_clean_acc > 0 else 0
        
        return AdversarialResult(
            attack_type=attack_type,
            epsilon=epsilon,
            original_accuracy=orig_clean_acc,
            adversarial_accuracy=comp_adv_acc,
            original_robustness=orig_robustness,
            compressed_robustness=comp_robustness,
            robustness_retention=(comp_robustness / orig_robustness * 100) if orig_robustness > 0 else 0,
            attack_success_rate=comp_success_rate,
            avg_perturbation_norm=comp_pert_norm
        )
    
    def _evaluate_clean_accuracy(
        self,
        model: nn.Module,
        loader: Any,
        device: torch.device
    ) -> float:
        """Evaluate clean accuracy."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        return correct / total
    
    def _evaluate_adversarial(
        self,
        model: nn.Module,
        loader: Any,
        attack_fn: callable,
        epsilon: float,
        device: torch.device
    ) -> Tuple[float, float, float]:
        """Evaluate on adversarial examples."""
        correct = 0
        total = 0
        successful_attacks = 0
        total_pert_norm = 0
        
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Generate adversarial examples
            adv_data = attack_fn(model, data, target, epsilon, device)
            
            # Evaluate
            with torch.no_grad():
                clean_output = model(data)
                clean_pred = clean_output.argmax(dim=1)
                
                adv_output = model(adv_data)
                adv_pred = adv_output.argmax(dim=1)
                
                correct += adv_pred.eq(target).sum().item()
                total += len(target)
                
                # Count successful attacks (changed prediction)
                successful_attacks += (clean_pred != adv_pred).sum().item()
                
                # Measure perturbation norm
                pert = adv_data - data
                pert_norm = pert.view(pert.size(0), -1).norm(2, dim=1).mean().item()
                total_pert_norm += pert_norm
        
        accuracy = correct / total
        success_rate = successful_attacks / total
        avg_pert_norm = total_pert_norm / len(loader)
        
        return accuracy, success_rate, avg_pert_norm
    
    def _fgsm_attack(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        device: torch.device
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        data.requires_grad = True
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        data_grad = data.grad.data
        perturbation = epsilon * data_grad.sign()
        
        # Add perturbation
        adv_data = data + perturbation
        adv_data = torch.clamp(adv_data, 0, 1)  # Ensure valid range
        
        return adv_data.detach()
    
    def _pgd_attack(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        device: torch.device
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        alpha = self.pgd_alpha or (2.5 * epsilon / self.pgd_steps)
        
        # Random initialization
        adv_data = data.clone().detach()
        noise = torch.empty_like(data).uniform_(-epsilon, epsilon)
        adv_data = adv_data + noise
        adv_data = torch.clamp(adv_data, 0, 1)
        
        for _ in range(self.pgd_steps):
            adv_data.requires_grad = True
            
            output = model(adv_data)
            loss = F.cross_entropy(output, target)
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial data
            adv_data = adv_data.detach() + alpha * adv_data.grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(adv_data - data, min=-epsilon, max=epsilon)
            adv_data = torch.clamp(data + delta, 0, 1).detach()
        
        return adv_data


def generate_adversarial_examples(
    model: nn.Module,
    data_loader: Any,
    attack_type: str = "pgd",
    epsilon: float = 0.03,
    num_examples: int = 10,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate adversarial examples for visualization.
    
    Args:
        model: Model to attack
        data_loader: Data loader
        attack_type: Type of attack
        epsilon: Perturbation budget
        num_examples: Number of examples to generate
        device: Device to use
        
    Returns:
        (original_images, adversarial_images, perturbations)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    evaluator = AdversarialEvaluator()
    
    original_images = []
    adversarial_images = []
    perturbations = []
    
    for data, target in data_loader:
        if len(original_images) >= num_examples:
            break
        
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        if attack_type == "fgsm":
            adv_data = evaluator._fgsm_attack(model, data, target, epsilon, device)
        elif attack_type == "pgd":
            adv_data = evaluator._pgd_attack(model, data, target, epsilon, device)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Store examples
        for i in range(min(len(data), num_examples - len(original_images))):
            original_images.append(data[i].cpu())
            adversarial_images.append(adv_data[i].cpu())
            perturbations.append((adv_data[i] - data[i]).cpu())
    
    return (
        torch.stack(original_images),
        torch.stack(adversarial_images),
        torch.stack(perturbations)
    )


def evaluate_adversarial_robustness(
    model: nn.Module,
    test_loader: Any,
    attacks: List[str] = ["fgsm", "pgd"],
    epsilons: List[float] = [0.01, 0.03, 0.1],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Comprehensive adversarial robustness evaluation.
    
    Args:
        model: Model to evaluate
        test_loader: Test data
        attacks: Attack types
        epsilons: Perturbation budgets
        device: Device to use
        
    Returns:
        Robustness evaluation results
    """
    evaluator = AdversarialEvaluator(
        attack_types=attacks,
        epsilons=epsilons
    )
    
    # For single model evaluation, compare against itself
    results = evaluator.evaluate(model, model, test_loader, device)
    
    # Organize results
    robustness_dict = {
        'clean_accuracy': results[0].original_accuracy if results else 0,
        'attacks': {}
    }
    
    for result in results:
        key = f"{result.attack_type}_eps{result.epsilon}"
        robustness_dict['attacks'][key] = {
            'accuracy': result.adversarial_accuracy,
            'robustness': result.compressed_robustness,
            'success_rate': result.attack_success_rate,
            'avg_perturbation': result.avg_perturbation_norm
        }
    
    # Calculate average robustness
    if results:
        avg_robustness = np.mean([r.compressed_robustness for r in results])
        robustness_dict['average_robustness'] = avg_robustness
    
    return robustness_dict


def analyze_gradient_obfuscation(
    model: nn.Module,
    test_loader: Any,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Analyze potential gradient obfuscation in compressed models.
    
    Args:
        model: Model to analyze
        test_loader: Test data
        device: Device to use
        
    Returns:
        Gradient analysis metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    gradient_norms = []
    gradient_variances = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Analyze gradients
        grad = data.grad.data
        grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
        
        gradient_norms.extend(grad_norm.cpu().numpy())
        gradient_variances.append(grad.var().item())
    
    return {
        'mean_gradient_norm': np.mean(gradient_norms),
        'std_gradient_norm': np.std(gradient_norms),
        'mean_gradient_variance': np.mean(gradient_variances),
        'gradient_sparsity': (np.array(gradient_norms) < 1e-8).mean()
    }