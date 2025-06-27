"""Natural language processing models for experiments."""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging


logger = logging.getLogger(__name__)


def load_bert_base(
    task_name: str = "sst2",
    pretrained: bool = True,
    num_labels: int = 2,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Any]:
    """
    Load BERT-Base model for GLUE tasks.
    
    Args:
        task_name: GLUE task name
        pretrained: Whether to load pretrained weights
        num_labels: Number of output labels
        device: Device to load model on
        
    Returns:
        BERT model and tokenizer
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "bert-base-uncased" if pretrained else None
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model = model.to(device)
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded BERT-Base with {total_params/1e6:.1f}M parameters")
    
    return model, tokenizer


def prepare_glue_dataset(
    task_name: str,
    tokenizer: Any,
    max_length: int = 128,
    batch_size: int = 32,
    split: str = "validation"
) -> DataLoader:
    """
    Prepare GLUE dataset for evaluation.
    
    Args:
        task_name: GLUE task name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        split: Dataset split
        
    Returns:
        DataLoader for GLUE task
    """
    # Task configurations
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    
    # Load dataset
    if task_name == "mnli-mm":
        dataset = load_dataset("glue", "mnli", split="validation_matched")
    else:
        dataset = load_dataset("glue", task_name, split=split)
    
    # Get task keys
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    def preprocess_function(examples):
        # Handle single vs paired sentences
        if sentence2_key is None:
            texts = examples[sentence1_key]
        else:
            texts = list(zip(examples[sentence1_key], examples[sentence2_key]))
        
        # Tokenize
        result = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Add labels
        if "label" in examples:
            result["labels"] = examples["label"]
        
        return result
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Convert to PyTorch dataset
    tokenized_dataset.set_format("torch")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Created GLUE {task_name} {split} loader with {len(dataset)} samples")
    
    return dataloader


def evaluate_glue_model(
    model: nn.Module,
    dataloader: DataLoader,
    task_name: str,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate model on GLUE task.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        task_name: GLUE task name
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Task-specific metrics
    if task_name in ["mrpc", "qqp"]:
        # F1 and accuracy
        from sklearn.metrics import accuracy_score, f1_score
        metric_fns = {
            "accuracy": accuracy_score,
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary")
        }
    elif task_name in ["stsb"]:
        # Pearson and Spearman correlation
        from scipy.stats import pearsonr, spearmanr
        metric_fns = {
            "pearson": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
            "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred)[0]
        }
    else:
        # Accuracy only
        from sklearn.metrics import accuracy_score
        metric_fns = {"accuracy": accuracy_score}
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            if task_name == "stsb":
                predictions = outputs.logits.squeeze()
            else:
                predictions = outputs.logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    results = {}
    for metric_name, metric_fn in metric_fns.items():
        results[metric_name] = metric_fn(all_labels, all_predictions)
    
    return results


class GLUEModelWrapper(nn.Module):
    """Wrapper for GLUE models to standardize interface."""
    
    def __init__(self, model: nn.Module, task_name: str):
        super().__init__()
        self.model = model
        self.task_name = task_name
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_accuracy(self, output, labels):
        """Calculate accuracy for classification tasks."""
        if self.task_name == "stsb":
            # Regression task
            return 0.0  # Not applicable
        else:
            predictions = output.logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            return correct / labels.size(0)