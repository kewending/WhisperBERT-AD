import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

def compute_binary_metrics(eval_pred):
    """
    Compute classification metrics.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        
    Returns:
        Dictionary of metrics
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
 
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall = recall_score(labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Per-class metrics
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1], zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "HC_precision": precisions[0],
        "AD_precision": precisions[1],
        "HC_recall": recalls[0],
        "AD_recall": recalls[1],
        "HC_f1": f1s[0],
        "AD_f1": f1s[1]
    }

def compute_multi_metrics(eval_pred):
    """
    Compute classification metrics for multi-class tasks.
    """
    predictions, labels = eval_pred

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # Per-class metrics
    per_class = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    precisions, recalls, f1s, _ = per_class

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=np.unique(labels))
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": cm.tolist()
    }

    # Add per-class scores
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
        metrics[f"class_{i}_precision"] = p
        metrics[f"class_{i}_recall"] = r
        metrics[f"class_{i}_f1"] = f

    return metrics

class MultimodalTrainer(Trainer):
    """Custom trainer for multimodal audio-text classification."""
    
    def __init__(self, *args, num_labels=2, verbose=False, dataType = "multimodal", **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.dataType = dataType
        self.num_labels = num_labels

    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Create evaluation dataloader."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for the model.
        
        Args:
            model: The model to train
            inputs: Dictionary of input tensors
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        labels = inputs.pop("labels", None)
        # Prepare input arguments based on dataType
        input_args = {}
        if self.dataType in ["multimodal", "audio"]:
            input_args["audio_input_features"] = inputs["audio_input_features"]
        if self.dataType in ["multimodal", "text"]:
            input_args["text_input_ids"] = inputs["text_input_ids"]
            input_args["text_attention_mask"] = inputs["text_attention_mask"]

        outputs = model(**input_args)
        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss = loss_fct(outputs.squeeze(), labels.squeeze())
            predictions = outputs.squeeze()
        elif self.num_labels > 1:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            predictions = torch.argmax(outputs, dim=-1)
            
        if self.verbose:
            idx = inputs["idx"]
            print(f"idx: {idx}")
            print(f"labels: {labels}")
            print(f"outputs: {outputs.view(-1, self.num_labels)}")
            print(f"predictions: {predictions}")
        
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step.
        
        Args:
            model: The model to evaluate
            inputs: Dictionary of input tensors
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore in inputs
            
        Returns:
            Tuple of (loss, predictions, labels)
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].float()
        preds = None
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            if self.num_labels == 1:
                preds = outputs.squeeze()
            elif self.num_labels > 1:
                preds = torch.argmax(outputs, dim=-1)

        return (loss, preds, labels)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if loss is None:
            raise ValueError("Loss is None. Check model outputs and loss calculation.")
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()

        # print(f" training_step loss: {loss}")
        return loss.detach()

class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on evaluation loss."""
    
    def __init__(self, early_stopping_patience: int = 3, metric: str = "eval_loss", mode: str = "min"):
        """
        Initialize early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations to wait before stopping
            metric: Metric to monitor
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.early_stopping_patience = early_stopping_patience
        self.metric = metric
        self.mode = mode
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Called after evaluation."""
        current_metric = metrics.get(self.metric)
        
        if current_metric is None:
            return
        
        improved = (
            (self.mode == "min" and current_metric < self.best_metric) or
            (self.mode == "max" and current_metric > self.best_metric)
        )
        
        if improved:
            self.best_metric = current_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.no_improvement_count} evaluations without improvement.")
                control.should_training_stop = True