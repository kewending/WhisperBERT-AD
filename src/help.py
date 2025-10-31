import random
import logging
import json
import torch
import numpy as np
import os
from datetime import datetime

def set_seed(seed=42):
    """
    Set seed for deep learning reproducibility

    Args:
        seed (int): Random seed
    """
    logging.info(f"Seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_and_save(message, file_path="training_log.txt"):
    """
    Print message to console and save to file with timestamp.
    
    Args:
        message: Message to print and save
        file_path: Path to log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_with_timestamp = f"{timestamp} - {message}"
    
    print(message_with_timestamp)
    
    with open(file_path, "a") as file:
        file.write(message_with_timestamp + "\n")

def save_results(results, output_dir, filename="results.json"):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Name of output file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")

def compute_statistics(results_list):
    """
    Compute mean and std across multiple runs.
    Handles both scalar metrics and confusion matrices.
    
    Args:
        results_list: List of result dictionaries (each containing metrics, possibly including 'confusion_matrix')
    
    Returns:
        Tuple (mean_results, std_results)
    """
    if not results_list:
        return {}, {}
    
    mean_results = {}
    std_results = {}
    
    keys = results_list[0].keys()
    
    for key in keys:
        values = [res[key] for res in results_list if key in res]

        # --- Handle confusion matrices ---
        if isinstance(values[0], (list, np.ndarray)) and np.array(values[0]).ndim == 2:
            matrices = []
            for v in values:
                cm = np.array(v, dtype=float)
                row_sums = cm.sum(axis=1, keepdims=True)
                # Avoid division by zero for empty rows
                row_sums[row_sums == 0] = 1
                cm_percent = (cm / row_sums) * 100  # Convert to percentage
                matrices.append(cm_percent)
            
            arr = np.stack(matrices)  # shape (num_runs, n_classes, n_classes)
            mean_results[key] = np.mean(arr, axis=0).tolist()
            std_results[key] = np.std(arr, axis=0).tolist()
        else:
            # --- Handle scalar metrics ---
            mean_results[key] = float(np.mean(values))
            std_results[key] = float(np.std(values))
    
    return mean_results, std_results


def format_metrics(metrics, prefix=" "):
    """
    Format metrics dictionary for pretty printing.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix to add to metric names
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def get_device():
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device