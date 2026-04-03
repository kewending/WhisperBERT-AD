import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Audio
from sklearn.model_selection import StratifiedKFold

class MultimodalDataCollator:
    """Custom data collator for multimodal audio-text data."""
    
    def __call__(self, features):
        batch = {
            "audio_input_features": [f["audio_input_features"] for f in features],
            "text_input_ids": torch.tensor(
                np.array([f["text_input_ids"] for f in features]), 
                dtype=torch.long
            ),
            "text_attention_mask": torch.tensor(
                np.array([f["text_attention_mask"] for f in features]), 
                dtype=torch.float32
            ),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            "idx": [f["idx"] for f in features],
        }
        return batch

def create_preprocess_function(tokenizer, feature_extractor, audio_chunk_duration=30, max_text_length=512):
    """
    Create a preprocessing function with specific tokenizer and feature extractor.
    
    Args:
        tokenizer: Tokenizer for text processing
        feature_extractor: Feature extractor for audio processing
        chunk_duration: Duration of audio chunks in seconds
        
    Returns:
        Preprocessing function
    """
    sample_rate = feature_extractor.sampling_rate
    samples_per_chunk = audio_chunk_duration * sample_rate
    
    def preprocess_function(examples):
        audio_inputs = []
        
        # Process audio: split into chunks and extract features
        for audio_array in examples["audio"]:
            audio_chunks = [
                audio_array["array"][i:i + samples_per_chunk] 
                for i in range(0, len(audio_array["array"]), samples_per_chunk)
            ]
            audio_input = []
            for chunk in audio_chunks:
                features = feature_extractor(
                    chunk, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                )
                audio_input.append(features.input_features[0])
            audio_inputs.append(audio_input)

        # Tokenize text
        text_inputs = tokenizer(
            examples["transcript"],
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt"
        )

        # Convert labels to integers
        labels = [int(label) for label in examples["label"]]

        return {
            "audio_input_features": audio_inputs,
            "labels": labels,
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            'idx': examples["id"]
        }
    
    return preprocess_function


def load_train_test_dataset(
    train_path, 
    test_path, 
    feature_extractor, 
    tokenizer,
    val_split=0.35, 
    seed=42,
    batch_size=200,
    num_proc=1,
    audio_chunk_duration=30, 
    max_text_length=512
):
    """
    Load and prepare dataset for training.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        feature_extractor: Audio feature extractor
        tokenizer: Text tokenizer
        val_split: Validation split ratio
        seed: Random seed for reproducibility
        batch_size: Batch size for preprocessing
        num_proc: Number of processes for preprocessing
        
    Returns:
        Encoded dataset dictionary with train, eval, and test splits
    """
    # Load datasets
    train_dataset = load_dataset("audiofolder", data_dir=train_path, split="all").shuffle(seed=seed)
    test_dataset = load_dataset("audiofolder", data_dir=test_path, split="all")

    # Split training data into train and validation
    train_val_split = train_dataset.train_test_split(test_size=val_split, seed=seed)
    
    dataset_dict = DatasetDict({
        "train": train_val_split['train'],
        "eval": train_val_split['test'],
        "test": test_dataset
    })

    # Cast audio column to correct sampling rate
    dataset_dict = dataset_dict.cast_column(
        "audio", 
        Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    # Preprocess dataset
    preprocess_fn = create_preprocess_function(tokenizer, feature_extractor, audio_chunk_duration, max_text_length)
    dataset_encoded = dataset_dict.map(
        preprocess_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset_dict["train"].column_names,
        num_proc=num_proc,
    )

    return dataset_encoded

def load_kfold_dataset(data_path, feature_extractor, tokenizer, seed, batch_size, num_proc, audio_chunk_duration, max_text_length, num_folds):
    cv_dataset = load_dataset("audiofolder", data_dir=data_path, split="all").shuffle(seed=seed)
    cv_dataset_encoded = []

    # Preprocess dataset
    preprocess_fn = create_preprocess_function(tokenizer, feature_extractor, audio_chunk_duration, max_text_length)
    dataset_encoded = cv_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    labels = np.array(dataset_encoded["label"])
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        train_dataset = dataset_encoded.select(train_idx)
        test_dataset  = dataset_encoded.select(test_idx)

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        cv_dataset_encoded.append(dataset_dict)
    return cv_dataset_encoded