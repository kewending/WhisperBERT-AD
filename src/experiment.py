from src.model import TextClassifier, AudioClassifier, AudioOTKEClassifier, BERTWhisper, BERTWhisperOTKE,CrossAttention, CrossAttentionOTKE
from src.dataloader import load_train_test_dataset, MultimodalDataCollator, load_kfold_dataset
from src.trainer import MultimodalTrainer, EarlyStoppingCallback, compute_binary_metrics, compute_multi_metrics
from src.help import set_seed, print_and_save, compute_statistics, format_metrics, get_device

import warnings
from transformers import AutoTokenizer, AutoFeatureExtractor, TrainingArguments
import wandb
import os

warnings.filterwarnings("ignore", message=".*1Torch was not compiled with flash attention.*")

class Experiment:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]
        self.train_cfg = config["training"]
        self.mode = config["mode"]
        self.name = config["experiment_name"]
        self.model_cfg["device"] = get_device()

        os.makedirs(self.config['log_dir'], exist_ok=True)
        self.log_file = os.path.join(self.config['log_dir'], f"{self.name}.txt")

    def build_model(self):
        if self.config["model_name"] == "TextClassifier":
            return TextClassifier(**self.model_cfg)
        elif self.config["model_name"] == "AudioClassifier":
            return AudioClassifier(**self.model_cfg)
        elif self.config["model_name"] == "AudioOTKEClassifier":
            return AudioOTKEClassifier(**self.model_cfg)
        elif self.config["model_name"] == "BERTWhisper":
            return BERTWhisper(**self.model_cfg)
        elif self.config["model_name"] == "BERTWhisperOTKE":
            return BERTWhisperOTKE(**self.model_cfg)
        elif self.config["model_name"] == "CrossAttention":
            return CrossAttention(**self.model_cfg)
        elif self.config["model_name"] == "CrossAttentionOTKE":
            return CrossAttentionOTKE(**self.model_cfg)
        else:
            raise ValueError(f"Unknown model type: {self.model_cfg['type']}")
    
    def run(self):
        results_list = []
        if self.mode == "train_test":
            results_list = self.run_train_test()
        elif self.mode == "kfold":
            results_list = self.run_kfold()
        else:
            raise ValueError(f"Unsupported experiment mode: {self.mode}")
        
        if len(results_list) > 0:
            print_and_save("FINAL RESULTS ACROSS ALL SEEDS", self.log_file)
            mean_results, std_results = compute_statistics(results_list)
            print_and_save("Mean Test Results:", self.log_file)
            print_and_save(format_metrics(mean_results), self.log_file)
            print_and_save("\nStd Test Results:", self.log_file)
            print_and_save(format_metrics(std_results), self.log_file)
    
    def run_train_test(self):

        self.data_cfg["feature_extractor"] = AutoFeatureExtractor.from_pretrained(self.config["audio_encoder"])
        self.data_cfg["tokenizer"] = AutoTokenizer.from_pretrained(self.config["text_encoder"])

        test_results_list = []
        for seed in self.config["seeds"]:

            if self.config.get("use_wandb", False):
                wandb.login()
                wandb.init(project="WhisperBERT", name=self.name)

            set_seed(seed)
            self.data_cfg["seed"] = seed
            dataset_encoded = load_train_test_dataset(**self.data_cfg)

            # Setup training arguments
            self.train_cfg["run_name"] = self.name
            self.train_cfg["output_dir"]= os.path.join(self.config['model_output_dir'], f"seed_{seed}")
            self.train_cfg["seed"] = seed
            training_args = TrainingArguments(**self.train_cfg)
        
            # Initialize trainer
            data_collator = MultimodalDataCollator()
            trainer = MultimodalTrainer(
                model=self.build_model(),
                args=training_args,
                train_dataset=dataset_encoded["train"],
                eval_dataset=dataset_encoded["eval"],
                compute_metrics=compute_binary_metrics,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=self.config["early_stopping_patience"]
                )],
                verbose=self.config["verbose"],  # Set based on args if needed
                dataType=self.config["data_type"],
                num_labels = self.model_cfg["num_labels"]
            )

            # Setup logging
            trainer.train()
            val_results = trainer.evaluate(dataset_encoded["eval"])
            print_and_save(f"Validation Results (seed {seed}):", self.log_file)
            print_and_save(format_metrics(val_results), self.log_file)

            test_results = trainer.evaluate(dataset_encoded["test"])
            print_and_save(f"Test Results (seed {seed}):", self.log_file)
            print_and_save(format_metrics(test_results), self.log_file)

            wandb.finish()
            test_results_list.append(test_results)
        return test_results_list
    
    def run_kfold(self):
        self.data_cfg["feature_extractor"] = AutoFeatureExtractor.from_pretrained(self.config["audio_encoder"])
        self.data_cfg["tokenizer"] = AutoTokenizer.from_pretrained(self.config["text_encoder"])

        test_results_list = []
        for seed in self.config["seeds"]:
            set_seed(seed)
            self.data_cfg["seed"] = seed
            kfold_dataset_encoded = load_kfold_dataset(**self.data_cfg)
            kfold_results_list = []
            for fold, fold_dataset in enumerate(kfold_dataset_encoded):
                if self.config.get("use_wandb", False):
                    wandb.login()
                    wandb.init(project="WhisperBERT", name=self.name)
                print_and_save(f"Fold {fold+1}/{self.data_cfg['num_folds']}", self.log_file)

                # Setup training arguments
                self.train_cfg["run_name"] = self.name
                self.train_cfg["output_dir"]= os.path.join(self.config['model_output_dir'], f"seed_{seed}")
                self.train_cfg["seed"] = seed
                training_args = TrainingArguments(**self.train_cfg)
            
                # Initialize trainer
                data_collator = MultimodalDataCollator()
                trainer = MultimodalTrainer(
                    model=self.build_model(),
                    args=training_args,
                    train_dataset=fold_dataset["train"],
                    eval_dataset=fold_dataset["test"],
                    compute_metrics=compute_multi_metrics,
                    data_collator=data_collator,
                    callbacks=[EarlyStoppingCallback(
                        early_stopping_patience=self.config["early_stopping_patience"]
                    )],
                    verbose=self.config["verbose"],  # Set based on args if needed
                    dataType=self.config["data_type"],
                    num_labels = self.model_cfg["num_labels"]
                )

                # Setup logging
                trainer.train()

                test_results = trainer.evaluate(fold_dataset["test"])
                print_and_save(f"Test Results (seed {seed}):", self.log_file)
                print_and_save(format_metrics(test_results), self.log_file)

                wandb.finish()
                kfold_results_list.append(test_results)
            
            print_and_save(f"Kfold result for Seed: {seed}", self.log_file)
            mean_results, std_results = compute_statistics(kfold_results_list)
            print_and_save("Mean Test Results:", self.log_file)
            print_and_save(format_metrics(mean_results), self.log_file)
            print_and_save("\nStd Test Results:", self.log_file)
            print_and_save(format_metrics(std_results), self.log_file)
            test_results_list.append(mean_results)
        
        return test_results_list