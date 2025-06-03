"""
Ananta Training Pipeline
========================

Fine-tunes DeepSeek-Math-7B using LoRA (Low-Rank Adaptation) for scientific
reasoning tasks. This script implements parameter-efficient fine-tuning optimized
for consumer GPUs while maintaining competitive performance on mathematical
problem-solving tasks.

The training pipeline uses Supervised Fine-Tuning (SFT) with careful memory
management and monitoring suitable for academic research environments.

Author: Ananta Team
Usage: python train_ananta.py [--config config.json]
"""

import os
import json
import logging
import torch
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import numpy as np

# Configure logging for training monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnantaTrainer:
    """
    Handles the complete fine-tuning pipeline for Ananta scientific LLM.
    
    This class encapsulates model loading, LoRA configuration, dataset preparation,
    and the actual training process with comprehensive monitoring and checkpointing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trainer with configuration.
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
        # Set random seeds for reproducibility
        set_seed(self.config["training"]["seed"])
        
        # Initialize experiment tracking
        if self.config["monitoring"]["use_wandb"]:
            self.init_wandb()
    
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load training configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "model": {
                "name": "deepseek-ai/deepseek-math-7b",
                "load_in_8bit": True,
                "device_map": "auto",
                "trust_remote_code": True
            },
            "lora": {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "training": {
                "output_dir": "./deepseek_finetuned",
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "num_train_epochs": 3,
                "learning_rate": 5e-5,
                "max_seq_length": 1024,
                "fp16": True,
                "optim": "adamw_torch",
                "seed": 42,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 100,
                "save_total_limit": 3,
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine"
            },
            "data": {
                "dataset_path": "formatted_math_dataset.json",
                "max_train_samples": None,
                "validation_split": 0.1
            },
            "monitoring": {
                "use_wandb": False,
                "wandb_project": "ananta-scientific-llm",
                "report_to": ["tensorboard"]
            }
        }
        
        if config_path and Path(config_path).exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Deep merge configurations
                default_config.update(user_config)
        
        return default_config
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking for experiment monitoring."""
        wandb.init(
            project=self.config["monitoring"]["wandb_project"],
            config=self.config,
            name=f"ananta-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Configure quantization for memory-efficient training on consumer GPUs.
        
        Returns:
            BitsAndBytesConfig for 8-bit quantization
        """
        if not self.config["model"]["load_in_8bit"]:
            return None
        
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        )
    
    def load_model_and_tokenizer(self):
        """
        Load the base model and tokenizer with appropriate configurations.
        
        This method handles model loading with quantization, device mapping,
        and prepares the model for LoRA training.
        """
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
            padding_side="right"  # Important for decoder-only models
        )
        
        # Add padding token if missing (common issue with decoder-only models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        quantization_config = self.setup_quantization_config()
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            quantization_config=quantization_config,
            device_map=self.config["model"]["device_map"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
            torch_dtype=torch.float16 if self.config["training"]["fp16"] else torch.float32,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        # Prepare model for k-bit training if using quantization
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
    
    def setup_lora_config(self) -> LoraConfig:
        """
        Configure LoRA parameters for parameter-efficient fine-tuning.
        
        Returns:
            LoraConfig with specified target modules and hyperparameters
        """
        lora_config = LoraConfig(
            task_type=getattr(TaskType, self.config["lora"]["task_type"]),
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            target_modules=self.config["lora"]["target_modules"],
            bias=self.config["lora"]["bias"],
            inference_mode=False
        )
        
        logger.info(f"LoRA configuration: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        return lora_config
    
    def apply_lora_adaptation(self):
        """Apply LoRA adaptation to the model."""
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters information
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def load_and_prepare_dataset(self) -> Dataset:
        """
        Load and prepare the mathematics dataset for training.
        
        Returns:
            Prepared dataset ready for training
        """
        logger.info(f"Loading dataset from {self.config['data']['dataset_path']}")
        
        # Load the formatted dataset
        dataset = load_dataset(
            "json", 
            data_files=self.config["data"]["dataset_path"],
            split="train"
        )
        
        # Limit dataset size if specified (useful for testing)
        if self.config["data"]["max_train_samples"]:
            dataset = dataset.select(range(self.config["data"]["max_train_samples"]))
        
        # Create instruction template for consistent formatting
        def format_instruction(example):
            """Format the instruction-input-output into a single text field."""
            instruction_text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
            return {"text": instruction_text}
        
        # Apply formatting
        dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
        
        # Split into train/validation if specified
        if self.config["data"]["validation_split"] > 0:
            dataset = dataset.train_test_split(
                test_size=self.config["data"]["validation_split"],
                seed=self.config["training"]["seed"]
            )
            self.dataset = dataset["train"]
            self.eval_dataset = dataset["test"]
        else:
            self.dataset = dataset
            self.eval_dataset = None
        
        logger.info(f"Training dataset size: {len(self.dataset):,}")
        if self.eval_dataset:
            logger.info(f"Validation dataset size: {len(self.eval_dataset):,}")
        
        return self.dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """
        Configure training arguments with optimizations for scientific LLM training.
        
        Returns:
            TrainingArguments configured for the training run
        """
        # Ensure output directory exists
        output_dir = Path(self.config["training"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            learning_rate=self.config["training"]["learning_rate"],
            fp16=self.config["training"]["fp16"],
            optim=self.config["training"]["optim"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"] if self.eval_dataset else None,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_total_limit=self.config["training"]["save_total_limit"],
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            warmup_ratio=self.config["training"]["warmup_ratio"],
            lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
            report_to=self.config["monitoring"]["report_to"],
            run_name=f"ananta-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed=self.config["training"]["seed"],
            data_seed=self.config["training"]["seed"],
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Can help with RTX 3050 memory constraints
        )
        
        return args
    
    def train(self):
        """
        Execute the complete training pipeline.
        
        This method orchestrates the entire training process including model loading,
        dataset preparation, LoRA setup, and the actual training loop.
        """
        logger.info("Starting Ananta training pipeline...")
        
        try:
            # Step 1: Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Step 2: Apply LoRA adaptation
            self.apply_lora_adaptation()
            
            # Step 3: Prepare dataset
            self.load_and_prepare_dataset()
            
            # Step 4: Setup training arguments
            training_args = self.setup_training_arguments()
            
            # Step 5: Initialize trainer
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                args=training_args,
                max_seq_length=self.config["training"]["max_seq_length"],
                dataset_text_field="text",
                packing=False,  # Disable packing for clearer instruction following
            )
            
            # Step 6: Start training
            logger.info("Beginning model training...")
            train_result = self.trainer.train()
            
            # Step 7: Save final model
            logger.info("Saving final model...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config["training"]["output_dir"])
            
            # Step 8: Log training summary
            self.log_training_summary(train_result)
            
            logger.info("✅ Training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.config["monitoring"]["use_wandb"]:
                wandb.finish()
    
    def log_training_summary(self, train_result):
        """
        Log comprehensive training summary for academic reporting.
        
        Args:
            train_result: Training result object from trainer
        """
        logger.info("=== Training Summary ===")
        logger.info(f"Total training time: {train_result.metrics.get('train_runtime', 0):.2f} seconds")
        logger.info(f"Training samples per second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
        logger.info(f"Final training loss: {train_result.metrics.get('train_loss', 0):.4f}")
        
        if hasattr(train_result.metrics, 'eval_loss'):
            logger.info(f"Final validation loss: {train_result.metrics.get('eval_loss', 0):.4f}")
        
        # Log hardware utilization if available
        if torch.cuda.is_available():
            logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        # Save metrics to file for analysis
        metrics_file = Path(self.config["training"]["output_dir"]) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {metrics_file}")


def main():
    """
    Main execution function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Ananta scientific LLM using LoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # Verify CUDA availability for GPU training
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be very slow on CPU.")
        response = input("Continue with CPU training? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled by user.")
            return
    else:
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize and run training
    trainer = AnantaTrainer(args.config)
    
    # Handle checkpoint resumption if specified
    if args.resume_from_checkpoint:
        trainer.config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
        logger.info(f"Will resume training from: {args.resume_from_checkpoint}")
    
    trainer.train()


if __name__ == "__main__":
    main() 