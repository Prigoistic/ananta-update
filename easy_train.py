"""
Easy Ananta Training Script
===========================

This is a simple script to fine-tune the DeepSeek math model with your dataset.
I tried to keep it as straightforward as possible - no overcomplicated stuff.

Before running:
1. Make sure you have your dataset ready (run simple_data_converter.py first)
2. Install requirements: pip install -r requirements.txt
3. Run: python easy_train.py

The script will save checkpoints as it goes, so you can stop and resume anytime.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_dataset(file_path="formatted_math_dataset.json"):
    """
    Load our converted math dataset.
    Returns a HuggingFace dataset ready for training.
    """
    print(f"Loading dataset from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} training examples")
        
        # Convert to HuggingFace dataset format
        dataset = Dataset.from_list(data)
        return dataset
        
    except FileNotFoundError:
        print(f"Dataset file {file_path} not found!")
        print("Run 'python simple_data_converter.py' first to create your dataset.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def format_example(example):
    """
    Format each training example into a text that the model can learn from.
    We'll use a simple instruction format.
    """
    text = f"### Instruction:\n{example['instruction']}\n\n"
    text += f"### Problem:\n{example['input']}\n\n"
    text += f"### Solution:\n{example['output']}"
    
    return {"text": text}

def setup_model_and_tokenizer():
    """
    Load the base DeepSeek model and set up LoRA for efficient training.
    """
    model_name = "deepseek-ai/deepseek-math-7b"
    
    print(f"Loading model: {model_name}")
    print("This might take a few minutes the first time...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,  # This helps with RTX 3050's limited VRAM
        trust_remote_code=True
    )
    
    # Set up LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Low rank - keeps it efficient
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_data(dataset, tokenizer):
    """
    Convert our text data into tokens the model can understand.
    """
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Tokenize the formatted text
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,  # Reasonable max length for math problems
            return_tensors="pt"
        )
        
        # For language modeling, labels are the same as input_ids
        tokens["labels"] = tokens["input_ids"].clone()
        
        return tokens
    
    # Apply tokenization to all examples
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def main():
    """
    Main training pipeline - this is where the magic happens!
    """
    print("=" * 60)
    print("Starting Ananta Fine-tuning")
    print("=" * 60)
    
    # Step 1: Load dataset
    dataset = load_dataset()
    if dataset is None:
        return
    
    # Format examples for training
    print("Formatting examples...")
    dataset = dataset.map(format_example, desc="Formatting")
    
    # Step 2: Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 3: Tokenize data
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    # Split into train/validation (90/10 split)
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    
    # Step 4: Set up training
    output_dir = "./ananta_checkpoints"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # How many times to go through the data
        per_device_train_batch_size=1,  # Small batch for RTX 3050
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Simulate larger batches
        learning_rate=5e-5,  # Conservative learning rate
        warmup_steps=100,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,  # Keep only 3 checkpoints to save space
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,  # Use half precision to save memory
        dataloader_pin_memory=False,  # Can help with memory
        remove_unused_columns=False,
        report_to=None,  # Disable wandb for simplicity
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Step 5: Start training!
    print("\nStarting training...")
    print("This will take a while. You can stop anytime with Ctrl+C and resume later.")
    print(f"Checkpoints will be saved to: {output_dir}")
    
    try:
        trainer.train()
        
        # Save the final model
        final_model_path = "./ananta_final"
        print(f"\nSaving final model to {final_model_path}...")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print("Training complete! 🎉")
        print(f"Your trained model is ready at: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Your progress is saved in checkpoints.")
        print("To resume, you can use the latest checkpoint in the output directory.")
    
    except Exception as e:
        print(f"\nTraining error: {e}")
        print("Check the error message above for details.")

if __name__ == "__main__":
    # Quick check for GPU
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("No GPU detected - training will be very slow on CPU!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    main() 