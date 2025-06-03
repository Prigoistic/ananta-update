import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset

model_name = "deepseek-ai/deepseek-math-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  
    device_map="auto"
)


dataset = load_dataset("json", data_files="formatted_math_dataset.json", split="train")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              
    lora_alpha=32,    
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  
)

model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./deepseek_finetuned",
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=16,  
    optim="adamw_torch",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=5e-5,
    save_total_limit=2,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()
