"""
Test Your Ananta Model
======================

Simple script to test your fine-tuned Ananta model.
You can ask it math problems and see how well it does!

Usage: python test_ananta.py

Make sure you've trained your model first with easy_train.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

def load_model(model_path="./ananta_final"):
    """
    Load your trained Ananta model.
    """
    print(f"Loading your trained model from {model_path}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-math-7b",
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        # Load your fine-tuned weights
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print("Model loaded successfully! ✓")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained the model first with easy_train.py")
        return None, None

def ask_math_question(model, tokenizer, question):
    """
    Ask Ananta a math question and get the answer.
    """
    # Format the question like we did during training
    prompt = f"### Instruction:\nSolve the following mathematical problem step-by-step, showing your reasoning clearly.\n\n### Problem:\n{question}\n\n### Solution:\n"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate response
    print("Thinking...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # How long the answer can be
            temperature=0.7,     # Controls randomness (lower = more focused)
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the solution part
    if "### Solution:" in full_response:
        solution = full_response.split("### Solution:")[-1].strip()
    else:
        solution = full_response
    
    end_time = time.time()
    
    return solution, end_time - start_time

def interactive_mode(model, tokenizer):
    """
    Chat with Ananta interactively.
    """
    print("\n" + "="*50)
    print("Interactive Math Solver")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        # Get question from user
        question = input("\n🤔 Ask me a math problem: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            continue
        
        # Get answer from model
        try:
            answer, response_time = ask_math_question(model, tokenizer, question)
            
            print(f"\n🧠 Ananta's answer (took {response_time:.1f}s):")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except Exception as e:
            print(f"Error getting answer: {e}")

def test_sample_problems(model, tokenizer):
    """
    Test with some sample math problems.
    """
    print("\n" + "="*50)
    print("Testing with sample problems...")
    print("="*50)
    
    sample_problems = [
        "What is 25 + 17?",
        "Solve for x: 2x + 5 = 13",
        "What is the area of a circle with radius 3?",
        "Factor the expression x² - 9",
        "What is the derivative of x³ + 2x?",
    ]
    
    for i, problem in enumerate(sample_problems, 1):
        print(f"\n{i}. Problem: {problem}")
        
        try:
            answer, response_time = ask_math_question(model, tokenizer, problem)
            print(f"   Answer ({response_time:.1f}s): {answer[:100]}...")  # Show first 100 chars
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """
    Main function - test your Ananta model!
    """
    print("🧮 Ananta Math Model Tester")
    print("="*40)
    
    # Load the model
    model, tokenizer = load_model()
    
    if model is None:
        return
    
    # Check what the user wants to do
    print("\nWhat would you like to do?")
    print("1. Test with sample problems")
    print("2. Interactive mode (ask your own questions)")
    print("3. Both")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        test_sample_problems(model, tokenizer)
    
    if choice in ['2', '3']:
        interactive_mode(model, tokenizer)
    
    print("\nThanks for testing Ananta! 🎉")

if __name__ == "__main__":
    # Quick GPU check
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU (this will be slower)")
    
    main() 