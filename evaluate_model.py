"""
Ananta Model Evaluation
=======================

Comprehensive evaluation suite for the fine-tuned Ananta scientific LLM.
This script provides quantitative metrics, qualitative analysis, and comparative
benchmarks to assess model performance on mathematical reasoning tasks.

The evaluation covers multiple dimensions:
- Mathematical accuracy across difficulty levels
- Reasoning quality and step-by-step analysis
- Computational efficiency and response time
- Comparison with baseline models

Author: Ananta Team
Usage: python evaluate_model.py --model_path ./deepseek_finetuned
"""

import os
import json
import logging
import torch
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnantaEvaluator:
    """
    Comprehensive evaluation framework for Ananta scientific LLM.
    
    This class provides methods for loading models, running evaluations,
    computing metrics, and generating detailed analysis reports.
    """
    
    def __init__(self, model_path: str, base_model: str = "deepseek-ai/deepseek-math-7b"):
        """
        Initialize evaluator with model paths and configuration.
        
        Args:
            model_path: Path to fine-tuned model directory
            base_model: Base model identifier for comparison
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation results storage
        self.results = {
            "accuracy_by_difficulty": {},
            "response_times": [],
            "qualitative_scores": {},
            "error_analysis": {},
            "sample_predictions": []
        }
        
        logger.info(f"Initializing evaluator for model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, load_in_8bit: bool = True):
        """
        Load the fine-tuned model and tokenizer.
        
        Args:
            load_in_8bit: Whether to use 8-bit quantization for inference
        """
        logger.info("Loading fine-tuned model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            padding_side="left"  # For generation
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA weights
        if (self.model_path / "adapter_config.json").exists():
            logger.info("Loading LoRA adapter weights...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            logger.warning("No adapter config found. Using base model.")
            self.model = base_model
        
        # Set to evaluation mode
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def load_test_dataset(self, dataset_path: str = "formatted_math_dataset.json", 
                         max_samples: int = 1000) -> List[Dict]:
        """
        Load test dataset for evaluation.
        
        Args:
            dataset_path: Path to the dataset file
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            List of test samples
        """
        logger.info(f"Loading test dataset from {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Sample from each difficulty level for balanced evaluation
            test_samples = []
            difficulty_counts = {}
            
            for item in data:
                difficulty = item.get('difficulty', 'unknown')
                if difficulty not in difficulty_counts:
                    difficulty_counts[difficulty] = 0
                
                # Limit samples per difficulty
                samples_per_difficulty = max_samples // 5  # Assuming 5 difficulty levels
                if difficulty_counts[difficulty] < samples_per_difficulty:
                    test_samples.append(item)
                    difficulty_counts[difficulty] += 1
                
                if len(test_samples) >= max_samples:
                    break
            
            logger.info(f"Loaded {len(test_samples)} test samples")
            logger.info(f"Distribution: {difficulty_counts}")
            
            return test_samples
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.1) -> Tuple[str, float]:
        """
        Generate model response for a given prompt.
        
        Args:
            prompt: Input prompt for the model
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_text, generation_time)
        """
        # Format prompt consistently
        formatted_prompt = f"""### Instruction:
Solve the following mathematical problem step-by-step, showing your reasoning clearly.

### Input:
{prompt}

### Response:
"""
        
        # Tokenize input
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        response_start = generated_text.find("### Response:") + len("### Response:")
        response = generated_text[response_start:].strip()
        
        return response, generation_time
    
    def extract_numerical_answer(self, text: str) -> str:
        """
        Extract numerical answer from model response.
        
        Args:
            text: Generated response text
            
        Returns:
            Extracted numerical answer or empty string
        """
        # Common patterns for mathematical answers
        patterns = [
            r'(?:answer is|equals?|=)\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)\s*(?:is the answer|is correct)',
            r'final answer:\s*([+-]?\d*\.?\d+)',
            r'therefore[,:]?\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)$'  # Number at end of text
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        
        # If no pattern matches, try to find any number
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        if numbers:
            return numbers[-1]  # Return last number found
        
        return ""
    
    def evaluate_accuracy(self, test_samples: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model accuracy across different difficulty levels.
        
        Args:
            test_samples: List of test samples
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Starting accuracy evaluation...")
        
        accuracy_results = {}
        difficulty_results = {}
        
        for sample in test_samples:
            problem = sample['input']
            correct_answer = self.extract_numerical_answer(sample['output'])
            difficulty = sample.get('difficulty', 'unknown')
            
            # Generate model response
            response, gen_time = self.generate_response(problem)
            predicted_answer = self.extract_numerical_answer(response)
            
            # Store timing information
            self.results["response_times"].append(gen_time)
            
            # Check accuracy
            is_correct = self.compare_answers(correct_answer, predicted_answer)
            
            # Update difficulty-specific results
            if difficulty not in difficulty_results:
                difficulty_results[difficulty] = {"correct": 0, "total": 0}
            
            difficulty_results[difficulty]["total"] += 1
            if is_correct:
                difficulty_results[difficulty]["correct"] += 1
            
            # Store sample for analysis
            self.results["sample_predictions"].append({
                "problem": problem,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "response": response,
                "is_correct": is_correct,
                "difficulty": difficulty,
                "generation_time": gen_time
            })
        
        # Calculate accuracy by difficulty
        for difficulty, results in difficulty_results.items():
            accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
            accuracy_results[difficulty] = accuracy
            self.results["accuracy_by_difficulty"][difficulty] = accuracy
        
        # Overall accuracy
        total_correct = sum(r["correct"] for r in difficulty_results.values())
        total_samples = sum(r["total"] for r in difficulty_results.values())
        accuracy_results["overall"] = total_correct / total_samples if total_samples > 0 else 0
        
        logger.info(f"Evaluation completed. Overall accuracy: {accuracy_results['overall']:.3f}")
        
        return accuracy_results
    
    def compare_answers(self, correct: str, predicted: str, tolerance: float = 1e-6) -> bool:
        """
        Compare numerical answers with tolerance for floating point errors.
        
        Args:
            correct: Correct answer string
            predicted: Predicted answer string
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if answers match within tolerance
        """
        if not correct or not predicted:
            return False
        
        try:
            correct_num = float(correct)
            predicted_num = float(predicted)
            return abs(correct_num - predicted_num) <= tolerance
        except ValueError:
            # Fallback to string comparison
            return correct.strip().lower() == predicted.strip().lower()
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Analyze common error patterns in model responses.
        
        Returns:
            Dictionary with error analysis results
        """
        logger.info("Analyzing error patterns...")
        
        error_analysis = {
            "common_errors": {},
            "error_by_difficulty": {},
            "response_length_analysis": {},
            "reasoning_quality": {}
        }
        
        incorrect_samples = [s for s in self.results["sample_predictions"] if not s["is_correct"]]
        
        # Analyze error patterns by difficulty
        for sample in incorrect_samples:
            difficulty = sample["difficulty"]
            if difficulty not in error_analysis["error_by_difficulty"]:
                error_analysis["error_by_difficulty"][difficulty] = 0
            error_analysis["error_by_difficulty"][difficulty] += 1
        
        # Analyze response lengths
        response_lengths = [len(s["response"].split()) for s in self.results["sample_predictions"]]
        error_analysis["response_length_analysis"] = {
            "mean_length": np.mean(response_lengths),
            "std_length": np.std(response_lengths),
            "min_length": np.min(response_lengths),
            "max_length": np.max(response_lengths)
        }
        
        self.results["error_analysis"] = error_analysis
        return error_analysis
    
    def generate_evaluation_report(self, output_dir: str = "evaluation_results"):
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        logger.info("Generating evaluation report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary = {
            "model_path": str(self.model_path),
            "evaluation_date": datetime.now().isoformat(),
            "overall_accuracy": self.results["accuracy_by_difficulty"].get("overall", 0),
            "accuracy_by_difficulty": self.results["accuracy_by_difficulty"],
            "average_response_time": np.mean(self.results["response_times"]),
            "total_samples_evaluated": len(self.results["sample_predictions"])
        }
        
        with open(output_path / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualizations
        self.create_visualizations(output_path)
        
        # Generate sample analysis
        self.create_sample_analysis(output_path)
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    def create_visualizations(self, output_path: Path):
        """
        Create visualization plots for evaluation results.
        
        Args:
            output_path: Path to save visualizations
        """
        plt.style.use('seaborn-v0_8')
        
        # Accuracy by difficulty plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy by difficulty
        difficulties = list(self.results["accuracy_by_difficulty"].keys())
        accuracies = list(self.results["accuracy_by_difficulty"].values())
        
        ax1.bar(difficulties, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot 2: Response time distribution
        ax2.hist(self.results["response_times"], bins=30, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax2.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Response Time (seconds)')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Accuracy vs Response Time
        sample_data = self.results["sample_predictions"]
        correct_times = [s["generation_time"] for s in sample_data if s["is_correct"]]
        incorrect_times = [s["generation_time"] for s in sample_data if not s["is_correct"]]
        
        ax3.boxplot([correct_times, incorrect_times], labels=['Correct', 'Incorrect'])
        ax3.set_title('Response Time: Correct vs Incorrect', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Response Time (seconds)')
        
        # Plot 4: Sample distribution by difficulty
        difficulty_counts = {}
        for sample in sample_data:
            diff = sample["difficulty"]
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        ax4.pie(difficulty_counts.values(), labels=difficulty_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Sample Distribution by Difficulty', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / "evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved")
    
    def create_sample_analysis(self, output_path: Path):
        """
        Create detailed sample analysis with examples.
        
        Args:
            output_path: Path to save analysis
        """
        # Select representative samples
        correct_samples = [s for s in self.results["sample_predictions"] if s["is_correct"]]
        incorrect_samples = [s for s in self.results["sample_predictions"] if not s["is_correct"]]
        
        analysis_data = []
        
        # Add best and worst examples
        if correct_samples:
            best_sample = min(correct_samples, key=lambda x: x["generation_time"])
            analysis_data.append(("Best Correct Example", best_sample))
        
        if incorrect_samples:
            worst_sample = max(incorrect_samples, key=lambda x: x["generation_time"])
            analysis_data.append(("Challenging Example", worst_sample))
        
        # Save sample analysis
        with open(output_path / "sample_analysis.txt", 'w', encoding='utf-8') as f:
            f.write("ANANTA MODEL EVALUATION - SAMPLE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            for title, sample in analysis_data:
                f.write(f"{title}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Problem: {sample['problem']}\n\n")
                f.write(f"Correct Answer: {sample['correct_answer']}\n")
                f.write(f"Predicted Answer: {sample['predicted_answer']}\n")
                f.write(f"Difficulty: {sample['difficulty']}\n")
                f.write(f"Generation Time: {sample['generation_time']:.3f}s\n")
                f.write(f"Is Correct: {sample['is_correct']}\n\n")
                f.write(f"Full Response:\n{sample['response']}\n\n")
                f.write("=" * 50 + "\n\n")


def main():
    """
    Main evaluation execution with command-line interface.
    """
    parser = argparse.ArgumentParser(description="Evaluate Ananta scientific LLM")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset_path", default="formatted_math_dataset.json", 
                       help="Path to test dataset")
    parser.add_argument("--max_samples", type=int, default=1000, 
                       help="Maximum samples to evaluate")
    parser.add_argument("--output_dir", default="evaluation_results", 
                       help="Directory to save results")
    parser.add_argument("--load_in_8bit", action="store_true", 
                       help="Use 8-bit quantization for inference")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AnantaEvaluator(args.model_path)
    
    try:
        # Load model
        evaluator.load_model(load_in_8bit=args.load_in_8bit)
        
        # Load test dataset
        test_samples = evaluator.load_test_dataset(args.dataset_path, args.max_samples)
        
        if not test_samples:
            logger.error("No test samples loaded. Evaluation cannot proceed.")
            return
        
        # Run evaluation
        accuracy_results = evaluator.evaluate_accuracy(test_samples)
        
        # Analyze errors
        error_analysis = evaluator.analyze_errors()
        
        # Generate report
        evaluator.generate_evaluation_report(args.output_dir)
        
        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        for difficulty, accuracy in accuracy_results.items():
            logger.info(f"{difficulty}: {accuracy:.3f}")
        
        logger.info(f"Average response time: {np.mean(evaluator.results['response_times']):.3f}s")
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 