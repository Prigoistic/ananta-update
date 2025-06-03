"""
Ananta Complete Training Pipeline
================================

Orchestrates the complete workflow for training and deploying the Ananta scientific LLM.
This script provides a single entry point for data processing, training, evaluation,
and deployment preparation.

Features:
- Data processing from DeepMind dataset
- Model fine-tuning with LoRA
- Comprehensive evaluation
- Deployment preparation for multiple platforms

Author: Ananta Team
Usage: python run_pipeline.py [--step <step_name>] [--config <config_file>]
"""

import argparse
import subprocess
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnantaPipeline:
    """
    Complete pipeline manager for Ananta scientific LLM development.
    
    This class orchestrates data processing, training, evaluation, and deployment
    with proper error handling and progress tracking.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the pipeline with optional configuration.
        
        Args:
            config_file: Path to training configuration JSON file
        """
        self.config_file = config_file
        self.start_time = time.time()
        self.steps_completed = []
        self.steps_failed = []
        
        # Define pipeline steps
        self.available_steps = {
            "data": "Process DeepMind dataset into training format",
            "train": "Fine-tune model with LoRA adaptation", 
            "evaluate": "Evaluate trained model performance",
            "deploy_prep": "Prepare deployment files for various platforms",
            "all": "Run complete pipeline from start to finish"
        }
        
        logger.info("Ananta Pipeline initialized")
        logger.info(f"Available steps: {list(self.available_steps.keys())}")
    
    def run_command(self, command: List[str], step_name: str) -> bool:
        """
        Execute a command with proper logging and error handling.
        
        Args:
            command: Command to execute as list of strings
            step_name: Name of the pipeline step for logging
            
        Returns:
            True if command succeeded, False otherwise
        """
        logger.info(f"Starting step: {step_name}")
        logger.info(f"Command: {' '.join(command)}")
        
        try:
            # Run command with real-time output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line.rstrip())
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                logger.info(f"✅ Step '{step_name}' completed successfully")
                self.steps_completed.append(step_name)
                return True
            else:
                logger.error(f"❌ Step '{step_name}' failed with return code {return_code}")
                self.steps_failed.append(step_name)
                return False
                
        except Exception as e:
            logger.error(f"❌ Step '{step_name}' failed with exception: {e}")
            self.steps_failed.append(step_name)
            return False
    
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met for running the pipeline.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        logger.info("Checking prerequisites...")
        
        # Check Python packages
        required_packages = [
            "torch", "transformers", "datasets", "peft", "trl", 
            "gradio", "accelerate", "bitsandbytes"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install requirements: pip install -r requirements.txt")
            return False
        
        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.warning("⚠️ CUDA not available. Training will be slow on CPU.")
        except Exception as e:
            logger.warning(f"Could not check CUDA status: {e}")
        
        logger.info("✅ Prerequisites check completed")
        return True
    
    def step_data_processing(self) -> bool:
        """Process the mathematics dataset."""
        logger.info("=== Data Processing Step ===")
        
        # Check if dataset directory exists
        if not Path("mathematics_dataset-v1.0").exists():
            logger.error("Mathematics dataset directory not found!")
            logger.error("Please download and extract mathematics_dataset-v1.0.tar.gz")
            return False
        
        command = [sys.executable, "data_processor.py"]
        return self.run_command(command, "data_processing")
    
    def step_training(self) -> bool:
        """Train the model with LoRA fine-tuning."""
        logger.info("=== Model Training Step ===")
        
        # Check if processed dataset exists
        if not Path("formatted_math_dataset.json").exists():
            logger.error("Processed dataset not found. Run data processing first.")
            return False
        
        command = [sys.executable, "train_ananta.py"]
        if self.config_file:
            command.extend(["--config", self.config_file])
        
        return self.run_command(command, "training")
    
    def step_evaluation(self) -> bool:
        """Evaluate the trained model."""
        logger.info("=== Model Evaluation Step ===")
        
        # Check if trained model exists
        if not Path("deepseek_finetuned").exists():
            logger.error("Trained model not found. Run training first.")
            return False
        
        command = [
            sys.executable, "evaluate_model.py",
            "--model_path", "deepseek_finetuned",
            "--max_samples", "500"  # Reduced for faster evaluation
        ]
        
        return self.run_command(command, "evaluation")
    
    def step_deployment_prep(self) -> bool:
        """Prepare deployment files."""
        logger.info("=== Deployment Preparation Step ===")
        
        command = [sys.executable, "deploy_hf_spaces.py"]
        return self.run_command(command, "deployment_preparation")
    
    def run_step(self, step: str) -> bool:
        """
        Run a specific pipeline step.
        
        Args:
            step: Name of the step to run
            
        Returns:
            True if step succeeded, False otherwise
        """
        step_methods = {
            "data": self.step_data_processing,
            "train": self.step_training,
            "evaluate": self.step_evaluation,
            "deploy_prep": self.step_deployment_prep
        }
        
        if step not in step_methods:
            logger.error(f"Unknown step: {step}")
            return False
        
        return step_methods[step]()
    
    def run_all_steps(self) -> bool:
        """Run the complete pipeline."""
        logger.info("=== Running Complete Ananta Pipeline ===")
        
        steps = ["data", "train", "evaluate", "deploy_prep"]
        
        for step in steps:
            success = self.run_step(step)
            if not success:
                logger.error(f"Pipeline failed at step: {step}")
                return False
        
        return True
    
    def generate_report(self) -> None:
        """Generate a final pipeline execution report."""
        elapsed_time = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("ANANTA PIPELINE EXECUTION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info(f"Steps completed: {len(self.steps_completed)}")
        logger.info(f"Steps failed: {len(self.steps_failed)}")
        
        if self.steps_completed:
            logger.info(f"✅ Successful steps: {', '.join(self.steps_completed)}")
        
        if self.steps_failed:
            logger.info(f"❌ Failed steps: {', '.join(self.steps_failed)}")
        
        # Provide next steps guidance
        if not self.steps_failed:
            logger.info("\n🎉 Pipeline completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Review evaluation results in evaluation_results/")
            logger.info("2. Test the model: python app.py")
            logger.info("3. Deploy to HF Spaces using files in hf_spaces_deploy/")
        else:
            logger.info("\n⚠️ Pipeline completed with errors.")
            logger.info("Please check the logs and resolve issues before proceeding.")
        
        logger.info("=" * 60)


def main():
    """Main pipeline execution with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ananta Scientific LLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --step all                    # Run complete pipeline
  python run_pipeline.py --step data                   # Process dataset only
  python run_pipeline.py --step train --config my.json # Train with custom config
  python run_pipeline.py --step evaluate               # Evaluate existing model
        """
    )
    
    parser.add_argument(
        "--step",
        choices=["data", "train", "evaluate", "deploy_prep", "all"],
        default="all",
        help="Pipeline step to run"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration JSON file"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AnantaPipeline(args.config)
    
    try:
        # Check prerequisites unless skipped
        if not args.skip_checks and not pipeline.check_prerequisites():
            logger.error("Prerequisite check failed. Use --skip-checks to override.")
            sys.exit(1)
        
        # Run requested step(s)
        if args.step == "all":
            success = pipeline.run_all_steps()
        else:
            success = pipeline.run_step(args.step)
        
        # Generate report
        pipeline.generate_report()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        pipeline.generate_report()
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        pipeline.generate_report()
        sys.exit(1)


if __name__ == "__main__":
    main() 