"""
Ananta Data Processor
====================

Converts DeepMind mathematics_dataset-v1.0 from text format into a structured
JSON dataset optimized for scientific LLM fine-tuning. This script handles
various difficulty levels and ensures consistent formatting for instruction-tuning.

The output format follows the standard instruction-input-output paradigm used
in modern LLM training pipelines.

Author: Ananta Team
Usage: python data_processor.py
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

# Configure logging for academic-style output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = r"C:\Users\r0b0t1x\Desktop\pri\Ananta-updated\mathematics_dataset-v1.0\mathematics_dataset-v1.0"


class MathDatasetProcessor:
    """
    Processes DeepMind mathematics dataset for Ananta fine-tuning.
    
    This class handles the conversion of raw text files into a structured
    JSON format suitable for instruction-following model training.
    """
    
    def __init__(self, dataset_dir: str = DEFAULT_DATASET_PATH, 
                 output_file: str = "formatted_math_dataset.json"):
        self.dataset_dir = Path(dataset_dir)
        self.output_file = Path(output_file)
        self.difficulty_levels = [
            "train-hard",     # Changed to match exact directory names
            "train-medium",
            "train-easy",
            "extrapolate",
            "interpolate"
        ]
        
        # Statistics tracking for academic reporting
        self.stats = {
            "total_problems": 0,
            "problems_by_difficulty": {},
            "skipped_entries": 0,
            "malformed_entries": 0
        }
    
    def validate_dataset_structure(self) -> bool:
        """
        Validates that the dataset directory exists and contains expected structure.
        """
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return False
        
        logger.info(f"Dataset directory found: {self.dataset_dir.absolute()}")
        
        # List actual contents of directory
        logger.info("Actual directory contents:")
        for item in self.dataset_dir.iterdir():
            logger.info(f"Found: {item.name}")
        
        # Check for difficulty level subdirectories
        missing_dirs = []
        for difficulty in self.difficulty_levels:
            difficulty_path = self.dataset_dir / difficulty
            logger.info(f"Checking for directory: {difficulty_path}")
            if not difficulty_path.exists():
                missing_dirs.append(difficulty)
        
        if missing_dirs:
            logger.warning(f"Missing difficulty directories: {missing_dirs}")
        
        return len(missing_dirs) < len(self.difficulty_levels)
    
    def extract_problem_solution_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Extracts problem-solution pairs from a single text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of (problem, solution) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            pairs = []
            # DeepMind format: alternating problem and solution lines
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    problem = lines[i]
                    solution = lines[i + 1]
                    
                    # Basic validation
                    if problem and solution:
                        pairs.append((problem, solution))
                    else:
                        self.stats["malformed_entries"] += 1
                        logger.debug(f"Skipped malformed entry in {file_path}")
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def create_instruction_entry(self, problem: str, solution: str, 
                               difficulty: str) -> Dict[str, str]:
        """
        Creates a structured instruction-following entry for fine-tuning.
        
        Args:
            problem: Mathematical problem statement
            solution: Correct solution
            difficulty: Difficulty level identifier
            
        Returns:
            Dictionary with instruction, input, output, and metadata
        """
        return {
            "instruction": "Solve the following mathematical problem step-by-step, showing your reasoning clearly.",
            "input": problem,
            "output": solution,
            "difficulty": difficulty,
            "domain": "mathematics",
            "task_type": "problem_solving"
        }
    
    def process_dataset(self) -> None:
        """
        Main processing pipeline that converts the entire dataset.
        
        This method iterates through all difficulty levels and text files,
        extracting problems and formatting them for instruction-tuning.
        """
        logger.info("Starting dataset processing pipeline...")
        
        if not self.validate_dataset_structure():
            raise ValueError("Dataset validation failed. Please check dataset structure.")
        
        # Use streaming JSON writing to handle large datasets efficiently
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("[\n")
            first_entry = True
            
            for difficulty in self.difficulty_levels:
                difficulty_path = self.dataset_dir / difficulty
                
                if not difficulty_path.exists():
                    logger.warning(f"Skipping {difficulty} - directory not found")
                    continue
                
                # Initialize stats for this difficulty
                self.stats["problems_by_difficulty"][difficulty] = 0
                
                # Process all text files in this difficulty level
                txt_files = list(difficulty_path.glob("*.txt"))
                logger.info(f"Processing {len(txt_files)} files from {difficulty}")
                
                for file_path in tqdm(txt_files, desc=f"Processing {difficulty}"):
                    problem_pairs = self.extract_problem_solution_pairs(file_path)
                    
                    for problem, solution in problem_pairs:
                        entry = self.create_instruction_entry(problem, solution, difficulty)
                        
                        # Write entry with proper JSON formatting
                        if not first_entry:
                            f_out.write(",\n")
                        
                        json.dump(entry, f_out, indent=4, ensure_ascii=False)
                        first_entry = False
                        
                        # Update statistics
                        self.stats["total_problems"] += 1
                        self.stats["problems_by_difficulty"][difficulty] += 1
            
            f_out.write("\n]")
        
        logger.info(f"Dataset processing complete. Output saved to: {self.output_file}")
        self.log_processing_statistics()
    
    def log_processing_statistics(self) -> None:
        """
        Logs comprehensive statistics about the processed dataset.
        """
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total problems processed: {self.stats['total_problems']}")
        logger.info(f"Malformed entries skipped: {self.stats['malformed_entries']}")
        
        logger.info("\nBreakdown by difficulty:")
        for difficulty, count in self.stats["problems_by_difficulty"].items():
            percentage = (count / self.stats["total_problems"]) * 100 if self.stats["total_problems"] > 0 else 0
            logger.info(f"  {difficulty}: {count} problems ({percentage:.1f}%)")
        
        # Calculate file size for memory planning
        file_size_mb = self.output_file.stat().st_size / (1024 * 1024) if self.output_file.exists() else 0
        logger.info(f"\nOutput file size: {file_size_mb:.2f} MB")
    
    def validate_output(self) -> bool:
        """
        Validates the generated JSON file structure and content.
        
        Returns:
            bool: True if output validation passes
        """
        try:
            logger.info("Validating output JSON structure...")
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("Output is not a valid JSON array")
                return False
            
            # Validate sample entries
            sample_size = min(100, len(data))
            required_keys = {"instruction", "input", "output", "difficulty"}
            
            for i, entry in enumerate(data[:sample_size]):
                if not isinstance(entry, dict):
                    logger.error(f"Entry {i} is not a valid dictionary")
                    return False
                
                if not required_keys.issubset(entry.keys()):
                    missing_keys = required_keys - entry.keys()
                    logger.error(f"Entry {i} missing keys: {missing_keys}")
                    return False
            
            logger.info(f"Output validation passed. {len(data)} entries validated.")
            return True
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False


def main():
    """
    Main execution function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Process DeepMind mathematics dataset for Ananta fine-tuning"
    )
    parser.add_argument(
        "--dataset_dir", 
        default=DEFAULT_DATASET_PATH,
        help="Path to mathematics dataset directory"
    )
    parser.add_argument(
        "--output_file",
        default="formatted_math_dataset.json", 
        help="Output JSON file path"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate existing output file"
    )
    
    args = parser.parse_args()
    
    processor = MathDatasetProcessor(args.dataset_dir, args.output_file)
    
    if args.validate_only:
        if processor.output_file.exists():
            success = processor.validate_output()
            exit(0 if success else 1)
        else:
            logger.error(f"Output file {args.output_file} not found")
            exit(1)
    
    try:
        processor.process_dataset()
        
        # Validate the output
        if processor.validate_output():
            logger.info("[SUCCESS] Dataset processing completed successfully!")
        else:
            logger.error("[ERROR] Output validation failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

"""#2025-06-04 14:28:30,856 - ERROR - Output validation failed: 
--- Logging error ---
Traceback (most recent call last):
  File "C:\python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
  File "C:\python310\lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u274c' in position 34: character maps to <undefined>
Call stack:
  File "C:\Users\r0b0t1x\Desktop\pri\Ananta-updated\ananta-update\data_processor.py", line 310, in <module>
    main()
  File "C:\Users\r0b0t1x\Desktop\pri\Ananta-updated\ananta-update\data_processor.py", line 301, in main
    logger.error("[ERROR] Output validation failed")
Message: '[ERROR] Output validation failed'
Arguments: ()
2025-06-04 14:28:31,838 - ERROR - [ERROR] Output validation failed"""