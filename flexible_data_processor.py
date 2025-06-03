"""
Flexible Mathematics Dataset Processor
=====================================

A more flexible version of the data processor that can handle different 
dataset structures and automatically detect the correct format for the
DeepMind mathematics dataset.

This processor can handle:
- Standard mathematics_dataset-v1.0 structure
- Flat directory with .txt files
- Alternative directory names
- Mixed structures

Author: Ananta Team
Usage: python flexible_data_processor.py [--dataset_dir <path>]
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import argparse
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flexible_data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FlexibleMathDatasetProcessor:
    """
    Flexible processor for mathematics datasets with auto-detection capabilities.
    """
    
    def __init__(self, dataset_dir: str = ".", output_file: str = "formatted_math_dataset.json"):
        """
        Initialize the flexible processor.
        
        Args:
            dataset_dir: Directory to search for dataset files
            output_file: Output JSON file path
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_file = Path(output_file)
        self.stats = {
            "total_problems": 0,
            "problems_by_source": {},
            "malformed_entries": 0,
            "files_processed": 0
        }
        
        # Difficulty mapping for various naming conventions
        self.difficulty_mapping = {
            'train-easy': 'train-easy',
            'train_easy': 'train-easy',
            'easy': 'train-easy',
            'train-medium': 'train-medium',
            'train_medium': 'train-medium', 
            'medium': 'train-medium',
            'train-hard': 'train-hard',
            'train_hard': 'train-hard',
            'hard': 'train-hard',
            'extrapolate': 'extrapolate',
            'interpolate': 'interpolate',
            'test': 'test',
            'valid': 'validation',
            'validation': 'validation'
        }
    
    def detect_dataset_structure(self) -> Dict[str, List[Path]]:
        """
        Auto-detect the dataset structure and return organized file paths.
        
        Returns:
            Dictionary mapping difficulty levels to lists of file paths
        """
        logger.info(f"Detecting dataset structure in: {self.dataset_dir.absolute()}")
        
        file_groups = {}
        
        # Strategy 1: Look for subdirectories with expected names
        subdirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            logger.info(f"Found {len(subdirs)} subdirectories")
            
            for subdir in subdirs:
                # Map directory name to difficulty
                dir_name = subdir.name.lower()
                difficulty = None
                
                # Direct mapping
                if dir_name in self.difficulty_mapping:
                    difficulty = self.difficulty_mapping[dir_name]
                else:
                    # Pattern matching for variations
                    for pattern, mapped_diff in self.difficulty_mapping.items():
                        if pattern.replace('-', '_') in dir_name or pattern.replace('_', '-') in dir_name:
                            difficulty = mapped_diff
                            break
                    
                    # Fallback: use directory name as-is
                    if difficulty is None:
                        difficulty = dir_name
                
                # Find .txt files in this directory
                txt_files = list(subdir.glob("*.txt"))
                if txt_files:
                    file_groups[difficulty] = txt_files
                    logger.info(f"Found {len(txt_files)} .txt files in {subdir.name}/ -> {difficulty}")
        
        # Strategy 2: Look for .txt files in the main directory
        main_txt_files = list(self.dataset_dir.glob("*.txt"))
        if main_txt_files:
            logger.info(f"Found {len(main_txt_files)} .txt files in main directory")
            
            # Try to group by filename patterns
            for txt_file in main_txt_files:
                filename = txt_file.stem.lower()
                difficulty = "unknown"
                
                # Try to extract difficulty from filename
                for pattern, mapped_diff in self.difficulty_mapping.items():
                    if pattern in filename:
                        difficulty = mapped_diff
                        break
                
                if difficulty not in file_groups:
                    file_groups[difficulty] = []
                file_groups[difficulty].append(txt_file)
        
        # Strategy 3: Recursive search for .txt files
        if not file_groups:
            logger.info("No organized structure found, searching recursively...")
            all_txt_files = list(self.dataset_dir.rglob("*.txt"))
            
            if all_txt_files:
                file_groups["all_files"] = all_txt_files
                logger.info(f"Found {len(all_txt_files)} .txt files recursively")
        
        # Log summary
        logger.info("Dataset structure detection complete:")
        for difficulty, files in file_groups.items():
            logger.info(f"  {difficulty}: {len(files)} files")
        
        return file_groups
    
    def extract_problem_solution_pairs(self, file_path: Path, difficulty: str) -> List[Tuple[str, str]]:
        """
        Extract problem-solution pairs from a text file.
        Handles different formats flexibly.
        
        Args:
            file_path: Path to the text file
            difficulty: Difficulty level for this file
            
        Returns:
            List of (problem, solution) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try different parsing strategies
            pairs = []
            
            # Strategy 1: DeepMind format (alternating lines)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            if len(lines) >= 2:
                # Check if it looks like alternating problem-solution format
                for i in range(0, len(lines) - 1, 2):
                    if i + 1 < len(lines):
                        problem = lines[i]
                        solution = lines[i + 1]
                        
                        # Basic validation
                        if problem and solution and len(problem) > 5 and len(solution) > 1:
                            pairs.append((problem, solution))
                        else:
                            self.stats["malformed_entries"] += 1
            
            # Strategy 2: Look for explicit separators
            if not pairs and ('---' in content or '===' in content):
                sections = re.split(r'[-=]{3,}', content)
                for section in sections:
                    section_lines = [line.strip() for line in section.split('\n') if line.strip()]
                    if len(section_lines) >= 2:
                        problem = section_lines[0]
                        solution = '\n'.join(section_lines[1:])
                        if problem and solution:
                            pairs.append((problem, solution))
            
            # Strategy 3: Single problem-solution pair per file
            if not pairs and len(lines) >= 2:
                # Assume first line is problem, rest is solution
                problem = lines[0]
                solution = '\n'.join(lines[1:])
                if problem and solution:
                    pairs.append((problem, solution))
            
            self.stats["files_processed"] += 1
            return pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def create_instruction_entry(self, problem: str, solution: str, difficulty: str, source_file: str) -> Dict[str, str]:
        """
        Create a structured instruction-following entry.
        
        Args:
            problem: Mathematical problem statement
            solution: Correct solution
            difficulty: Difficulty level
            source_file: Source file name for tracking
            
        Returns:
            Dictionary with instruction format
        """
        return {
            "instruction": "Solve the following mathematical problem step-by-step, showing your reasoning clearly.",
            "input": problem,
            "output": solution,
            "difficulty": difficulty,
            "domain": "mathematics",
            "task_type": "problem_solving",
            "source_file": source_file
        }
    
    def process_dataset(self) -> None:
        """
        Main processing pipeline with flexible structure detection.
        """
        logger.info("Starting flexible dataset processing...")
        
        # Detect dataset structure
        file_groups = self.detect_dataset_structure()
        
        if not file_groups:
            raise ValueError("No .txt files found in the specified directory!")
        
        # Process files
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("[\n")
            first_entry = True
            
            for difficulty, file_list in file_groups.items():
                logger.info(f"Processing {len(file_list)} files for difficulty: {difficulty}")
                
                # Initialize stats for this difficulty
                self.stats["problems_by_source"][difficulty] = 0
                
                for file_path in tqdm(file_list, desc=f"Processing {difficulty}"):
                    problem_pairs = self.extract_problem_solution_pairs(file_path, difficulty)
                    
                    for problem, solution in problem_pairs:
                        entry = self.create_instruction_entry(
                            problem, solution, difficulty, file_path.name
                        )
                        
                        # Write entry
                        if not first_entry:
                            f_out.write(",\n")
                        
                        json.dump(entry, f_out, indent=4, ensure_ascii=False)
                        first_entry = False
                        
                        # Update statistics
                        self.stats["total_problems"] += 1
                        self.stats["problems_by_source"][difficulty] += 1
            
            f_out.write("\n]")
        
        logger.info(f"Processing complete. Output saved to: {self.output_file}")
        self.log_processing_statistics()
    
    def log_processing_statistics(self) -> None:
        """Log comprehensive processing statistics."""
        logger.info("=== FLEXIBLE PROCESSING STATISTICS ===")
        logger.info(f"Total problems processed: {self.stats['total_problems']}")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Malformed entries skipped: {self.stats['malformed_entries']}")
        
        logger.info("\nBreakdown by source:")
        for source, count in self.stats["problems_by_source"].items():
            percentage = (count / self.stats["total_problems"]) * 100 if self.stats["total_problems"] > 0 else 0
            logger.info(f"  {source}: {count} problems ({percentage:.1f}%)")
        
        # File size
        if self.output_file.exists():
            file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
            logger.info(f"\nOutput file size: {file_size_mb:.2f} MB")
    
    def validate_output(self) -> bool:
        """Validate the generated JSON file."""
        try:
            logger.info("Validating output JSON...")
            
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("Output is not a valid JSON array")
                return False
            
            # Sample validation
            sample_size = min(10, len(data))
            required_keys = {"instruction", "input", "output", "difficulty"}
            
            for i, entry in enumerate(data[:sample_size]):
                if not isinstance(entry, dict):
                    logger.error(f"Entry {i} is not a valid dictionary")
                    return False
                
                if not required_keys.issubset(entry.keys()):
                    missing_keys = required_keys - entry.keys()
                    logger.error(f"Entry {i} missing keys: {missing_keys}")
                    return False
            
            logger.info(f"✅ Output validation passed. {len(data)} entries validated.")
            return True
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main execution with flexible dataset handling."""
    parser = argparse.ArgumentParser(
        description="Flexible Mathematics Dataset Processor for Ananta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flexible_data_processor.py                           # Process current directory
  python flexible_data_processor.py --dataset_dir "math_data" # Process specific directory
  python flexible_data_processor.py --output "my_dataset.json" # Custom output file
        """
    )
    
    parser.add_argument(
        "--dataset_dir",
        default=".",
        help="Directory containing dataset files (default: current directory)"
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
    
    processor = FlexibleMathDatasetProcessor(args.dataset_dir, args.output_file)
    
    if args.validate_only:
        if processor.output_file.exists():
            success = processor.validate_output()
            exit(0 if success else 1)
        else:
            logger.error(f"Output file {args.output_file} not found")
            exit(1)
    
    try:
        processor.process_dataset()
        
        # Validate output
        if processor.validate_output():
            logger.info("✅ Dataset processing completed successfully!")
            logger.info(f"📁 Output file: {processor.output_file}")
            logger.info("🚀 Ready for training! Run: python train_ananta.py")
        else:
            logger.error("❌ Output validation failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main() 