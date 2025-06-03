"""
Simple Math Dataset Converter
=============================

Hey! This script converts your mathematics dataset txt files into the JSON format 
needed for training. I wrote this to be as straightforward as possible.

Just run: python simple_data_converter.py

It'll look for your dataset files and convert them automatically.
No fancy stuff, just gets the job done.
"""

import os
import json
from pathlib import Path

def find_txt_files():
    """
    Look around for txt files that might be our dataset.
    Returns a list of txt files we found.
    """
    print("Looking for txt files...")
    
    current_folder = Path(".")
    txt_files = []
    
    # Check current folder first
    for file in current_folder.glob("*.txt"):
        txt_files.append(file)
        print(f"Found: {file.name}")
    
    # Check subfolders too
    for folder in current_folder.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            for file in folder.glob("*.txt"):
                txt_files.append(file)
                print(f"Found: {folder.name}/{file.name}")
    
    print(f"Total files found: {len(txt_files)}")
    return txt_files

def read_math_problems(txt_file):
    """
    Read a txt file and extract math problems and solutions.
    DeepMind format usually has problems and answers on alternating lines.
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Clean up the lines - remove empty ones and strip whitespace
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # only keep non-empty lines
                clean_lines.append(stripped)
        
        # Extract problem-answer pairs
        # Usually it's: problem line, answer line, problem line, answer line...
        pairs = []
        for i in range(0, len(clean_lines), 2):
            if i + 1 < len(clean_lines):
                problem = clean_lines[i]
                answer = clean_lines[i + 1]
                
                # Make sure we actually have content
                if len(problem) > 3 and len(answer) > 0:
                    pairs.append((problem, answer))
        
        return pairs
    
    except Exception as e:
        print(f"Couldn't read {txt_file}: {e}")
        return []

def figure_out_difficulty(file_path):
    """
    Try to guess the difficulty level from the file path or name.
    """
    path_str = str(file_path).lower()
    
    if 'easy' in path_str:
        return 'train-easy'
    elif 'medium' in path_str:
        return 'train-medium'
    elif 'hard' in path_str:
        return 'train-hard'
    elif 'extrapolate' in path_str:
        return 'extrapolate'
    elif 'interpolate' in path_str:
        return 'interpolate'
    else:
        # If we can't figure it out, just use the folder name or 'unknown'
        if file_path.parent.name != '.':
            return file_path.parent.name
        else:
            return 'unknown'

def convert_to_training_format(problem, answer, difficulty, filename):
    """
    Convert a problem-answer pair into the format our training script expects.
    """
    return {
        "instruction": "Solve the following mathematical problem step-by-step, showing your reasoning clearly.",
        "input": problem,
        "output": answer,
        "difficulty": difficulty,
        "domain": "mathematics", 
        "task_type": "problem_solving",
        "source": filename
    }

def main():
    """
    Main function - this is where everything happens.
    """
    print("=" * 50)
    print("Math Dataset Converter")
    print("=" * 50)
    
    # Step 1: Find all the txt files
    txt_files = find_txt_files()
    
    if not txt_files:
        print("Hmm, I didn't find any txt files.")
        print("Make sure your dataset files are in this folder or its subfolders.")
        return
    
    # Step 2: Process each file
    all_problems = []
    total_processed = 0
    
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file}")
        
        # Read the problems from this file
        problems = read_math_problems(txt_file)
        
        if problems:
            difficulty = figure_out_difficulty(txt_file)
            print(f"  Found {len(problems)} problems (difficulty: {difficulty})")
            
            # Convert each problem to training format
            for problem, answer in problems:
                formatted = convert_to_training_format(
                    problem, answer, difficulty, txt_file.name
                )
                all_problems.append(formatted)
                total_processed += 1
        else:
            print(f"  No problems found in {txt_file}")
    
    # Step 3: Save everything to JSON
    if all_problems:
        output_file = "formatted_math_dataset.json"
        print(f"\nSaving {total_processed} problems to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_problems, f, indent=2, ensure_ascii=False)
        
        print("Done! ✓")
        print(f"Your dataset is ready: {output_file}")
        print(f"Total problems: {total_processed}")
        
        # Show a breakdown by difficulty
        difficulty_counts = {}
        for problem in all_problems:
            diff = problem['difficulty']
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        print("\nBreakdown:")
        for diff, count in difficulty_counts.items():
            print(f"  {diff}: {count} problems")
        
        print("\nNext step: Run 'python train_ananta.py' to start training!")
        
    else:
        print("No problems were extracted. Check your txt files format.")
        print("Expected format: alternating lines of problems and solutions.")

if __name__ == "__main__":
    main() 