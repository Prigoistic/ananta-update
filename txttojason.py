import os
import json
from tqdm import tqdm

# Define dataset directory and output file
dataset_dir = "mathematics_dataset-v1.0"
output_file = "formatted_math_dataset.json"

# Debug: Print current working directory and dataset path
print(f"Current working directory: {os.getcwd()}")
print(f"Dataset directory: {dataset_dir}")
print(f"Full dataset path: {os.path.abspath(dataset_dir)}")
print(f"Dataset directory exists: {os.path.exists(dataset_dir)}")

# Open JSON file for writing in streaming mode
with open(output_file, "w") as f_out:
    f_out.write("[\n")  # Start JSON array

    first_entry = True  # Track first entry to handle commas properly

    # Loop through all difficulty levels
    for difficulty in ["train-easy", "train-medium", "train-hard", "extrapolate", "interpolate"]:
        difficulty_path = os.path.join(dataset_dir, difficulty) 
        
        # Debug: Print path being checked
        print(f"Checking path: {difficulty_path}")
        print(f"Path exists: {os.path.exists(difficulty_path)}")
        print(f"Is directory: {os.path.isdir(difficulty_path)}")

        if not os.path.isdir(difficulty_path):
            print(f"Skipping {difficulty} (folder not found)")
            continue

        for file_name in tqdm(os.listdir(difficulty_path), desc=f"Processing {difficulty}"):
            if file_name.endswith(".txt"):
                file_path = os.path.join(difficulty_path, file_name)

                with open(file_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]

                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        problem = lines[i]
                        solution = lines[i + 1]

                        entry = {
                            "instruction": "Solve the following math problem step-by-step.",
                            "input": problem,
                            "output": solution,
                            "difficulty": difficulty
                        }

                        # Write JSON object without keeping all in memory
                        if not first_entry:
                            f_out.write(",\n")  # Add comma between JSON objects
                        json.dump(entry, f_out, indent=4)
                        first_entry = False  # Only the first entry skips the comma

    f_out.write("\n]")  # Close JSON array

print(f"Formatted dataset saved as {output_file}")
