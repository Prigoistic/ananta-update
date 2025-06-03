import gradio as gr
import json
import os
import random

def load_dataset():
    dataset_file = "formatted_math_dataset.json"
    if os.path.exists(dataset_file):
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        return data
    return None

def get_random_problem(difficulty):
    data = load_dataset()
    if data is None:
        return "Dataset not found. Please run txttojason.py first.", "", ""
    
    if difficulty == "All":
        filtered_data = data
    else:
        filtered_data = [item for item in data if item['difficulty'] == difficulty]
    
    if not filtered_data:
        return "No problems found for this difficulty.", "", ""
    
    problem = random.choice(filtered_data)
    return problem['input'], problem['output'], problem['difficulty']

def get_dataset_info():
    data = load_dataset()
    if data is None:
        return "Dataset not found. Please run txttojason.py first."
    
    total = len(data)
    difficulties = {}
    for item in data:
        diff = item['difficulty']
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    info = f"Total problems: {total}\n\nBreakdown by difficulty:\n"
    for diff, count in difficulties.items():
        info += f"- {diff}: {count} problems\n"
    
    return info

# Create Gradio interface
with gr.Blocks(title="Math Dataset Viewer") as demo:
    gr.Markdown("# Math Dataset Viewer")
    
    with gr.Tab("Dataset Info"):
        info_output = gr.Textbox(label="Dataset Information", lines=10)
        info_btn = gr.Button("Load Dataset Info")
        info_btn.click(get_dataset_info, outputs=info_output)
    
    with gr.Tab("Browse Problems"):
        difficulty_dropdown = gr.Dropdown(
            choices=["All", "train-easy", "train-medium", "train-hard", "extrapolate", "interpolate"],
            value="All",
            label="Select Difficulty"
        )
        
        generate_btn = gr.Button("Generate Random Problem")
        
        problem_output = gr.Textbox(label="Problem", lines=3)
        solution_output = gr.Textbox(label="Solution", lines=3)
        diff_output = gr.Textbox(label="Difficulty")
        
        generate_btn.click(
            get_random_problem,
            inputs=[difficulty_dropdown],
            outputs=[problem_output, solution_output, diff_output]
        )

if __name__ == "__main__":
    demo.launch() 