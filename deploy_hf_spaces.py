"""
Hugging Face Spaces Deployment Script
====================================

Prepares the Ananta project for deployment to Hugging Face Spaces with
proper configuration files and structure optimized for cloud deployment.

This script creates the necessary files for HF Spaces deployment including
app configuration, README for the space, and optimized requirements.

Author: Ananta Team
Usage: python deploy_hf_spaces.py
"""

import os
import shutil
from pathlib import Path
import json

def create_hf_app_file():
    """Create optimized app.py for Hugging Face Spaces deployment."""
    app_content = '''"""
Ananta: Scientific LLM for Mathematical Reasoning
===============================================

Hugging Face Spaces deployment of the Ananta scientific reasoning model.
Optimized for cloud deployment with efficient memory management.
"""

import gradio as gr
import json
import os
import random
import logging
from pathlib import Path

# Configure for HF Spaces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified demo for cloud deployment
class AnantaDemo:
    def __init__(self):
        self.dataset = self.load_dataset()
    
    def load_dataset(self):
        """Load dataset if available."""
        dataset_file = "formatted_math_dataset.json"
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
        return None
    
    def get_random_problem(self, difficulty="All"):
        """Get random problem from dataset."""
        if not self.dataset:
            return "Dataset not available", "", ""
        
        if difficulty == "All":
            filtered_data = self.dataset
        else:
            filtered_data = [item for item in self.dataset if item.get('difficulty') == difficulty]
        
        if not filtered_data:
            return "No problems found for this difficulty.", "", ""
        
        problem = random.choice(filtered_data)
        return problem['input'], problem['output'], problem.get('difficulty', 'unknown')
    
    def get_dataset_info(self):
        """Generate dataset statistics."""
        if not self.dataset:
            return "Dataset not available."
        
        total = len(self.dataset)
        difficulties = {}
        for item in self.dataset:
            diff = item.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        info = f"**Total Problems:** {total:,}\\n\\n**By Difficulty:**\\n"
        for diff, count in difficulties.items():
            percentage = (count / total) * 100
            info += f"- {diff}: {count:,} ({percentage:.1f}%)\\n"
        
        return info

# Initialize demo
demo_instance = AnantaDemo()

# Create Gradio interface
with gr.Blocks(title="Ananta Scientific LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Ananta: Scientific LLM Demo
    
    **Mathematical Reasoning with DeepSeek-Math + LoRA Fine-tuning**
    
    This is a demonstration of the Ananta scientific reasoning model. 
    The full model requires GPU resources not available in this space.
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Dataset Explorer"):
            with gr.Row():
                with gr.Column():
                    dataset_info_btn = gr.Button("📈 Load Dataset Statistics")
                    dataset_info_output = gr.Markdown()
                    
                    gr.Markdown("### 🎲 Random Problem Generator")
                    difficulty_dropdown = gr.Dropdown(
                        choices=["All", "train-easy", "train-medium", "train-hard", "extrapolate", "interpolate"],
                        value="All",
                        label="Select Difficulty Level"
                    )
                    generate_btn = gr.Button("🎯 Generate Random Problem")
                
                with gr.Column():
                    problem_output = gr.Textbox(label="Problem Statement", lines=4)
                    solution_output = gr.Textbox(label="Expected Solution", lines=4)
                    diff_output = gr.Textbox(label="Difficulty Level")
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Ananta
            
            Ananta is a specialized scientific LLM based on DeepSeek-Math-7B, fine-tuned using LoRA 
            for mathematical reasoning tasks. 
            
            ### Key Features:
            - Mathematical problem solving
            - Step-by-step reasoning
            - Scientific computation
            - Efficient LoRA fine-tuning
            
            ### Repository:
            [GitHub: Ananta Scientific LLM](https://github.com/yourusername/ananta-update)
            
            ### Deployment:
            For full model inference, deploy locally or on GPU-enabled infrastructure.
            """)
    
    # Event handlers
    dataset_info_btn.click(
        fn=demo_instance.get_dataset_info,
        outputs=[dataset_info_output]
    )
    
    generate_btn.click(
        fn=demo_instance.get_random_problem,
        inputs=[difficulty_dropdown],
        outputs=[problem_output, solution_output, diff_output]
    )

if __name__ == "__main__":
    demo.launch()
'''
    
    with open("app_hf.py", "w") as f:
        f.write(app_content)
    
    print("✅ Created app_hf.py for Hugging Face Spaces")

def create_hf_requirements():
    """Create optimized requirements for HF Spaces."""
    requirements_content = """gradio>=3.50.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
"""
    
    with open("requirements_hf.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Created requirements_hf.txt for Hugging Face Spaces")

def create_hf_readme():
    """Create README for Hugging Face Spaces."""
    readme_content = """---
title: Ananta Scientific LLM Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.50.0
app_file: app_hf.py
pinned: false
license: mit
---

# Ananta: Scientific LLM for Mathematical Reasoning

This is a demonstration space for the Ananta scientific reasoning model, a specialized language model fine-tuned for mathematical problem-solving.

## About Ananta

Ananta is built on DeepSeek-Math-7B and uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning on mathematical datasets. It specializes in:

- **Mathematical Problem Solving**: Step-by-step reasoning for complex problems
- **Scientific Computing**: Symbolic mathematics and calculations
- **Efficient Training**: LoRA fine-tuning optimized for consumer GPUs

## Features in this Demo

- **Dataset Explorer**: Browse the DeepMind Mathematics Dataset used for training
- **Problem Generator**: Sample random problems by difficulty level
- **Statistics**: View dataset distribution and characteristics

## Full Model Access

For complete model inference capabilities, please visit the [GitHub repository](https://github.com/yourusername/ananta-update) to run locally or deploy on GPU infrastructure.

## Technical Details

- **Base Model**: deepseek-ai/deepseek-math-7b
- **Fine-tuning**: LoRA with target modules: q_proj, v_proj
- **Dataset**: DeepMind Mathematics Dataset v1.0
- **Training**: RTX 3050+ optimized with 8-bit quantization

## Citation

If you use Ananta in your research, please cite:

```bibtex
@misc{ananta2024,
  title={Ananta: Scientific LLM for Mathematical Reasoning},
  author={Ananta Team},
  year={2024},
  url={https://github.com/yourusername/ananta-update}
}
```
"""
    
    with open("README_HF.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README_HF.md for Hugging Face Spaces")

def create_deployment_package():
    """Create a complete deployment package."""
    deployment_dir = Path("hf_spaces_deploy")
    deployment_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    files_to_copy = [
        "app_hf.py",
        "requirements_hf.txt", 
        "README_HF.md"
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy(file, deployment_dir / file.replace("_hf", "").replace("_HF", ""))
    
    # Copy dataset if it exists
    if Path("formatted_math_dataset.json").exists():
        shutil.copy("formatted_math_dataset.json", deployment_dir)
        print("✅ Copied dataset to deployment package")
    
    print(f"✅ Created deployment package in {deployment_dir}")
    print("\nTo deploy to Hugging Face Spaces:")
    print("1. Create a new Space on Hugging Face")
    print(f"2. Upload all files from {deployment_dir}")
    print("3. Set Space to public and enable GPU if needed")

def main():
    """Main deployment preparation function."""
    print("🚀 Preparing Ananta for Hugging Face Spaces deployment...\n")
    
    create_hf_app_file()
    create_hf_requirements()
    create_hf_readme()
    create_deployment_package()
    
    print("\n✅ Hugging Face Spaces deployment files ready!")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Test locally: python app_hf.py")
    print("3. Deploy to HF Spaces using the hf_spaces_deploy/ folder")

if __name__ == "__main__":
    main() 