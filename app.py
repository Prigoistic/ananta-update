"""
Ananta Scientific LLM Demo Interface
===================================

Interactive Gradio interface for the Ananta scientific reasoning model.
This demo showcases the model's capabilities in mathematical problem-solving,
step-by-step reasoning, and scientific computation.

Features:
- Mathematical problem solving with step-by-step explanations
- Dataset exploration and statistics
- Model comparison (base vs fine-tuned)
- Real-time inference with performance metrics

Author: Ananta Team
Usage: python app.py
Deployment: Can be deployed to Hugging Face Spaces or run locally
"""

import gradio as gr
import json
import os
import random
import torch
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import model components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logging.warning("Model libraries not available. Demo will run in dataset-only mode.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnantaDemo:
    """
    Main demo class handling model loading, inference, and UI components.
    """
    
    def __init__(self):
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_dataset()
        
        # Performance tracking
        self.inference_stats = {
            "total_queries": 0,
            "total_time": 0,
            "avg_response_time": 0
        }
    
    def load_dataset(self) -> None:
        """Load the mathematics dataset for exploration."""
        dataset_file = "formatted_math_dataset.json"
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                logger.info(f"Loaded {len(self.dataset)} problems from dataset")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                self.dataset = None
        else:
            logger.warning("Dataset file not found. Dataset features disabled.")
            self.dataset = None
    
    def load_model(self, model_path: str = "./deepseek_finetuned") -> Tuple[bool, str]:
        """
        Load the fine-tuned Ananta model for inference.
        
        Args:
            model_path: Path to the fine-tuned model
            
        Returns:
            Tuple of (success, message)
        """
        if not MODEL_AVAILABLE:
            return False, "Model libraries not available. Please install transformers and peft."
        
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/deepseek-math-7b",
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-math-7b",
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Load LoRA weights if available
            if Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
                self.model = PeftModel.from_pretrained(base_model, model_path)
                model_type = "Fine-tuned Ananta"
            else:
                self.model = base_model
                model_type = "Base DeepSeek-Math"
                logger.warning("Fine-tuned model not found. Using base model.")
            
            self.model.eval()
            self.model_loaded = True
            
            return True, f"✅ {model_type} model loaded successfully!"
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False, f"❌ Model loading failed: {str(e)}"
    
    def generate_response(self, problem: str, max_length: int = 512, 
                         temperature: float = 0.1) -> Tuple[str, float, Dict]:
        """
        Generate response for a mathematical problem.
        
        Args:
            problem: Mathematical problem to solve
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Tuple of (response, generation_time, stats)
        """
        if not self.model_loaded:
            return "❌ Model not loaded. Please load a model first.", 0.0, {}
        
        # Format the prompt
        prompt = f"""### Instruction:
Solve the following mathematical problem step-by-step, showing your reasoning clearly.

### Input:
{problem}

### Response:
"""
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate with timing
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            response_start = full_response.find("### Response:") + len("### Response:")
            response = full_response[response_start:].strip()
            
            # Update statistics
            self.inference_stats["total_queries"] += 1
            self.inference_stats["total_time"] += generation_time
            self.inference_stats["avg_response_time"] = (
                self.inference_stats["total_time"] / self.inference_stats["total_queries"]
            )
            
            # Generate performance stats
            stats = {
                "generation_time": generation_time,
                "tokens_generated": len(outputs[0]) - len(inputs[0]),
                "tokens_per_second": (len(outputs[0]) - len(inputs[0])) / generation_time,
                "model_parameters": getattr(self.model, 'num_parameters', lambda: 0)(),
                "gpu_memory_used": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            }
            
            return response, generation_time, stats
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"❌ Generation failed: {str(e)}", 0.0, {}
    
    def get_random_problem(self, difficulty: str = "All") -> Tuple[str, str, str]:
        """Get a random problem from the dataset."""
        if self.dataset is None:
            return "Dataset not available", "", ""
        
        if difficulty == "All":
            filtered_data = self.dataset
        else:
            filtered_data = [item for item in self.dataset if item.get('difficulty') == difficulty]
        
        if not filtered_data:
            return "No problems found for this difficulty.", "", ""
        
        problem = random.choice(filtered_data)
        return problem['input'], problem['output'], problem.get('difficulty', 'unknown')
    
    def get_dataset_info(self) -> str:
        """Generate dataset statistics and information."""
        if self.dataset is None:
            return "❌ Dataset not available. Please ensure formatted_math_dataset.json exists."
        
        total = len(self.dataset)
        difficulties = {}
        domains = {}
        
        for item in self.dataset:
            diff = item.get('difficulty', 'unknown')
            domain = item.get('domain', 'mathematics')
            
            difficulties[diff] = difficulties.get(diff, 0) + 1
            domains[domain] = domains.get(domain, 0) + 1
        
        info = f"""# 📊 Ananta Dataset Statistics

**Total Problems:** {total:,}

## 📈 Distribution by Difficulty:
"""
        for diff, count in difficulties.items():
            percentage = (count / total) * 100
            info += f"- **{diff}**: {count:,} problems ({percentage:.1f}%)\n"
        
        info += f"\n## 🎯 Domains Covered:\n"
        for domain, count in domains.items():
            info += f"- **{domain}**: {count:,} problems\n"
        
        info += f"""
## 📝 Dataset Characteristics:
- **Format**: Instruction-Input-Output tuples
- **Task Type**: Mathematical problem solving
- **Reasoning Style**: Step-by-step explanations
- **Source**: DeepMind Mathematics Dataset v1.0
- **Preprocessing**: Formatted for instruction-tuning
        """
        
        return info
    
    def create_performance_chart(self) -> go.Figure:
        """Create a performance visualization chart."""
        if self.inference_stats["total_queries"] == 0:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No inference data yet.<br>Try asking the model a question!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(
                title="Inference Performance Metrics",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Create performance metrics visualization
        metrics = ["Total Queries", "Avg Response Time (s)", "GPU Memory (GB)"]
        values = [
            self.inference_stats["total_queries"],
            self.inference_stats["avg_response_time"],
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=['lightblue', 'lightgreen', 'lightcoral'])
        ])
        
        fig.update_layout(
            title="Real-time Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Values",
            showlegend=False
        )
        
        return fig


# Initialize demo instance
demo_instance = AnantaDemo()


def solve_math_problem(problem: str, max_length: int, temperature: float):
    """Gradio interface function for solving math problems."""
    if not problem.strip():
        return "Please enter a mathematical problem to solve.", "", ""
    
    response, gen_time, stats = demo_instance.generate_response(
        problem, max_length, temperature
    )
    
    # Format performance info
    perf_info = f"""**Generation Time:** {gen_time:.3f} seconds
**Tokens per Second:** {stats.get('tokens_per_second', 0):.1f}
**GPU Memory Used:** {stats.get('gpu_memory_used', 0):.2f} GB"""
    
    return response, perf_info, demo_instance.create_performance_chart()


def load_model_interface(model_path: str):
    """Gradio interface function for loading models."""
    success, message = demo_instance.load_model(model_path)
    return message


# Create the Gradio interface
with gr.Blocks(
    title="Ananta: Scientific LLM Demo",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    """
) as demo:
    
    # Header
    gr.Markdown("""
    <div class="header">
        <h1>🧠 Ananta: Scientific LLM Demo</h1>
        <p><i>Advanced Mathematical Reasoning with DeepSeek-Math + LoRA Fine-tuning</i></p>
    </div>
    """)
    
    with gr.Tabs():
        
        # Tab 1: Model Inference
        with gr.Tab("🔬 Model Inference"):
            gr.Markdown("""
            ### Interactive Mathematical Problem Solving
            Ask Ananta to solve mathematical problems with step-by-step reasoning.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        problem_input = gr.Textbox(
                            label="Mathematical Problem",
                            placeholder="Enter your mathematical problem here...\nExample: What is the derivative of x^2 + 3x + 1?",
                            lines=3
                        )
                        
                        with gr.Row():
                            max_length = gr.Slider(
                                minimum=128, maximum=1024, value=512, step=64,
                                label="Max Response Length"
                            )
                            temperature = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                                label="Temperature"
                            )
                        
                        solve_btn = gr.Button("🚀 Solve Problem", variant="primary")
                        
                        # Model loading section
                        with gr.Accordion("🔧 Model Management", open=False):
                            model_path_input = gr.Textbox(
                                label="Model Path",
                                value="./deepseek_finetuned",
                                placeholder="Path to fine-tuned model directory"
                            )
                            load_model_btn = gr.Button("📥 Load Model")
                            model_status = gr.Textbox(label="Model Status", interactive=False)
                
                with gr.Column(scale=3):
                    response_output = gr.Textbox(
                        label="Model Response",
                        lines=15,
                        interactive=False
                    )
                    
                    with gr.Row():
                        perf_info = gr.Textbox(
                            label="Performance Metrics",
                            lines=3,
                            interactive=False
                        )
                        perf_chart = gr.Plot(label="Real-time Metrics")
        
        # Tab 2: Dataset Explorer
        with gr.Tab("📊 Dataset Explorer"):
            gr.Markdown("""
            ### Mathematics Dataset Analysis
            Explore the training dataset and generate random problems for testing.
            """)
            
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
        
        # Tab 3: Model Information
        with gr.Tab("ℹ️ About Ananta"):
            gr.Markdown("""
            ## 🧮 Ananta: Scientific LLM for Mathematical Reasoning
            
            Ananta is a specialized language model fine-tuned for scientific and mathematical problem-solving.
            Built on the foundation of DeepSeek-Math-7B, it uses LoRA (Low-Rank Adaptation) for efficient
            parameter-efficient fine-tuning.
            
            ### 🎯 Key Features:
            - **Mathematical Reasoning**: Step-by-step problem solving
            - **Scientific Computing**: Symbolic mathematics and calculations  
            - **Efficient Training**: LoRA fine-tuning on consumer GPUs
            - **Block-level Generation**: Optimized for structured outputs
            
            ### 🔧 Technical Specifications:
            - **Base Model**: deepseek-ai/deepseek-math-7b (7B parameters)
            - **Fine-tuning**: LoRA with target modules: q_proj, v_proj
            - **Training Data**: DeepMind Mathematics Dataset v1.0
            - **Optimization**: 8-bit quantization, gradient accumulation
            - **Hardware**: Optimized for RTX 3050 and above
            
            ### 🚀 Deployment Options:
            
            #### Local Deployment:
            ```bash
            python app.py  # Launch Gradio interface
            ```
            
            #### Hugging Face Spaces:
            1. Upload repository to HF Spaces
            2. Add requirements.txt with dependencies
            3. Set hardware to GPU for optimal performance
            
            #### Production API:
            ```bash
            # FastAPI deployment (create deploy/api_server.py)
            uvicorn api_server:app --host 0.0.0.0 --port 8000
            ```
            
            #### Docker Deployment:
            ```dockerfile
            FROM python:3.9-slim
            COPY . /app
            WORKDIR /app
            RUN pip install -r requirements.txt
            CMD ["python", "app.py"]
            ```
            
            ### 📚 Academic Usage:
            This codebase follows academic standards with:
            - Comprehensive documentation and comments
            - Reproducible training procedures
            - Detailed evaluation metrics
            - Performance benchmarking tools
            
            ### 🤝 Contributing:
            - Fork the repository
            - Create feature branches
            - Submit pull requests with detailed descriptions
            - Follow code quality standards (black, flake8)
            """)
    
    # Event handlers
    solve_btn.click(
        fn=solve_math_problem,
        inputs=[problem_input, max_length, temperature],
        outputs=[response_output, perf_info, perf_chart]
    )
    
    load_model_btn.click(
        fn=load_model_interface,
        inputs=[model_path_input],
        outputs=[model_status]
    )
    
    dataset_info_btn.click(
        fn=demo_instance.get_dataset_info,
        outputs=[dataset_info_output]
    )
    
    generate_btn.click(
        fn=demo_instance.get_random_problem,
        inputs=[difficulty_dropdown],
        outputs=[problem_output, solution_output, diff_output]
    )


# Launch configuration
if __name__ == "__main__":
    # Try to load model automatically if available
    if MODEL_AVAILABLE and Path("./deepseek_finetuned").exists():
        success, message = demo_instance.load_model()
        if success:
            logger.info("Model loaded automatically")
        else:
            logger.warning(f"Auto-load failed: {message}")
    
    # Launch demo
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False,           # Set to True for public sharing
        debug=True,            # Enable debug mode
        show_error=True,       # Show detailed errors
        quiet=False           # Show startup logs
    ) 