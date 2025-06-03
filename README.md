# Ananta: Scientific LLM Fine-tuning Pipeline

Ananta is a specialized scientific reasoning language model based on DeepSeek-Math-7B, fine-tuned for symbolic mathematics and scientific problem-solving. This repository contains the complete pipeline for data processing, fine-tuning, and deployment.

## 🔬 Project Overview

Ananta focuses on:

- **Symbolic reasoning** and mathematical problem-solving
- **Block-level output generation** optimized for scientific contexts
- **Parameter-efficient fine-tuning** using LoRA on consumer GPUs
- **Clean academic codebase** with comprehensive documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (RTX 3050 or better)
- 16GB+ RAM recommended
- DeepMind mathematics_dataset-v1.0 in txt format

### Installation

```bash
git clone https://github.com/yourusername/ananta-update.git
cd ananta-update
pip install -r requirements.txt
```

### Usage Pipeline

1. **Data Processing**: Convert raw dataset to training format

   ```bash
   python data_processor.py
   ```

2. **Fine-tuning**: Train the model with LoRA

   ```bash
   python train_ananta.py
   ```

3. **Evaluation**: Test model performance

   ```bash
   python evaluate_model.py
   ```

4. **Demo Interface**: Launch Gradio interface
   ```bash
   python app.py
   ```

## 📊 Model Specifications

- **Base Model**: deepseek-ai/deepseek-math-7b
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, v_proj
- **Training Configuration**:
  - Batch size: 1 (with gradient accumulation: 16)
  - Learning rate: 5e-5
  - Epochs: 3
  - Precision: FP16

## 🔧 Deployment Options

### Local Deployment

- Use `app.py` for local Gradio interface
- Load model in LM Studio for interactive testing

### Cloud Deployment

- **Hugging Face Spaces**: Upload to HF Spaces with Gradio
- **FastAPI**: Production API using `deploy/api_server.py`
- **Docker**: Containerized deployment (see `Dockerfile`)

## 📁 Project Structure

```
ananta-update/
├── data_processor.py      # Dataset conversion and preprocessing
├── train_ananta.py        # Main training script with LoRA
├── evaluate_model.py      # Model evaluation and metrics
├── app.py                 # Gradio demo interface
├── deploy/                # Deployment configurations
├── configs/               # Training configurations
├── utils/                 # Utility functions
└── docs/                  # Documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📚 References

- [DeepSeek-Math Paper](https://arxiv.org/abs/2402.03300)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Mathematics Dataset](https://github.com/deepmind/mathematics_dataset)

## 📄 License

MIT License - see LICENSE file for details.
