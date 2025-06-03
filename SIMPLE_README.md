# Ananta Math Model - Simple Version

Hey! This is a simplified version of the Ananta fine-tuning project. I wrote these scripts to be as straightforward as possible - no fancy stuff, just what you need to get your math model working.

## What is this?

Ananta is a fine-tuned version of DeepSeek's math model. It's trained on mathematical problems to make it better at solving math questions step-by-step.

## What you need

- A computer with an NVIDIA GPU (RTX 3050 or better)
- Python 3.8 or newer
- Your mathematics dataset in .txt format

## Quick start

### 1. Install everything

```bash
pip install -r simple_requirements.txt
```

This might take a while the first time. Grab some coffee! ☕

### 2. Convert your dataset

Put your .txt files in this folder, then run:

```bash
python simple_data_converter.py
```

This will:

- Find all your .txt files
- Convert them to the JSON format needed for training
- Save everything as `formatted_math_dataset.json`

### 3. Train the model

```bash
python easy_train.py
```

This will:

- Load the DeepSeek math model
- Fine-tune it on your dataset
- Save checkpoints as it goes
- Take a few hours (depending on your dataset size)

You can stop anytime with Ctrl+C and it'll save your progress.

### 4. Test your model

```bash
python test_ananta.py
```

This lets you:

- Ask your model math questions
- See how well it performs
- Chat with it interactively

## Files explained

- `simple_data_converter.py` - Converts txt files to training format
- `easy_train.py` - Trains your model (the main script)
- `test_ananta.py` - Test and chat with your trained model
- `simple_requirements.txt` - All the packages you need

## Tips

**Memory issues?** The scripts are set up for RTX 3050 with 8GB VRAM. If you have less, try:

- Reducing batch size in `easy_train.py`
- Using CPU instead (much slower though)

**Training taking forever?** That's normal! A few hours is typical. The script saves progress every 500 steps, so you can stop and resume anytime.

**Model giving weird answers?** Try:

- Training for more epochs
- Using more data
- Adjusting the temperature in `test_ananta.py`

## Troubleshooting

**"CUDA out of memory"** - Your GPU doesn't have enough memory. Try reducing the batch size or use CPU.

**"Model not found"** - Make sure you've trained the model first with `easy_train.py`.

**"Dataset not found"** - Run `simple_data_converter.py` first to convert your txt files.

## What's different from the full version?

This simplified version:

- Has fewer features but is easier to understand
- Uses simpler file names and structure
- Has more comments explaining what's happening
- Focuses on just getting things working

If you want more advanced features, check out the full version with all the bells and whistles!

## Need help?

The scripts print helpful messages as they run. If something goes wrong, read the error message - it usually tells you what to fix.

Good luck training your math model! 🚀
