# AAMRBP
## The code will be released upon acceptance.

We will progressively refine this project and make the following deliverables publicly available:
- [ ] Training and test datasets
- [ ] Source code
- [ ] Experimental results

## 🛠️ Environment Setup

Basic Installation

```bash
# Create and activate the environment
conda create -n aamrbp python=3.10 -y
conda activate aamrbp

# Install the dependency required by the core demo
pip install torch

# Run the architecture and loss-function demo
python aamrbp_core.py
```

For integration with the complete Qwen3-VL project:

```bash
pip install transformers peft accelerate scipy pillow pandas safetensors
```

## 🚀 Training

Before training, please prepare the required datasets and pretrained models.

### 1. Data Preparation

Download the required datasets from the following link:

👉 [Download Datasets](DATA_LINK)

After downloading, organize the datasets according to the expected directory structure.

### 2. Model Preparation

Our model is built upon **Qwen3-VL-4B** as the base vision-language model.

Please download the **Qwen3-VL-4B** model weights from the following link:

👉 [Download Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

After downloading, place the model files in the corresponding directory before starting training.

### 3. Start Training

After completing the data and model preparation, you can start training with:

```bash
# training command
```


