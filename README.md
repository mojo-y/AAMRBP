
# AAMRBP

## 📢 Release Plan

The **full code** will be released upon acceptance.

We will progressively refine this project and make the following resources publicly available:

- [ ] Training and testing datasets
- [ ] Full source code
- [ ] Experimental results
- [ ] Model checkpoints

## 🔬 Phytolith Microscopy Gallery

Representative microscopic images of phytoliths collected from archaeological and modern reference samples.

| Archaeological Sample I | Archaeological Sample II | Modern Reference Sample |
| :---: | :---: | :---: |
| <img src="assest/images/鸡叫城TN025w125④b%201-007.jpg" width="280" alt="Archaeological Sample I"> | <img src="assest/images/鸡叫城台VI⑧-2%207-027.jpg" width="280" alt="Archaeological Sample II"> | <img src="assest/images/鸡叫城现代样品3%201-003.jpg" width="280" alt="Modern Reference Sample"> |

These representative images illustrate the morphological characteristics and diversity of phytoliths from archaeological contexts and modern reference materials, providing visual examples for phytolith recognition and analysis.

## 🛠️ Environment Setup

### Basic Installation

```bash
# Create and activate the environment
conda create -n aamrbp python=3.10 -y
conda activate aamrbp

# Install the dependency required by the core demo
pip install torch

For integration with the complete **Qwen3-VL** framework, install the additional dependencies:

```bash
pip install transformers peft accelerate scipy pillow pandas safetensors
```

## 🚀 Training

Before training, please prepare the required datasets and pretrained model.

### 1. Data Preparation

The training and testing datasets will be made publicly available upon acceptance.

After downloading, organize the datasets according to the expected directory structure.

### 2. Model Preparation

Our model is built upon **Qwen3-VL-4B** as the base vision-language model.

Please download the pretrained **Qwen3-VL-4B-Instruct** model from:

👉 [Download Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

After downloading, place the model files in the corresponding directory before starting training.

### 3. Training

After completing the data and model preparation, training can be performed using the provided training scripts.

Detailed training instructions and the full training code will be released upon acceptance.
