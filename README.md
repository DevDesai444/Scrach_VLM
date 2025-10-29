# Building a Vision-Language Model (VLM) from Scratch

This project demonstrates the complete process of building and training our own Vision-Language Model (VLM) from scratch, showcasing different approaches and evolutionary stages in development.

## Development Journey

### Stage 1: Dataset Creation using SmolVLM
- **Initial Data Generation**
  - Utilized SmolVLM as a teacher model to create high-quality training data
  - Generated descriptions for CIFAR10 images
  - Created a curated dataset of 100 image-text pairs
  - Ensured diverse and detailed descriptions across all CIFAR10 classes
- **Quality Control**
  - Manually verified generated descriptions
  - Ensured consistency in description style and detail level
  - Established a baseline for description quality

### Stage 2: Experimentation with SigLIP
- **Architecture Exploration**
  - Implemented SigLIP for image-text alignment
  - Explored vision-language bridging techniques
  - Tested different embedding approaches
- **Learning Outcomes**
  - Understood limitations of separate embedding models
  - Identified need for more direct fine-tuning approach
  - Gained insights into vision-language architectures

### Stage 3: Final Implementation with QLoRA
- **Model Architecture**
  - Base Model: Microsoft Phi-3-mini-4k-instruct
  - Training Method: QLoRA (Quantized Low-Rank Adaptation)
  - Direct image processing capability
- **Advantages**
  - Efficient fine-tuning process
  - Reduced model complexity
  - Eliminated need for separate image encoder
  - Better performance with simpler architecture

## Current Implementation

### Features
- Direct image understanding without separate encoders
- Efficient CPU inference
- Customizable generation parameters
- Pre-loaded CIFAR10 examples

### Technical Details
- Model: Fine-tuned Phi-3-mini with QLoRA
- Input: 32x32 RGB images
- Output: Detailed image descriptions
- Deployment: CPU-optimized for accessibility

## Hugging Face Space
- **Space Name**: [jatingocodeo/phi-vlm](https://huggingface.co/spaces/jatingocodeo/phi-vlm)
- **Features**:
  - Interactive web interface using Gradio
  - Example images from CIFAR10
  - Adjustable generation parameters
  - Real-time inference
- **Usage**:
  - Upload any image or use provided examples
  - Adjust generation parameters if needed
  - Get instant image descriptions

## Requirements
```python
transformers
torch
gradio
Pillow
peft
torchvision
```

## Key Learnings
- Evolution from complex multi-model setup to streamlined single model
- Importance of quality training data
- Benefits of direct fine-tuning over embedding-based approaches
- Practical considerations in deploying VLMs

## Future Improvements
- Expand training dataset
- Experiment with higher resolution inputs
- Optimize inference speed
- Add support for more diverse image types

## Acknowledgments
- Microsoft for Phi-3-mini model
- Hugging Face for infrastructure and tools
- CIFAR10 dataset creators
- SmolVLM team for initial data generation support
