High-Fidelity Denoising Diffusion Probabilistic Model (DDPM)

A complete PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating high-quality 28×28 MNIST digits.
This project leverages an enhanced U-Net architecture with residual connections and self-attention to produce sharp, realistic samples.

🚀 Key Features
Advanced U-Net Architecture

Residual Blocks

GELU activations

Time-step feature injection at every block

Self-Attention

Multi-head self-attention at the bottleneck (7×7 resolution) for capturing global dependencies

Time Embedding MLP

Deep MLP for continuous timestep embeddings

Optimized Diffusion Process

Linear beta schedule

β₁ = 1e-4 → βₜ = 0.02

1,000 diffusion timesteps for a fine-grained reverse denoising process

Robust Evaluation

Integrated CNN classifier to evaluate:

Generation confidence

Class diversity across generated samples

🏗️ Architecture Overview

The model follows the standard DDPM framework, with an enhanced reverse (denoising) process implemented using a sophisticated U-Net:

Down-sampling Path

Residual blocks followed by max-pooling

Bottleneck

Residual block combined with multi-head self-attention

Up-sampling Path

ConvTranspose2d layers

Skip connections and residual blocks

Time Embedding

Timestep t is embedded using a sinusoidal-style MLP and injected into every residual block

📊 Results & Performance

The model was trained for 30 epochs on the MNIST dataset with the following results:

Metric	Value
Final Training MSE	~0.023
Mean Classifier Confidence	~0.85+
Class Diversity	9/10 to 10/10 per batch
🖼️ Sample Generation

Generation starts from pure Gaussian noise 
𝑁(0,1)
N(0,1) and iteratively denoises the samples over 1,000 steps, producing clean, MNIST-style handwritten digits.

🛠️ Setup & Usage
Prerequisites

Python 3.8+

PyTorch & Torchvision

tqdm, matplotlib

Running the Project

Open the Jupyter Notebook:
TaskSet-1/Project-1.ipynb

Ensure a GPU (e.g., NVIDIA T4) is available for optimal performance.

Run all cells to:

Train the CNN evaluator

Train the High-Fidelity DDPM

Generate and visualize a 4×4 grid of realistic MNIST digits

📜 Acknowledgments

Inspired by the original paper:
“Denoising Diffusion Probabilistic Models” — Ho et al.
