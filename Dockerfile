# Use Kaggle's Python-GPU image as the base
FROM gcr.io/kaggle-gpu-images/python:latest

# Uninstall existing PyTorch and dependencies to avoid conflicts
RUN pip uninstall torch torchvision torchaudio -y

# Using RUN to install PyTorch nightly build that may support CUDA 12.1
RUN pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121

