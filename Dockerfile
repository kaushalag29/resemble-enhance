# Base image with Anaconda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies
# git is needed for some pip installs from git repos
# ffmpeg is a common dependency for audio processing libraries
# build-essential and other build tools are needed for compiling packages
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    && apt-get clean

# Copy the application files into the container
COPY . .

# Create a conda environment for the application
# Resemble Enhance requires Python 3.11.13 according to TODO.md
RUN conda create -n resemble_enhance python=3.11 -y

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "resemble_enhance", "/bin/bash", "-c"]

# Upgrade pip to get latest wheel support
RUN pip install --upgrade pip

# Install dependencies from requirements.txt with timeout and retry
RUN pip install -r requirements.txt --timeout=100 --retries=3

# Install Resemble Enhance package in development mode
RUN pip install -e .

# Pre-download the model to ensure it's cached during build
# This triggers the model download if not already present
RUN python -c "from resemble_enhance.enhancer.inference import load_enhancer; import torch; device = 'cpu'; enhancer = load_enhancer(None, device); print('Resemble Enhance model loaded successfully')"

# Expose the port the server will run on (port 8014 as specified in server.py)
EXPOSE 8014

# Command to run the application server with conda environment activated
CMD ["conda", "run", "-n", "resemble_enhance", "python", "server.py"] 