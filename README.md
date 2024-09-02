PanDiff is a state-of-the-art approach for pansharpening satellite images using Denoising Diffusion Probabilistic Models (DDPM) in combination with a U-Net architecture. The goal of this project is to enhance the spatial resolution of low-resolution satellite images (e.g., Sentinel-1) by leveraging high-resolution images (e.g., Venus) in a generative process.

Features : 
-Denoising Diffusion Probabilistic Model (DDPM): Implements a Gaussian diffusion process to iteratively denoise images and improve their resolution.
-U-Net Architecture: A robust neural network architecture that serves as the denoising function in the diffusion process.
-Image Preprocessing: Handles the loading and preprocessing of satellite image datasets, resizing and normalizing them for training.
-Training Pipeline: Includes scripts for training the diffusion model on paired low-resolution and high-resolution images.
-Testing Pipeline: Evaluates the trained model on test data to generate high-resolution images from low-resolution inputs.


Project Structure:

- diffusion.py: Implements the DDPM model, handling the forward and reverse diffusion processes.
- unet.py: Defines the U-Net architecture used for denoising in the DDPM.
- dataset.py: Contains the dataset handling and preprocessing logic, including loading, resizing, and batching the images.
- train.py: Script for training the PanDiff model using the provided dataset.
- test.py: Script for testing and evaluating the performance of the trained model.

  
How It Works : 
Dataset Preparation: The project uses a private dataset containing pairs of low-resolution (Sentinel-1) and high-resolution (Venus) satellite images. The dataset is processed to ensure images are resized and normalized for training.

Model Training: The DDPM and U-Net models are trained using the low-resolution images as input and the high-resolution images as targets. The diffusion process iteratively refines the low-resolution images to match the high-resolution targets.

Inference: After training, the model can generate high-resolution images from new low-resolution inputs, leveraging the learned diffusion process.

#Requirements : 
Python 3.8+
PyTorch
torchvision
numpy
PIL
tqdm
