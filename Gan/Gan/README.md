# GAN (Generative Adversarial Network) - Bird Image Generator 🦅

A PyTorch implementation of Deep Convolutional GAN (DCGAN) for generating realistic bird images.

## 📋 Project Overview

This project implements a GAN architecture that learns to generate 64×64 RGB bird images from random noise. The model consists of two competing neural networks:

- **Generator**: Creates fake images from random latent vectors
- **Discriminator**: Distinguishes between real and generated images

## 🎯 Model Performance

**Discriminator Scores:**
- Real images (average): **0.7313**
- Fake images (average): **0.3312**

These scores indicate a healthy GAN training state where:
- The Discriminator successfully identifies real images (~73% confidence)
- The Generator produces images that fool the Discriminator ~33% of the time
- Both networks are learning in balance without mode collapse

## 🏗️ Architecture

### Generator
```
Input: 100-dimensional latent vector
↓
Linear → 512×4×4 feature maps
↓
ConvTranspose2d (4×4 → 8×8) → 256 channels
↓
ConvTranspose2d (8×8 → 16×16) → 128 channels
↓
ConvTranspose2d (16×16 → 32×32) → 64 channels
↓
ConvTranspose2d (32×32 → 64×64) → 3 channels (RGB)
↓
Output: 64×64×3 RGB image
```

**Key Features:**
- BatchNorm2d for training stability
- ReLU activation (except final layer)
- Tanh output (range: [-1, 1])

### Discriminator
```
Input: 64×64×3 RGB image
↓
Conv2d (64×64 → 32×32) → 64 channels
↓
Conv2d (32×32 → 16×16) → 128 channels
↓
Conv2d (16×16 → 8×8) → 256 channels
↓
Flatten → Linear → Sigmoid
↓
Output: Single probability (0: fake, 1: real)
```

**Key Features:**
- LeakyReLU(0.2) to prevent dead neurons
- Dropout(0.3) to avoid overfitting
- BatchNorm2d (except first layer, per DCGAN best practices)

## 📦 Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow
```

## 🚀 Quick Start

### 1. Prepare Your Data

The notebook expects a pickle file containing 64×64 bird images:

```python
# Your data should be in format:
# List of numpy arrays (H, W, C) in 0-255 range
# Example: [np.array([64, 64, 3]), np.array([64, 64, 3]), ...]
```

### 2. Run the Notebook

Open `gan.ipynb` and run cells in order:

1. **Import libraries** - Load PyTorch and dependencies
2. **Load data** - Read pickle file and visualize samples
3. **Normalize data** - Transform images to [-1, 1] range
4. **Define models** - Generator and Discriminator classes
5. **Setup training** - Optimizers, loss function, DataLoader
6. **Train** - Run the GAN training loop
7. **Visualize** - Generate sample images every 2 epochs

### 3. Training Parameters

```python
latent_dim = 100      # Noise vector dimension
lr = 0.0002           # Learning rate (Adam)
beta1 = 0.5           # Adam beta1
beta2 = 0.999         # Adam beta2
num_epochs = 10       # Training epochs
batch_size = 32       # Batch size
```

## 📊 Training Process

The GAN alternates between two training phases each batch:

### Phase 1: Train Discriminator
1. Show real images → Target: 1 (real)
2. Generate fake images → Target: 0 (fake)
3. Backpropagate combined loss
4. Update Discriminator weights

### Phase 2: Train Generator
1. Generate new fake images
2. Pass through Discriminator → Target: 1 (fool the D)
3. Backpropagate loss
4. Update Generator weights

**Loss Function:** Binary Cross Entropy (BCELoss)

## 📈 Monitoring Training

Watch for these signs of healthy training:

✅ **Good Signs:**
- D loss stable around 0.5-1.0
- G loss gradually decreasing
- Generated images improve over epochs
- D scores: real ~0.6-0.8, fake ~0.3-0.5

❌ **Warning Signs:**
- D loss → 0 (Discriminator too strong)
- G loss → high values (Generator failing)
- Mode collapse (same images repeated)
- Training instability (losses oscillating wildly)

## 🎨 Sample Results

The notebook generates 4×4 grids of sample images every 2 epochs:

```
Epoch 2:  [16 generated bird images]
Epoch 4:  [16 generated bird images]
Epoch 6:  [16 generated bird images]
...
```

Images are automatically denormalized from [-1, 1] to [0, 1] for visualization.

## 🐛 Troubleshooting

### Issue: Discriminator too strong (D loss → 0)
**Solution:**
- Decrease D learning rate
- Add more dropout to Discriminator
- Use label smoothing (real labels = 0.9 instead of 1.0)

### Issue: Mode collapse
**Solution:**
- Increase data diversity
- Try minibatch discrimination
- Switch to WGAN-GP loss

### Issue: Out of memory
**Solution:**
- Reduce batch_size (try 16 or 8)
- Use mixed precision training
- Train on smaller images (32×32)


## 🔬 Technical Details

**Data Preprocessing:**
- Input: NumPy arrays (H, W, C) in [0, 255]
- Transform: ToTensor → (C, H, W) in [0, 1]
- Normalize: mean=0.5, std=0.5 → [-1, 1]

**Weight Initialization:**
- Default PyTorch initialization
- Consider: `normal(0, 0.02)` for DCGAN best practices

**Hardware:**
- Automatic GPU detection (CUDA)
- Falls back to CPU if GPU unavailable
- Training time: ~2-5 min/epoch (GPU), ~20-30 min/epoch (CPU)

## 📚 References
- [https://medium.com/@buslu4700/understanding-gans-from-scratch-with-pytorch-7921bd5c88b7]
- [Goodfellow et al., 2014 - Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Radford et al., 2015 - DCGAN](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License - feel free to use this code for your projects.

---

**Happy Training! 🚀**

*If you find this useful, consider starring the repo ⭐*
