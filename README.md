# GPU-Accelerated CNN on CIFAR-10 (PyTorch + CUDA)

## Objective
The purpose of this project was to **compare CPU vs GPU performance** when training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
The experiment demonstrates the practical benefits of GPU acceleration.

---

## ️ What I Did
1. **Defined a small CNN** in PyTorch to classify CIFAR-10 images (10 classes).
2. **Trained the model twice**:
   - Once on **CPU** for 2 epochs.
   - Once on an **NVIDIA GPU (Tesla T4)** for 20 epochs.
3. **Measured runtime and accuracy**:
   - Total training time.
   - Normalized seconds per epoch.
   - Final test accuracy.
4. **Plotted graphs** to visualize time and performance.

---

## What the Code Does
- **Data preparation:** Loads CIFAR-10 images with standard normalization.
- **Model:** Implements a 3-layer CNN (Conv → ReLU → Pool) followed by a fully connected layer.
- **Training loop:**
  - Forward pass, compute loss, backpropagation, optimizer step.
  - Tracks loss and accuracy each epoch.
- **Evaluation:** Runs on the test set to measure accuracy.
- **Timing:** Uses Python timers and `torch.cuda.synchronize()` to ensure fair GPU timings.
- **Comparison:** Prints CPU vs GPU runtime, accuracy, and speedup.
- **Visualization:** Generates bar charts for total time, seconds/epoch, and accuracy.

---

## Findings

| Device | Epochs | Total Time (s) | Seconds / Epoch | Test Accuracy (%) |
|--------|--------|----------------|-----------------|-------------------|
| CPU    | 2      | 133.8          | 66.9            | 48.7              |
| GPU    | 20     | 242.9          | 12.1            | 72.1              |

---

## Interpretation
- **Per-epoch speedup:** GPU training was **~5.5× faster** per epoch.
- **Accuracy:** CPU plateaued at ~49% after 2 epochs; GPU reached ~72% in 20 epochs.
- **Counterintuitive finding:**
  - At first glance, GPU total time (242.9s) looks *longer* than CPU (133.8s).
  - But this is misleading: the GPU trained **10× more epochs** in less than double the time.
  - When normalized (seconds/epoch), GPU is clearly much faster and enables higher accuracy.

---

## What This Means
- CPUs have a few powerful cores → great for sequential tasks.
- GPUs have thousands of smaller cores → perfect for parallel matrix multiplications in deep learning.
- This project shows that:
  1. **GPUs cut training time per epoch by ~5.5×.**
  2. **GPUs enable deeper training in the same wall-clock time**, yielding better accuracy.
  3. GPU acceleration is not just about speed — it’s about making training *practically feasible*.

---

## Takeaway
> With the same CNN and dataset, the GPU achieved **5.5× faster training per epoch** and reached **72% accuracy in 20 epochs**, compared to the CPU stuck at **49% after 2 epochs**.
> This experiment highlights why GPU acceleration is critical in modern AI development.
