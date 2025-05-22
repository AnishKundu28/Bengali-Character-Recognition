# Bengali Handwritten Character Recognition using CNN (PyTorch)

## Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to recognize handwritten Bengali characters. The primary goal is to accurately classify characters from the Ekush dataset, which consists of a diverse set of Bengali characters, including basic characters, compound characters (juktakkhor), and numerals.

The `Bengali-CNN-Agent.ipynb` Jupyter notebook contains the complete workflow, from data loading and preprocessing to model training, evaluation, and visualization of results.

## Dataset

The model is trained on the **Ekush Dataset**.
- This dataset contains 122 classes of Bengali characters.
- Each class corresponds to a folder named from '0' to '121'.
- Images are typically 28x28 pixels, with a black background and white characters.
- The `Character-Mapping.csv` file provides the mapping from folder names (class labels) to the actual Bengali characters.

## Model Architecture

A Convolutional Neural Network (CNN) is used for character recognition. The architecture includes:
- Multiple convolutional layers with Batch Normalization, ReLU activation, Max Pooling, and Dropout for feature extraction.
- A fully connected classifier head with Batch Normalization, ReLU activation, and Dropout, leading to the output layer with softmax activation for classification among the 122 classes.

The specific model is defined in the `BengaliCNN` class within the notebook.

## Setup and Dependencies

To run this project, you'll need Python 3 and the following libraries:

- PyTorch (version 2.x recommended)
- Matplotlib
- NumPy
- Pandas
- scikit-learn
- Pillow (PIL)
- tqdm
- (Optional, for detailed GPU utilization) pynvml

You can install the required packages using pip:
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas scikit-learn pillow tqdm pynvml
```
It's recommended to use a virtual environment.

**GPU Support:**
The notebook is configured to utilize a CUDA-enabled GPU if available, significantly speeding up training. Ensure your PyTorch installation includes CUDA support compatible with your GPU drivers.

## Running the Notebook

1.  **Clone the repository (if applicable) or download the files.**
2.  **Ensure the Ekush dataset is downloaded and the `dataset_path` variable in the notebook (cell `47b3adf4`) points to its location.**
    ```python
    # Update this path in the notebook
    dataset_path = '/path/to/your/Ekush/dataset'
    ```
3.  **Ensure the `Character-Mapping.csv` file is available and the `csv_path` variable in the notebook (cell `47b3adf4`) points to its location.**
    ```python
    # Update this path in the notebook
    csv_path = '/path/to/your/Character-Mapping.csv'
    ```
4.  **Open and run the `Bengali-CNN-Agent.ipynb` notebook using Jupyter Notebook or JupyterLab.**
    ```bash
    jupyter notebook Bengali-CNN-Agent.ipynb
    # or
    jupyter lab Bengali-CNN-Agent.ipynb
    ```
5.  Execute the cells sequentially.

## Notebook Structure

The notebook is organized into the following main sections:

1.  **Import Libraries:** Imports all necessary Python packages.
2.  **GPU Configuration and Optimization:** Sets up the device (GPU/CPU) and applies various PyTorch optimizations for GPU performance (e.g., cuDNN benchmarking, TF32 precision, mixed-precision training settings).
3.  **Data Loading and Preprocessing:**
    *   Loads character mappings from `Character-Mapping.csv`.
    *   Defines the `EkushDataset` class for loading images.
    *   Applies data augmentation (random rotations, shifts) to the training set and normalization to all sets.
    *   Splits the data into training, validation, and test sets.
    *   Creates `DataLoaders` with optimized parameters for efficient data feeding.
4.  **Optimized CNN Model Architecture:** Defines the `BengaliCNN` class.
5.  **Training and Evaluation Functions:**
    *   `train_epoch`: Function to train the model for one epoch using mixed-precision (AMP) if a GPU is used.
    *   `validate`: Function to evaluate the model on the validation set.
6.  **Training Loop:**
    *   Initializes the model, optimizer (Adam), loss function (CrossEntropyLoss), and learning rate scheduler (ReduceLROnPlateau).
    *   Trains the model, implementing early stopping based on validation loss.
    *   Monitors GPU memory usage during training.
7.  **Visualize Training Results:** Plots training and validation loss and accuracy curves.
8.  **Evaluate Model on Test Set:**
    *   Evaluates the best model (chosen via early stopping) on the test set.
    *   Displays test loss and accuracy.
9.  **Performance Metrics & Analysis:**
    *   Generates and prints a `classification_report` (precision, recall, F1-score for each class).
    *   Displays a `confusion_matrix`, potentially for the most confused classes if the total number of classes is large, to identify misclassification patterns.
10. **Bengali Font Configuration and Visualization:**
    *   Includes `configure_bengali_fonts` to set up Matplotlib for displaying Bengali characters in plots.
    *   `visualize_bengali_predictions` function to show sample test images with their true and predicted Bengali character labels.
    *   `test_bengali_rendering` to verify font setup.
11. **Save and Load Model:** Functions to save the trained model state and load it back.
12. **Inference Example:** `predict_single_image` function to demonstrate how to use the trained model for inference on a new image.
13. **GPU Training Performance Analysis (Optional):** Code to measure inference speed.

## Interpreting Results

-   **Loss/Accuracy Plots:** Observe if the training and validation curves converge. A large gap between training and validation accuracy might indicate overfitting. Low training accuracy compared to validation accuracy might suggest issues with the training data/process or excessive regularization (this was an initial concern in the issue).
-   **Classification Report:** Check precision, recall, and F1-score for individual classes to understand model performance on specific characters.
-   **Confusion Matrix:** Identify which characters are often confused with each other.
-   **Bengali Character Visualization:** Visually inspect if the predictions and font rendering are correct.

## GPU Optimizations Applied

-   **Device Agnostic Code:** Uses `torch.device` to run on CUDA if available, otherwise CPU.
-   **cuDNN Benchmarking:** `torch.backends.cudnn.benchmark = True` for faster execution with fixed input sizes.
-   **TF32 Precision:** Enabled on compatible NVIDIA GPUs for faster computations with minimal precision loss.
-   **Mixed-Precision Training (AMP):** Uses `torch.cuda.amp.autocast` and `GradScaler` to speed up training and reduce memory usage on GPUs.
-   **Optimized DataLoaders:** Uses `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` for efficient data loading.
-   **Gradient Management:** `optimizer.zero_grad(set_to_none=True)` for slightly better performance.
-   **In-place Operations:** ReLU and other operations are used in-place where appropriate to save memory.
