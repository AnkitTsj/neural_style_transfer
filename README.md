# Neural Style Transfer

This repository contains an implementation of Neural Style Transfer (NST) using PyTorch.
The primary objective was to experiment with the contributions of different layers in generating stylized images and
understand how these layers affect the transfer process.

## Motivation

I initially attempted to utilize models like YOLO and VGG16 for style transfer. However, YOLO, being primarily designed for object detection,
did not yield meaningful results due to its task-specific nature. In contrast, VGG16, widely used for feature extraction and tasks like NST,
proved to be a better fit for this purpose.

After encountering limitations with YOLO, I explored Grad-CAM to visualize and understand layer contributions better. This approach allowed me to identify how specific layers learn and contribute to the overall results. The Grad-CAM implementation can be found in a separate repository: [GRAD_CAM](https://github.com/AnkitTsj/GRAD_CAM).

## Project Structure

- **`model.py`**: Contains the implementation of the NST model and the configuration of layers used for style and content extraction.
- **`preprocess.py`**: Provides preprocessing utilities for input images and postprocessing for output images.
- **`plotr.py`**: Includes visualization utilities for displaying the content, style, and generated images.
- **`train.ipynb`**: Jupyter Notebook demonstrating the training process for NST, from loading images to generating stylized outputs.
- **`input.jpg`**: The content image to which styles will be applied.
- **`images/`**: Directory containing style images.
- **`output/`**: Directory for saving the generated stylized images.
- **`requirements.txt`**: Lists all required Python dependencies.

## How to Use

### Prerequisites

Ensure Python 3.x is installed. Using a virtual environment is recommended to manage dependencies.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AnkitTsj/neural_style_transfer.git
   cd neural_style_transfer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model

1. Prepare your content and style images:
   - Place the content image in the root directory with the name `input.jpg`.
   - Place the style image in the `images/` directory.

2. Open and run the `train.ipynb` notebook to perform style transfer. Generated images will be saved in the `output/` directory.

## Results and Observations

This implementation demonstrates how different layers of a model, such as VGG16, contribute to extracting and blending content and style features. By experimenting with layer configurations, it is possible to fine-tune the results to achieve the desired level of stylization.

For further insights into model layer contributions, refer to the [GRAD_CAM](https://github.com/AnkitTsj/GRAD_CAM) repository, which visualizes how models learn and contribute at various levels.

## Here are some arts,
Style :
![image](https://github.com/user-attachments/assets/a14dc552-d766-4e31-91ea-3813ea97a57d)


Content1:
![image](https://github.com/user-attachments/assets/344e7afd-d3e8-4602-bbcf-dbaf5c759aaa)


Style + content1:
![image](https://github.com/user-attachments/assets/4daca666-4036-46c1-9150-e8f1e9153ac5)

content2:
![image](https://github.com/user-attachments/assets/eb7a8a58-4dfc-4252-a085-b0426d7e44c7)


style + content2:
![image](https://github.com/user-attachments/assets/622f82fc-1234-4cf1-b97a-5b56034ddc90)



