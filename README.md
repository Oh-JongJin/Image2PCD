# Image2PCD

Single image to Point Cloud Data converter using Depth Anything V2. This project provides a simple pipeline to generate 3D point clouds from 2D images by estimating depth information.



## Features
- Single image depth estimation using Depth Anything V2
- 3D point cloud generation with RGB color information
- Optional mask support for filtered point cloud generation
- Outputs both PCD file and depth visualization



## Dependencies
```bash
torch
opencv-python
numpy
open3d
depth-anything-v2
```



## Project Structure
```
├── images/           # Input images directory
├── result/         # Output directory for PCD and depth visualization
├── checkpoint/     # Directory for model weights
├── main.py         # Main script
└── README.md
```



## Installation

1. Clone the repository:
```bash
git clone https://github.com/Oh-JongJin/Image2PCD.git
cd Image2PCD
```

2. Install dependencies:
```bash
pip install torch opencv-python numpy open3d
# Install Depth Anything V2 following their official repository
```

3. Download the pre-trained weights for Depth Anything V2 (vitl model):
```bash
mkdir checkpoint
# Download depth_anything_v2_vitl.pth and place it in the checkpoint directory
```



## Usage

1. Place your input images in the `images/` directory

2. Run the script:
```bash
python run.py
```

3. Check the results in the `result/` directory:
- `{image_name}.pcd`: Point cloud data file
- `{image_name}.jpg`: Depth visualization image



## Example Code
```python
# Load the model
depth_anything = DepthAnythingV2(**model_configs["vitl"])
depth_anything.load_state_dict(torch.load(
    "checkpoint/depth_anything_v2_vitl.pth",
    map_location=DEVICE,
))
```



## Configuration

The script supports different model configurations:
- vits: ViT-Small model
- vitb: ViT-Base model
- vitl: ViT-Large model (default)
- vitg: ViT-Giant model

Model selection can be modified in the script by changing the model_configs parameter.



## Output Format
- Point Cloud: Standard PCD format with RGB colors
- Depth Map: Grayscale image (8-bit) with normalized depth values



## Directory Setup
```bash
mkdir data result checkpoint
```



## Acknowledgements
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the depth estimation model
- Open3D for point cloud processing capabilities



## Contact
For questions and feedback, please create an issue or contact-><5jx2oh@gmail.com>

