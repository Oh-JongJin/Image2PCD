# Image2PCD

Single image to Point Cloud Data converter using Depth Anything V2. 
This project provides a simple pipeline to generate 3D point clouds from 2D images by estimating depth information.



## Features
- Single image depth estimation using Depth-Anything-V2
- 3D point cloud generation with RGB color information
- Optional mask support for filtered point cloud generation
- Outputs both PCD file and depth image



## Dependencies
```bash
torch
opencv-python
numpy
open3d
depth-anything-v2
```



## Structure
```
├── Depth-Anything-V2/	# Refer to the Installation section
├── images/             # Input images directory
├── result/         	# Output directory for PCD and depth visualization
├── checkpoint/     	# Directory for model weights
├── run.py         		# Main script
├── LICENSE
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
# Install Depth Anything V2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

3. Download the pre-trained weights for Depth Anything V2 (vitl model):
```bash
mkdir checkpoint
# Download depth_anything_v2_vitl.pth and place it in the checkpoint directory
```

[model download link](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models)



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



## Output Format
- Point Cloud: Standard **PCD** format with RGB colors
- Depth Map: **Grayscale image** (8-bit) with normalized depth values



## Directory Setup
```bash
mkdir images result checkpoint
```



## Acknowledgements
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the depth estimation model
- Open3D for point cloud processing capabilities



## Contact
For questions and feedback, please create an issue or contact-><5jx2oh@gmail.com>

