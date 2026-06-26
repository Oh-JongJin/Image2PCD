# Image2PCD

Single image to Point Cloud Data converter using Depth Anything V2. 
This project provides a robust pipeline and an interactive Web UI to generate 3D point clouds from 2D images by estimating depth information.

## Features
- Single image depth estimation using Depth-Anything-V2
- Interactive Web UI powered by Gradio
- Browser-based 3D point cloud visualization using Plotly
- Multi-language support (English and Korean)
- Point cloud density control (Voxel Downsampling)
- 3D point cloud generation with RGB color information
- Mask support for extracting specific objects into 3D
- Outputs both 3D formats (PCD/PLY) and depth images

## Dependencies
```bash
torch
opencv-python
numpy
open3d
gradio
plotly

```

## Directory Setup

Initialize the required directories. To maintain the directory structure without tracking the contents of the submodule, create a `.gitkeep` file.

```bash
mkdir images result checkpoint
mkdir Depth-Anything-V2
touch Depth-Anything-V2/.gitkeep

```

## Installation

1. Clone the repository:

```bash
git clone [https://github.com/Oh-JongJin/Image2PCD.git](https://github.com/Oh-JongJin/Image2PCD.git)
cd Image2PCD

```

2. Install dependencies (Using `uv` is highly recommended for fast virtual environment setup and avoiding Open3D version conflicts):

```bash
uv venv --python 3.12
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install torch torchvision opencv-python numpy open3d gradio plotly

```

3. If you need the Depth Anything V2 architecture, proceed to clone with the command below in the project root:

```bash
git clone [https://github.com/DepthAnything/Depth-Anything-V2.git](https://github.com/DepthAnything/Depth-Anything-V2.git)

```

4. Download the pre-trained weights for Depth Anything V2 (vitl model):

```bash
# Download depth_anything_v2_vitl.pth and place it in the checkpoint directory

```

[Model download link](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models)

### Folder Tree

```text
├── Depth-Anything-V2/  # Architecture directory (Contents ignored via .gitignore)
│   └── .gitkeep        # Keeps the folder structure tracked
├── images/             # Input images directory
├── result/             # Output directory for 3D files and depth visualization
├── checkpoint/         # Directory for model weights
├── app.py              # Gradio Web UI script (Interactive mode)
├── run.py              # CLI script (Batch or single execution)
├── LICENSE
└── README.md

```

## Usage

### Interactive Web UI (Recommended)

Launch the Gradio application to use the interactive interface with an embedded Plotly 3D viewer:

```bash
python app.py
```

Access the local server via your browser (typically `http://127.0.0.1:7860`).

### Command Line Interface

1. Place your input images in the `images/` directory
2. Run the script:

```bash
python run.py
```

3. Check the results in the `result/` directory:

* `IMAGE_NAME.pcd` or `output_3d.ply`: Point cloud data file
* `IMAGE_NAME.jpg` or `output_depth.jpg`: Depth visualization image

## Output Format

* Point Cloud: Standard **PCD / PLY** format with RGB colors
* Depth Map: **Grayscale image** (8-bit) with normalized depth values

## Acknowledgements

* [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the depth estimation model
* [Open3D](https://github.com/isl-org/Open3D) for point cloud processing capabilities
* [Gradio](https://github.com/gradio-app/gradio) for the interactive Web UI framework
* [Plotly](https://github.com/plotly/plotly.py) for the interactive 3D scatter visualization