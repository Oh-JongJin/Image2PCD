import cv2
import torch
import numpy as np
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2


# Function to convert depth map and mask to 3D point cloud
def depth_to_3d_point_cloud(depth_map, raw_image, mask=None):
    """
    Converts a depth map and a raw image into a 3D point cloud with colors.

    Parameters:
    - depth_map (np.ndarray): 2D array of depth values with shape (height, width).
    - raw_image (np.ndarray): 3D array of RGB values with shape (height, width, 3).
    - mask (np.ndarray, optional): 2D boolean array to filter points with shape (height, width).

    Returns:
    - points (np.ndarray): Nx3 array of 3D points.
    - colors (np.ndarray): Nx3 array of RGB colors normalized to [0, 1].
    """
    # Get the height and width from the depth map
    height, width = depth_map.shape

    # Generate grid of coordinates (v, u)
    v_coords, u_coords = np.indices((height, width))

    # If a mask is provided, apply it to filter valid points
    if mask is not None:
        valid = mask > 0
    else:
        valid = np.ones_like(depth_map, dtype=bool)

    # Extract the valid coordinates and depth values
    v_valid = v_coords[valid]
    u_valid = u_coords[valid]
    depth_valid = depth_map[valid]

    # Stack the coordinates to form the point cloud (v, u, depth)
    points = np.stack((v_valid, u_valid, depth_valid), axis=-1)

    # Extract and normalize the corresponding colors
    colors = raw_image[valid] / 255.0

    return points, colors


if __name__ == '__main__':
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    depth_anything = DepthAnythingV2(**model_configs["vitl"])
    depth_anything.load_state_dict(torch.load(
        "checkpoint/depth_anything_v2_vitl.pth",
        map_location=DEVICE,
    )
    )
    depth_anything = depth_anything.to(DEVICE).eval()

    image_name = "test"     # input image file name
    raw = cv2.imread(f"data/{image_name}.jpg")
    mask = None  # cv2.imread('/path/to/mask.png') # if you do not have a mask, skip this line
    depth = depth_anything.infer_image(raw)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Generate the 3D point cloud
    point_cloud, color = depth_to_3d_point_cloud(depth, raw, mask=mask)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(f'result/{image_name.split(".")[0]}.pcd', pcd)
    print(f"Saved point cloud to result/{image_name}.pcd")

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cv2.imwrite(f'result/{image_name.split(".")[0]}.jpg', depth)
    print(f"Saved depth image to result/{image_name}.jpg")
