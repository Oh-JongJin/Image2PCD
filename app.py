import os
import cv2
import torch
import numpy as np
import open3d as o3d
import gradio as gr
import plotly.graph_objects as go

import sys
sys.path.append(os.path.abspath("Depth-Anything-V2"))
from depth_anything_v2.dpt import DepthAnythingV2

custom_css = """
.generating {
    opacity: 0.6 !important;
    filter: blur(2px) !important;
    transition: all 0.3s ease-in-out;
    pointer-events: none;
}
"""

I18N = {
    "en": {
        "title": "# Image2PCD: 2D Image to 3D Point Cloud Converter",
        "desc": "Convert standard 2D images into 3D spatial data.",
        "input_img": "Upload Original 2D Image",
        "mask_img": "Upload Mask Image (Optional, B&W)",
        "density": "Point Cloud Density (Voxel Size: 0 = Original, 0.05 = Sparse)",
        "btn_convert": "Generate 3D Point Cloud",
        "out_depth": "Generated Depth Map",
        "out_3d": "3D Interactive Viewer",
        "out_file": "Download 3D File (.ply)",
        "tip": "Tip: Scroll to zoom, drag to rotate.",
        "err_no_img": "Please upload an original image."
    },
    "ko": {
        "title": "# Image2PCD",
        "input_img": "원본 2D 이미지 업로드",
        "mask_img": "마스크 이미지 업로드 (선택, 흑백)",
        "density": "포인트 클라우드 밀도 (Voxel Size: 0 = 원본, 0.05 = 듬성듬성)",
        "btn_convert": "3D 포인트 클라우드 생성",
        "out_depth": "생성된 Depth Map",
        "out_3d": "3D 인터랙티브 뷰어",
        "out_file": "3D 파일 다운로드 (.ply)",
        "tip": "Tip: 마우스 휠로 확대/축소, 드래그로 회전이 가능합니다.",
        "err_no_img": "원본 이미지를 업로드해주세요."
    }
}

model_configs = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 전역 모델 로드
depth_anything = DepthAnythingV2(**model_configs["vitl"])
depth_anything.load_state_dict(torch.load("checkpoint/depth_anything_v2_vitl.pth", map_location=DEVICE))
depth_anything = depth_anything.to(DEVICE).eval()

def depth_to_3d_point_cloud(depth_map, raw_image, mask_image=None):
    height, width = depth_map.shape
    v_coords, u_coords = np.indices((height, width))
    
    if mask_image is not None:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_resized = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = mask_resized > 127
    else:
        valid = np.ones_like(depth_map, dtype=bool)
        
    v_valid = v_coords[valid]
    u_valid = u_coords[valid]
    depth_valid = depth_map[valid]
    
    # [수정됨] 이미지의 X(u), Y(v) 좌표계 보정 (Z는 깊이)
    points = np.stack((u_valid, v_valid, depth_valid), axis=-1)
    colors = raw_image[valid] / 255.0
    
    return points, colors

def create_plotly_fig(pcd):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 브라우저 렌더링 과부하 방지를 위해 최대 5만개의 점만 샘플링하여 시각화 (원본 파일은 유지)
    max_points = 50000
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]
        colors = colors[::step]
        
    colors_255 = (colors * 255).astype(np.uint8)
    color_strings = [f'rgb({c[0]},{c[1]},{c[2]})' for c in colors_255]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=-points[:, 1], # 이미지 좌표계의 상하를 뒤집어 실제 눈에 보이는 3D 방향과 일치시킴
        z=-points[:, 2], 
        mode='markers',
        marker=dict(size=2, color=color_strings, opacity=0.9)
    )])
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data' # 모델의 실제 찌그러짐 없는 비율 유지
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def process_image(image_path, mask_path, voxel_size, lang, progress=gr.Progress()):
    if image_path is None:
        raise gr.Error(I18N[lang]["err_no_img"])

    progress(0.1, desc="이미지 로딩 중...")
    raw = cv2.imread(image_path)
    mask = cv2.imread(mask_path) if mask_path is not None else None

    progress(0.4, desc="Depth-Anything-V2 모델 추론 중...")
    depth = depth_anything.infer_image(raw)
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    progress(0.6, desc="3D 좌표계 매핑 중...")
    point_cloud, color = depth_to_3d_point_cloud(depth, raw_rgb, mask_image=mask)
    
    center = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - center
    max_bound = np.max(np.abs(point_cloud))
    if max_bound > 0:
        point_cloud = point_cloud / max_bound

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)

    progress(0.8, desc="다운샘플링 및 파일 저장 중...")
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    os.makedirs("result", exist_ok=True)
    output_ply_path = 'result/output_3d.ply'
    o3d.io.write_point_cloud(output_ply_path, pcd)

    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    output_depth_path = 'result/output_depth.jpg'
    cv2.imwrite(output_depth_path, depth_vis)

    progress(0.9, desc="웹 시각화 렌더링 중...")
    fig = create_plotly_fig(pcd)

    progress(1.0, desc="완료!")
    return output_depth_path, fig, output_ply_path

def update_ui_language(lang):
    t = I18N[lang]
    return (
        gr.update(value=t["title"]), gr.update(value=t["desc"]), gr.update(label=t["input_img"]),
        gr.update(label=t["mask_img"]), gr.update(label=t["density"]), gr.update(value=t["btn_convert"]),
        gr.update(label=t["out_depth"]), gr.update(label=t["out_3d"]), gr.update(label=t["out_file"]), 
        gr.update(value=t["tip"])
    )

with gr.Blocks(title="Image2PCD", css=custom_css) as demo:
    current_lang = gr.State("en")
    
    with gr.Row():
        gr.HTML("<div style='flex-grow: 1;'></div>")
        lang_dropdown = gr.Dropdown(choices=["en", "ko"], value="en", label="Language", interactive=True)

    title_md = gr.Markdown(I18N["en"]["title"])
    desc_md = gr.Markdown(I18N["en"]["desc"])
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label=I18N["en"]["input_img"])
            mask_image = gr.Image(type="filepath", label=I18N["en"]["mask_img"])
            density_slider = gr.Slider(minimum=0.0, maximum=0.05, value=0.0, step=0.005, label=I18N["en"]["density"])
            submit_btn = gr.Button(I18N["en"]["btn_convert"], variant="primary")
            
        with gr.Column():
            output_depth = gr.Image(type="filepath", label=I18N["en"]["out_depth"])
            # [수정됨] Model3D 대신 Plot과 File 컴포넌트로 분리
            output_3d = gr.Plot(label=I18N["en"]["out_3d"])
            output_file = gr.File(label=I18N["en"]["out_file"])

    tip_md = gr.Markdown(I18N["en"]["tip"])

    lang_dropdown.change(
        fn=update_ui_language, inputs=[lang_dropdown],
        outputs=[title_md, desc_md, input_image, mask_image, density_slider, submit_btn, output_depth, output_3d, output_file, tip_md]
    )

    submit_btn.click(
        fn=process_image,
        inputs=[input_image, mask_image, density_slider, lang_dropdown],
        outputs=[output_depth, output_3d, output_file]
    )

if __name__ == "__main__":
    demo.launch(share=False)