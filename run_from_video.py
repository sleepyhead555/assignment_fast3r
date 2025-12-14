import os
import tempfile
import shutil
from pathlib import Path

import torch
import cv2
from PIL import Image
import numpy as np

# optional: safetensors loader
try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None

# 下面两行假定 fast3r 可 import；若是本地 repo，请确保 PYTHONPATH 指向该 repo
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# ---------------------------
# 配置
# ---------------------------
VIDEO_PATH = "/root/autodl-tmp/fast3r/demo_examples/kitchen/family/Family.mp4"   # 要抽帧的视频路径
WEIGHT_FILE = "/root/autodl-tmp/fast3r/Fast3R_ViT_Large_512/model.safetensors"  # 下载好的 safetensors 权重的完整路径
FRAME_INTERVAL = 30   # 每隔多少帧抽一帧；例如 30 表示每秒若视频30fps则取1帧/s
MAX_FRAMES = 3        # 最多用多少张视角（模型原来是三张，现在可用多张）；注意显存
IMG_SIZE = 512        # load_images 的 size 参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # 或 torch.bfloat16 若支持

# ---------------------------
# 视频抽帧函数：保存到临时目录，返回文件路径列表
# ---------------------------
def extract_frames_to_temp(video_path, interval=30, max_frames=6):
    tmpdir = Path(tempfile.mkdtemp(prefix="fast3r_frames_"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frame_paths = []
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            fname = tmpdir / f"frame_{saved:04d}.jpg"
            img.save(str(fname), quality=95)
            frame_paths.append(str(fname))
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1

    cap.release()
    if len(frame_paths) == 0:
        raise RuntimeError("No frames extracted — check interval/max_frames/video length.")
    return tmpdir, frame_paths

def save_points_to_ply(points_np, ply_path, colors_np=None):
    """
    points_np: (N,3) numpy array
    colors_np: optional (N,3) uint8 array, or None
    """
    N = points_np.shape[0]
    has_color = colors_np is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    with open(ply_path, "w") as f:
        f.write("\n".join(header) + "\n")
        if has_color:
            for p, c in zip(points_np, colors_np):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points_np:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

# ---------------------------
# 模型加载函数：优先尝试 from_pretrained(local_dir)；否则尝试 safetensors 直接加载 state_dict
# ---------------------------
def load_model_from_safetensors_or_dir(weight_path, device):
    weight_path = Path(weight_path)
    # # 如果权重是一个目录（含 config + checkpoint），直接调用 from_pretrained
    # if weight_path.is_dir():
    #     print("Loading model from directory via Fast3R.from_pretrained(...)")
    #     model = Fast3R.from_pretrained(str(weight_path))
    #     return model.to(device)

    # # 如果权重是单个文件
    # # 先尝试把它放到一个临时目录，命名成 huggingface 风格的 repo folder，再 from_pretrained
    # # 许多 HF models expect a folder; 有时 from_pretrained 可以接受文件路径——试试
    # try:
    #     print("Trying Fast3R.from_pretrained with file path...")
    #     model = Fast3R.from_pretrained(str(weight_path))
    #     return model.to(device)
    # except Exception as e:
    #     print("from_pretrained(file) failed:", e)

    # 如果是 safetensors，并且安装了 safetensors 库，尝试加载 state_dict
    if weight_path.suffix in (".safetensors",) and safetensors_load is not None:
        print("Loading safetensors state dict...")
        sd = safetensors_load(str(weight_path))
        # sd 是一个 dict of tensors，可能和模型的 state_dict 名称有前缀差异
        # 创建模型结构（使用默认配置），然后尝试 load_state_dict
        print("Instantiating model architecture (uninitialized weights)...")

        decoder_args = {
            "attn_bias_for_inference_enabled": False,
            "attn_drop": 0.0,
            "attn_implementation": "flash_attention",
            "decoder_type": "fast3r",
            "depth": 24,
            "drop": 0.0,
            "embed_dim": 1024,
            "enc_embed_dim": 1024,
            "mlp_ratio": 4.0,
            "num_heads": 16,
            "qkv_bias": True,
            "random_image_idx_embedding": True
        }
        encoder_args = {
            "attn_implementation": "flash_attention",
            "depth": 24,
            "embed_dim": 1024,
            "encoder_type": "croco",
            "img_size": 512,
            "mlp_ratio": 4,
            "num_heads": 16,
            "patch_embed_cls": "PatchEmbedDust3R",
            "patch_size": 16,
            "pos_embed": "RoPE100"
        }
        head_args = {
            "conf_mode": [
            "exp",
            1,
            "Infinity"
            ],
            "depth_mode": [
            "exp",
            "-Infinity",
            "Infinity"
            ],
            "head_type": "dpt",
            "landscape_only": False,
            "output_mode": "pts3d",
            "patch_size": 16,
            "with_local_head": True
        }
        model = Fast3R(encoder_args, decoder_args, head_args)  
        model = model.to(device)
        try:
            # 尝试直接加载（严格匹配）
            model.load_state_dict(sd, strict=True)
            print("Loaded state_dict strict=True")
        except Exception as e:
            print("Strict load failed, trying non-strict load:", e)
            try:
                model.load_state_dict(sd, strict=False)
                print("Loaded state_dict strict=False")
            except Exception as e2:
                print("Non-strict load also failed:", e2)
                raise RuntimeError("无法把 safetensors 权重匹配到模型（名称不匹配或需要额外的 config）")
        return model

    raise RuntimeError("无法加载权重：既不是目录也无法用 safetensors 加载，或缺少 safetensors 库。")

# ---------------------------
# 主流程
# ---------------------------
def main():
    # 1) 抽帧
    tmpdir, frame_paths = extract_frames_to_temp(VIDEO_PATH, interval=FRAME_INTERVAL, max_frames=MAX_FRAMES)
    print(f"Extracted {len(frame_paths)} frames to {tmpdir}")

    # 2) 加载模型
    model = load_model_from_safetensors_or_dir(WEIGHT_FILE, DEVICE)
    model = model.to(DEVICE)
    model.eval()

    # 3) 包装成 lit module（你之前用的）
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    lit_module.eval()

    # 4) load_images（注意：load_images 期望路径列表；我们正好传 frame_paths）
    images = load_images(frame_paths, size=IMG_SIZE, verbose=True)

    # 5) inference
    output_dict, profiling_info = inference(
        images,
        model,
        DEVICE,
        dtype=DTYPE,
        verbose=True,
        profiling=True,
    )

    # 6) 估计相机位姿
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    camera_poses = poses_c2w_batch[0]
    for i, pose in enumerate(camera_poses):
        print(f"Camera Pose for view {i}: shape={pose.shape}")

    # 7) 输出每个视图的点云形状（示例）
    for view_idx, pred in enumerate(output_dict['preds']):
        point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
        print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")

    # all_points = []
    # for view_idx, pred in enumerate(output_dict['preds']):
    #     # 假设 pred['pts3d_in_other_view'] 是 tensor，形状 (1, H, W, 3) 或 (H, W, 3)
    #     pts = pred['pts3d_in_other_view'].cpu().numpy()
    #     # 一般可能是 (1, H, W, 3)
    #     if pts.ndim == 4 and pts.shape[0] == 1:
    #         pts = pts[0]
    #     H, W, C = pts.shape
    #     pts = pts.reshape(-1, 3)

    #     # 可选：根据置信度掩码过滤（如果有 pred['conf'] 或 pred['mask']）
    #     # conf = pred.get('conf', None)
    #     # if conf is not None:
    #     #     conf_np = conf.cpu().numpy().reshape(-1)
    #     #     mask = conf_np > 0.5
    #     #     pts = pts[mask]

    #     ply_path = f"view_{view_idx:02d}.ply"
    #     save_points_to_ply(pts, ply_path)
    #     print(f"Saved {pts.shape[0]} points to {ply_path}")
    #     all_points.append(pts)

    # # 合并并保存
    # if len(all_points) > 0:
    #     merged = np.vstack(all_points)
    #     save_points_to_ply(merged, "merged_all_views.ply")
    #     print(f"Saved merged point cloud with {merged.shape[0]} points to merged_all_views.ply")

    # 清理临时帧文件夹（如需保留可注释掉）
    shutil.rmtree(tmpdir)
    print("Temporary frames removed.")

if __name__ == "__main__":
    main()
