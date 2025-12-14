import glob
import os
import cv2
import numpy as np
import torch
import open3d as o3d

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference, find_opt_scaling
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# -------- 配置：7-Scenes 抽帧与评估 --------
DATA_ROOT = "/root/autodl-tmp/OpenDataLab___7-Scenes/raw"
SCENE = "chess"       # 可改：fire, heads, office, pumpkin, redkitchen, stairs
SEQ = "seq-01"        # 可改：seq-01, seq-02 ...
NUM_VIEWS = 5         # 采样的帧数
SAMPLING_STRATEGY = "stride"  # 可选: linspace, window, stride
FIXED_STRIDE = 100             # SAMPLING_STRATEGY=stride 时使用；None 表示自动

# 内参与深度尺度（参考 seven_scenes.py / SimpleRecon）
INTRINSICS_7SCENES = {
    "fx": 525.0,
    "fy": 525.0,
    "cx": 320.0,
    "cy": 240.0,
    "depth_scale": 1000.0,  # 深度存储为毫米
}

# 点云处理
CONF_THRESH = 0.3          # 置信度阈值
APPLY_SCALE = False        # 是否尝试用 find_opt_scaling 做尺度对齐（需要真值点云）


# -------- 工具函数 --------
def select_7scenes_frames(data_root, scene, seq, num_views, strategy="stride", fixed_stride=None):
    """
    从指定 7-Scenes 序列中抽取 color 帧，并返回对应 pose/depth 路径。
    抽帧策略:
        - linspace (默认): 首尾必选，其余等距索引 (np.linspace)。
        - window: 取一段连续窗口，长度为 num_views，窗口居中（若不足则靠前）。
        - stride: 按固定步长抽帧；若 fixed_stride=None，则自动估计 stride= max(1, len/num_views)。
    """
    seq_dir = os.path.join(data_root, scene, seq)
    color_glob = os.path.join(seq_dir, "frame-*.color.png")
    color_files = sorted(glob.glob(color_glob))
    if len(color_files) < num_views:
        raise ValueError(f"{scene}/{seq} 颜色帧不足 {num_views} 张，只有 {len(color_files)}")

    if strategy == "window":
        # 连续窗口，尽量居中
        start = max(0, (len(color_files) - num_views) // 2)
        end = start + num_views
        selected_colors = color_files[start:end]
    elif strategy == "stride":
        stride = fixed_stride
        if stride is None:
            stride = max(1, len(color_files) // num_views)
        selected_colors = color_files[::stride][:num_views]
        if len(selected_colors) < num_views:
            # 回退：若不足，再补尾部
            selected_colors = (color_files[::stride] + color_files[-num_views:])[:num_views]
    else:
        # 默认 linspace：首尾必选，中间等间隔
        indices = np.linspace(0, len(color_files) - 1, num_views, dtype=int)
        selected_colors = [color_files[i] for i in indices]

    pose_files = []
    for cf in selected_colors:
        stem = os.path.basename(cf).replace(".color.png", "")
        pose_path = os.path.join(seq_dir, f"{stem}.pose.txt")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"未找到对应 pose: {pose_path}")
        pose_files.append(pose_path)

    depth_files = [cf.replace(".color.png", ".depth.png") for cf in selected_colors]
    return selected_colors, depth_files, pose_files


def load_depth_png(path, depth_scale=1000.0):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"读取深度失败: {path}")
    depth = depth.astype(np.float32) / depth_scale
    return depth


def backproject_depth(depth, fx, fy, cx, cy):
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = depth
    mask = z > 0
    z = z[mask]
    x = (uu[mask] - cx) * z / fx
    y = (vv[mask] - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def build_gt_pointcloud(depth_paths, pose_paths, intrinsics, max_points=None):
    """
    使用真值位姿 + 深度生成 GT 点云（世界坐标）。
    """
    all_pts = []
    for dpath, ppath in zip(depth_paths, pose_paths):
        depth = load_depth_png(dpath, depth_scale=intrinsics.get("depth_scale", 1000.0))
        cam_pts = backproject_depth(depth, intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])
        pose = np.loadtxt(ppath).astype(np.float32)  # c2w
        homo = np.concatenate([cam_pts, np.ones((cam_pts.shape[0], 1), dtype=np.float32)], axis=1)
        world = (pose @ homo.T).T[:, :3]
        all_pts.append(world)
    merged = np.concatenate(all_pts, axis=0) if len(all_pts) > 0 else np.zeros((0, 3), dtype=np.float32)
    if max_points is not None and merged.shape[0] > max_points:
        idx = np.random.choice(merged.shape[0], max_points, replace=False)
        merged = merged[idx]
    return merged.astype(np.float32)


def estimate_normals_o3d(points_np, radius=0.05, max_nn=30):
    """
    使用 Open3D 估计点云法线。
    """
    if points_np.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def transform_points_hom(points_np, pose4x4):
    if points_np.size == 0:
        return points_np
    N = points_np.shape[0]
    homo = np.concatenate([points_np, np.ones((N, 1), dtype=points_np.dtype)], axis=1)  # (N,4)
    trans = (pose4x4 @ homo.T).T
    return trans[:, :3]


def save_ply(path, points_np, colors_np=None):
    N = points_np.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors_np is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors_np is None:
            for p in points_np:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(points_np, colors_np):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def load_gt_poses_txt(pose_txt_paths):
    poses = []
    for pp in pose_txt_paths:
        mat = np.loadtxt(pp)  # 4x4
        if mat.shape != (4, 4):
            raise ValueError(f"Pose file {pp} 形状异常: {mat.shape}")
        poses.append(mat.astype(np.float32))
    return np.stack(poses, axis=0)


# -------- 主流程 --------
def run_fast3r_pipeline(
    data_root,
    scene,
    seq,
    num_views=5,
    size=512,
    sampling_strategy="linspace",
    fixed_stride=None,
    conf_thresh=0.3,
    apply_scale=False,
    model_path=r"/root/autodl-tmp/fast3r/Fast3R_ViT_Large_512",
    out_dir="/root/autodl-tmp/output",
    device_str=None,
):
    # 选择 7-Scenes 帧
    filelist, depth_paths, pose_paths = select_7scenes_frames(
        data_root, scene, seq, num_views, strategy=sampling_strategy, fixed_stride=fixed_stride
    )
    print("Using frames:")
    for p in filelist:
        print("  ", p)

    # 加载图像
    images = load_images(filelist, size=size, verbose=True)

    # 模型加载
    model = Fast3R.from_pretrained(model_path)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()

    # 推理
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )

    # 估计相机位姿
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    camera_poses = poses_c2w_batch[0]  # list of np (4,4)

    # 打印形状
    for view_idx, pose in enumerate(camera_poses):
        print(f"Camera Pose for view {view_idx}:", pose.shape)

    # 构建真值：位姿与点云
    gt_poses_array = load_gt_poses_txt(pose_paths)
    print("Building GT point cloud from depth + poses ...")
    gt_points_array = build_gt_pointcloud(
        depth_paths,
        pose_paths,
        intrinsics=INTRINSICS_7SCENES,
        max_points=2_000_000
    )
    gt_normals_array = estimate_normals_o3d(gt_points_array, radius=0.05, max_nn=30)
    print("GT point cloud size:", gt_points_array.shape)

    # 逐视角点云 -> 合并
    all_world_points = []
    all_world_colors = []
    for view_idx, pred in enumerate(output_dict['preds']):
        pts_t = None
        if 'pts3d_in_other_view' in pred:
            pts_t = pred['pts3d_in_other_view']
        elif 'pts3d' in pred:
            pts_t = pred['pts3d']
        elif 'depth' in pred:
            print("view", view_idx, "has depth but no pts3d; skipping depth->pts conversion.")
            continue
        else:
            print("view", view_idx, "no recognized pts field; skipping.")
            continue

        if isinstance(pts_t, torch.Tensor):
            pts_np = pts_t.squeeze(0).cpu().numpy()
        else:
            pts_np = np.array(pts_t).squeeze(0)

        H, W, _ = pts_np.shape
        pts_flat = pts_np.reshape(-1, 3)

        try:
            img_t = images[view_idx]["img"].squeeze(0).cpu()
            img_np = ((img_t.numpy().transpose(1, 2, 0) * 0.5) + 0.5) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            if (img_np.shape[0] != H) or (img_np.shape[1] != W):
                color_img_np = cv2.resize(img_np, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                color_img_np = img_np
            colors_flat = color_img_np.reshape(-1, 3)
        except Exception:
            colors_flat = np.ones((pts_flat.shape[0], 3), dtype=np.uint8) * 255

        conf_t = pred.get('conf', None)
        if conf_t is not None:
            if isinstance(conf_t, torch.Tensor):
                conf_np = conf_t.squeeze(0).cpu().numpy().reshape(-1)
            else:
                conf_np = np.array(conf_t).squeeze(0).reshape(-1)
            mask = (conf_np > conf_thresh)
        else:
            mask = np.isfinite(pts_flat).all(axis=1)

        pts_selected = pts_flat[mask]
        colors_selected = colors_flat[mask]
        finite_mask = np.isfinite(pts_selected).all(axis=1)
        pts_selected = pts_selected[finite_mask]
        colors_selected = colors_selected[finite_mask]

        pose = camera_poses[view_idx]  # 4x4 numpy
        world_pts = transform_points_hom(pts_selected, pose)
        all_world_points.append(world_pts)
        all_world_colors.append(colors_selected)

    if len(all_world_points) == 0:
        print("No points collected from views.")
        return

    merged = np.concatenate(all_world_points, axis=0)
    merged_colors = np.concatenate(all_world_colors, axis=0) if len(all_world_colors) > 0 else None
    print("Merged points count:", merged.shape)

    # 可选尺度对齐（需 GT 点云）
    if apply_scale:
        try:
            gt_t = torch.from_numpy(gt_points_array[None, None, ...].astype(np.float32))
            pr_t = torch.from_numpy(merged[None, None, ...].astype(np.float32))
            scale = find_opt_scaling(gt_t, None, pr_t, None).item()
            print("Estimated scale:", scale)
            merged = merged * scale
        except Exception as e:
            print("Scaling failed:", e)

    # 输出
    os.makedirs(out_dir, exist_ok=True)

    out_ply = os.path.join(out_dir, f"{scene}_{seq}_{sampling_strategy}_points.ply")
    save_ply(out_ply, merged, merged_colors)
    print("Saved merged PLY to", out_ply)

    # 保存预测/真值位姿与真值点云
    poses_array = np.stack(camera_poses, axis=0)  # (V,4,4)
    poses_out_npy = os.path.join(out_dir, "camera_poses.npy")
    np.save(poses_out_npy, poses_array)
    print("Saved camera poses (npy) to", poses_out_npy)

    gt_poses_out_npy = os.path.join(out_dir, "gt_camera_poses.npy")
    np.save(gt_poses_out_npy, gt_poses_array)
    print("Saved GT camera poses (npy) to", gt_poses_out_npy)

    gt_points_out_npy = os.path.join(out_dir, "gt_points.npy")
    np.save(gt_points_out_npy, gt_points_array)
    print("Saved GT points (npy) to", gt_points_out_npy)

    gt_normals_out_npy = os.path.join(out_dir, "gt_normals.npy")
    np.save(gt_normals_out_npy, gt_normals_array)
    print("Saved GT normals (npy) to", gt_normals_out_npy)

    # 预测点云法线
    pred_normals_array = estimate_normals_o3d(merged.astype(np.float32), radius=0.05, max_nn=30)
    pred_normals_out_npy = os.path.join(out_dir, "pred_normals.npy")
    np.save(pred_normals_out_npy, pred_normals_array)
    print("Saved predicted normals (npy) to", pred_normals_out_npy)

    # 相机信息
    import json
    cam_info = {
        "scene": scene,
        "sequence": seq,
        "filelist": filelist,
        "gt_pose_path": gt_poses_out_npy,
        "gt_points_path": gt_points_out_npy,
        "gt_normals_path": gt_normals_out_npy,
        "estimated_focals": [float(x) for x in np.asarray(estimated_focals).reshape(-1).tolist()] if estimated_focals is not None else None,
    }
    cam_info_path = os.path.join(out_dir, "camera_info.json")
    with open(cam_info_path, "w") as jf:
        json.dump(cam_info, jf, indent=2)
    print("Saved camera info (json) to", cam_info_path)

    # ===== 评估 =====
    # 相机位姿评估
    try:
        from fast3r.eval import cam_pose_metric, recon_metric
    except Exception:
        import importlib.util
        spec1 = importlib.util.spec_from_file_location("cam_pose_metric", "/root/autodl-tmp/fast3r/fast3r/eval/cam_pose_metric.py")
        cam_pose_metric = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(cam_pose_metric)
        spec2 = importlib.util.spec_from_file_location("recon_metric", "/root/autodl-tmp/fast3r/fast3r/eval/recon_metric.py")
        recon_metric = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(recon_metric)

    # 相机位姿评估
    pred_poses = np.stack(camera_poses, axis=0)
    pred_t = torch.from_numpy(pred_poses.astype(np.float32)).to(device)
    gt_t = torch.from_numpy(gt_poses_array.astype(np.float32)).to(device)
    rel_r_deg, rel_t_deg = cam_pose_metric.camera_to_rel_deg(pred_t, gt_t, device, batch_size=pred_t.shape[0])
    if isinstance(rel_r_deg, torch.Tensor):
        rel_r_deg = rel_r_deg.cpu().numpy()
    if isinstance(rel_t_deg, torch.Tensor):
        rel_t_deg = rel_t_deg.cpu().numpy()
    print("Pose evaluation (relative pairwise):")
    print(f"Rotation error mean: {np.mean(rel_r_deg):.4f} deg, median: {np.median(rel_r_deg):.4f} deg")
    print(f"Translation-angle error mean: {np.mean(rel_t_deg):.4f} deg, median: {np.median(rel_t_deg):.4f} deg")

    # 额外指标：AUC@5，ATE，ARE，RPE-rot，RPE-trans
    auc5 = cam_pose_metric.calculate_auc_np(rel_r_deg, rel_t_deg, max_threshold=5)
    ate = cam_pose_metric.absolute_translation_error(pred_poses, gt_poses_array)
    are = cam_pose_metric.absolute_rotation_error(pred_poses, gt_poses_array)
    rpe_rot, rpe_trans = cam_pose_metric.relative_pose_error(pred_poses, gt_poses_array)
    print(f"AUC@5: {auc5:.6f}")
    print(f"ATE mean: {np.mean(ate):.6f}, median: {np.median(ate):.6f}")
    print(f"ARE mean: {np.mean(are):.6f}, median: {np.median(are):.6f}")
    print(f"RPE-rot mean: {np.mean(rpe_rot):.6f}, median: {np.median(rpe_rot):.6f}")
    print(f"RPE-trans mean: {np.mean(rpe_trans):.6f}, median: {np.median(rpe_trans):.6f}")

    # 点云重建评估（含法线一致性）
    gt_pts = gt_points_array.astype(np.float32)
    gt_normals = gt_normals_array.astype(np.float32)
    rec_pts = merged.astype(np.float32)
    rec_normals = pred_normals_array.astype(np.float32)

    acc_res = recon_metric.accuracy(gt_pts, rec_pts, gt_normals, rec_normals)
    comp_res = recon_metric.completion(gt_pts, rec_pts, gt_normals, rec_normals)
    acc_mean, acc_median, nc_acc_mean, nc_acc_median = acc_res
    comp_mean, comp_median, nc_comp_mean, nc_comp_median = comp_res
    comp_ratio = recon_metric.completion_ratio(gt_pts, rec_pts, dist_th=0.05)

    print("Reconstruction evaluation:")
    print(f"Accuracy mean: {acc_mean:.6f}, median: {acc_median:.6f}")
    print(f"Completion mean: {comp_mean:.6f}, median: {comp_median:.6f}")
    print(f"NC (accuracy) mean: {nc_acc_mean:.6f}, median: {nc_acc_median:.6f}")
    print(f"NC (completion) mean: {nc_comp_mean:.6f}, median: {nc_comp_median:.6f}")
    print(f"Completion ratio (@0.05 m): {comp_ratio:.6f}")

    return {
        "filelist": filelist,
        "depth_paths": depth_paths,
        "pose_paths": pose_paths,
        "images": images,
        "output_dict": output_dict,
        "profiling_info": profiling_info,
        "camera_poses": camera_poses,
        "estimated_focals": estimated_focals,
        "merged_points": merged,
        "merged_colors": merged_colors,
        "gt_points": gt_points_array,
        "gt_normals": gt_normals_array,
        "pred_normals": pred_normals_array,
        "paths": {
            "out_dir": out_dir,
            "out_ply": out_ply,
            "poses_npy": poses_out_npy,
            "gt_poses_npy": gt_poses_out_npy,
            "gt_points_npy": gt_points_out_npy,
            "gt_normals_npy": gt_normals_out_npy,
            "pred_normals_npy": pred_normals_out_npy,
            "cam_info_json": cam_info_path,
        },
        "metrics": {
            "rel_r_deg_mean": np.mean(rel_r_deg),
            "rel_r_deg_median": np.median(rel_r_deg),
            "rel_t_deg_mean": np.mean(rel_t_deg),
            "rel_t_deg_median": np.median(rel_t_deg),
            #
            "auc@5": auc5,
            "ate": ate.tolist(),
            "are": are.tolist(),
            "rpe_rot": rpe_rot.tolist(),
            "rpe_trans": rpe_trans.tolist(),
            #
            "acc_mean": acc_mean,
            "acc_median": acc_median,
            "comp_mean": comp_mean,
            "comp_median": comp_median,
            "nc_acc_mean": nc_acc_mean,
            "nc_acc_median": nc_acc_median,
            "nc_comp_mean": nc_comp_mean,
            "nc_comp_median": nc_comp_median,
            "comp_ratio_0_05": comp_ratio,
        },
    }


if __name__ == "__main__":
    run_fast3r_pipeline(
        DATA_ROOT,
        SCENE,
        SEQ,
        num_views=NUM_VIEWS,
        size=512,
        sampling_strategy=SAMPLING_STRATEGY,
        fixed_stride=FIXED_STRIDE,
        conf_thresh=CONF_THRESH,
        apply_scale=APPLY_SCALE,
        model_path=r"/root/autodl-tmp/fast3r/Fast3R_ViT_Large_512",
        out_dir="/root/autodl-tmp/output",
        device_str=None,
    )

