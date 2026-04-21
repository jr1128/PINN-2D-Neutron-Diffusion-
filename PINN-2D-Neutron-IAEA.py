
# -*- coding: utf-8 -*-
"""2D IAEA 基准题 PINN 训练代码（整理版）"""

import atexit
import copy
import json
import os
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# 1. 实验配置：统一管理超参数
# ============================================================

RUNTIME_CONFIG = {
    "seed": 42,
    "allow_tf32": True,
    "matmul_precision": "high",
}

PHYSICS_CONFIG = {
    "x_max": 170.0,
    "y_max": 170.0,
    "b_z_sq": 0.8e-4,
    "raw_materials": {
        1: [1.5, 0.4, 0.02, 0.01, 0.080, 0.135],  # Fuel 1
        2: [1.5, 0.4, 0.02, 0.01, 0.085, 0.135],  # Fuel 2
        3: [1.5, 0.4, 0.02, 0.01, 0.130, 0.135],  # Fuel 3 (Rod)
        4: [2.0, 0.3, 0.04, 0.00, 0.010, 0.0],    # Reflector
    },
}

GEOMETRY_CONFIG = {
    "x_grid_lines": [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0],
    "y_grid_lines": [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0],
    "mat_grid": [
        [3, 2, 2, 2, 3, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 1, 1, 4],
        [2, 2, 2, 2, 2, 2, 1, 4, 4],
        [3, 2, 2, 2, 3, 1, 1, 4, 0],
        [2, 2, 2, 2, 1, 1, 4, 4, 0],
        [2, 2, 1, 1, 1, 4, 4, 0, 0],
        [1, 1, 1, 4, 4, 4, 0, 0, 0],
        [4, 4, 4, 4, 0, 0, 0, 0, 0],
    ],
    "true_interface_scan_eps": 1e-2,
    "true_interface_scan_points": 512,
    "interface_grid_lines": [10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
    "external_segments": [
        (0, 70, 170, 170, 0, 1),
        (170, 170, 0, 70, 1, 0),
        (70, 110, 150, 150, 0, 1),
        (110, 130, 130, 130, 0, 1),
        (130, 150, 110, 110, 0, 1),
        (150, 170, 70, 70, 0, 1),
        (70, 70, 150, 170, 1, 0),
        (110, 110, 130, 150, 1, 0),
        (130, 130, 110, 130, 1, 0),
        (150, 150, 70, 110, 1, 0),
    ],
}

SAMPLING_CONFIG = {
    "quad_samples": 20000,
    "fixed_collocation_total": 30000,
    "weighted_region_weights": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
    "adaptive_start_epoch": 1000,
    "fixed_ratio": 0.5,
    "adaptive_candidate_ratio": 1.2,
    "adaptive_min_ratio": 0.2,
    "adaptive_reflector_multiplier": 2.0,
    "adaptive_extra_fuel_frac": 0.1,
    "adaptive_weight_eps": 1e-8,
    "boundary_segment_ratio": 12,
    "mirror_multiplier": 1.5,
    "ext_segment_multiplier": 1.2,
    "interface_eps": 0.1,
    "interface_direction_ratio": 0.6,
    "interface_min_points_per_line": 10,
    "bc_inner_eps": 1e-3,
}

MODEL_CONFIG = {
    "n_mat": 4,
    "subnet_type": "fourier",
    "embedding_dim": 32,
    "hidden_dim": 50,
    "feature_dim": 64,
    "sigmas": (0.6, 2.2, 5.0),
    "learnable_sigma": False,
    "sigma_min": 0.3,
    "sigma_max": 20.0,
    "plain_hidden_dim":75,
    "plain_n_hidden": 8,
}

LOSS_CONFIG = {
    "w_pde": 1.0,
    "w_bc": 1.0,
    "w_ifc": 1.0,
    "material_weights": [1.0, 1.0, 1.0, 1.6],
    "interface_train_points": 4000,
}

TRAINING_CONFIG = {
    "outer_iters": 200,
    "inner_epochs": 1500,
    "n_coll": 20000,
    "n_bound": 2000,
    "sample_interval": 50,
    "if_sample_interval": 50,
    "history_interval": 100,
    "visualize_interval": 10000,
    "auto_tune_block_size": 1000,
    "target_bc_over_pde": 0.1,
    "w_bc_min": 1e-2,
    "w_bc_max": 10.0,
    "optimizer_lr": 1e-3,
    "scheduler_eta_min": 1e-6,
    "grad_clip_norm": 1.0,
    "always_clip_before_epoch": 20000,
    "clip_every_after_epoch": 5,
    "print_inner_interval": 4000,
    "init_keff": 1.0,
}

KEFF_UPDATE_CONFIG = {
    "alpha": 0.8,
    "min_keff": 0.8,
    "max_keff": 1.2,
}

PLOT_CONFIG = {
    "grid_size": 100,
    "sampling_figure_name": "fig_sampling.png",
    "loss_figure_name": "loss_keff.png",
    "pde_figure_name": "pde.png",
    "flux_figure_name": "phi.png",
}

SEED = int(os.environ.get("SEED", str(RUNTIME_CONFIG["seed"])))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if RUNTIME_CONFIG["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(RUNTIME_CONFIG["matmul_precision"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), f"run_{RUN_ID}"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_MAX = PHYSICS_CONFIG["x_max"]
Y_MAX = PHYSICS_CONFIG["y_max"]
B_Z_SQ = PHYSICS_CONFIG["b_z_sq"]

X_GRID_LINES = GEOMETRY_CONFIG["x_grid_lines"]
Y_GRID_LINES = GEOMETRY_CONFIG["y_grid_lines"]
X_CAND_LINES = X_GRID_LINES[1:-1]
Y_CAND_LINES = Y_GRID_LINES[1:-1]

TRUE_INTERFACE_SCAN_EPS = GEOMETRY_CONFIG["true_interface_scan_eps"]
TRUE_INTERFACE_SCAN_POINTS = GEOMETRY_CONFIG["true_interface_scan_points"]
INTERFACE_GRID_LINES = GEOMETRY_CONFIG["interface_grid_lines"]
EXT_SEGMENTS = GEOMETRY_CONFIG["external_segments"]

WEIGHTED_REGION_WEIGHTS = SAMPLING_CONFIG["weighted_region_weights"]
INTERFACE_EPS = SAMPLING_CONFIG["interface_eps"]
BC_INNER_EPS = SAMPLING_CONFIG["bc_inner_eps"]


def out_path(fname: str) -> str:
    return os.path.join(OUTPUT_DIR, fname)


LOG_PATH = out_path("train.log")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


_original_stdout = sys.stdout
_original_stderr = sys.stderr
_log_fp = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
sys.stdout = Tee(_original_stdout, _log_fp)
sys.stderr = Tee(_original_stderr, _log_fp)


def _close_log_file():
    try:
        _log_fp.flush()
        _log_fp.close()
    except Exception:
        pass


atexit.register(_close_log_file)

print(f"[Run] OUTPUT_DIR = {OUTPUT_DIR}")
print(f"[Run] LOG_PATH   = {LOG_PATH}")
print(f"[Run] START_TIME = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


RAW_MATERIALS = PHYSICS_CONFIG["raw_materials"]

MATERIALS = {}
for mat_id, params in RAW_MATERIALS.items():
    d1, d2, sigma12, sigma_a1, sigma_a2, nu_sigma_f2 = params
    sigma_a1_eff = sigma_a1 + d1 * B_Z_SQ
    sigma_a2_eff = sigma_a2 + d2 * B_Z_SQ
    MATERIALS[mat_id] = [d1, sigma_a1_eff + sigma12, d2, sigma_a2_eff, sigma12, nu_sigma_f2]

MATERIALS_TENSOR = torch.tensor(
    [[MATERIALS[i][j] for j in range(6)] for i in range(1, 5)],
    device=DEVICE,
    dtype=torch.float32,
)

mat_grid = torch.tensor(GEOMETRY_CONFIG["mat_grid"], device=DEVICE, dtype=torch.long)
_X_BUCKETS = torch.tensor(X_GRID_LINES[1:-1], device=DEVICE, dtype=torch.float32)
_Y_BUCKETS = torch.tensor(Y_GRID_LINES[1:-1], device=DEVICE, dtype=torch.float32)


def normalize_xy(xy: torch.Tensor) -> torch.Tensor:
    xy_norm = xy.clone()
    xy_norm[:, 0] = (xy[:, 0] / X_MAX) * 2.0 - 1.0
    xy_norm[:, 1] = (xy[:, 1] / Y_MAX) * 2.0 - 1.0
    return xy_norm


def get_material_id(x_coords, y_coords):
    """基于查表的 IAEA 材料 ID 映射。返回 0,1,2,3,4，其中 0 表示物理域外。"""
    if x_coords.dim() > 1:
        x_coords = x_coords.squeeze(-1)
    if y_coords.dim() > 1:
        y_coords = y_coords.squeeze(-1)

    x = x_coords
    y = y_coords
    mat_ids = torch.zeros_like(x, dtype=torch.long, device=x.device)

    in_rect = (x >= 0.0) & (x <= X_MAX) & (y >= 0.0) & (y <= Y_MAX)
    if not in_rect.any():
        return mat_ids

    x_valid = x[in_rect].to(_X_BUCKETS.dtype)
    y_valid = y[in_rect].to(_Y_BUCKETS.dtype)

    i_idx = torch.bucketize(x_valid, _X_BUCKETS, right=False)
    j_idx = torch.bucketize(y_valid, _Y_BUCKETS, right=False)

    mat_ids[in_rect] = mat_grid[j_idx, i_idx].to(torch.long)
    return mat_ids


def build_region_boxes_from_get_material_id():
    """使用真实网格线精确构建每种材料的采样盒。"""
    region_boxes = {1: [], 2: [], 3: [], 4: []}

    for i in range(len(X_GRID_LINES) - 1):
        x1, x2 = X_GRID_LINES[i], X_GRID_LINES[i + 1]
        for j in range(len(Y_GRID_LINES) - 1):
            y1, y2 = Y_GRID_LINES[j], Y_GRID_LINES[j + 1]
            xc = 0.5 * (x1 + x2)
            yc = 0.5 * (y1 + y2)
            with torch.no_grad():
                mat_id = get_material_id(
                    torch.tensor([xc], device=DEVICE),
                    torch.tensor([yc], device=DEVICE),
                )[0].item()
            if mat_id in (1, 2, 3, 4):
                region_boxes[mat_id].append((x1, x2, y1, y2))
    return region_boxes


REGION_BOXES = build_region_boxes_from_get_material_id()


@torch.no_grad()
def build_true_interface_lines(eps=TRUE_INTERFACE_SCAN_EPS, n_scan=TRUE_INTERFACE_SCAN_POINTS, device=DEVICE):
    """从内部候选网格线中筛出真实存在的材料界面。"""
    y = torch.linspace(0.0, Y_MAX, n_scan, device=device).view(-1, 1)
    x = torch.linspace(0.0, X_MAX, n_scan, device=device).view(-1, 1)

    true_x = []
    for x0 in X_CAND_LINES:
        x0_t = torch.full_like(y, x0)
        id_left = get_material_id(x0_t - eps, y)
        id_right = get_material_id(x0_t + eps, y)
        mask = (id_left != id_right) & (id_left > 0) & (id_right > 0)
        if mask.any():
            true_x.append(x0)

    true_y = []
    for y0 in Y_CAND_LINES:
        y0_t = torch.full_like(x, y0)
        id_down = get_material_id(x, y0_t - eps)
        id_up = get_material_id(x, y0_t + eps)
        mask = (id_down != id_up) & (id_down > 0) & (id_up > 0)
        if mask.any():
            true_y.append(y0)

    tx = (
        torch.tensor(true_x, device=device, dtype=torch.float32).view(1, -1)
        if true_x
        else torch.empty(1, 0, device=device)
    )
    ty = (
        torch.tensor(true_y, device=device, dtype=torch.float32).view(1, -1)
        if true_y
        else torch.empty(1, 0, device=device)
    )
    return tx, ty


TRUE_X_IF, TRUE_Y_IF = build_true_interface_lines()
print("TRUE_X_IF:", TRUE_X_IF.flatten().tolist())
print("TRUE_Y_IF:", TRUE_Y_IF.flatten().tolist())


def lhs_sample_from_boxes(boxes, n_points, device=DEVICE):
    if n_points <= 0:
        return torch.empty(0, 2, device=device)

    areas = torch.tensor(
        [(x2 - x1) * (y2 - y1) for (x1, x2, y1, y2) in boxes],
        dtype=torch.float32,
        device=device,
    )
    probs = areas / areas.sum()

    base_counts = torch.floor(probs * n_points).long()
    remain = int(n_points - base_counts.sum().item())
    if remain > 0:
        extra_idx = torch.multinomial(probs, remain, replacement=True)
        for idx in extra_idx:
            base_counts[idx] += 1

    pts_list = []
    for box, count in zip(boxes, base_counts.tolist()):
        if count <= 0:
            continue

        x_min, x_max, y_min, y_max = box
        n = int(count)

        u = torch.rand(n, 2, device=device)
        idx = torch.arange(n, device=device).unsqueeze(1)
        u = (idx + u) / n

        perm = torch.randperm(n, device=device)
        u[:, 1] = u[perm, 1]

        x = x_min + (x_max - x_min) * u[:, 0:1]
        y = y_min + (y_max - y_min) * u[:, 1:2]
        pts_list.append(torch.cat([x, y], dim=1))

    if not pts_list:
        return torch.empty(0, 2, device=device)

    return torch.cat(pts_list, dim=0)


QUAD_POINTS_BY_MAT = {}
QUAD_AREAS_BY_MAT = {}
FIXED_COLL_POINTS = None


def init_quad_points(n_samples=SAMPLING_CONFIG["quad_samples"], device=DEVICE):
    global QUAD_POINTS_BY_MAT, QUAD_AREAS_BY_MAT

    QUAD_POINTS_BY_MAT = {}
    QUAD_AREAS_BY_MAT = {}

    n_per_mat = n_samples // 4

    for mat_id in [1, 2, 3, 4]:
        boxes = REGION_BOXES[mat_id]
        area_m = sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes)
        QUAD_AREAS_BY_MAT[mat_id] = area_m

        if area_m == 0 or n_per_mat == 0:
            continue

        QUAD_POINTS_BY_MAT[mat_id] = lhs_sample_from_boxes(boxes, n_per_mat, device=device)


def weighted_sampling(n_total, device=DEVICE):
    n_per_region = {}
    for mat_id, weight in WEIGHTED_REGION_WEIGHTS.items():
        n_per_region[mat_id] = int(n_total * weight)

    total_assigned = sum(n_per_region.values())
    if total_assigned < n_total:
        max_mat = max(WEIGHTED_REGION_WEIGHTS, key=WEIGHTED_REGION_WEIGHTS.get)
        n_per_region[max_mat] += (n_total - total_assigned)
    elif total_assigned > n_total:
        min_mat = min(WEIGHTED_REGION_WEIGHTS, key=WEIGHTED_REGION_WEIGHTS.get)
        n_per_region[min_mat] -= (total_assigned - n_total)

    pts_list = []
    for mat_id in [1, 2, 3, 4]:
        n_region = n_per_region[mat_id]
        if n_region <= 0:
            continue
        pts_list.append(lhs_sample_from_boxes(REGION_BOXES[mat_id], n_region, device=device))

    if not pts_list:
        return torch.empty(0, 2, device=device)

    return torch.cat(pts_list, dim=0)


def init_fixed_collocation_points(n_fixed_total=SAMPLING_CONFIG["fixed_collocation_total"], device=DEVICE):
    global FIXED_COLL_POINTS
    FIXED_COLL_POINTS = weighted_sampling(n_fixed_total, device=device)
    print(f"[init_fixed_collocation_points] Generated {FIXED_COLL_POINTS.shape[0]} fixed points.")


def adaptive_sampling(model, n_coll_target, device=DEVICE, current_keff=None, source_model=None):
    n_candidates = int(n_coll_target * SAMPLING_CONFIG["adaptive_candidate_ratio"])
    candidate_pts = weighted_sampling(n_candidates, device=device)

    model.eval()
    x_c = candidate_pts[:, 0:1].detach().requires_grad_(True)
    y_c = candidate_pts[:, 1:2].detach().requires_grad_(True)
    xy_c = torch.cat([x_c, y_c], dim=1)

    mat_ids = get_material_id(x_c, y_c).long().view(-1)
    phi_1, phi_2 = model(xy_c, mat_ids)

    if source_model is not None:
        with torch.no_grad():
            _, phi_2_src = source_model(xy_c, mat_ids)
    else:
        phi_2_src = phi_2.detach()

    lap1 = compute_laplacian(phi_1, x_c, y_c, create_graph=False)
    lap2 = compute_laplacian(phi_2, x_c, y_c, create_graph=False)

    if current_keff is None:
        k_eff_val = 1.0
    elif isinstance(current_keff, torch.Tensor):
        k_eff_val = float(current_keff.item())
    else:
        k_eff_val = float(current_keff)

    coeffs = MATERIALS_TENSOR[mat_ids - 1]
    d1 = coeffs[:, 0:1]
    sig_r1 = coeffs[:, 1:2]
    d2 = coeffs[:, 2:3]
    sig_r2 = coeffs[:, 3:4]
    sig12 = coeffs[:, 4:5]
    nu_sf2 = coeffs[:, 5:6]

    res1 = -d1 * lap1 + sig_r1 * phi_1 - (1.0 / k_eff_val) * nu_sf2 * phi_2_src
    res2 = -d2 * lap2 + sig_r2 * phi_2 - sig12 * phi_1

    res = res1.abs() + res2.abs()
    multiplier = torch.ones_like(res)
    multiplier[mat_ids.view(-1) == 4] = SAMPLING_CONFIG["adaptive_reflector_multiplier"]
    res = res * multiplier
    residuals = res.squeeze(-1)

    model.train()
    residuals = residuals + SAMPLING_CONFIG["adaptive_weight_eps"]
    weights = residuals / residuals.sum()

    if torch.isnan(weights).any() or (weights.sum() == 0):
        selected_indices = torch.randperm(len(candidate_pts), device=device)[:n_coll_target]
    else:
        try:
            selected_indices = torch.multinomial(weights, n_coll_target, replacement=False)
        except RuntimeError:
            selected_indices = torch.randperm(len(candidate_pts), device=device)[:n_coll_target]

    adaptive_pts = candidate_pts[selected_indices]

    n_extra = int(n_coll_target * SAMPLING_CONFIG["adaptive_extra_fuel_frac"])
    if n_extra > 0:
        fuel_boxes = REGION_BOXES[1] + REGION_BOXES[2] + REGION_BOXES[3]
        fuel_pts = lhs_sample_from_boxes(fuel_boxes, n_extra, device=device)
        if fuel_pts.numel() > 0:
            if adaptive_pts.shape[0] + fuel_pts.shape[0] > n_coll_target:
                adaptive_pts = adaptive_pts[: n_coll_target - fuel_pts.shape[0]]
            adaptive_pts = torch.cat([adaptive_pts, fuel_pts], dim=0)

    return adaptive_pts


def generate_points(
    n_coll,
    n_b,
    model=None,
    global_epoch=0,
    current_keff=None,
    source_model=None,
    fixed_ratio=SAMPLING_CONFIG["fixed_ratio"],
):
    if model is None or global_epoch < SAMPLING_CONFIG["adaptive_start_epoch"]:
        collocation_points = weighted_sampling(n_coll, device=DEVICE)
    else:
        n_fixed = int(n_coll * fixed_ratio)
        n_adapt = n_coll - n_fixed
        n_adapt = max(n_adapt, int(SAMPLING_CONFIG["adaptive_min_ratio"] * n_coll))
        n_fixed = n_coll - n_adapt

        total_fixed = FIXED_COLL_POINTS.shape[0]
        idx = torch.randint(0, total_fixed, (n_fixed,), device=FIXED_COLL_POINTS.device)
        fixed_pts = FIXED_COLL_POINTS[idx]

        adapt_pts = adaptive_sampling(
            model,
            n_adapt,
            device=DEVICE,
            current_keff=current_keff,
            source_model=source_model,
        )
        collocation_points = torch.cat([fixed_pts, adapt_pts], dim=0)

    boundary_points_list = []
    n_per_segment = max(n_b // SAMPLING_CONFIG["boundary_segment_ratio"], 1)
    n_mirror = int(n_per_segment * SAMPLING_CONFIG["mirror_multiplier"])

    y_b1 = torch.rand(n_mirror, 1, device=DEVICE) * Y_MAX
    x_b1 = torch.zeros_like(y_b1)
    boundary_points_list.append((torch.cat([x_b1, y_b1], dim=1), -1, 0, "MIRROR"))

    x_b2 = torch.rand(n_mirror, 1, device=DEVICE) * X_MAX
    y_b2 = torch.zeros_like(x_b2)
    boundary_points_list.append((torch.cat([x_b2, y_b2], dim=1), 0, -1, "MIRROR"))

    for x1, x2, y1, y2, nx, ny in EXT_SEGMENTS:
        t = torch.rand(int(n_per_segment * SAMPLING_CONFIG["ext_segment_multiplier"]), 1, device=DEVICE)
        x_b = x1 + t * (x2 - x1)
        y_b = y1 + t * (y2 - y1)
        boundary_points_list.append((torch.cat([x_b, y_b], dim=1), nx, ny, "EXT_NULL"))

    return collocation_points, boundary_points_list


def sample_interface_points(n_if=LOSS_CONFIG["interface_train_points"], eps=INTERFACE_EPS, device=DEVICE):
    """只在真实材料界面附近采样。"""
    pts_list = []
    n_per_direction = int(n_if * SAMPLING_CONFIG["interface_direction_ratio"])
    n_per_line = max(n_per_direction // len(INTERFACE_GRID_LINES), SAMPLING_CONFIG["interface_min_points_per_line"])

    for x_val in INTERFACE_GRID_LINES:
        y = torch.rand(n_per_line, 1, device=device) * Y_MAX
        x = torch.full_like(y, x_val)
        id_left = get_material_id(x - eps, y)
        id_right = get_material_id(x + eps, y)
        mask = (id_left != id_right) & (id_left > 0) & (id_right > 0)
        if mask.any():
            count = int(mask.sum().item())
            pts_list.append(
                {
                    "xL": (x - eps)[mask],
                    "yL": y[mask],
                    "matL": id_left[mask],
                    "xR": (x + eps)[mask],
                    "yR": y[mask],
                    "matR": id_right[mask],
                    "nx": torch.ones(count, 1, device=device),
                    "ny": torch.zeros(count, 1, device=device),
                }
            )

    for y_val in INTERFACE_GRID_LINES:
        x = torch.rand(n_per_line, 1, device=device) * X_MAX
        y = torch.full_like(x, y_val)
        id_down = get_material_id(x, y - eps)
        id_up = get_material_id(x, y + eps)
        mask = (id_down != id_up) & (id_down > 0) & (id_up > 0)
        if mask.any():
            count = int(mask.sum().item())
            pts_list.append(
                {
                    "xL": x[mask],
                    "yL": (y - eps)[mask],
                    "matL": id_down[mask],
                    "xR": x[mask],
                    "yR": (y + eps)[mask],
                    "matR": id_up[mask],
                    "nx": torch.zeros(count, 1, device=device),
                    "ny": torch.ones(count, 1, device=device),
                }
            )

    if not pts_list:
        return (torch.empty(0, 1, device=device),) * 8

    xL = torch.cat([d["xL"] for d in pts_list], dim=0)
    yL = torch.cat([d["yL"] for d in pts_list], dim=0)
    matL = torch.cat([d["matL"] for d in pts_list], dim=0)
    xR = torch.cat([d["xR"] for d in pts_list], dim=0)
    yR = torch.cat([d["yR"] for d in pts_list], dim=0)
    matR = torch.cat([d["matR"] for d in pts_list], dim=0)
    nx = torch.cat([d["nx"] for d in pts_list], dim=0)
    ny = torch.cat([d["ny"] for d in pts_list], dim=0)

    total_valid = xL.shape[0]
    if total_valid > n_if:
        idx = torch.randperm(total_valid, device=device)[:n_if]
        xL, yL, matL = xL[idx], yL[idx], matL[idx]
        xR, yR, matR = xR[idx], yR[idx], matR[idx]
        nx, ny = nx[idx], ny[idx]

    return xL, yL, matL.long(), xR, yR, matR.long(), nx, ny


def compute_laplacian(phi, x, y, create_graph=True):
    grad_x, grad_y = torch.autograd.grad(
        outputs=phi,
        inputs=[x, y],
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )

    lap_x = torch.autograd.grad(
        grad_x,
        x,
        grad_outputs=torch.ones_like(grad_x),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    lap_y = torch.autograd.grad(
        grad_y,
        y,
        grad_outputs=torch.ones_like(grad_y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    return lap_x + lap_y


def bc_loss_dirichlet(model, points_b_list=None, eps=BC_INNER_EPS, bc_cache=None):
    if bc_cache is not None and "ext" in bc_cache:
        data = bc_cache["ext"]
        coords_all = data["coords"]
        mat_ids_b = data["mat_ids"]
        x_norm = data["x_norm"]
        phi1_b, phi2_b = model(coords_all, mat_ids_b, x_norm=x_norm)
        return (phi1_b ** 2).mean() + (phi2_b ** 2).mean()

    coords_list = []
    nx_list, ny_list = [], []

    for coords, nx, ny, b_type in points_b_list:
        if b_type == "EXT_NULL" and coords is not None and len(coords) > 0:
            coords_list.append(coords)
            n = coords.shape[0]
            nx_list.append(torch.full((n, 1), float(nx), device=coords.device))
            ny_list.append(torch.full((n, 1), float(ny), device=coords.device))

    if not coords_list:
        return torch.tensor(0.0, device=DEVICE)

    coords_all = torch.cat(coords_list, dim=0)
    nx_all = torch.cat(nx_list, dim=0)
    ny_all = torch.cat(ny_list, dim=0)

    x_b = coords_all[:, 0:1]
    y_b = coords_all[:, 1:2]
    x_inner = x_b - eps * nx_all
    y_inner = y_b - eps * ny_all
    mat_ids_b = get_material_id(x_inner, y_inner).long().view(-1)

    xy_b = torch.cat([x_b, y_b], dim=1)
    phi1_b, phi2_b = model(xy_b, mat_ids_b)

    return (phi1_b ** 2).mean() + (phi2_b ** 2).mean()


def bc_loss_mirror(model, points_b_list=None, bc_cache=None):
    if bc_cache is not None and "mir" in bc_cache:
        data = bc_cache["mir"]
        coords_all = data["coords"]
        nx_all = data["nx"]
        ny_all = data["ny"]
        mat_ids_b = data["mat_ids"]
    else:
        coords_list = []
        nx_list, ny_list = [], []

        for coords, nx, ny, b_type in points_b_list:
            if b_type != "MIRROR" or coords is None or len(coords) == 0:
                continue
            coords_list.append(coords)
            n = coords.shape[0]
            nx_list.append(torch.full((n, 1), float(nx), device=coords.device))
            ny_list.append(torch.full((n, 1), float(ny), device=coords.device))

        if not coords_list:
            return torch.tensor(0.0, device=DEVICE)

        coords_all = torch.cat(coords_list, dim=0)
        nx_all = torch.cat(nx_list, dim=0)
        ny_all = torch.cat(ny_list, dim=0)
        mat_ids_b = get_material_id(coords_all[:, 0:1], coords_all[:, 1:2]).long().view(-1)

    x_b = coords_all[:, 0:1].detach().requires_grad_(True)
    y_b = coords_all[:, 1:2].detach().requires_grad_(True)
    xy_b = torch.cat([x_b, y_b], dim=1)

    phi1_b, phi2_b = model(xy_b, mat_ids_b)

    dphi1_dx, dphi1_dy = torch.autograd.grad(phi1_b.sum(), [x_b, y_b], create_graph=True)
    dphi2_dx, dphi2_dy = torch.autograd.grad(phi2_b.sum(), [x_b, y_b], create_graph=True)

    dphi1_n = dphi1_dx * nx_all + dphi1_dy * ny_all
    dphi2_n = dphi2_dx * nx_all + dphi2_dy * ny_all

    return (dphi1_n ** 2).mean() + (dphi2_n ** 2).mean()


def interface_loss(model, pts_if=None, n_if=LOSS_CONFIG["interface_train_points"], eps=INTERFACE_EPS, if_cache=None):
    if if_cache is not None:
        xy0 = if_cache["xy0"]
        matL_flat = if_cache["matL"]
        matR_flat = if_cache["matR"]
        nx_view = if_cache["nx"].view(-1, 1)
        ny_view = if_cache["ny"].view(-1, 1)
        coeffL = if_cache["coeffL"]
        coeffR = if_cache["coeffR"]

        if xy0.numel() == 0:
            return torch.tensor(0.0, device=DEVICE)

        xI = xy0[:, 0:1].detach().requires_grad_(True)
        yI = xy0[:, 1:2].detach().requires_grad_(True)
    else:
        if pts_if is None:
            xL, yL, matL, xR, yR, matR, nx, ny = sample_interface_points(n_if=n_if, eps=eps, device=DEVICE)
        else:
            xL, yL, matL, xR, yR, matR, nx, ny = pts_if

        if xL.numel() == 0:
            return torch.tensor(0.0, device=DEVICE)

        matL_flat = matL.view(-1)
        matR_flat = matR.view(-1)
        xI = (0.5 * (xL + xR)).clone().detach().requires_grad_(True)
        yI = (0.5 * (yL + yR)).clone().detach().requires_grad_(True)
        nx_view = nx.view(-1, 1)
        ny_view = ny.view(-1, 1)
        coeffL = MATERIALS_TENSOR[matL_flat - 1]
        coeffR = MATERIALS_TENSOR[matR_flat - 1]

    xyI = torch.cat([xI, yI], dim=1)

    phi1_L, phi2_L = model(xyI, matL_flat)
    phi1_R, phi2_R = model(xyI, matR_flat)

    l_flux = ((phi1_L - phi1_R) ** 2 + (phi2_L - phi2_R) ** 2).mean()

    dphi1L_dx, dphi1L_dy = torch.autograd.grad(phi1_L.sum(), [xI, yI], create_graph=True)
    dphi2L_dx, dphi2L_dy = torch.autograd.grad(phi2_L.sum(), [xI, yI], create_graph=True)
    dphi1R_dx, dphi1R_dy = torch.autograd.grad(phi1_R.sum(), [xI, yI], create_graph=True)
    dphi2R_dx, dphi2R_dy = torch.autograd.grad(phi2_R.sum(), [xI, yI], create_graph=True)

    dphi1n_L = dphi1L_dx * nx_view + dphi1L_dy * ny_view
    dphi1n_R = dphi1R_dx * nx_view + dphi1R_dy * ny_view
    dphi2n_L = dphi2L_dx * nx_view + dphi2L_dy * ny_view
    dphi2n_R = dphi2R_dx * nx_view + dphi2R_dy * ny_view

    d1_L, d2_L = coeffL[:, 0:1], coeffL[:, 2:3]
    d1_R, d2_R = coeffR[:, 0:1], coeffR[:, 2:3]

    l_curr = ((d1_L * dphi1n_L - d1_R * dphi1n_R) ** 2 +
              (d2_L * dphi2n_L - d2_R * dphi2n_R) ** 2).mean()

    return l_flux + l_curr


def build_colloc_cache(points_c, source_model=None):
    xy = points_c.detach()
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    mat_ids = get_material_id(x, y).long().view(-1)
    coeffs = MATERIALS_TENSOR[mat_ids - 1]

    phi1_src, phi2_src = None, None
    if source_model is not None:
        x_norm = normalize_xy(xy)
        with torch.no_grad():
            phi1_src, phi2_src = source_model(xy, mat_ids, x_norm=x_norm)

    return {
        "xy": xy,
        "mat_ids": mat_ids,
        "coeffs": coeffs.detach(),
        "phi1_src": phi1_src,
        "phi2_src": phi2_src,
    }


def build_bc_cache(points_b_list, eps=BC_INNER_EPS):
    ext_coords, ext_nx, ext_ny = [], [], []
    mir_coords, mir_nx, mir_ny = [], [], []

    for coords, nx, ny, b_type in points_b_list:
        if coords is None or len(coords) == 0:
            continue
        n = coords.shape[0]
        if b_type == "EXT_NULL":
            ext_coords.append(coords)
            ext_nx.append(torch.full((n, 1), float(nx), device=coords.device))
            ext_ny.append(torch.full((n, 1), float(ny), device=coords.device))
        elif b_type == "MIRROR":
            mir_coords.append(coords)
            mir_nx.append(torch.full((n, 1), float(nx), device=coords.device))
            mir_ny.append(torch.full((n, 1), float(ny), device=coords.device))

    cache = {}

    if ext_coords:
        coords_all = torch.cat(ext_coords, dim=0)
        nx_all = torch.cat(ext_nx, dim=0)
        ny_all = torch.cat(ext_ny, dim=0)

        x_b = coords_all[:, 0:1]
        y_b = coords_all[:, 1:2]
        x_inner = x_b - eps * nx_all
        y_inner = y_b - eps * ny_all
        mat_ids_inner = get_material_id(x_inner, y_inner).long().view(-1)

        cache["ext"] = {
            "coords": coords_all.detach(),
            "mat_ids": mat_ids_inner,
            "x_norm": normalize_xy(coords_all).detach(),
        }

    if mir_coords:
        coords_all = torch.cat(mir_coords, dim=0)
        nx_all = torch.cat(mir_nx, dim=0)
        ny_all = torch.cat(mir_ny, dim=0)
        x_b = coords_all[:, 0:1]
        y_b = coords_all[:, 1:2]
        mat_ids = get_material_id(x_b, y_b).long().view(-1)

        cache["mir"] = {
            "coords": coords_all.detach(),
            "nx": nx_all.detach(),
            "ny": ny_all.detach(),
            "mat_ids": mat_ids,
        }

    return cache


def build_if_cache(pts_if):
    xL, yL, matL, xR, yR, matR, nx, ny = pts_if

    matL_flat = matL.view(-1).long()
    matR_flat = matR.view(-1).long()

    xI0 = 0.5 * (xL + xR)
    yI0 = 0.5 * (yL + yR)
    xy0 = torch.cat([xI0, yI0], dim=1).detach()

    coeffL = MATERIALS_TENSOR[matL_flat - 1].detach()
    coeffR = MATERIALS_TENSOR[matR_flat - 1].detach()

    return {
        "xy0": xy0,
        "matL": matL_flat,
        "matR": matR_flat,
        "nx": nx.detach(),
        "ny": ny.detach(),
        "coeffL": coeffL,
        "coeffR": coeffR,
    }


def visualize_sampling_all(pts_c, pts_b_list, title=None):
    pts_c_cpu = pts_c.detach().cpu()
    x_c = pts_c_cpu[:, 0]
    y_c = pts_c_cpu[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_c, y_c, s=1, alpha=0.3)

    for coords, _, _, b_type in pts_b_list:
        if coords is None or len(coords) == 0:
            continue
        coords_cpu = coords.detach().cpu()
        if b_type == "MIRROR":
            plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=1, alpha=0.4)
        elif b_type == "EXT_NULL":
            plt.scatter(coords_cpu[:, 0], coords_cpu[:, 1], s=1, alpha=0.4)

    plt.xlim(0, X_MAX)
    plt.ylim(0, Y_MAX)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title or "All sampling points (interior + boundary)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path(PLOT_CONFIG["sampling_figure_name"]))
    plt.close()


def visualize_material_pde(history):
    if len(history["epoch"]) < 2:
        return

    epochs = np.array(history["epoch"])
    plt.figure(figsize=(9, 6))

    for i in range(1, 5):
        p_arr = np.array(history[f"PDE_m{i}"])
        plt.semilogy(epochs, p_arr, label=f"Material {i}")

    plt.xlabel("Global Epoch")
    plt.ylabel("Mean PDE residual per material")
    plt.title("Per-material PDE residual during training")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path(PLOT_CONFIG["pde_figure_name"]))
    plt.close()


def visualize_loss_and_keff(history):
    if len(history["epoch"]) < 2:
        return

    epochs = np.array(history["epoch"])
    loss_total = np.array(history["loss"])
    loss_pde = np.array(history["PDE"])
    loss_bc = np.array(history["BC"])
    loss_ic = np.array(history["IFC"])
    keff = np.array(history["keff"])

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax0 = axes[0]
    ax0.semilogy(epochs, loss_total, label="Total")
    ax0.semilogy(epochs, loss_pde, label="PDE", linestyle="--")
    ax0.semilogy(epochs, loss_bc, label="BC", linestyle=":")
    ax0.semilogy(epochs, loss_ic, label="IFC", linestyle=":")
    ax0.set_xlabel("Global Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend()
    ax0.grid(True, which="both", alpha=0.3)

    ax1 = axes[1]
    ax1.plot(epochs, keff, color="orange")
    ax1.set_xlabel("Global Epoch")
    ax1.set_ylabel("k_eff")
    ax1.set_title(f"Current k_eff: {keff[-1]:.6f}")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path(PLOT_CONFIG["loss_figure_name"]))
    plt.close()


@torch.no_grad()
def visualize_during_training(model, device, global_epoch, outer_idx, n_plot=PLOT_CONFIG["grid_size"]):
    model.eval()
    x_np = np.linspace(0, X_MAX, n_plot)
    y_np = np.linspace(0, Y_MAX, n_plot)
    X, Y = np.meshgrid(x_np, y_np)
    xy_tensor = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float32,
        device=device,
    )

    xv = xy_tensor[:, 0:1]
    yv = xy_tensor[:, 1:2]
    mat_ids = get_material_id(xv, yv).long().view(-1)
    phi1, phi2 = model(xy_tensor, mat_ids)

    phi1 = phi1.view(n_plot, n_plot).cpu().numpy()
    phi2 = phi2.view(n_plot, n_plot).cpu().numpy()

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].contourf(X, Y, phi1, levels=60, cmap="jet")
    axes[0].set_title(f"Outer={outer_idx + 1}, Epoch={global_epoch}, Fast Flux φ1")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[0].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0], label="Fast Flux")

    im2 = axes[1].contourf(X, Y, phi2, levels=60, cmap="jet")
    axes[1].set_title(f"Outer={outer_idx + 1}, Epoch={global_epoch}, Thermal Flux φ2")
    axes[1].set_xlabel("x (cm)")
    axes[1].set_ylabel("y (cm)")
    axes[1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1], label="Thermal Flux")

    plt.tight_layout()
    plt.savefig(out_path(PLOT_CONFIG["flux_figure_name"]))
    plt.close()

    model.train()


class SharedDirections(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.register_buffer("B0", torch.randn(in_dim, emb_dim))


class FourierEmbedding(nn.Module):
    def __init__(
        self,
        in_dim=2,
        emb_dim=MODEL_CONFIG["embedding_dim"],
        sigma=1.0,
        learnable_sigma=False,
        sigma_min=MODEL_CONFIG["sigma_min"],
        sigma_max=MODEL_CONFIG["sigma_max"],
        shared_dirs=None,
    ):
        super().__init__()
        self.shared_dirs = shared_dirs
        if self.shared_dirs is None:
            self.register_buffer("B0", torch.randn(in_dim, emb_dim))
        self.out_dim = emb_dim * 2

        self.learnable_sigma = learnable_sigma
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(float(sigma)), dtype=torch.float32))
        else:
            self.register_buffer("sigma_const", torch.tensor(float(sigma), dtype=torch.float32))

    def sigma(self):
        if self.learnable_sigma:
            s = torch.exp(self.log_sigma)
            return s.clamp(self.sigma_min, self.sigma_max)
        return self.sigma_const

    def forward(self, x):
        B0 = self.shared_dirs.B0 if self.shared_dirs is not None else self.B0
        proj = 2 * np.pi * (x @ B0) * self.sigma()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class SubDomainFeatureMoE(nn.Module):
    def __init__(
        self,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        feature_dim=MODEL_CONFIG["feature_dim"],
        output_dim=2,
        shared_dirs=None,
        sigmas=MODEL_CONFIG["sigmas"],
        learnable_sigma=MODEL_CONFIG["learnable_sigma"],
    ):
        super().__init__()

        self.shared_dirs = (
            shared_dirs
            if shared_dirs is not None
            else SharedDirections(in_dim=2, emb_dim=MODEL_CONFIG["embedding_dim"])
        )

        s1, s2, s3 = sigmas
        self.embed_low = FourierEmbedding(
            in_dim=2,
            emb_dim=MODEL_CONFIG["embedding_dim"],
            sigma=s1,
            learnable_sigma=learnable_sigma,
            shared_dirs=self.shared_dirs,
        )
        self.embed_mid = FourierEmbedding(
            in_dim=2,
            emb_dim=MODEL_CONFIG["embedding_dim"],
            sigma=s2,
            learnable_sigma=learnable_sigma,
            shared_dirs=self.shared_dirs,
        )
        self.embed_high = FourierEmbedding(
            in_dim=2,
            emb_dim=MODEL_CONFIG["embedding_dim"],
            sigma=s3,
            learnable_sigma=learnable_sigma,
            shared_dirs=self.shared_dirs,
        )

        input_emb_dim = self.embed_low.out_dim

        def build_expert_net():
            return nn.Sequential(
                nn.Linear(input_emb_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, feature_dim),
            )

        self.expert_low = build_expert_net()
        self.expert_mid = build_expert_net()
        self.expert_high = build_expert_net()

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.register_buffer(
            "fixed_expert_weights",
            torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=torch.float32),
        )

    def forward(self, x, manual_weights=None, return_weights=False):
        embL = self.embed_low(x)
        embM = self.embed_mid(x)
        embH = self.embed_high(x)

        feat_L = self.expert_low(embL)
        feat_M = self.expert_mid(embM)
        feat_H = self.expert_high(embH)

        if manual_weights is not None:
            if manual_weights.dim() == 1:
                weights = manual_weights.view(1, 3).expand(x.size(0), 3)
            else:
                weights = manual_weights
        else:
            weights = self.fixed_expert_weights.to(device=x.device, dtype=x.dtype).view(1, 3).expand(x.size(0), 3)

        wL = weights[:, 0:1]
        wM = weights[:, 1:2]
        wH = weights[:, 2:3]
        fused_feat = wL * feat_L + wM * feat_M + wH * feat_H

        out = self.decoder(fused_feat)
        if return_weights:
            return out, weights
        return out

class SubDomainPlainMLP(nn.Module):
    def __init__(
        self,
        hidden_dim=MODEL_CONFIG["plain_hidden_dim"],
        n_hidden=MODEL_CONFIG["plain_n_hidden"],
        output_dim=2,
    ):
        super().__init__()

        layers = []
        in_dim = 2
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class MultiMatPINN(nn.Module):
    def __init__(
        self,
        n_mat=MODEL_CONFIG["n_mat"],
        subnet_type=MODEL_CONFIG["subnet_type"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        feature_dim=MODEL_CONFIG["feature_dim"],
        sigmas=MODEL_CONFIG["sigmas"],
        learnable_sigma=MODEL_CONFIG["learnable_sigma"],
        plain_hidden_dim=MODEL_CONFIG["plain_hidden_dim"],
        plain_n_hidden=MODEL_CONFIG["plain_n_hidden"],
    ):
        super().__init__()
        self.n_mat = n_mat
        self.subnet_type = subnet_type
        self.shared_dirs = SharedDirections(in_dim=2, emb_dim=MODEL_CONFIG["embedding_dim"])

        if subnet_type == "fourier":
            self.nets = nn.ModuleList([
                SubDomainFeatureMoE(
                    hidden_dim=hidden_dim,
                    feature_dim=feature_dim,
                    output_dim=2,
                    shared_dirs=self.shared_dirs,
                    sigmas=sigmas,
                    learnable_sigma=learnable_sigma,
                )
                for _ in range(n_mat)
            ])

        elif subnet_type == "plain":
            self.nets = nn.ModuleList([
                SubDomainPlainMLP(
                    hidden_dim=plain_hidden_dim,
                    n_hidden=plain_n_hidden,
                    output_dim=2,
                )
                for _ in range(n_mat)
            ])

        else:
            raise ValueError(f"Unknown subnet_type: {subnet_type}")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, mat_ids, x_norm=None):
        if mat_ids.dim() > 1:
            mat_ids = mat_ids.view(-1)

        if x_norm is None:
            x_norm = normalize_xy(x)

        out = torch.zeros(x.size(0), 2, device=x.device)

        for m in range(self.n_mat):
            mask = (mat_ids == (m + 1))
            if mask.any():
                out[mask] = self.nets[m](x_norm[mask])

        phi = F.softplus(out)

        zero_mask = (mat_ids == 0)
        if zero_mask.any():
            phi[zero_mask] = 0.0

        phi1 = phi[:, 0:1]
        phi2 = phi[:, 1:2]
        return phi1, phi2


def pinn_loss(
    model,
    points_c,
    points_b_list,
    current_keff,
    source_model=None,
    w_pde=LOSS_CONFIG["w_pde"],
    w_bc=LOSS_CONFIG["w_bc"],
    w_ifc=LOSS_CONFIG["w_ifc"],
    pts_if=None,
    colloc_cache=None,
    bc_cache=None,
    if_cache=None,
):
    x_c = points_c[:, 0:1].detach().requires_grad_(True)
    y_c = points_c[:, 1:2].detach().requires_grad_(True)
    xy_c = torch.cat([x_c, y_c], dim=1)

    if colloc_cache is not None:
        mat_ids = colloc_cache["mat_ids"]
        coeffs = colloc_cache["coeffs"]
        phi_1_src = colloc_cache["phi1_src"]
        phi_2_src = colloc_cache["phi2_src"]
        phi_1, phi_2 = model(xy_c, mat_ids)
        if (phi_1_src is None) or (phi_2_src is None):
            phi_1_src, phi_2_src = phi_1.detach(), phi_2.detach()
    else:
        mat_ids = get_material_id(x_c, y_c).long().view(-1)
        phi_1, phi_2 = model(xy_c, mat_ids)
        coeffs = MATERIALS_TENSOR[mat_ids - 1]

        if source_model is not None:
            with torch.no_grad():
                phi_1_src, phi_2_src = source_model(xy_c, mat_ids)
        else:
            phi_1_src, phi_2_src = phi_1.detach(), phi_2.detach()

    lap1 = compute_laplacian(phi_1, x_c, y_c)
    lap2 = compute_laplacian(phi_2, x_c, y_c)

    d1 = coeffs[:, 0:1]
    sig_r1 = coeffs[:, 1:2]
    d2 = coeffs[:, 2:3]
    sig_r2 = coeffs[:, 3:4]
    sig12 = coeffs[:, 4:5]
    nu_sf2 = coeffs[:, 5:6]

    res1 = -d1 * lap1 + sig_r1 * phi_1 - (1.0 / current_keff) * nu_sf2 * phi_2_src
    res2 = -d2 * lap2 + sig_r2 * phi_2 - sig12 * phi_1

    r2 = (res1 ** 2 + res2 ** 2).squeeze(-1)

    valid_mask = (mat_ids >= 1) & (mat_ids <= 4)
    r2_valid = r2[valid_mask]
    mat_idx_valid = (mat_ids[valid_mask] - 1).view(-1)

    sums = torch.zeros(4, device=DEVICE, dtype=r2.dtype)
    if r2_valid.numel() > 0:
        sums.index_add_(0, mat_idx_valid, r2_valid)
        counts = torch.bincount(mat_idx_valid, minlength=4).float().to(r2.device)
    else:
        counts = torch.zeros(4, device=DEVICE)

    means = sums / (counts + 1e-12)
    w_mat = torch.tensor(LOSS_CONFIG["material_weights"], device=DEVICE, dtype=r2.dtype)
    l_pde_raw = (w_mat[counts > 0] * means[counts > 0]).sum()

    l_bc_dir = bc_loss_dirichlet(model, points_b_list, bc_cache=bc_cache)
    l_bc_mir = bc_loss_mirror(model, points_b_list, bc_cache=bc_cache)
    l_bc_raw = l_bc_dir + l_bc_mir

    l_if_raw = interface_loss(model, pts_if=pts_if, if_cache=if_cache)

    total_loss = w_pde * l_pde_raw + w_bc * l_bc_raw + w_ifc * l_if_raw
    l_norm = torch.tensor(0.0, device=DEVICE)

    return total_loss, (l_pde_raw, l_bc_raw, l_if_raw, means, l_norm)


def update_keff_source_ratio(model_new, model_old, k_old):
    if model_old is None:
        return k_old

    model_new.eval()
    model_old.eval()

    total_new = torch.tensor(0.0, device=DEVICE)
    total_old = torch.tensor(0.0, device=DEVICE)

    for m_id, xy_m in QUAD_POINTS_BY_MAT.items():
        area_m = QUAD_AREAS_BY_MAT.get(m_id, 0.0)
        if xy_m.numel() == 0 or area_m == 0:
            continue

        x_m = xy_m[:, 0:1]
        y_m = xy_m[:, 1:2]
        xy_in = torch.cat([x_m, y_m], dim=1)

        mat_ids_m = torch.full((xy_in.size(0),), m_id, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            _, phi2_new = model_new(xy_in, mat_ids_m)
            _, phi2_old = model_old(xy_in, mat_ids_m)

        coeffs = MATERIALS_TENSOR[m_id - 1]
        nu_sf2 = coeffs[5]

        prod_new_local = (nu_sf2 * phi2_new).mean()
        prod_old_local = (nu_sf2 * phi2_old).mean()

        total_new += prod_new_local * area_m
        total_old += prod_old_local * area_m

    ratio = (total_new / (total_old + 1e-10)).item()
    return k_old * ratio


def train_main(resume_path=None):
    start_time = time.time()
    global_epoch = 0

    outer_iters = TRAINING_CONFIG["outer_iters"]
    inner_epochs = TRAINING_CONFIG["inner_epochs"]
    n_coll = TRAINING_CONFIG["n_coll"]
    n_bound = TRAINING_CONFIG["n_bound"]
    total_epochs = outer_iters * inner_epochs

    sample_interval = TRAINING_CONFIG["sample_interval"]
    if_sample_interval = TRAINING_CONFIG["if_sample_interval"]
    n_if_train = LOSS_CONFIG["interface_train_points"]

    init_quad_points()
    init_fixed_collocation_points()

    model = MultiMatPINN(
        n_mat=MODEL_CONFIG["n_mat"],
        subnet_type=MODEL_CONFIG["subnet_type"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        feature_dim=MODEL_CONFIG["feature_dim"],
        sigmas=MODEL_CONFIG["sigmas"],
        learnable_sigma=MODEL_CONFIG["learnable_sigma"],
        plain_hidden_dim=MODEL_CONFIG["plain_hidden_dim"],
        plain_n_hidden=MODEL_CONFIG["plain_n_hidden"],
    ).to(DEVICE)
    current_keff_tensor = torch.tensor(TRAINING_CONFIG["init_keff"], device=DEVICE, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG["optimizer_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=TRAINING_CONFIG["scheduler_eta_min"],
    )

    w_pde = LOSS_CONFIG["w_pde"]
    w_bc = LOSS_CONFIG["w_bc"]
    w_ifc = LOSS_CONFIG["w_ifc"]

    block_size = TRAINING_CONFIG["auto_tune_block_size"]
    block_l_pde_sum = 0.0
    block_l_bc_sum = 0.0
    block_count = 0

    history = {"epoch": [], "loss": [], "PDE": [], "BC": [], "IFC": [], "keff": []}
    for i in range(1, 5):
        history[f"PDE_m{i}"] = []

    start_outer = 0
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        current_keff_tensor.fill_(ckpt["keff"])

        start_outer = ckpt["outer"] + 1
        global_epoch = ckpt["global_epoch"]
        w_bc = ckpt["W_BC"]
        history = ckpt["history"]

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=TRAINING_CONFIG["scheduler_eta_min"],
        )
        scheduler.load_state_dict(ckpt["scheduler"])
        print(
            f"[Resume] from outer={ckpt['outer']}, "
            f"global_epoch={global_epoch}, k_eff={ckpt['keff']:.6f}"
        )

    for outer in range(start_outer, outer_iters):
        time_start = time.time()

        source_model = copy.deepcopy(model)
        source_model.eval()
        for param in source_model.parameters():
            param.requires_grad_(False)

        current_k = current_keff_tensor.item()
        print(f"\n=== Outer Iter {outer + 1}, k_eff={current_k:.5f} ===")

        cached_pts_c = None
        cached_pts_b = None
        cached_pts_if = None

        for inner in range(inner_epochs):
            if (inner == 0) or (inner % sample_interval == 0):
                pts_c, pts_b = generate_points(
                    n_coll,
                    n_bound,
                    model=model,
                    global_epoch=global_epoch,
                    current_keff=current_keff_tensor,
                    source_model=source_model,
                )
                cached_pts_c, cached_pts_b = pts_c, pts_b
                colloc_cache = build_colloc_cache(pts_c, source_model=source_model)
                bc_cache = build_bc_cache(pts_b)
            else:
                pts_c, pts_b = cached_pts_c, cached_pts_b

            if (inner == 0) or (inner % if_sample_interval == 0):
                pts_if = sample_interface_points(n_if=n_if_train, eps=INTERFACE_EPS, device=DEVICE)
                cached_pts_if = pts_if
                if_cache = build_if_cache(pts_if)
            else:
                pts_if = cached_pts_if

            optimizer.zero_grad()
            loss, (l_pde_raw, l_bc_raw, l_if_raw, mat_means, l_norm) = pinn_loss(
                model,
                pts_c,
                pts_b,
                current_keff_tensor,
                source_model=source_model,
                colloc_cache=colloc_cache,
                bc_cache=bc_cache,
                if_cache=if_cache,
                w_pde=w_pde,
                w_bc=w_bc,
                w_ifc=w_ifc,
                pts_if=pts_if,
            )
            loss.backward()

            if global_epoch < TRAINING_CONFIG["always_clip_before_epoch"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["grad_clip_norm"])
            elif global_epoch % TRAINING_CONFIG["clip_every_after_epoch"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["grad_clip_norm"])

            optimizer.step()
            scheduler.step()

            l_pde_eff = (w_pde * l_pde_raw).item()
            l_bc_eff = (w_bc * l_bc_raw).item()

            block_l_pde_sum += l_pde_raw.item()
            block_l_bc_sum += l_bc_raw.item()
            block_count += 1

            if inner % TRAINING_CONFIG["history_interval"] == 0:
                history["epoch"].append(global_epoch)
                history["loss"].append(loss.item())
                history["PDE"].append(l_pde_eff)
                history["BC"].append(l_bc_eff)
                history["IFC"].append(w_ifc * l_if_raw.item())
                history["keff"].append(current_keff_tensor.item())
                for m_idx in range(4):
                    history[f"PDE_m{m_idx + 1}"].append(float(mat_means[m_idx].item()))

            if block_count >= block_size:
                mean_l_pde = block_l_pde_sum / block_count
                mean_l_bc = block_l_bc_sum / block_count
                ratio = mean_l_pde / (mean_l_bc + 1e-10)

                new_w_bc = TRAINING_CONFIG["target_bc_over_pde"] * ratio
                new_w_bc = float(
                    np.clip(
                        new_w_bc,
                        TRAINING_CONFIG["w_bc_min"],
                        TRAINING_CONFIG["w_bc_max"],
                    )
                )

                print(
                    f"[Auto-tune] epoch={global_epoch}, "
                    f"mean L_PDE={mean_l_pde:.3e}, mean L_BC={mean_l_bc:.3e}, "
                    f"target(BC/PDE)={TRAINING_CONFIG['target_bc_over_pde']:.2f}, "
                    f"W_BC: {w_bc:.2f} -> {new_w_bc:.2f}"
                )

                w_bc = new_w_bc
                block_l_pde_sum = 0.0
                block_l_bc_sum = 0.0
                block_count = 0

            if inner % TRAINING_CONFIG["print_inner_interval"] == 0:
                print(
                    f"  Inner {inner}: Loss={loss.item():.2e} "
                    f"(PDE_eff={l_pde_eff:.2e}, BC_eff={l_bc_eff:.2e}, "
                    f"W_BC={w_bc:.5f}), Norm={l_norm.item():.2e}"
                )

            if global_epoch % TRAINING_CONFIG["visualize_interval"] == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[LR] global_epoch={global_epoch}, lr={current_lr:.6e}")
                visualize_during_training(model, DEVICE, global_epoch, outer)
                visualize_loss_and_keff(history)
                visualize_sampling_all(
                    pts_c,
                    pts_b,
                    title=f"Sampling (outer={outer + 1}, epoch={global_epoch})",
                )
                visualize_material_pde(history)

            global_epoch += 1

        used_time = time.time() - time_start
        print(f"used_time={used_time:.5f}")

        k_old = current_keff_tensor.item()
        new_k_sr = update_keff_source_ratio(model, source_model, k_old)
        alpha = KEFF_UPDATE_CONFIG["alpha"]
        new_k_sr = (1 - alpha) * k_old + alpha * new_k_sr

        valid_k = max(KEFF_UPDATE_CONFIG["min_keff"], min(KEFF_UPDATE_CONFIG["max_keff"], new_k_sr))
        current_keff_tensor.fill_(valid_k)

        print(f"k_eff = {valid_k:.5f}")
        print(f"source-ratio={new_k_sr:.5f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "keff": float(current_keff_tensor.item()),
            "outer": outer,
            "global_epoch": global_epoch,
            "W_BC": w_bc,
            "history": history,
            "model_config": MODEL_CONFIG,
        }
        torch.save(checkpoint, out_path("pinn_checkpoint.pt"))
        torch.save(model.state_dict(), out_path("RTXpinn.pth"))
        with open(out_path("history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f)

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.1f} s ({elapsed / 60:.2f} min)")

    print("\nTraining Finished. Generating Analysis Plots...")
    visualize_loss_and_keff(history)


if __name__ == "__main__":
    resume_path = os.environ.get("RESUME_PATH", None)
    train_main(resume_path=resume_path)
