#!/usr/bin/env python3
"""./check_ncore.py data/242/hytech_242.json"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ncore.data.v4 import (
    CameraSensorComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.common.transformations import PoseInterpolator

seq = Path(sys.argv[1]).expanduser().absolute()
out = seq.parent / "check"
out.mkdir(exist_ok=True)

reader = SequenceComponentGroupsReader([seq])
poses = reader.open_component_readers(PosesComponent.Reader)["default"]
intrinsics = reader.open_component_readers(IntrinsicsComponent.Reader)["default"]
cameras = reader.open_component_readers(CameraSensorComponent.Reader)
lidar_id, lidar = next(iter(reader.open_component_readers(LidarSensorComponent.Reader).items()))

T_rig_world, pose_t = poses.get_dynamic_pose("rig", "world")
rig_pose = PoseInterpolator(T_rig_world, pose_t.astype(np.float64))
T_lidar_rig = poses.get_static_pose(lidar_id, "rig")
spin_ends = lidar.frames_timestamps_us[:, 1].astype(np.int64)
path_m = np.linalg.norm(np.diff(T_rig_world[:, :3, 3], axis=0), axis=1).sum()
counts = ", ".join(f"{k} {len(v.frames_timestamps_us)}" for k, v in {**cameras, lidar_id: lidar}.items())
print(f"{reader.sequence_id}: {(pose_t[-1] - pose_t[0]) / 1e6:.1f} s, {path_m:.1f} m; frames: {counts}")


def spin_world(t_end):
    """LiDAR returns of one spin in the world frame, each ray placed with the rig pose at its own time."""
    d = lidar.get_frame_ray_bundle_data(t_end, "direction").astype(np.float64)
    t = lidar.get_frame_ray_bundle_data(t_end, "timestamp_us").astype(np.float64)
    r = lidar.get_frame_ray_bundle_return_data(t_end, "distance_m", 0).astype(np.float64)
    hit = np.isfinite(r)
    p_rig = (d[hit] * r[hit, None]) @ T_lidar_rig[:3, :3].T + T_lidar_rig[:3, 3]
    T = rig_pose.interpolate_to_timestamps(t[hit], np.float64)
    return np.einsum("nij,nj->ni", T[:, :3, :3], p_rig) + T[:, :3, 3]


# --- LiDAR on camera frames ------------------------------------------------------------------------
for cam_id, cam in cameras.items():
    m = intrinsics.get_camera_model_parameters(cam_id)
    (fx, fy), (cx, cy) = m.focal_length, m.principal_point
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    k1, k2, k3, k4, k5, k6 = m.radial_coeffs
    p1, p2 = m.tangential_coeffs
    dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], float)
    T_cam_rig = poses.get_static_pose(cam_id, "rig")

    cam_ts = cam.frames_timestamps_us[:, 1].astype(np.int64)
    for n, t in enumerate(cam_ts[np.linspace(0, len(cam_ts) - 1, 6).astype(int)]):
        img = cv2.cvtColor(np.array(cam.get_frame_image(t).convert("RGB")), cv2.COLOR_RGB2BGR)
        spin = spin_ends[np.argmin(np.abs(spin_ends - t))]
        pts = spin_world(spin)
        T_rig = rig_pose.interpolate_to_timestamps([t], np.float64)[0]
        # height above the local ground, to draw cones and poles apart from the asphalt
        near = np.linalg.norm(pts[:, :2] - T_rig[:2, 3], axis=1) < 20
        height = pts[:, 2] - np.percentile(pts[near, 2], 10)
        T_world_cam = np.linalg.inv(T_rig @ T_cam_rig)
        p = pts @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        # in front and roughly in view (the distortion polynomial folds far off-image points back in)
        keep = (p[:, 2] > 0.5) & (np.abs(p[:, 0]) < 0.45 * p[:, 2]) & (np.abs(p[:, 1]) < 0.4 * p[:, 2])
        p, height = p[keep], height[keep]
        uv = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist_coeffs)[0].reshape(-1, 2).astype(int)
        ground = height < 0.12
        for u, v in uv[ground][::3]:
            cv2.circle(img, (u, v), 1, (90, 90, 90), -1)
        depth = np.linalg.norm(p[~ground], axis=1)  # colour by distance, 1-40 m on a log scale
        shade = np.clip(255 * (1 - np.log(depth) / np.log(40)), 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO).reshape(-1, 3)
        for (u, v), c in zip(uv[~ground], colors):
            cv2.circle(img, (u, v), 3, c.tolist(), -1)
        label = f"{cam_id} t={(t - pose_t[0]) / 1e6:.2f}s  lidar spin end {(spin - t) / 1e3:+.0f} ms"
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"overlay_{cam_id}_{n}.jpg"), img)

# --- accumulated LiDAR map: every 5th spin, every 8th point, 0.15-2.5 m above ground ---------------
cloud = np.concatenate([spin_world(t)[::8] for t in spin_ends[::5]])
ground_z = np.percentile(cloud[:, 2], 5)
cloud = cloud[(cloud[:, 2] > ground_z + 0.15) & (cloud[:, 2] < ground_z + 2.5)]
plt.figure(figsize=(12, 12), dpi=150)
plt.scatter(cloud[:, 0], cloud[:, 1], s=0.2, c=cloud[:, 2], linewidths=0)
plt.plot(T_rig_world[:, 0, 3], T_rig_world[:, 1, 3], "r-", lw=1.5, label="rig path")
plt.axis("equal")
plt.legend()
plt.title(f"{reader.sequence_id}: LiDAR 0.15-2.5 m above ground")
plt.savefig(out / "map_bev.png", bbox_inches="tight")
print(f"wrote {out}")
