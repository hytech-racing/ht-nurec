#!/usr/bin/env python3
"""drivebrain MCAP -> NCore V4 sequence for NuRec.
./mcap_to_ncore.py ~/Downloads/242-recovered.mcap data/242 --start 78 --end 122
"""

import argparse
import ast
import re
from pathlib import Path

import cv2
import numpy as np
import zstandard as zstd
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from ncore.data import OpenCVPinholeCameraModelParameters, RowOffsetStructuredSpinningLidarModelParameters, ShutterType
from ncore.data.v4 import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.common.transformations import HalfClosedInterval, PoseInterpolator
from ouster.sdk import core as ouster
from scipy.spatial.transform import Rotation
from upath import UPath

CAMERAS = ["blackfly_26191748", "blackfly_26291609"]
LIDAR = "lidar_ouster_os1"
INTRINSICS_DIR = Path.home() / "cars/drivebrain/intrinsics"
TAI_MINUS_UTC_US = 37_000_000  # camera stamps come from PTP (TAI), everything else is UTC
# camera mounts relative to the lidar, in ConeClassifier's convention (forward, left, up in m; yaw, pitch,
# roll in deg). Config says yaw +-17, pitch/roll 0; angles here are refit from cone detections in run 242.
CAMERA_MOUNTS = {
    "blackfly_26191748": ((-1.561, 0.129, 0.452827), (16.400, -0.580, 0.006)),
    "blackfly_26291609": ((-1.561, -0.129, 0.452827), (-18.180, -0.560, 0.012)),
}
R_LIDAR_FLU = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)
R_FLU_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)

ap = argparse.ArgumentParser()
ap.add_argument("mcap", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float)
args = ap.parse_args()
out = args.out.expanduser()
seq_id = "hytech_" + args.mcap.stem.split("-")[0]

reader = make_reader(args.mcap.expanduser().open("rb"))
stats = reader.get_summary().statistics
win_start = stats.message_start_time + int(args.start * 1e9)
win_end = stats.message_start_time + int(args.end * 1e9) if args.end else stats.message_end_time
decoders = DecoderFactory()


def messages(topic, start=win_start, end=win_end):
    for schema, channel, msg in reader.iter_messages([topic], starting_at=start, ending_before=end):
        yield decoders.decoder_for(channel.message_encoding, schema)(msg.data)


def stamp_us(msg):
    return (msg.timestamp.seconds * 10**9 + msg.timestamp.nanos) // 1000


def se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def tf_to_T(tf):  # child -> parent
    q = [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]
    R = Rotation.from_quat(q if any(q) else [0, 0, 0, 1]).as_matrix()
    return se3(R, [tf.translation.x, tf.translation.y, tf.translation.z])


def bayer(msg):
    if msg.encoding == "bayer_rggb8":  # already decompressed
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
    planes = np.cumsum(np.frombuffer(zstd.decompress(msg.data, msg.width * msg.height), np.uint8), dtype=np.uint8)
    img = np.empty((msg.height, msg.width), np.uint8)
    n = img.size // 4
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        img[r::2, c::2] = planes[i * n : (i + 1) * n].reshape(msg.height // 2, msg.width // 2)
    return img


odom, map_odom = {}, {}
for tf in messages("tf", win_start - 5 * 10**9, win_end + 5 * 10**9):
    frames = (tf.parent_frame_id, tf.child_frame_id)
    if frames == ("odom", "base_link"):
        odom[stamp_us(tf)] = tf_to_T(tf)
    elif frames == ("map", "odom"):
        map_odom[stamp_us(tf)] = tf_to_T(tf)
    elif frames == ("base_link", "os_sensor"):
        T_lidar_rig = tf_to_T(tf)
pose_t = np.array([t for t in sorted(odom) if win_start // 1000 <= t <= win_end // 1000])
map_t = np.array(sorted(map_odom))
T_map_odom = PoseInterpolator(np.array([map_odom[t] for t in map_t]), map_t).interpolate_to_timestamps(
    np.clip(pose_t, map_t[0], map_t[-1]), np.float64
)
T_rig_world = T_map_odom @ np.array([odom[t] for t in pose_t])
interval = HalfClosedInterval(int(pose_t[0]), int(pose_t[-1]) + 1)
path_m = np.linalg.norm(np.diff(T_rig_world[:, :3, 3], axis=0), axis=1).sum()
print(f"{seq_id}: {(pose_t[-1] - pose_t[0]) / 1e6:.1f} s, {len(pose_t)} poses, {path_m:.1f} m driven")

writer = SequenceComponentGroupsWriter(
    output_dir_path=UPath(out),
    store_base_name=seq_id,
    sequence_id=seq_id,
    sequence_timestamp_interval_us=interval,
    generic_meta_data={"source": args.mcap.name, "window_s": [args.start, args.end]},
    store_type="itar",
)
poses = writer.register_component_writer(PosesComponent.Writer, "default")
intrinsics = writer.register_component_writer(IntrinsicsComponent.Writer, "default")
masks = writer.register_component_writer(MasksComponent.Writer, "default")
cuboids = writer.register_component_writer(CuboidsComponent.Writer, "default")
poses.store_dynamic_pose("rig", "world", T_rig_world, pose_t.astype(np.uint64))
poses.store_static_pose(LIDAR, "rig", T_lidar_rig)
cuboids.store_observations([])  # static scene, no tracked objects

for cam in CAMERAS:
    text = (INTRINSICS_DIR / f"{cam.split('_')[1]}-opencv8.cameramodel").read_text()
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = ast.literal_eval(
        re.search(r"'intrinsics'\s*:\s*(\[.*?\])", text)[1]
    )
    w, h = ast.literal_eval(re.search(r"'imagersize'\s*:\s*(\[.*?\])", text)[1])
    model = OpenCVPinholeCameraModelParameters(
        resolution=np.array([w, h], np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([cx, cy], np.float32),
        focal_length=np.array([fx, fy], np.float32),
        radial_coeffs=np.array([k1, k2, k3, k4, k5, k6], np.float32),
        tangential_coeffs=np.array([p1, p2], np.float32),
        thin_prism_coeffs=np.zeros(4, np.float32),  # no default in ncore 19.8.0
    )
    intrinsics.store_camera_intrinsics(cam, model)
    masks.store_camera_masks(cam, {})

    flu, ypr = CAMERA_MOUNTS[cam]
    R_mount = Rotation.from_euler("ZYX", ypr, degrees=True).as_matrix()
    T_cam_rig = T_lidar_rig @ se3(R_LIDAR_FLU @ R_mount @ R_FLU_CAM, R_LIDAR_FLU @ flu)
    poses.store_static_pose(cam, "rig", T_cam_rig)

    cam_writer = writer.register_component_writer(CameraSensorComponent.Writer, cam)
    last = n = 0
    for msg in messages(f"{cam}/raw"):
        t = stamp_us(msg) - TAI_MINUS_UTC_US
        if t not in interval or t <= last:
            continue
        bgr = cv2.cvtColor(bayer(msg), cv2.COLOR_BayerBG2BGR)  # OpenCV's "BG" = RGGB data
        jpeg = cv2.imencode(".jpg", cv2.rotate(bgr, cv2.ROTATE_180), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        cam_writer.store_frame(jpeg.tobytes(), "jpeg", np.array([t, t], np.uint64), {}, {})
        last, n = t, n + 1
    print(f"{cam}: {n} frames")

info = ouster.SensorInfo(next(messages("ouster/calibration", None, None)).data.decode())
xyz_lut = ouster.XYZLutFloat(info, use_extrinsics=True)  # same as drivebrain's XYZLut(info, true)
rows, cols, hz = info.format.pixels_per_column, info.format.columns_per_frame, float(info.format.fps)
# xyz = range * direction + offset: get each pixel's unit direction from two constant range images
p1, p2 = (xyz_lut(np.full((rows, cols), mm, np.uint32)).astype(np.float64) for mm in (1000, 2000))
dirs = (p2 - p1) / np.linalg.norm(p2 - p1, axis=-1, keepdims=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circmean(a, axis):
    return np.arctan2(np.sin(a).mean(axis), np.cos(a).mean(axis))


# NCore's row-offset-spinning model: elevation per row, azimuth per column + an azimuth offset per row
elev, az = np.arcsin(dirs[..., 2]), np.arctan2(dirs[..., 1], dirs[..., 0])
col_az = circmean(az, 0)
row_off = circmean(wrap(az - col_az), 1)
row_off -= circmean(row_off, 0)
col_az = circmean(wrap(az - row_off[:, None]), 0)
spin = "cw" if np.median(wrap(np.diff(col_az))) < 0 else "ccw"
residual = np.rad2deg(np.abs(wrap(az - col_az - row_off[:, None])).max())
print(f"{LIDAR}: {rows}x{cols} @ {hz:g} Hz, spin {spin}, model fit residual {residual:.4f} deg")
lidar_model = RowOffsetStructuredSpinningLidarModelParameters(
    spinning_frequency_hz=hz,
    spinning_direction=spin,
    n_rows=rows,
    n_columns=cols,
    row_elevations_rad=np.median(elev, axis=1).astype(np.float32),
    column_azimuths_rad=col_az.astype(np.float32),
    row_azimuth_offsets_rad=row_off.astype(np.float32),
)
intrinsics.store_lidar_intrinsics(LIDAR, lidar_model)

lidar_writer = writer.register_component_writer(LidarSensorComponent.Writer, LIDAR)
col_t = np.round(np.arange(cols) * 1e6 / hz / cols).astype(np.int64)  # columns spread evenly over the spin
elements = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij"), -1).reshape(-1, 2).astype(np.uint16)
n = 0
for msg in messages("ouster/range"):
    end = stamp_us(msg)
    start = end - int(col_t[-1])
    if start not in interval or end not in interval:
        continue
    ranges = np.cumsum(np.frombuffer(zstd.decompress(msg.data, rows * cols * 4), "<u4"), dtype=np.uint32)
    ranges = ranges.reshape(rows, cols)
    pts = xyz_lut(ranges).astype(np.float64)  # metres, includes the beam-origin offset
    hit = ranges > 0
    dist = np.linalg.norm(pts, axis=-1)
    direction = np.where(hit[..., None], pts / np.maximum(dist, 1e-9)[..., None], dirs)
    lidar_writer.store_frame(
        direction=direction.reshape(-1, 3).astype(np.float32),
        timestamp_us=np.tile(start + col_t, rows).astype(np.uint64),
        model_element=elements,
        distance_m=np.where(hit, dist, np.nan).reshape(1, -1).astype(np.float32),  # NaN = no return
        intensity=np.where(hit, 0, np.nan).reshape(1, -1).astype(np.float32),  # reflectivity isn't logged
        frame_timestamps_us=np.array([start, end], np.uint64),
        generic_data={},
        generic_meta_data={},
    )
    n += 1
print(f"{LIDAR}: {n} spins")

stores = writer.finalize()
(out / f"{seq_id}.json").write_text(SequenceComponentGroupsReader(stores).get_sequence_meta().to_json(indent=2))
print(f"wrote {out / seq_id}.json and {stores[0].name}")
