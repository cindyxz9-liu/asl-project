"""HaWoR-style world-space hand motion pipeline for the ASL stream.

HaWoR (Zhang et al., CVPR 2025, github.com/ThunderVVV/HaWoR) reconstructs hand
motion from egocentric video in three stages: a camera-frame hand estimator, an
egocentric SLAM module for the camera trajectory, and an infiller that fills
frames where the hand was missed. The official code needs CUDA, DROID-SLAM and
the MANO model, none of which run on this Mac, so this module reproduces the
same stage structure with what is available here and leaves a hook to swap in
the official model offline.

Stages, per frame:
  1. detect_track      MediaPipe hands (done in asl_stream.py, passed in)
  2. hand_cam          camera-frame 3D joints: MediaPipe's metric world
                       landmarks placed in the camera frame by solvePnP against
                       the image landmarks (HaWoR: pred_rot / pred_trans)
  3. camera            CameraPose from telemetry.py (HaWoR: hawor_slam / R_c2w, t_c2w)
  4. lift_to_world     X_w = R_c2w @ X_c + t_c2w
  5. infiller          short gaps in each hand's track are interpolated once
                       the hand reappears; longer gaps are left missing
  6. writer            one JSON record per frame to asl_world.jsonl

Joint order is MediaPipe's 21-point layout (0 wrist, 4 thumb tip, 8 index tip,
12 middle tip, 16 ring tip, 20 pinky tip). Units are metres.
"""
import json
import os
import subprocess
import numpy as np
import cv2
from collections import deque

INFILL_MAX_GAP = 12        # frames; HaWoR's infiller handles much longer gaps, but it has a learned prior
WRITE_LAG = INFILL_MAX_GAP # records are held this long so gaps can be back-filled before writing
POSE_SMOOTHING = 0.6       # EMA on hand translation; 0 = none, 1 = frozen

# Back-of-hand landmarks: wrist plus the four knuckles. Their centroid is the
# hand's lock point -- it does not move when the fingers open or close, and it
# is the same spot on a small hand and a large one.
PALM_IDS = [0, 5, 9, 13, 17]


def hand_center(points):
    """Centroid of the back of the hand, in whatever space `points` are in."""
    return np.asarray(points, dtype=np.float64)[PALM_IDS].mean(axis=0)


class HandCamSolver:
    """Stage 2: put MediaPipe's hand-centred metric landmarks into the camera frame."""

    def __init__(self, K):
        self.K = K
        self.dist = np.zeros(5)
        self.prev = {}          # label -> (rvec, tvec) for solvePnP warm start + smoothing

    def solve(self, label, image_points_norm, world_points, frame_w, frame_h):
        img = np.array([(x * frame_w, y * frame_h) for x, y in image_points_norm], dtype=np.float64)
        obj = np.asarray(world_points, dtype=np.float64)
        if obj.shape != (21, 3) or img.shape != (21, 2):
            return None
        warm = self.prev.get(label)
        if warm is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, warm[0].copy(), warm[1].copy(),
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_EPNP)
            if ok:
                ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, rvec, tvec,
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2] <= 0.05 or tvec[2] > 5.0:
            self.prev.pop(label, None)
            return None
        if warm is not None and POSE_SMOOTHING > 0:
            tvec = POSE_SMOOTHING * warm[1] + (1 - POSE_SMOOTHING) * tvec
        self.prev[label] = (rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        joints_cam = (R @ obj.T).T + tvec.ravel()
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        reproj = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
        return {"R": R, "t": tvec.ravel(), "joints_cam": joints_cam, "reproj_px": reproj}

    def forget(self, label):
        self.prev.pop(label, None)


def lift_to_world(joints_cam, pose):
    """Stage 4."""
    return (pose.R_c2w @ np.asarray(joints_cam).T).T + pose.t_c2w


class Infiller:
    """Stage 5: linear interpolation of world joints across short dropouts.

    Keeps the last WRITE_LAG records in a buffer. When a hand reappears after
    a gap of at most INFILL_MAX_GAP frames, the missing frames in the buffer
    get interpolated joints flagged infilled=True. Records older than the lag
    are released for writing.
    """

    def __init__(self):
        self.buffer = deque()
        self.last_seen = {}      # label -> (frame_index, joints_world)

    def push(self, record):
        present = {h["label"]: h for h in record["hands"]}
        for label, hand in present.items():
            prior = self.last_seen.get(label)
            if prior is not None:
                gap = record["frame"] - prior[0] - 1
                if 0 < gap <= INFILL_MAX_GAP:
                    self._fill(label, prior, record["frame"], np.asarray(hand["joints_world"]))
            self.last_seen[label] = (record["frame"], np.asarray(hand["joints_world"]))
        self.buffer.append(record)
        out = []
        while len(self.buffer) > WRITE_LAG:
            out.append(self.buffer.popleft())
        return out

    def _fill(self, label, prior, end_frame, end_joints):
        start_frame, start_joints = prior
        for rec in self.buffer:
            f = rec["frame"]
            if start_frame < f < end_frame and not any(h["label"] == label for h in rec["hands"]):
                a = (f - start_frame) / (end_frame - start_frame)
                joints = (1 - a) * start_joints + a * end_joints
                rec["hands"].append({
                    "label": label, "confidence": 0.0, "infilled": True,
                    "joints_world": np.round(joints, 4).tolist(),
                    "wrist_world": np.round(joints[0], 4).tolist(),
                })
                rec["hands"].sort(key=lambda h: h["label"])

    def flush(self):
        out = list(self.buffer)
        self.buffer.clear()
        return out


class WorldHandPipeline:
    """Glue for stages 2-6. Call `step` once per frame."""

    def __init__(self, K, telemetry, out_path="asl_world.jsonl"):
        self.solver = HandCamSolver(K)
        self.telemetry = telemetry
        self.infiller = Infiller()
        self.out_path = out_path
        self.out = open(out_path, "w")
        self.pose = None
        self.written = 0

    def reset(self):
        """Call when the view orientation changes: cached poses are for the old one."""
        self.solver.prev.clear()
        self.telemetry.reset()

    def set_intrinsics(self, K):
        """Frame size changed (e.g. rotated to portrait): update K everywhere."""
        self.solver.K = K
        if hasattr(self.telemetry, "set_intrinsics"):
            self.telemetry.set_intrinsics(K)

    def step(self, frame_index, t, frame, hands, frame_w, frame_h, mirrored=False):
        """hands: list of (label, confidence, image_points_norm[21], world_points[21]).

        `mirrored` says the frame (and the landmarks measured on it) are a
        mirror image of the scene. All geometry is done on the physical image:
        the frame is flipped back for odometry, image x becomes 1-x, and the
        world landmarks have x negated (MediaPipe's metric landmarks share the
        chirality of the image they were measured on). Handedness labels are
        already physical and are left alone.
        """
        if mirrored:
            if frame is not None:
                frame = cv2.flip(frame, 1)
            hands = [(label, conf, [(1.0 - x, y) for x, y in pts], [(-x, y, z) for x, y, z in world])
                     for label, conf, pts, world in hands]
        boxes = []
        for _, _, pts, _ in hands:
            xs, ys = zip(*pts)
            boxes.append((min(xs) - 0.05, min(ys) - 0.05, max(xs) + 0.05, max(ys) + 0.05))
        self.pose = self.telemetry.update(t, frame, boxes)

        seen = set()
        out_hands = []
        for label, conf, pts, world in hands:
            seen.add(label)
            cam = self.solver.solve(label, pts, world, frame_w, frame_h)
            if cam is None:
                continue
            jw = lift_to_world(cam["joints_cam"], self.pose)
            out_hands.append({
                "label": label,
                "confidence": round(float(conf), 3),
                "infilled": False,
                "reproj_px": round(cam["reproj_px"], 2),
                "R_hand2cam": np.round(cam["R"], 4).tolist(),
                "t_hand_cam": np.round(cam["t"], 4).tolist(),
                "joints_cam": np.round(cam["joints_cam"], 4).tolist(),
                "joints_world": np.round(jw, 4).tolist(),
                "wrist_world": np.round(jw[0], 4).tolist(),
                "center_image": [round(float(v), 4) for v in hand_center(pts)],
                "center_cam": np.round(hand_center(cam["joints_cam"]), 4).tolist(),
                "center_world": np.round(hand_center(jw), 4).tolist(),
                "depth_m": round(float(cam["t"][2]), 3),
            })
        for label in list(self.solver.prev):
            if label not in seen:
                self.solver.forget(label)

        record = {"frame": frame_index, "t": round(t, 4),
                  "camera": self.pose.to_json(), "hands": sorted(out_hands, key=lambda h: h["label"])}
        for rec in self.infiller.push(record):
            self._write(rec)
        return record

    def _write(self, rec):
        self.out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.written += 1

    def close(self):
        for rec in self.infiller.flush():
            self._write(rec)
        self.out.close()


def run_official_hawor(video_path, hawor_root=None, img_focal=None, vis_mode="world"):
    """Stage 2+3+5 with the real model, offline, on a recorded clip.

    Requires a working clone of github.com/ThunderVVV/HaWoR (CUDA, DROID-SLAM,
    MANO) at HAWOR_ROOT. Runs its demo.py and returns the output folder. This
    is the path to use on a GPU box; the live pipeline above is the on-device
    approximation.
    """
    hawor_root = hawor_root or os.environ.get("HAWOR_ROOT")
    if not hawor_root or not os.path.exists(os.path.join(hawor_root, "demo.py")):
        raise SystemExit("Set HAWOR_ROOT to a clone of https://github.com/ThunderVVV/HaWoR "
                         "(with weights downloaded) to run the official model.")
    cmd = ["python", "demo.py", "--video_path", os.path.abspath(video_path),
           "--input_type", "file", "--vis_mode", vis_mode]
    if img_focal:
        cmd += ["--img_focal", str(img_focal)]
    subprocess.run(cmd, cwd=hawor_root, check=True)
    return os.path.join(hawor_root, "outputs" if os.path.isdir(os.path.join(hawor_root, "outputs")) else "")
