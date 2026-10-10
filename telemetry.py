"""Camera telemetry for the Insta360 ASL stream.

Produces one CameraPose per video frame: where the camera is and which way it
points, in a fixed world frame. The HaWoR pipeline uses this to lift hands out
of the camera frame into world space. Three sources, picked by `make_telemetry`:

  static   The camera is clamped to the desk and never moves. Identity pose.
           Cheapest and the right answer for the current rig.
  visual   Live ego-motion from the video itself: ORB features on the
           background (hands masked out), essential matrix, recoverPose.
           Rotation is metric; translation direction only, scale is unknown.
           This is the stand-in for HaWoR's adaptive egocentric SLAM stage.
  insv     Gyro + accelerometer embedded in an Insta360 recording (.insv /
           .mp4), read with telemetry-parser and integrated to orientation.
           Only available for recorded files: Webcam Mode streams plain UVC
           video and carries no IMU data.

Conventions follow HaWoR: R_c2w / t_c2w map camera-frame points to world,
X_w = R_c2w @ X_c + t_c2w. Camera frame is OpenCV's (x right, y down, z
forward). World frame = the camera frame at the first frame, with the insv
source additionally rotated so that gravity points along +y (OpenCV: y down).
"""
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation


@dataclass
class CameraPose:
    t: float                      # seconds since start of stream
    R_c2w: np.ndarray             # 3x3
    t_c2w: np.ndarray             # 3
    source: str                   # static | visual | insv
    confidence: float = 1.0       # 0..1, how much to trust this pose
    gyro: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # rad/s, camera frame
    accel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # m/s^2, camera frame
    moving: bool = False

    def to_json(self):
        return {
            "t": round(self.t, 4),
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "moving": self.moving,
            "R_c2w": np.round(self.R_c2w, 5).tolist(),
            "t_c2w": np.round(self.t_c2w, 4).tolist(),
            "yaw_pitch_roll_deg": np.round(
                Rotation.from_matrix(self.R_c2w).as_euler("YXZ", degrees=True), 2).tolist(),
            "gyro_rad_s": [round(v, 4) for v in self.gyro],
            "accel_m_s2": [round(v, 3) for v in self.accel],
        }


def intrinsics(width, height, focal_px=None, hfov_deg=None):
    """Pinhole K for the frame. The Insta360 webcam feed is a de-warped
    fisheye; a pinhole approximation is good enough for pose work near the
    centre, which is where the hands live. Pass either focal_px or hfov_deg."""
    if focal_px is None:
        hfov_deg = hfov_deg or 100.0
        focal_px = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[focal_px, 0, width / 2],
                     [0, focal_px, height / 2],
                     [0, 0, 1]], dtype=np.float64)


class StaticTelemetry:
    source = "static"

    def __init__(self, **_):
        pass

    def reset(self):
        pass

    def update(self, t, frame=None, exclude_boxes=()):
        return CameraPose(t, np.eye(3), np.zeros(3), self.source, 1.0)


class VisualOdometry:
    """Monocular ego-motion from consecutive frames.

    Each frame is matched to the last keyframe. If the background barely moved
    the pose is held (no drift while the camera sits still). Otherwise the
    essential matrix gives the relative rotation and translation direction.
    Translation is accumulated at unit scale per keyframe, so t_c2w is a
    direction history, not metres; rotation is real.
    """
    source = "visual"

    def __init__(self, K, work_width=480, min_matches=40, still_px=1.0, **_):
        self.work_width = work_width
        self.set_intrinsics(K)
        self.orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_matches = min_matches
        self.still_px = still_px
        self.key = None               # (keypoints, descriptors)
        self.R = np.eye(3)
        self.tr = np.zeros(3)
        self.last_gyro = np.zeros(3)
        self.last_t = None
        self.confidence = 1.0
        self.failures = 0
        self.max_failures = 5     # consecutive frames without a usable match before the keyframe is re-seeded

    def reset(self):
        """Drop the keyframe (e.g. after the view is mirrored); pose is kept."""
        self.key = None
        self.failures = 0

    def set_intrinsics(self, K):
        self.K = K.copy()
        self.scale = self.work_width / (2 * K[0, 2])
        self.Ks = K.copy()
        self.Ks[:2] *= self.scale
        self.reset()

    def _features(self, frame, exclude_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.scale, fy=self.scale)
        mask = np.full(small.shape, 255, np.uint8)
        h, w = small.shape
        for x0, y0, x1, y1 in exclude_boxes:          # normalised boxes: hands move
            cv2.rectangle(mask, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), 0, -1)
        return self.orb.detectAndCompute(small, mask)

    def _relative_motion(self, p1, p2):
        """(R_rel, t_rel, inlier_ratio) from matched points, or None.

        A camera that only turns (the common egocentric case) makes the
        essential matrix degenerate, so both models are fitted and the one
        that explains more matches wins: a homography is read as pure
        rotation (R = K^-1 H K, zero translation), the essential matrix as
        rotation plus unit-length translation direction.
        """
        n = len(p1)
        Hm, h_inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
        h_ratio = float(h_inl.sum()) / n if h_inl is not None else 0.0
        E, e_inl = cv2.findEssentialMat(p1, p2, self.Ks, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        e_ratio = 0.0
        if E is not None and E.shape == (3, 3):
            _, R_e, t_e, e_mask = cv2.recoverPose(E, p1, p2, self.Ks, mask=e_inl)
            e_ratio = float(e_mask.sum()) / n
        if h_ratio >= 0.5 and h_ratio >= e_ratio:
            Rn = np.linalg.inv(self.Ks) @ Hm @ self.Ks
            U, _, Vt = np.linalg.svd(Rn)
            R_h = U @ Vt
            if np.linalg.det(R_h) < 0:
                R_h = -R_h
            return R_h, np.zeros(3), h_ratio
        if e_ratio >= 0.5:
            return R_e, t_e.ravel(), e_ratio
        return None

    def update(self, t, frame, exclude_boxes=()):
        kp, des = self._features(frame, exclude_boxes)
        moving = False
        estimated = False
        gyro = np.zeros(3)
        if self.key is not None and des is not None and len(des) >= 8:
            pairs = self.matcher.knnMatch(self.key[1], des, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) >= self.min_matches:
                p1 = np.float32([self.key[0][m.queryIdx].pt for m in good])
                p2 = np.float32([kp[m.trainIdx].pt for m in good])
                flow = np.median(np.linalg.norm(p2 - p1, axis=1))
                if flow <= self.still_px:
                    estimated = True              # background did not move: hold the pose
                else:
                    rel = self._relative_motion(p1, p2)
                    if rel is not None:
                        R_rel, t_rel, ratio = rel
                        # Convention: x2 = R_rel x1 + t_rel, i.e. T_c2<-c1.
                        # New camera pose in world: T_w_c2 = T_w_c1 @ inv(T_c2_c1)
                        self.tr = self.tr + self.R @ (-R_rel.T @ t_rel)
                        self.R = self.R @ R_rel.T
                        moving = True
                        estimated = True
                        self.confidence = ratio
                        if self.last_t is not None and t > self.last_t:
                            gyro = Rotation.from_matrix(R_rel.T).as_rotvec() / (t - self.last_t)
                        self.key = (kp, des)
                        self.last_t = t
        if estimated:
            self.failures = 0
        elif self.key is not None:
            # No usable match against the keyframe (fast pan, hand over the
            # background, lens covered). Decay confidence, and after a run of
            # these drop the keyframe so the next good frame re-seeds tracking.
            self.confidence = max(0.1, self.confidence * 0.9)
            self.failures += 1
            if self.failures >= self.max_failures:
                self.key = None
        if self.key is None and des is not None and len(des) >= 8:
            self.key = (kp, des)
            self.last_t = t
            self.failures = 0
        self.last_gyro = gyro
        return CameraPose(t, self.R.copy(), self.tr.copy(), self.source,
                          self.confidence, gyro.tolist(), [0.0, 0.0, 0.0], moving)


class InsvTelemetry:
    """Orientation from the IMU track of an Insta360 recording.

    telemetry-parser returns gyro (deg/s) and accelerometer (g) samples with
    millisecond timestamps, already normalised to the camera's axes. Gyro is
    integrated to a rotation per sample; the mean accelerometer over the first
    half second fixes "down", so the world frame is gravity-aligned. Position
    is not integrated (double-integrating consumer accelerometers drifts within
    seconds), so t_c2w stays zero and the camera is treated as rotating in
    place -- the same simplification HaWoR's infiller makes between SLAM keyframes.
    """
    source = "insv"
    G = 9.80665
    # telemetry-parser normalises IMU axes to Gyroflow's camera convention. Whether
    # that matches OpenCV's (x right, y down, z forward) has not been verified on
    # an Ace Pro 2 clip. If a recorded pan comes out with the opposite yaw sign
    # to `--telemetry visual` on the same clip, set this to np.diag([1, -1, -1]).
    AXES = np.eye(3)

    def __init__(self, path, **_):
        try:
            import telemetry_parser
        except ImportError:
            raise SystemExit("telemetry-parser is not installed:  python3 -m pip install --user telemetry-parser")
        try:
            tp = telemetry_parser.Parser(path)
        except Exception as e:            # telemetry-parser raises OSError on unknown containers
            raise SystemExit(f"telemetry-parser could not read {path}: {e}")
        self.camera = f"{tp.camera} {tp.model}".strip()
        try:
            samples = tp.normalized_imu() or []
        except Exception as e:
            raise SystemExit(f"telemetry-parser could not decode IMU data in {path}: {e}")
        samples = [s for s in samples if s.get("gyro") is not None and s.get("accl") is not None]
        if not samples:
            raise SystemExit(f"No IMU telemetry found in {path}. Webcam-mode captures have none; "
                             "record on the camera itself to get gyro data.")
        self.ts = np.array([s["timestamp_ms"] for s in samples]) / 1000.0
        self.gyro = np.radians(np.array([s["gyro"] for s in samples], dtype=np.float64)) @ self.AXES.T
        self.accl = (np.array([s["accl"] for s in samples], dtype=np.float64) * self.G) @ self.AXES.T
        self._integrate()

    def _integrate(self):
        n = len(self.ts)
        rots = [Rotation.identity()]
        for i in range(1, n):
            dt = self.ts[i] - self.ts[i - 1]
            w = 0.5 * (self.gyro[i] + self.gyro[i - 1])
            rots.append(rots[-1] * Rotation.from_rotvec(w * dt))
        # Gravity alignment: rotate the whole trajectory so the initial "down"
        # (mean accel while presumably still) maps to world -y.
        head = self.accl[self.ts <= self.ts[0] + 0.5]
        up = head.mean(axis=0)                  # a still accelerometer reads the reaction force: it points UP
        if np.linalg.norm(up) > 1.0:
            gravity = -up / np.linalg.norm(up)
            down = np.array([0.0, 1.0, 0.0])    # OpenCV convention: +y is down, so gravity -> world +y
            align, _ = Rotation.align_vectors([down], [gravity])
            rots = [align * r for r in rots]
        self.R = np.stack([r.as_matrix() for r in rots])

    def reset(self):
        pass

    def update(self, t, frame=None, exclude_boxes=()):
        i = int(np.searchsorted(self.ts, t, side="right") - 1)
        i = max(0, min(i, len(self.ts) - 1))
        gyro = self.gyro[i]
        moving = bool(np.linalg.norm(gyro) > math.radians(2.0))
        return CameraPose(t, self.R[i].copy(), np.zeros(3), self.source, 1.0,
                          gyro.tolist(), self.accl[i].tolist(), moving)


def make_telemetry(spec, K):
    """spec: 'static', 'visual', or 'insv:<path to recording>'."""
    if spec == "static":
        return StaticTelemetry()
    if spec == "visual":
        return VisualOdometry(K)
    if spec.startswith("insv:"):
        return InsvTelemetry(spec[len("insv:"):])
    raise SystemExit(f"Unknown telemetry source '{spec}'. Use static, visual, or insv:<file>.")
