"""HaMeR hand pose for the ASL stream.

MediaPipe is kept only for what it is good at here: finding hands quickly and
following them between frames. The hand POSE it reports is a single-view guess
that falls apart exactly where ASL needs it most -- a thumb tucked between or
under folded fingers (T, N, M, S) is occluded, so MediaPipe invents it.

HaMeR (Pavlakos et al., CVPR 2024, github.com/geopavlakos/hamer) fits a full
MANO hand model with a learned prior, so hidden joints are constrained by hand
anatomy instead of guessed. This module runs it on the boxes MediaPipe found
and hands back the same (label, confidence, image points, world points) tuples
the rest of the pipeline already consumes, with HaMeR's numbers substituted.

Joint order matches: HaMeR remaps MANO to OpenPose hand order, which is the
same 21-point layout MediaPipe uses (0 wrist, 1-4 thumb, 5-8 index, 9-12
middle, 13-16 ring, 17-20 pinky). So HAND_CONNECTIONS and every downstream
index stay valid.

Requires the MANO model, which is licensed and must be downloaded by hand:
register at https://mano.is.tue.mpg.de and put MANO_RIGHT.pkl at
<hamer_root>/_DATA/data/mano/MANO_RIGHT.pkl
"""
import os
import sys
import contextlib
import io
import cv2
import numpy as np

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "hamer")
RESCALE_FACTOR = 2.0      # box padding HaMeR's demo uses; its crops expect a loose box
BOX_PAD = 0.15            # extra padding on the MediaPipe landmark box, as a fraction of box size


def mano_path(root):
    return os.path.join(root, "_DATA", "data", "mano", "MANO_RIGHT.pkl")


def checkpoint_path(root):
    return os.path.join(root, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")


# The published hamer.ckpt is 2.56 GB. A file well under that is a download
# still in flight, not a usable checkpoint -- torch.load on a truncated file
# fails with an unpickling error that says nothing about the real cause.
CHECKPOINT_MIN_BYTES = 2_000_000_000


def missing_pieces(root=DEFAULT_ROOT):
    """Human-readable list of what is not yet in place, empty if ready."""
    missing = []
    if not os.path.isdir(root):
        return [f"HaMeR is not cloned at {root}"]
    if not os.path.exists(checkpoint_path(root)):
        missing.append(f"model weights ({checkpoint_path(root)}); run fetch_demo_data.sh in {root}")
    else:
        size = os.path.getsize(checkpoint_path(root))
        if size < CHECKPOINT_MIN_BYTES:
            missing.append(f"a complete checkpoint: {checkpoint_path(root)} is only "
                           f"{size / 1e9:.2f} GB of the expected 2.56 GB, so it is still downloading")
    if not os.path.exists(mano_path(root)):
        missing.append(f"the MANO model ({mano_path(root)}); register at https://mano.is.tue.mpg.de, "
                       "download the right-hand model, and put MANO_RIGHT.pkl there")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("pytorch (pip install torch torchvision)")
    return missing


def fix_config_paths(root=DEFAULT_ROOT):
    """Strip a doubled data-dir prefix from the model config.

    load_hamer prepends its cache dir (./_DATA) to every path under MANO in
    model_config.yaml. The config shipped on the Hugging Face Space already
    includes "_DATA/", so paths came out as ./_DATA/_DATA/... and the mean
    parameters were not found. Rewrites the file once, keeping a .orig copy.
    """
    path = os.path.join(root, "_DATA", "hamer_ckpts", "model_config.yaml")
    if not os.path.exists(path):
        return
    text = open(path).read()
    import re
    fixed = re.sub(r"^(\s*[A-Z_]+:\s*)_DATA/", r"\1", text, flags=re.M)
    if fixed != text:
        if not os.path.exists(path + ".orig"):
            open(path + ".orig", "w").write(text)
        open(path, "w").write(fixed)


def ensure_mano_loadable(root=DEFAULT_ROOT, verbose=True):
    """Make MANO_RIGHT.pkl loadable without chumpy.

    The MANO release pickles its arrays as `chumpy` objects. chumpy is
    unmaintained and uses numpy aliases (np.bool, np.object) that were removed
    in numpy 1.24, so installing it alongside a current torch is not an
    option. Instead the pickle is read once through a stand-in module that
    hands back the underlying arrays, then written back as plain numpy. The
    original is kept alongside as .chumpy.bak.

    Safe to call repeatedly: it returns immediately once the file is plain.
    """
    import pickle
    import types
    import shutil
    import warnings

    path = mano_path(root)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            pickle.load(f, encoding="latin1")
        return True          # already plain: nothing to do
    except ModuleNotFoundError as error:
        if error.name != "chumpy":
            raise

    class _Stub:
        """Stands in for any chumpy class; keeps whatever state it is given."""
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {"x": state})

        def __reduce__(self):
            return (_Stub, ())

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            return _Stub

    created = []
    for name in ("chumpy", "chumpy.ch", "chumpy.ch_ops", "chumpy.reordering",
                 "chumpy.linalg", "chumpy.optimization", "chumpy.utils", "chumpy.logic"):
        if name not in sys.modules:
            sys.modules[name] = _StubModule(name)
            created.append(name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the pickle also pulls in a deprecated scipy namespace
            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
    finally:
        for name in created:
            sys.modules.pop(name, None)

    def plain(value):
        """Replace stand-in objects with the numpy array they stood for.

        A chumpy value is either a leaf holding its array in `.x`, or an
        operation over another value. The only operation MANO actually uses is
        a flat index-and-reshape (`shapedirs` picks 10 shape components out of
        a 20-component array), so that one is evaluated and anything else is
        reported rather than silently mangled.
        """
        if isinstance(value, _Stub):
            fields = value.__dict__
            for key in ("x", "r", "_data"):
                if key in fields:
                    return np.asarray(fields[key])
            if "a" in fields and "idxs" in fields and "preferred_shape" in fields:
                base = np.asarray(plain(fields["a"]))
                return base.ravel()[np.asarray(fields["idxs"])].reshape(fields["preferred_shape"])
            raise SystemExit("Unsupported chumpy value in the MANO file "
                             f"(fields: {sorted(fields)}). Please report this.")
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(plain(v) for v in value)
        return value

    converted = plain(data)             # before touching the file on disk
    shutil.copy2(path, path + ".chumpy.bak")
    temporary = path + ".converting"
    with open(temporary, "wb") as f:
        pickle.dump(converted, f, protocol=2)
    os.replace(temporary, path)         # atomic: the model is never left half-written
    if verbose:
        print(f"Converted {os.path.basename(path)} to plain numpy "
              f"(original kept as {os.path.basename(path)}.chumpy.bak).")
    return True


class HamerHands:
    """Wraps HaMeR so it can be called per frame with MediaPipe's boxes."""

    def __init__(self, root=DEFAULT_ROOT, device=None, rescale_factor=RESCALE_FACTOR):
        self.root = root
        self.rescale_factor = rescale_factor
        missing = missing_pieces(root)
        if missing:
            raise SystemExit("HaMeR is not ready:\n  - " + "\n  - ".join(missing))

        # HaMeR resolves its data paths relative to the working directory, so
        # it has to be imported with its own root on sys.path and cwd set.
        import torch
        ensure_mano_loadable(root)
        fix_config_paths(root)
        # HaMeR's renderer module forces PyOpenGL onto EGL unless told
        # otherwise; EGL does not exist on macOS. The renderer is imported
        # as a side effect of loading the model and never used here.
        os.environ.setdefault("PYOPENGL_PLATFORM", "darwin" if sys.platform == "darwin" else "egl")
        sys.path.insert(0, root)
        previous_cwd = os.getcwd()
        os.chdir(root)
        try:
            from hamer.models import load_hamer
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.utils.renderer import cam_crop_to_full
            import hamer.models.hamer as hamer_module

            class NoRenderer:
                """HAMER builds mesh/skeleton renderers for training-time
                visualisation. They need offscreen OpenGL (EGL/OSMesa), which
                macOS lacks, and nothing here ever renders, so they are
                replaced with an object that accepts anything and does nothing."""
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    return None

            hamer_module.MeshRenderer = NoRenderer
            hamer_module.SkeletonRenderer = NoRenderer
            self.model, self.cfg = load_hamer(checkpoint_path(root))
        finally:
            os.chdir(previous_cwd)
        self.ViTDetDataset = ViTDetDataset
        self.cam_crop_to_full = cam_crop_to_full
        self.torch = torch

        if device is None:
            # Apple Silicon: the Metal backend runs the ViT backbone. A few ops
            # still fall back to the CPU, which the env var below allows.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        # HaMeR feeds the backbone a strided slice of the input image. Metal's
        # conv2d refuses non-contiguous tensors, so make the slice contiguous
        # on the way in.
        backbone, original_forward = self.model.backbone, self.model.backbone.forward
        backbone.forward = lambda x, *a, **k: original_forward(x.contiguous(), *a, **k)
        self.image_size = self.cfg.MODEL.IMAGE_SIZE
        self.base_focal = self.cfg.EXTRA.FOCAL_LENGTH

    def _batch(self, frame_bgr, boxes, right):
        """Collate HaMeR's per-hand crops into one batch, without a DataLoader."""
        torch = self.torch
        with contextlib.redirect_stdout(io.StringIO()):     # the dataset prints a debug line per item
            dataset = self.ViTDetDataset(self.cfg, frame_bgr, boxes, right,
                                         rescale_factor=self.rescale_factor)
            items = [dataset[i] for i in range(len(dataset))]
        batch = {}
        for key in items[0]:
            values = [item[key] for item in items]
            if isinstance(values[0], (int, np.integer)):
                tensor = torch.tensor([int(v) for v in values])
            elif isinstance(values[0], (float, np.floating)):
                tensor = torch.tensor([float(v) for v in values])
            else:
                tensor = torch.from_numpy(np.stack(values))
            if tensor.is_floating_point():
                tensor = tensor.float()           # Metal has no float64
            batch[key] = tensor
        return {k: v.to(self.device) for k, v in batch.items()}

    def refine(self, frame_bgr, hands, mirrored=False):
        """Replace MediaPipe's pose with HaMeR's for each hand.

        `hands` and the return value are both lists of
        (label, confidence, image_points_norm[21], world_points[21]), in the
        same (possibly mirrored) space they came in. Hands HaMeR cannot
        process are passed through untouched.

        `mirrored` matters more here than anywhere else in the pipeline:
        HaMeR has only a right-hand model and mirrors the crop for a left
        hand, so it works on appearance, not on the label. In a mirrored
        picture a physically right hand LOOKS left. Rather than invert the
        flag and reason about chirality twice, the frame and landmarks are
        un-mirrored here, HaMeR runs on the physical image, and the results
        are mirrored back on the way out.
        """
        if not hands:
            return hands
        torch = self.torch
        if mirrored:
            frame_bgr = cv2.flip(frame_bgr, 1)
            hands = [(label, conf, [(1.0 - x, y) for x, y in pts], world)
                     for label, conf, pts, world in hands]
        fh, fw = frame_bgr.shape[:2]

        boxes, right = [], []
        for label, _, points, _ in hands:
            xs = [x * fw for x, _ in points]
            ys = [y * fh for _, y in points]
            pad = BOX_PAD * max(max(xs) - min(xs), max(ys) - min(ys))
            boxes.append([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad])
            # HaMeR has only a right-hand model; it mirrors the crop for a left
            # hand, so this flag has to be the true handedness or the fit is
            # of the wrong hand entirely.
            right.append(1 if label == "Right" else 0)
        boxes = np.array(boxes, dtype=np.float32)
        right = np.array(right, dtype=np.float32)

        batch = self._batch(frame_bgr, boxes, right)
        with torch.no_grad():
            out = self.model(batch)

        # Undo the left-hand mirroring, then place each hand in the full frame.
        flip = (2 * batch["right"] - 1).reshape(-1, 1)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = flip[:, 0] * pred_cam[:, 1]
        img_size = batch["img_size"].float()
        focal = self.base_focal / self.image_size * img_size.max()
        cam_t = self.cam_crop_to_full(pred_cam, batch["box_center"].float(),
                                      batch["box_size"].float(), img_size, focal)

        joints = out["pred_keypoints_3d"].clone()
        joints[:, :, 0] = flip * joints[:, :, 0]
        joints_cam = joints + cam_t.unsqueeze(1)

        focal = float(focal)
        joints_cam_np = joints_cam.detach().cpu().numpy()
        joints_np = joints.detach().cpu().numpy()

        refined = []
        for i, (label, confidence, points, world) in enumerate(hands):
            jc = joints_cam_np[i]
            if jc[:, 2].min() <= 1e-3:              # behind the lens: keep MediaPipe's answer
                refined.append((label, confidence, points, world))
                continue
            # HaMeR's own weak-perspective camera: principal point at the image
            # centre, focal as above. Project to full-frame pixels, then
            # normalise, which is what the rest of the pipeline expects.
            u = focal * jc[:, 0] / jc[:, 2] + fw / 2.0
            v = focal * jc[:, 1] / jc[:, 2] + fh / 2.0
            image_points = [(float(x / fw), float(y / fh)) for x, y in zip(u, v)]
            # Root-relative metric joints, in metres, standing in for
            # MediaPipe's world landmarks. The pipeline's solvePnP places them
            # in the real camera frame using the real intrinsics, so HaMeR's
            # long weak-perspective focal never leaks into the world output.
            world_points = [(float(x), float(y), float(z)) for x, y, z in (jc_rel for jc_rel in
                            (joints_np[i] - joints_np[i][0]))]
            refined.append((label, confidence, image_points, world_points))
        if mirrored:
            # Back to the display space the caller handed us: image x mirrors,
            # and the metric joints mirror with it so the pair stays consistent
            # with what the pipeline's own un-mirroring step expects.
            refined = [(label, conf, [(1.0 - x, y) for x, y in pts], [(-x, y, z) for x, y, z in world])
                       for label, conf, pts, world in refined]
        return refined
