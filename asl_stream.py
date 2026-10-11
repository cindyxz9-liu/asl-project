import os
import sys
import math
import cv2
import json
import time
import numpy as np
from collections import deque
import argparse
import threading
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from telemetry import intrinsics, make_telemetry
from hawor_pipeline import WorldHandPipeline, hand_center, PALM_IDS
from capture import Capture

# Live feed: the preview window is the main output. No video is recorded, but
# per-frame world-space hand records go to asl_world.jsonl and a 1 Hz status
# heartbeat to asl_output.json.
CAMERA_NAME = "insta360"     # the only camera this script will run on, matched by name (case-insensitive)
CAMERA_INDEX = 1             # device index the Insta360 gets on this MacBook (built-in camera is 0)
# Preview cost scales with the pixels drawn, so cap the LONG edge. Capping only
# width meant a portrait (rotated) frame became 960x1707 -- 1.6MP and ~21ms per
# frame to blit, three times the area of the same cap in landscape.
PREVIEW_MAX_LONG_EDGE = 720

# Handedness depends on two things and nothing else: rotation never swaps
# left and right, only these do.
#
#   EGOCENTRIC   The camera looks out from the signer's side (on the desk in
#                front of them pointing up, or worn), so hands come in from the
#                signer's own left and right. The signer's right hand is on
#                the RIGHT of an un-mirrored picture -- the same as a selfie
#                mirror, which is the case MediaPipe's handedness labels assume.
#                A camera FACING the signer (third person) is the opposite.
#   mirror_view  Whether the picture on screen is a mirror image of reality.
#                The Ace Pro 2 mirrors its Webcam Mode output, so with no flip
#                of our own the view starts out mirrored.
#
# MediaPipe labels are taken as-is when (EGOCENTRIC xor mirror_view) is true
# and swapped otherwise; the same test decides which screen side the Left
# hand belongs on when two hands share a label. 'm' flips the picture (and
# everything follows); 'h' says "the camera-mirrors assumption was wrong"
# and swaps labels while leaving the picture alone.
EGOCENTRIC = True
# Checked against a signer's real hands on 2026-10-10: with the Ace Pro 2 in
# Webcam Mode, rotated to portrait, the labels came out swapped under the
# assumption that the camera mirrors its own output. It does not. Flipping
# this leaves the picture exactly as it is and swaps the handedness, which is
# what the 'h' key does at runtime.
CAMERA_ALREADY_MIRRORS = False
# The Ace Pro 2 is clamped on its side for a portrait (upright) view. Frames
# are rotated right after capture, so tracking, telemetry and the world frame
# all see the upright image. Press 'r' while running to cycle orientations.
ROTATION = "cw"              # none | cw | ccw | 180
ROTATIONS = ["none", "cw", "ccw", "180"]
ROTATE_CODES = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
                "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
DETECT_WIDTH = 640           # hand tracking runs on a downscaled copy to stay real-time
STATUS_INTERVAL = 1.0        # how often the JSON heartbeat is refreshed
MODEL_PATH = "hand_landmarker.task"
WORLD_OUTPUT = "asl_world.jsonl"   # per-frame camera telemetry + world-space hand joints
CAMERA_HFOV_DEG = 100.0      # Ace Pro 2 webcam feed, de-warped; used for the pinhole K

# The 21 MediaPipe hand landmarks, grouped into the bones that connect them.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]
HAND_COLORS = {"Left": (255, 170, 0), "Right": (0, 170, 255)}
# Regions (x0, y0, x1, y1 as fractions of the portrait frame) swept one per
# frame while looking for a hand. The camera sits on the desk looking up at
# the signer, so hands come in from the left and right edges around the
# middle of the frame, palm or back half-turned to the lens. Each side gets
# its own crop so a hand fills the detector's input; the centre and full
# frame come after.
SEARCH_REGIONS = [
    (0.00, 0.30, 0.60, 0.85),
    (0.40, 0.30, 1.00, 0.85),
    (0.10, 0.25, 0.90, 0.90),
    (0.00, 0.45, 0.60, 1.00),
    (0.40, 0.45, 1.00, 1.00),
    (0.00, 0.00, 1.00, 1.00),
]
ROI_PATIENCE = 20         # frames the tracking crop survives without a hand; a fist can vanish for a while
HANDEDNESS_HISTORY = 60   # frames of handedness history kept per tracked hand
TRACK_MATCH_DIST = 0.18   # how close two hand centres must be to count as the same hand
LOST_HAND_TTL = 45        # frames to keep looking where a hand was last seen before giving up on it
LOST_HAND_PAD = 0.15      # how far around a lost hand's last box to look for it
DETECT_CONFIDENCE = 0.2   # palm detector threshold; fists are low-confidence detections, so keep this low
PRESENCE_CONFIDENCE = 0.2 # landmark model's "there is really a hand here" threshold; fists score low here too
TRACKING_CONFIDENCE = 0.15 # video-mode tracker keeps following the hand above this; fists score low, so very low
LOST_HAND_WIDTH = 896     # a lost hand is looked for in a small crop, so it can afford more pixels: fists are small
LOST_DETECT_CONFIDENCE = 0.05  # palm-detector threshold for the lost-hand search only: a closed fist against the
                          # face or hair scores below 0.2 there, and the search crop is small enough that
                          # false hits are rare (measured: 3 single-frame hits in 400 hand-free frames)
TRACK_CROP_MIN = 1.0      # the tracking crop covers the whole frame: the camera is strapped to the signer,
                          # so the hands are large and near and must never fall out of a box. (A tight crop
                          # is still used to re-find a hand that was just lost, where it measured as useful.)
TRACK_CROP_MARGIN = 0.10  # re-centre the crop when a hand comes this close to its edge, early enough
                          # that a moving hand is followed rather than lost off the edge
TRACK_DETECT_WIDTH = 768  # tracking crop is fed at this width; fists are small, give the model pixels
TRACK_TTL = 10            # frames a track survives without being seen
# With one hand followed, looking for the other costs a full palm-detector
# pass, and in a one-handed sign it ran on 60-70% of frames, pushing the
# worker past the camera's 33 ms. A hand that was just LOST (a fist, usually)
# is still searched for every frame -- that is what brings fists back -- but
# the sweep for a second hand that has never been seen runs every Nth frame.
SECOND_PASS_EVERY = 3
# Seen from below, a palm-up right hand and a back-up left hand have the same
# outline, so the model often calls both hands the same side. Two hands can
# never share a label, so when they do, screen position decides: which side
# each hand is on is learned from frames where the model confidently named
# both hands differently.
# Handedness evidence, weighted by how reliable each cue measured on recorded
# sessions where the answer was known (two hands clearly apart):
#   screen side, hand away from the centre ...... 100%
#   screen side, hand near the centre ........... 93%
#   finger direction (arm_side_label), strong ... 92%; weak ... 63%
#   MediaPipe's own Left/Right .................. worst of all from this angle
SIDE_WEIGHT = 2.0         # full weight once the hand is SIDE_FULL from the centre line
SIDE_DEADZONE = 0.10      # within this of the centre a lone hand's side says nothing (measured 23% right there)
SIDE_FULL = 0.25
ARM_MIN_STRENGTH = 0.8    # weaker finger-direction readings were measured wrong more often than right
SWITCH_RATIO = 3.0        # a lone hand changes label only when the other label has this many times the votes
PAIR_OVERLAP_GAP = 0.06   # two hands closer than this (fraction of width) are overlapping: screen order means nothing
PAIR_VOTE_WEIGHT = 3.0    # a clear two-hand frame is the best evidence there is; tracks remember it

# Main hands = the closest hands to the camera. Every detected hand gets a
# distance estimate (see hand_depth) and only the MAIN_HANDS closest are
# tracked. Background hands are further away by definition, so they can never
# displace the signer's. The detector is allowed more slots than that so a
# background hand cannot take a slot and hide one of the signer's hands
# before this ranking even sees it.
# Phantom hands. On this rig MediaPipe reliably "finds" a closed hand in a
# ceiling lamp (honeycomb shade plus cable), with the same confidence it
# gives real hands, and the sticky slots then hold on to it. A lamp cannot do
# two things a hand does: differ from the static background, and move.
#   Background: a slowly updated grey image of the scene (the camera is desk
#   mounted and never moves). A detection must sit on pixels that differ from
#   it. Hands move in and out; the lamp is the background.
#   Stillness: a tracked "hand" whose palm has not moved for STILL_FRAMES is
#   dropped and its spot is ignored for a while. ASL holds are shorter.
BG_SCALE = 0.25           # background model at quarter resolution
BG_ALPHA = 1.0 / 150      # how fast the background absorbs change (about 5 s)
BG_WARMUP_FRAMES = 45     # faster learning for the first 1.5 s
BG_DIFF = 28              # grey-level difference that counts as "not background"
FG_ADMIT = 0.25           # a NEW hand's box must be at least this much foreground
FG_KEEP = 0.08            # a hand already being tracked must keep at least this much
STILL_FRAMES = 90         # 3 s without moving: not a hand
STILL_MOVE = 0.012        # movement below this (fraction of the frame) counts as not moving
PHANTOM_TTL = 300         # frames to keep ignoring a spot where a phantom was found (10 s)
PHANTOM_RADIUS = 0.08

MAX_HANDS_DETECTED = 4
MAIN_HANDS = 2
DUPLICATE_IOU = 0.5       # two detections whose landmark boxes overlap this much are the same hand
SAME_HAND_PALMS = 1.0     # two palm centres closer than this many palm lengths are one hand: two real palms
                          # cannot be that close. Measured on SD clips 014/015: one hand fit twice (the second
                          # fit displaced along the arm, so the boxes barely overlap) 0.47-0.87 palm lengths;
                          # every pair of distinct hands 1.15 or more
DUPLICATE_CONTAIN = 0.6   # ...or when this much of the smaller box lies inside the larger: the same fist fit
                          # twice (once tight, once loose) has low IoU but is almost entirely contained
# Body anchor. With the camera strapped to the chest, the signer's chin sits at
# one spot in the image for the whole session and the hand model keeps reading
# it as a small closed hand. It is learned: small detections (landmark box
# diagonal under SMALL_HAND_F x focal) that come back to the same spot in at
# least ANCHOR_EPISODES separate episodes mark a body anchor, and small
# detections near it are rejected from then on. Measured on SD clip 183711:
# chin chains 0.14-0.19 x focal, the signer's smallest real hand 0.24 (a
# resting fist that also returns to one spot, but twice the size).
SMALL_HAND_F = 0.22
ANCHOR_RADIUS = 0.12      # fraction of the frame (the head moves a little relative to a chest camera; only
                          # small detections are rejected near the anchor, so a real hand at the face is safe)
ANCHOR_EPISODES = 3
ANCHOR_DETECTIONS = 30    # ...or this many small, steady detections at one spot in a single long run: on SD
                          # clip 017 the chin was detected almost continuously, so it never had 3 episodes
ANCHOR_GAP_FRAMES = 25    # frames without a small detection there that separate two episodes
MAX_BODY_ANCHORS = 3      # the chin, plus static spots like bystanders' hands resting on a table
ANCHOR_KEEP_F = 0.35      # near the anchor, only a hand at least this big (box diagonal / focal) is kept: a
                          # real hand at the face is about 0.3 m from a chest camera and reads about 0.6
ANCHOR_STEADY = 0.006     # median frame-to-frame movement of the small detections at a candidate spot must
                          # be under this (fraction of frame): the chin moves with the camera, a resting
                          # fist on an arm does not hold that still
MIN_HAND_SIDE_F = 0.09    # the shorter side of a hand's landmark box must be at least this fraction of the
                          # focal length in pixels: a hand within arm's reach (0.7 m) cannot look smaller.
                          # Measured on SD clip 183711: the signer's chin read as a hand at 54-60 px and
                          # bystanders' hands at 37-48 px, the signer's real hands at 121 px and up, with
                          # a 775 px focal length -> floor 70 px
DUPLICATE_DIST = 0.03     # ...or whose palm centres are practically on top of each other
PARTIAL_MIN_EXTENT = 0.12 # a partial hand is exempt from the distance rules only if it is at least this big
                          # (diagonal, fraction of the frame): the signer's hands at arm's length are large,
                          # a bystander's hand cut by the frame edge is small
PARTIAL_OUT_OF_FRAME = 6  # a hand with more landmarks than this outside the frame is "partial": still
                          # tracked and recorded (the phrase model needs every movement, and the glasses
                          # view will fill in what this camera cannot see), but its distance estimate is
                          # meaningless, so it is admitted by the attached-to-body rule alone
MIN_NEW_HAND_DEPTH_M = 0.21  # a hand appearing from nowhere closer than this is the forearm or upper arm
                             # right in front of a chest camera, not a hand; a real hand gets that close only
                             # by moving there while tracked (measured on SD clips: forearm fits 0.13-0.19 m; real hands first appear at 0.225 m or farther)
MAX_HAND_DEPTH_M = 0.7    # beyond this a hand is never a main hand: an arm is about 0.7 m long and the
                          # camera is strapped to the body those arms belong to
ARM_REACH_PALMS = 6       # a hand on this body: extending the forearm line from the wrist by this many palm
                          # lengths must leave the frame (the arm goes to the body, which is off-frame),
                          # or the wrist is already near a frame edge. A bystander's forearm leads to a
                          # body that is in the frame.
EDGE_NEAR = 0.2
# Arm direction. The signer's forearms point back toward the signer's body; a
# bystander's point to their own. Learned from the main hands as they are
# tracked (an average of the knuckles-to-wrist direction), then a newcomer
# whose forearm points against it is not on this body. Rig independent: on a
# chest camera the arms come from the top of the picture, on the desk rig
# from the bottom, and the average learns which.
ARM_DIR_ALPHA = 0.02
ARM_DIR_READY = 0.35      # the learned average must be at least this consistent (length of the mean unit vector)
ARM_DIR_MIN_DOT = -0.1    # newcomers whose forearm direction scores below this against it are rejected
                          # (the signer's hands measured under 0.6 m in 99% of recorded frames)
MAIN_DEPTH_BAND = 0.3     # a newcomer must be within this of the nearest main hand, present or recently lost;
                          # the signer's two hands sit within 0.04 m of each other in a median frame
DEPTH_STEP_FRACTION = 0.5 # a hand cannot change distance by more than this fraction of its distance between
MIN_DEPTH_STEP = 0.15     # consecutive frames (at least this many metres): the distance estimate is noisy in
                          # proportion to distance (about 30%), and a fixed step rejected real hands at 40 cm.
                          # A detection that "continues" a track with a bigger jump is someone else in the
                          # same spot -- a bystander's hand inheriting the signer's slot was how background
                          # people 1.7 m away got tracked
MAX_MAIN_DEPTH_M = 0.9    # even a hand continuing a main track is dropped beyond this: a bystander who
                          # inherited the slot in small depth steps (0.8 -> 1.04 -> 1.08 m) is cut off here,
                          # while the signer's own hand raised to the face (about 0.8 m) is kept
MAIN_TRACK_TTL = 45       # frames a main hand's track survives unseen (1.5 s). While it is lost its slot is
                          # reserved against hands farther away than it was, so a stranger's hand cannot
                          # take the place of a hand that just dropped out of view for a moment
ARM_CUE_FULL = 0.3        # wrist-to-knuckle x offset, as a fraction of hand size, at which the arm cue is fully trusted
ARM_VOTE_WEIGHT = 0.3     # finger-direction cue, scaled by its strength
MODEL_VOTE_WEIGHT = 0.3   # MediaPipe's palm-or-back guess: a tiebreaker only


def list_cameras():
    """Return [(index, name)] of video capture devices, in the same order OpenCV
    indexes them. Returns None if the AVFoundation binding is unavailable."""
    try:
        import AVFoundation as AVF
    except ImportError:
        return None
    devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    return [(i, str(d.localizedName())) for i, d in enumerate(devices)]


def resolve_camera(requested):
    """Work out which camera index to open.

    Indices shift when devices are plugged, unplugged or wake from sleep, so the
    camera is looked up by name rather than by a hardcoded number. An explicit
    index still wins if one is passed on the command line.
    """
    cameras = list_cameras()

    if requested is not None and requested.isdigit():
        index = int(requested)
        name = dict(cameras).get(index, "unknown") if cameras else "unknown"
        if CAMERA_NAME not in name.lower() and not allow_any_camera:
            print(f"Error: index {index} is '{name}', not the {CAMERA_NAME}. Refusing to use it.")
            print("  Pass --allow-any-camera if you really want this device.")
            sys.exit(1)
        return index, name

    # No argument: try the configured index first, as long as it really is the
    # Insta360. If the indices have shifted, fall through to the name lookup.
    if requested is None and cameras:
        name = dict(cameras).get(CAMERA_INDEX, "")
        if CAMERA_NAME in name.lower():
            return CAMERA_INDEX, name

    wanted = (requested or CAMERA_NAME).lower()

    if cameras is None:
        print("Error: cannot look up cameras by name -- the AVFoundation binding is missing.")
        print("  Install it with:  python3 -m pip install --user 'pyobjc-framework-AVFoundation==10.3.2'")
        print("  Or pass an explicit index, e.g.  python3 asl_stream.py 0")
        sys.exit(1)

    for index, name in cameras:
        if wanted in name.lower():
            if CAMERA_NAME not in name.lower() and not allow_any_camera:
                print(f"Error: '{name}' matched '{wanted}' but is not the {CAMERA_NAME}. Refusing to use it.")
                print("  Pass --allow-any-camera if you really want this device.")
                sys.exit(1)
            return index, name

    print(f"Error: no camera matching '{wanted}' is connected.")
    if cameras:
        print("Cameras currently available:")
        for index, name in cameras:
            print(f"  [{index}] {name}")
    else:
        print("  (no video capture devices found at all)")
    print("If this is the Insta360, connect it and put it in Webcam Mode.")
    sys.exit(1)


if not os.path.exists(MODEL_PATH):
    print(f"Error: missing {MODEL_PATH}.")
    print("Download it with:")
    print("  curl -sSL -o hand_landmarker.task \\")
    print("    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
          "hand_landmarker/float16/1/hand_landmarker.task")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Insta360 ASL stream with camera telemetry and a "
                                             "HaWoR-style world-space hand pipeline.")
parser.add_argument("camera", nargs="?",
                    help=f"camera name fragment or index (default: index {CAMERA_INDEX}, the Insta360)")
parser.add_argument("--video", help="process a recorded file instead of the live camera")
parser.add_argument("--telemetry", default=None,
                    help="camera pose source: static, visual, or insv:<recording>. "
                         "Default: static for live, insv:<file> for --video (falls back to visual)")
parser.add_argument("--hfov", type=float, default=CAMERA_HFOV_DEG, help="horizontal field of view, degrees")
parser.add_argument("--focal", type=float, default=None, help="focal length in pixels (overrides --hfov)")
parser.add_argument("--fisheye-hfov", type=float, default=None, metavar="DEG",
                    help="the source is an undewarped fisheye with this horizontal field of view (Insta360 "
                         "video files; Webcam Mode output is already de-warped). Frames are undistorted to a "
                         "pinhole of --hfov before anything looks at them.")
parser.add_argument("--thorough", action="store_true",
                    help="file processing: on every frame also run the full-frame detector and a high-"
                         "resolution search around each hand's last position, and merge the results. "
                         "About three times the detector cost; for offline runs where every frame matters.")
parser.add_argument("--no-enhance", action="store_true",
                    help="skip the adaptive exposure/contrast correction (see enhance())")
parser.add_argument("--no-preview", action="store_true",
                    help="do not open the preview window (file processing only; no keyboard controls). "
                         "The window's event pump costs a fixed ~13 ms per frame on macOS.")
parser.add_argument("--max-long-edge", type=int, default=None, metavar="PX",
                    help="downscale frames so the long edge is at most this many pixels before processing "
                         "(default: 1920 for --video files, which come at 2.7K/4K; untouched for the camera)")
parser.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                    help=f"hand detector input width in pixels (default: {DETECT_WIDTH}; smaller is faster)")
parser.add_argument("--out", default=WORLD_OUTPUT, help="world-space JSONL output path")
parser.add_argument("--fps", type=float, default=None,
                    help="request this capture rate from the camera (the Ace Pro 2 tops out at 30 in Webcam Mode)")
parser.add_argument("--rotate", default=ROTATION, choices=ROTATIONS,
                    help="rotate frames after capture for a portrait view (default: %(default)s)")
parser.add_argument("--record", metavar="PATH",
                    help="also save the raw camera feed to this .mp4, for replaying with --video")
parser.add_argument("--realtime", action="store_true",
                    help="with --video, play at the file's own frame rate so tracking behaves as it does live")
parser.add_argument("--hamer", action="store_true",
                    help="use HaMeR for hand pose (MediaPipe only finds and tracks the hands); "
                         "needs the HaMeR weights and the licensed MANO model")
parser.add_argument("--hamer-every", type=int, default=1, metavar="N",
                    help="run HaMeR on every Nth frame (1 = every frame). HaMeR is a large "
                         "transformer; raising this keeps the preview smooth on slower machines. "
                         "Frames it did not touch keep MediaPipe's pose and are marked as such.")
parser.add_argument("--view", choices=["egocentric", "facing"], default="egocentric" if EGOCENTRIC else "facing",
                    help="egocentric: camera looks out from the signer's side; facing: camera looks at the signer")
parser.add_argument("--camera-mirrors", choices=["yes", "no"], default=None,
                    help="whether the camera feed is already a mirror image "
                         f"(default: {'yes' if CAMERA_ALREADY_MIRRORS else 'no'}; 'h' key toggles it live)")
parser.add_argument("--allow-any-camera", action="store_true",
                    help="permit a camera other than the Insta360 (e.g. the built-in MacBook camera)")
args = parser.parse_args()
if args.detect_width <= 0:
    parser.error("--detect-width must be a positive integer")
allow_any_camera = args.allow_any_camera

if args.video:
    cam_index, cam_name = args.video, os.path.basename(args.video)
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
else:
    cam_index, cam_name = resolve_camera(args.camera)
    cap = cv2.VideoCapture(cam_index)
    video_fps = None
    if args.fps and cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        got = cap.get(cv2.CAP_PROP_FPS)
        if got and abs(got - args.fps) > 0.5:
            print(f"Camera would not accept {args.fps:g} fps; it reports {got:g} fps.")
if not cap.isOpened():
    print(f"Error: could not open '{cam_name}' at index {cam_index}.")
    print("If this is the Insta360, make sure it is in Webcam Mode and reconnect it.")
    sys.exit(1)
# Mirroring is a property of Webcam Mode; a file recorded on the camera is not mirrored.
camera_already_mirrors = (args.camera_mirrors == "yes") if args.camera_mirrors else (
    CAMERA_ALREADY_MIRRORS and not args.video)
# Start with the picture exactly as the camera gives it; 'm' flips it later.
mirror_view = camera_already_mirrors
egocentric = args.view == "egocentric"


def labels_as_is():
    """True when the picture is selfie-like, i.e. MediaPipe's handedness
    labels apply directly and the signer's Left hand sits on screen-left."""
    return egocentric != mirror_view

# A camera can open but deliver nothing -- an Insta360 that has dropped out of
# Webcam Mode does exactly this, and each failed read costs a one-second
# timeout, so without this check the script just hangs silently.
failures = 0
while failures < 5:
    ok, frame = cap.read()
    if ok:
        break
    failures += 1
else:
    print(f"Error: '{cam_name}' opened but is not delivering frames.")
    print("If this is the Insta360, re-enable Webcam Mode on the camera and reconnect it.")
    cap.release()
    sys.exit(1)

rotation = args.rotate

recorder = None
if args.record:
    rh, rw = frame.shape[:2]
    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps or 30.0, (rw, rh))
    # Frozen at creation: a file must be one chirality end to end, because
    # --video replays the whole file as un-mirrored.
    record_flip = camera_already_mirrors
    if not recorder.isOpened():
        print(f"Error: cannot write {args.record}.")
        sys.exit(1)
    print(f"Recording the raw feed to {args.record}")


# Adaptive exposure. The rig moves between rooms, backlit doorways and open
# sky, and the camera's own exposure follows the scene, not the hands. Every
# frame is corrected before anything looks at it: a gamma that pulls the
# frame's mean luminance toward a target, then local contrast equalisation
# (CLAHE) on the luminance channel so a hand in silhouette against the sky, or
# in a dark corner, keeps its creases and edges. The preview shows the
# corrected frame too, so what you see is what the tracker sees.
EXPOSURE_TARGET = 0.45    # mean luminance (0-1) the gamma aims for
EXPOSURE_GAMMA_LIMITS = (0.5, 2.2)
CLAHE_CLIP = 2.0
CLAHE_TILES = 8
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
_gamma_luts = {}


def enhance(frame):
    """Luminance-adaptive gamma + CLAHE on the luma channel. About 2.5 ms at
    1080x1920 (the channel is split out first: OpenCV is 5x slower on a
    strided channel view, and an INTER_AREA resize of one cost 7 ms alone)."""
    luma, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    mean = float(luma[::8, ::8].mean()) / 255.0
    gamma = math.log(EXPOSURE_TARGET) / math.log(min(max(mean, 1e-3), 0.999))
    gamma = round(min(max(gamma, EXPOSURE_GAMMA_LIMITS[0]), EXPOSURE_GAMMA_LIMITS[1]), 2)
    lut = _gamma_luts.get(gamma)
    if lut is None:
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma), 0, 255).astype(np.uint8)
        _gamma_luts[gamma] = lut
    return cv2.cvtColor(cv2.merge([_clahe.apply(cv2.LUT(luma, lut)), cr, cb]), cv2.COLOR_YCrCb2BGR)


undistort_maps = None      # (map1, map2) for the current frame size, or None when not a fisheye source
process_scale = None       # downscale factor applied to every frame before processing, or None


def build_undistort(w, h):
    """Remap tables turning an equidistant fisheye of --fisheye-hfov into the
    pinhole of --hfov at the same size. Hands live near the centre of the
    frame, which is the part a pinhole keeps; the extreme edges are lost."""
    fisheye_focal = (w / 2) / math.radians(args.fisheye_hfov / 2)      # equidistant: r = f * theta
    K_fish = np.array([[fisheye_focal, 0, w / 2], [0, fisheye_focal, h / 2], [0, 0, 1]], dtype=np.float64)
    K_pin = intrinsics(w, h, focal_px=args.focal, hfov_deg=args.hfov)
    return cv2.fisheye.initUndistortRectifyMap(K_fish, np.zeros(4), np.eye(3), K_pin, (w, h), cv2.CV_16SC2)


def rotate(frame):
    """Every frame passes through here first: rotate, downscale, undistort."""
    code = ROTATE_CODES[rotation]
    frame = frame if code is None else cv2.rotate(frame, code)
    if process_scale is not None:
        frame = cv2.resize(frame, None, fx=process_scale, fy=process_scale, interpolation=cv2.INTER_AREA)
    if undistort_maps is not None:
        frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)
    if not args.no_enhance:
        frame = enhance(frame)
    return frame


def configure_frame_pipeline(raw):
    """Set the downscale and undistortion for the current rotation from a raw frame."""
    global process_scale, undistort_maps
    process_scale = undistort_maps = None
    shape = rotate(raw).shape[:2]
    limit = args.max_long_edge or (1920 if args.video else None)
    if limit and max(shape) > limit:
        process_scale = limit / max(shape)
        shape = rotate(raw).shape[:2]
    if args.fisheye_hfov:
        undistort_maps = build_undistort(shape[1], shape[0])


def frame_geometry(frame):
    """(width, height, preview_size, K) for a frame in the current orientation."""
    h, w = frame.shape[:2]
    scale = min(1.0, PREVIEW_MAX_LONG_EDGE / max(w, h))
    preview = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return w, h, preview, intrinsics(w, h, focal_px=args.focal, hfov_deg=args.hfov)


configure_frame_pipeline(frame)
frame = rotate(frame)
width, height, preview_size, K = frame_geometry(frame)

telemetry = None
if args.telemetry is None and args.video:
    try:
        telemetry = make_telemetry(f"insv:{args.video}", K)
    except (SystemExit, Exception) as e:
        print(f"No usable IMU telemetry in {args.video} ({e}); falling back to visual odometry.")
        telemetry = make_telemetry("visual", K)
if telemetry is None:
    telemetry = make_telemetry(args.telemetry or "static", K)
pipeline = WorldHandPipeline(K, telemetry, args.out)

def make_landmarker(mode, detect_confidence=DETECT_CONFIDENCE):
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            # Metal via MediaPipe's GPU delegate: measured ~45% cheaper per
            # inference than the CPU path on this Mac (19 -> 11 ms tracker,
            # 18 -> 10 ms detector) with the same hands found. macOS GPU input
            # needs an alpha channel, see detect_in.
            base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.GPU),
            running_mode=mode,
            num_hands=MAX_HANDS_DETECTED,
            min_hand_detection_confidence=detect_confidence,
            min_hand_presence_confidence=min(PRESENCE_CONFIDENCE, detect_confidence),
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
    )


# The palm detector is trained on open palms and gives up on a fist, worst of
# all seen from above. The landmark model has no such trouble once it has a
# hand. So hands are FOUND with a per-image detector on search crops, then
# FOLLOWED with a video-mode landmarker, which carries the landmarks from one
# frame to the next and only asks the palm detector again when it truly
# loses the hand. The video tracker needs a steady picture, so its crop is
# held in place and only moves when a hand nears its edge.
landmarker = make_landmarker(vision.RunningMode.IMAGE)
tracker = make_landmarker(vision.RunningMode.VIDEO)
# The lost-hand search looks in a small crop where a hand was a moment ago, so
# it can afford a much lower palm-detector threshold than the frame sweep: a
# fist against the face or hair is found at 0.05 where 0.2 misses it.
lost_finder = make_landmarker(vision.RunningMode.IMAGE, LOST_DETECT_CONFIDENCE)

# HaMeR replaces MediaPipe's hand POSE (see hamer_hands.py); MediaPipe keeps
# finding and following the hands, which is what it is fast and reliable at.
hamer = None
if args.hamer:
    from hamer_hands import HamerHands
    hamer = HamerHands()
    print(f"HaMeR hand pose active on {hamer.device}.")


def detect_in(frame, region, model=None, timestamp_ms=None, detect_width=None, as_is=None, pad=None):
    """Run the landmarker on one sub-region and return (label, score,
    image landmarks in whole-frame normalised coordinates, metric world
    landmarks in metres centred on the hand).

    pad=(left, top, right, bottom), as fractions of the crop, adds black
    margin around it. A hand cut off by the frame edge is invisible to the
    palm detector at the very edge; giving it room past the edge finds it
    (measured on SD clip 183711, frames 35 and 100)."""
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = (int(region[0] * fw), int(region[1] * fh),
                      int(region[2] * fw), int(region[3] * fh))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    pl = pt = 0
    if pad is not None:
        ch, cw = crop.shape[:2]
        pl, pt = int(pad[0] * cw), int(pad[1] * ch)
        crop = cv2.copyMakeBorder(crop, pt, int(pad[3] * ch), pl, int(pad[2] * cw), cv2.BORDER_CONSTANT)
    detect_width = detect_width or args.detect_width
    scale = detect_width / crop.shape[1]
    small = cv2.resize(crop, (detect_width, max(1, int(crop.shape[0] * scale))))
    rgba = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA))   # GPU delegate needs alpha on macOS
    image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
    if timestamp_ms is None:
        result = (model or landmarker).detect(image)
    else:
        result = (model or tracker).detect_for_video(image, timestamp_ms)

    hands = []
    for index, marks in enumerate(result.hand_landmarks):
        label, score = "Right", 0.0
        if index < len(result.handedness) and result.handedness[index]:
            category = result.handedness[index][0]
            score = float(category.score)
            # A mirrored right hand is anatomically identical to a left hand, so
            # the model's label is only meaningful relative to image orientation.
            # It is taken as-is when the picture is selfie-like (egocentric xor
            # mirrored, see the header comment) and swapped otherwise.
            if as_is is None:
                as_is = labels_as_is()
            label = category.category_name if as_is else (
                "Left" if category.category_name == "Right" else "Right")
        cw_full, ch_full = crop.shape[1], crop.shape[0]
        points = [((x0 - pl + p.x * cw_full) / fw, (y0 - pt + p.y * ch_full) / fh) for p in marks]
        world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[index]] \
            if index < len(result.hand_world_landmarks) else []
        hands.append((label, score, points, world))
    return hands


def arm_side_label(points, as_is):
    """Handedness from which side the arm comes in, or (None, 0).

    MediaPipe decides Left/Right from palm-versus-back cues, which fail from
    this angle: a left hand palm-up and a right hand palm-down look the same.
    On an egocentric rig the arm is the giveaway instead. The signer's right
    arm enters from the right of a selfie-like picture (labels_as_is), so the
    wrist sitting to the right of the knuckles means the right hand. The cue
    fades to nothing as the fingers point straight up or down.
    """
    wx, wy = points[0]
    mx, my = points[9]
    dx = wx - mx
    size = max(1e-6, abs(wy - my) + abs(dx))
    strength = min(1.0, abs(dx) / (ARM_CUE_FULL * size))
    if strength < 0.2:
        return None, 0.0
    from_right = dx > 0
    label = ("Right" if from_right else "Left") if as_is else ("Left" if from_right else "Right")
    return label, strength


def match_tracks(centers):
    """One-to-one assignment of detections to existing tracks.

    Each detection used to pick its nearest track independently, so two hands
    close together could both land on the same track, mixing their label
    histories. Pairs are now assigned closest-first, and a track is used once.
    """
    pairs = sorted((math.hypot(t["pos"][0] - c[0], t["pos"][1] - c[1]), i, k)
                   for i, c in enumerate(centers) for k, t in enumerate(tracks))
    assigned, used = {}, set()
    for distance, i, k in pairs:
        if distance >= TRACK_MATCH_DIST:
            break
        if i in assigned or k in used:
            continue
        assigned[i] = tracks[k]
        used.add(k)
    return assigned


def tally(track):
    """Summed votes over the track's history; one entry per frame."""
    totals = {"Left": 0.0, "Right": 0.0}
    for frame_votes in track["votes"]:
        for label, weight in frame_votes.items():
            totals[label] += weight
    return totals


def side_label(center, as_is):
    """Which hand the screen position suggests, and how strongly."""
    offset = center[0] - 0.5
    strength = min(1.0, max(0.0, (abs(offset) - SIDE_DEADZONE) / (SIDE_FULL - SIDE_DEADZONE)))
    on_left = offset < 0
    return ("Left" if on_left == as_is else "Right"), strength


def smooth_handedness(hands, as_is=None):
    """Label each hand Left or Right, steadily.

    Every frame, each hand collects weighted votes from screen side, finger
    direction and MediaPipe's own guess (weights above, from measured cue
    reliability). Votes accumulate on the hand's track over the last
    HANDEDNESS_HISTORY frames. With two hands clearly apart, screen order
    decides outright and that decision is written into both tracks as a
    strong vote, so a hand that is then left on its own keeps the label it
    had -- the memory the old permanent lock provided, but fading, so a wrong
    one cannot stick. Overlapping hands keep the labels they already had.
    """
    if as_is is None:
        as_is = labels_as_is()
    for track in tracks:
        track["idle"] += 1

    centers = [tuple(hand_center(points)) for _, _, points, _ in hands]
    assigned = match_tracks(centers)
    out = []
    for i, (label, score, points, world) in enumerate(hands):
        track = assigned.get(i)
        if track is None:
            track = {"pos": centers[i], "votes": deque(maxlen=HANDEDNESS_HISTORY), "idle": 0, "last": None}
            tracks.append(track)
        if track["idle"] > TRACK_TTL:
            # Back after a long absence: the slot and its distance are still
            # trusted (that is what kept a stranger out), but the old label
            # history is not -- re-decide left/right from current evidence.
            track["votes"].clear()
            track["last"] = None
        hist = track.setdefault("hist", deque(maxlen=STILL_FRAMES))
        hist.append(centers[i])
        if len(hist) == hist.maxlen:
            xs = [p[0] for p in hist]; ys = [p[1] for p in hist]
            track["static"] = (max(xs) - min(xs)) < STILL_MOVE and (max(ys) - min(ys)) < STILL_MOVE
        track["pos"] = centers[i]
        track["idle"] = 0
        track["main"] = True                      # only main hands reach this function
        track["depth"] = hand_depth(points, world)
        votes = {"Left": 0.0, "Right": 0.0}
        votes[label if label in votes else "Right"] += max(score, 1e-3) * MODEL_VOTE_WEIGHT
        side, side_strength = side_label(centers[i], as_is)
        votes[side] += SIDE_WEIGHT * side_strength
        arm, arm_strength = arm_side_label(points, as_is)
        if arm is not None and arm_strength >= ARM_MIN_STRENGTH:
            votes[arm] += ARM_VOTE_WEIGHT * arm_strength
        track["votes"].append(votes)          # one entry per frame, so the history length is in frames
        out.append([None, 0.0, points, world, track])

    if len(out) == 2:
        first, second = out
        gap = hand_center(second[2])[0] - hand_center(first[2])[0]     # > 0: first is on screen-left
        if abs(gap) >= PAIR_OVERLAP_GAP:
            left_first = (gap > 0) == as_is
            labels = ("Left", "Right") if left_first else ("Right", "Left")
        else:
            # Overlapping. Continuity first: hands that already have distinct
            # labels keep them, and a newcomer takes whichever label the other
            # hand does not have. Only with no usable history do the votes
            # decide, and if they are close, screen order -- weak this close,
            # but never arbitrary, unlike the arrival order of the detections.
            other = {"Left": "Right", "Right": "Left"}
            l1, l2 = first[4].get("last"), second[4].get("last")
            if l1 and l2 and l1 != l2:
                labels = (l1, l2)
            elif l1 and not l2:
                labels = (l1, other[l1])
            elif l2 and not l1:
                labels = (other[l2], l2)
            else:
                t1, t2 = tally(first[4]), tally(second[4])
                f1 = t1["Left"] / max(1e-9, sum(t1.values()))
                f2 = t2["Left"] / max(1e-9, sum(t2.values()))
                if abs(f1 - f2) > 0.1:
                    labels = ("Left", "Right") if f1 > f2 else ("Right", "Left")
                else:
                    left_first = (gap >= 0) == as_is
                    labels = ("Left", "Right") if left_first else ("Right", "Left")
        # Added, not reset: memory comes from many clear two-hand frames piling
        # up votes. Wiping a track's history at each one was tried and scored
        # worse on recorded sessions (88% -> 81% for a lone hand).
        for hand, label in zip(out, labels):
            hand[4]["votes"][-1][label] += PAIR_VOTE_WEIGHT
            hand[0] = label
    for hand in out:
        totals = tally(hand[4])
        if hand[0] is None:
            # A lone hand keeps the label it had unless the evidence against it
            # is overwhelming: which hand this was a moment ago is a better
            # guess than any single-frame cue (see the measured rates above).
            current = hand[4].get("last")
            best = max(totals, key=totals.get)
            if current is None or totals[best] > SWITCH_RATIO * totals[current]:
                hand[0] = best
            else:
                hand[0] = current
        hand[1] = totals[hand[0]] / max(1e-9, sum(totals.values()))
        hand[4]["last"] = hand[0]
        if not is_partial(hand[2]):
            learn_arm_direction(hand[2], hand[0])        # with the settled label, not the model's guess

    tracks[:] = [t for t in tracks if t["idle"] <= (MAIN_TRACK_TTL if t.get("main") else TRACK_TTL)]
    return [tuple(hand[:4]) for hand in out]


# Bones used to judge distance: wrist to each knuckle, and across the knuckles.
DEPTH_BONES = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 17), (5, 9), (9, 13), (13, 17)]


def hand_depth(points, world):
    """Distance from the camera to a hand, in metres.

    MediaPipe gives each hand twice: in the image (normalised points) and as
    metric 3D landmarks centred on the hand. Fitting the metric hand to the
    image points (the same solvePnP the pipeline uses for its records) gives
    the distance. A quicker bone-ratio estimate was tried first and let
    bystanders through: for a tiny distant hand the metric landmarks are
    unreliable and a single bone can read as close. Falls back to the bone
    ratio only when the fit fails.
    """
    p = np.asarray(points, dtype=np.float64) * [width, height]
    if len(world) == 21:
        obj = np.asarray(world, dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, p, K, np.zeros(5), flags=cv2.SOLVEPNP_EPNP)
            if ok and 0.02 < tvec[2] < 20.0:
                R, _ = cv2.Rodrigues(rvec)
                palm = (R @ obj[PALM_IDS].T).T + tvec.ravel()
                return float(palm[:, 2].mean())
        except cv2.error:
            pass
        fx, fy = K[0, 0], K[1, 1]
        best = 0.0
        for i, j in DEPTH_BONES:
            real = np.linalg.norm(obj[i] - obj[j])
            if real > 1e-4:
                best = max(best, np.hypot((p[i, 0] - p[j, 0]) * (fy / fx), p[i, 1] - p[j, 1]) / real)
        if best > 0:
            return float(fy / best)
    return float(K[1, 1] / max(np.hypot(*(p[0] - p[9])) / 0.09, 1e-6))   # average palm, 9 cm


arm_dirs = {"Left": np.zeros(2), "Right": np.zeros(2)}   # per arm: running mean of forearm direction


def forearm_direction(points):
    p = np.asarray(points, dtype=np.float64) * [width, height]
    d = p[0] - p[9]
    n = float(np.hypot(*d))
    return d / n if n > 1e-6 else None


def learn_arm_direction(points, label):
    d = forearm_direction(points)
    if d is not None and label in arm_dirs:
        arm_dirs[label] = (1 - ARM_DIR_ALPHA) * arm_dirs[label] + ARM_DIR_ALPHA * d


def points_toward_body(points):
    """False only when every learned arm direction disagrees with this forearm.
    Each arm is learned separately: the signer's left arm comes in from one
    side and the right from the other, so one average would point at neither."""
    ready = [v / np.hypot(*v) for v in arm_dirs.values() if float(np.hypot(*v)) >= ARM_DIR_READY]
    if len(ready) < 2:
        return True
    d = forearm_direction(points)
    return d is None or any(float(d @ v) >= ARM_DIR_MIN_DOT for v in ready)


def attached_to_body(points):
    """Does this hand's forearm lead out of the frame, as the signer's own must?"""
    if not points_toward_body(points):
        return False
    p = np.asarray(points, dtype=np.float64)
    wrist, mcp = p[0], p[9]
    if min(wrist[0], 1 - wrist[0], wrist[1], 1 - wrist[1]) < EDGE_NEAR:
        return True
    direction = wrist - mcp                      # from the knuckles through the wrist, toward the elbow
    palm = np.linalg.norm(direction)
    if palm < 1e-6:
        return True
    end = wrist + direction / palm * (palm * ARM_REACH_PALMS)
    return not (0.0 <= end[0] <= 1.0 and 0.0 <= end[1] <= 1.0)


body_anchors = []          # [(x, y)] learned this session
_small_episodes = []       # [[x, y, first_frame, last_frame, episodes]]


def box_diag_f(points):
    p = np.asarray(points) * [width, height]
    return float(np.hypot(*(p.max(axis=0) - p.min(axis=0)))) / K[0, 0]


def learn_body_anchor(found, index):
    """Feed every small detection's position into the episode tracker."""
    if len(body_anchors) >= MAX_BODY_ANCHORS:
        return
    for hand in found:
        if box_diag_f(hand[2]) >= SMALL_HAND_F:
            continue
        c = hand_center(hand[2])
        for ep in _small_episodes:
            if math.hypot(c[0] - ep[0], c[1] - ep[1]) < ANCHOR_RADIUS:
                if index - ep[3] > ANCHOR_GAP_FRAMES:
                    ep[4] += 1
                elif index - ep[3] == 1:
                    ep[5].append(math.hypot(c[0] - ep[6][0], c[1] - ep[6][1]))
                ep[6] = c
                ep[7] += 1
                ep[0] += 0.1 * (c[0] - ep[0]); ep[1] += 0.1 * (c[1] - ep[1]); ep[3] = index
                steady = len(ep[5]) >= 5 and float(np.median(ep[5])) < ANCHOR_STEADY
                known = any(math.hypot(ep[0] - a[0], ep[1] - a[1]) < ANCHOR_RADIUS for a in body_anchors)
                recurring = ep[4] >= ANCHOR_EPISODES or ep[7] >= ANCHOR_DETECTIONS
                if recurring and steady and not known and len(body_anchors) < MAX_BODY_ANCHORS:
                    body_anchors.append((ep[0], ep[1]))
                    print(f"learned a body anchor at ({ep[0]:.2f}, {ep[1]:.2f}) on frame {index}: "
                          "small detections there are ignored from now on")
                break
        else:
            _small_episodes.append([c[0], c[1], index, index, 1, [], c, 1])


def on_body_anchor(points):
    if not body_anchors or box_diag_f(points) >= ANCHOR_KEEP_F:
        return False
    c = hand_center(points)
    return any(math.hypot(c[0] - a[0], c[1] - a[1]) < ANCHOR_RADIUS for a in body_anchors)


def mostly_out_of_frame(points):
    """Palm centre outside the frame, or more than half the landmarks outside."""
    p = np.asarray(points)
    c = p[PALM_IDS].mean(axis=0)
    outside = int(((p < 0.0) | (p > 1.0)).any(axis=1).sum())
    return not (0.0 <= c[0] <= 1.0 and 0.0 <= c[1] <= 1.0) or outside > 10


def is_partial(points):
    """Partial AND large: the signer's own hand cut off by the frame edge."""
    pts = np.asarray(points)
    if int(((pts < 0.0) | (pts > 1.0)).any(axis=1).sum()) <= PARTIAL_OUT_OF_FRAME:
        return False
    extent = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    return extent >= PARTIAL_MIN_EXTENT


_frame_index = [0]


def signer_hands(found):
    """The main hands, with memory.

    Pure "closest two" has no idea whose hands they are: the moment one of the
    signer's hands drops out, a stranger's hand reaching into the frame fills
    the free slot. So the two slots are sticky:

      * A detection that continues a main hand's track keeps its slot whatever
        its distance. A track that was lost up to MAIN_TRACK_TTL frames ago
        still counts, provided the hand reappears at about the same distance.
      * Free slots go to the closest remaining hands, but only ones within
        MAIN_DEPTH_BAND of the nearest main hand (present or recently lost)
        and within MAX_HAND_DEPTH_M. A hand reaching in from further back
        than the signer's hands is never admitted while any main hand is
        known; with no main hands known at all, closest wins.

    Pure given the track state (updated in smooth_handedness), so it can be
    called more than once per frame.
    """
    if not found:
        return found
    # The detector has four slots and sometimes fills two with the same hand
    # (seen at the frame edge, labelled Left and Right at once). One per spot.
    unique = []
    min_side = MIN_HAND_SIDE_F * K[0, 0]
    learn_body_anchor(found, _frame_index[0])
    for hand in found:
        if on_body_anchor(hand[2]):
            continue
        p = np.asarray(hand[2]) * [width, height]
        if min(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()) < min_side:
            continue                      # too small to be a hand within arm's reach (chin, bystander)
        dup = next((k for k, u in enumerate(unique) if same_hand(hand, u)), None)
        if dup is not None:
            unique[dup] = better_fit(unique[dup], hand)   # one hand fit twice: keep the fuller fit
        else:
            unique.append(hand)
    found = unique
    depths = [hand_depth(h[2], h[3]) for h in found]
    centers = [tuple(hand_center(h[2])) for h in found]
    assigned = match_tracks(centers)

    continuing, others = [], []
    for i, hand in enumerate(found):
        track = assigned.get(i)
        if track is not None and track.get("main"):
            if is_partial(hand[2]) and not points_toward_body(hand[2]):
                others.append((depths[i], i, hand))
                continue
            known = track.get("depth")
            allowed = max(MIN_DEPTH_STEP, DEPTH_STEP_FRACTION * (known or 0)) if track["idle"] == 0 else MAIN_DEPTH_BAND
            if is_partial(hand[2]) or (depths[i] <= MAX_MAIN_DEPTH_M and (known is None or abs(depths[i] - known) <= allowed)):
                continuing.append((depths[i], i, hand))
                continue
        others.append((depths[i], i, hand))
    continuing.sort()
    continuing = continuing[:MAIN_HANDS]

    known_depths = [d for d, _, _ in continuing]
    known_depths += [t["depth"] for t in tracks
                     if t.get("main") and t.get("depth") is not None and t["idle"] > 0]
    reference = min(known_depths) if known_depths else None

    free = MAIN_HANDS - len(continuing)
    admitted = []
    if free > 0:
        for depth, i, hand in sorted(others):
            if not attached_to_body(hand[2]):
                continue
            # A hand mostly outside the frame is only kept as the continuation
            # of a hand already being tracked (the signer's hand moving out of
            # view). As a newcomer there is nothing to tell whose it is, and on
            # the outdoor clip it was a bystander below the frame in 28 frames.
            if is_partial(hand[2]) or mostly_out_of_frame(hand[2]):
                continue
            if depth > MAX_HAND_DEPTH_M or depth < MIN_NEW_HAND_DEPTH_M:
                continue
            if reference is not None and depth > reference + MAIN_DEPTH_BAND:
                continue
            admitted.append((depth, i, hand))
            if len(admitted) == free:
                break
    return [hand for _, _, hand in continuing + admitted]


def tracking_crop(hands, previous):
    """A generous, steady crop for the video tracker.

    Big enough to hold both hands with room to move (never below
    TRACK_CROP_MIN of the frame), and only re-centred when a hand gets within
    TRACK_CROP_MARGIN of the current crop's edge. Keeping it still is what
    lets the tracker follow a closed fist from frame to frame.
    """
    xs = [x for _, _, pts, _ in hands for x, _ in pts]
    ys = [y for _, _, pts, _ in hands for _, y in pts]
    if previous is not None:
        x0, y0, x1, y1 = previous
        inside = (min(xs) > x0 + TRACK_CROP_MARGIN or x0 <= 0.0) and (max(xs) < x1 - TRACK_CROP_MARGIN or x1 >= 1.0) \
             and (min(ys) > y0 + TRACK_CROP_MARGIN or y0 <= 0.0) and (max(ys) < y1 - TRACK_CROP_MARGIN or y1 >= 1.0)
        if inside:
            return previous
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max(TRACK_CROP_MIN / 2, (max(xs) - min(xs)) * 0.9)
    half_h = max(TRACK_CROP_MIN / 2 * width / height, (max(ys) - min(ys)) * 0.9)
    return (max(0.0, cx - half_w), max(0.0, cy - half_h), min(1.0, cx + half_w), min(1.0, cy + half_h))


def box_around_points(points, pad):
    xs = [x for x, _ in points]; ys = [y for _, y in points]
    return (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(1.0, max(xs) + pad), min(1.0, max(ys) + pad))


def same_hand(a, b):
    """Are two detections the same physical hand? Judged by how much their
    landmark boxes overlap: a duplicate from a second detector pass overlaps
    almost entirely, while two hands signing close together overlap little
    even when their centres are near. (A centre-distance test merged adjacent
    hands and let duplicates through.)"""
    pa, pb = np.asarray(a[2]), np.asarray(b[2])
    ax0, ay0 = pa.min(axis=0); ax1, ay1 = pa.max(axis=0)
    bx0, by0 = pb.min(axis=0); bx1, by1 = pb.max(axis=0)
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    contain = inter / smaller if smaller > 0 else 0.0
    if iou > DUPLICATE_IOU or contain > DUPLICATE_CONTAIN:
        return True
    pxa, pxb = pa * [width, height], pb * [width, height]
    palm = max(np.hypot(*(pxa[9] - pxa[0])), np.hypot(*(pxb[9] - pxb[0])), 1e-6)
    gap = np.hypot(*(pxa[PALM_IDS].mean(axis=0) - pxb[PALM_IDS].mean(axis=0)))
    return gap / palm < SAME_HAND_PALMS


def better_fit(a, b):
    """Of two fits of one hand, the one covering more of it (the other is
    usually compressed onto the wrist and knuckles)."""
    return a if box_diag_f(a[2]) >= box_diag_f(b[2]) else b


def merge_hands(found, extra):
    """Add detections from a second pass; where one duplicates a hand already
    found, keep whichever fit covers more of the hand."""
    for hand in extra:
        for k, h in enumerate(found):
            if same_hand(hand, h):
                found[k] = better_fit(h, hand)
                break
        else:
            found.append(hand)
    return found


def draw_label(img, text, org, color, scale=0.6, thickness=2):
    """Text with a dark outline so it stays readable over skin and bright backgrounds."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

print(f"Live feed from '{cam_name}' (index {cam_index}) at {width}x{height} (rotation: {rotation}).")
print(f"Camera telemetry: {telemetry.source}. World-space hand records -> {args.out}")
print("Hand landmark overlay active. Press 'q' in the preview window to quit.")
print("Press Tab to toggle training-data capture (then a letter key saves that sign).")

# Hand inference costs ~15ms. Run it on a worker thread so it overlaps the
# camera's ~33ms pacing instead of adding to it: the capture loop then stays
# well inside one frame interval and no longer drops every second frame.
# Landmarks trail the displayed frame by at most one frame.
detect_lock = threading.Lock()
detect_input = {"index": -1, "frame": None, "as_is": True, "mirrored": False}
detect_output = {"index": -1, "hands": [], "roi": None, "pose_model": "mediapipe"}
detect_stop = threading.Event()
detect_reset = threading.Event()


class Background:
    """Slowly learned grey image of the scene, for telling hands from fixtures."""

    def __init__(self):
        self.model = None
        self.mask = None
        self.frames = 0

    def update(self, frame, keep_out_boxes):
        small = cv2.resize(frame, None, fx=BG_SCALE, fy=BG_SCALE, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.model is None:
            self.model = grey.copy()
        self.mask = np.abs(grey - self.model) > BG_DIFF
        alpha = 1.0 / 30 if self.frames < BG_WARMUP_FRAMES else BG_ALPHA
        learn = np.ones(grey.shape, bool)
        h, w = grey.shape
        for x0, y0, x1, y1 in keep_out_boxes:                 # hands must not be learned as background
            learn[int(y0 * h):int(y1 * h) + 1, int(x0 * w):int(x1 * w) + 1] = False
        self.model[learn] += alpha * (grey[learn] - self.model[learn])
        self.frames += 1

    def foreground_fraction(self, points):
        if self.mask is None:
            return 1.0
        h, w = self.mask.shape
        x0, y0, x1, y1 = box_around_points(points, 0.02)
        region = self.mask[int(y0 * h):int(y1 * h) + 1, int(x0 * w):int(x1 * w) + 1]
        return float(region.mean()) if region.size else 0.0


def detector_worker():
    crop = None              # steady crop the video tracker is following hands in
    lost = {}                # label -> [last box, frames since last seen]
    last_box = {}            # label -> box where that hand was last seen
    misses = 0
    search_slot = 0
    done = -1
    background = Background()
    phantom_spots = []       # [(centre, frame it was found)]: places that produced a phantom hand

    def not_phantom(found, index):
        """Drop detections that are background, are static fixtures, or sit on
        a spot that recently produced one. Tracked hands get the lenient
        foreground bar; newcomers the strict one."""
        if not found:
            return found
        centers = [tuple(hand_center(h[2])) for h in found]
        assigned = match_tracks(centers)
        kept = []
        for i, hand in enumerate(found):
            track = assigned.get(i)
            c = centers[i]
            if track is not None and track.get("static"):
                phantom_spots.append((c, index))
                continue
            if any(math.hypot(c[0] - p[0], c[1] - p[1]) < PHANTOM_RADIUS for p, _ in phantom_spots):
                continue
            needed = FG_KEEP if (track is not None and track.get("main")) else FG_ADMIT
            if background.foreground_fraction(hand[2]) < needed:
                continue
            kept.append(hand)
        return kept

    def process_frame(index, work, as_is, mirrored_now, ts_ms, crop, misses):
        nonlocal search_slot
        _frame_index[0] = index

        # Follow: the video tracker carries landmarks across frames inside a
        # steady crop, which is what keeps a fist from being dropped.
        if crop is not None:
            found = detect_in(work, crop, tracker, ts_ms, TRACK_DETECT_WIDTH, as_is)
            if found:
                misses = 0
            else:
                misses += 1
                if misses > ROI_PATIENCE:
                    crop = None
        else:
            found = []

        # Nothing found at all. Before sweeping the frame, look where a main
        # hand was last seen: a hand closed into a fist is usually still right
        # there, it has just dropped below the tracker's confidence. This used
        # to happen only when ANOTHER hand was still in view, so a lone hand
        # making a fist was left to the slow region sweep -- the "fist drops
        # out for half a second" problem in the recordings.
        if not found and lost:
            label, (box, _) = min(lost.items(), key=lambda kv: kv[1][1])
            region = (max(0.0, box[0] - LOST_HAND_PAD), max(0.0, box[1] - LOST_HAND_PAD),
                      min(1.0, box[2] + LOST_HAND_PAD), min(1.0, box[3] + LOST_HAND_PAD))
            found = detect_in(work, region, lost_finder, detect_width=LOST_HAND_WIDTH, as_is=as_is)

        # Find: no hand being followed, so sweep one search region per frame.
        if not found and crop is None:
            found = detect_in(work, SEARCH_REGIONS[search_slot % len(SEARCH_REGIONS)], as_is=as_is)
            search_slot += 1

        if args.thorough:
            # Offline: spend detector passes freely. A full-frame pass catches a
            # hand the tracker never picked up; a tight high-resolution pass
            # around each hand's last position catches fists and edge hands.
            extra = detect_in(work, (0.0, 0.0, 1.0, 1.0), as_is=as_is)
            # Edge sweeps: hands enter a chest camera's view from the sides and
            # bottom, cut off by the frame edge. Each sweep pads past its edge.
            for region, pad in (((0.0, 0.3, 0.5, 1.0), (0.5, 0.0, 0.0, 0.0)),
                                ((0.5, 0.3, 1.0, 1.0), (0.0, 0.0, 0.5, 0.0)),
                                ((0.1, 0.55, 0.9, 1.0), (0.0, 0.0, 0.0, 0.5))):
                extra = merge_hands(extra, detect_in(work, region, lost_finder, as_is=as_is, pad=pad))
            for box in list(last_box.values()) + [b for b, _ in lost.values()]:
                region = (max(0.0, box[0] - LOST_HAND_PAD), max(0.0, box[1] - LOST_HAND_PAD),
                          min(1.0, box[2] + LOST_HAND_PAD), min(1.0, box[3] + LOST_HAND_PAD))
                extra = merge_hands(extra, detect_in(work, region, lost_finder, detect_width=LOST_HAND_WIDTH, as_is=as_is))
            found = merge_hands(found, extra)

        # Keep only the closest hands before deciding whether one is missing:
        # with spare detector slots, background hands would otherwise make the
        # count look complete and the search for a lost main hand never run.
        found = signer_hands(not_phantom(found, index))

        # One hand followed but the other is missing: spend a second detection
        # looking for it. While a hand is known to be lost, look where it was
        # last seen EVERY frame: a fist dropped by the tracker is still right
        # there, and the big tracking crop never re-acquires it on its own, so
        # alternating with the sweep found it only every other frame. The
        # sweep still runs when no lost hand is known.
        if found and len(found) < 2:
            extra = []
            if lost:
                box, _ = min(lost.values(), key=lambda v: v[1])
                region = (max(0.0, box[0] - LOST_HAND_PAD), max(0.0, box[1] - LOST_HAND_PAD),
                          min(1.0, box[2] + LOST_HAND_PAD), min(1.0, box[3] + LOST_HAND_PAD))
                extra = detect_in(work, region, lost_finder, detect_width=LOST_HAND_WIDTH, as_is=as_is)
            elif index % SECOND_PASS_EVERY == 0:
                extra = detect_in(work, SEARCH_REGIONS[search_slot % len(SEARCH_REGIONS)],
                                  detect_width=LOST_HAND_WIDTH, as_is=as_is)
                search_slot += 1
            if extra:
                found = signer_hands(not_phantom(merge_hands(found, extra), index))

        # From here on `found` holds only the main hands, so background hands
        # never get a label, never reach HaMeR, and never stretch the tracking
        # box away from the signer.
        found = smooth_handedness(found, as_is)
        # Handedness has to be settled before HaMeR runs: it fits a right-hand
        # model and mirrors the crop for a left hand, so a wrong label fits the
        # wrong hand.
        pose_model = "mediapipe"
        if hamer is not None and found and index % max(1, args.hamer_every) == 0:
            try:
                found = hamer.refine(work, found, mirrored=mirrored_now)
                pose_model = "hamer"
            except Exception as error:                 # never kill the stream over one frame
                print(f"HaMeR frame {index} failed: {error}")
        seen_now = set()
        for label, _, pts, _ in found:
            seen_now.add(label)
            lost.pop(label, None)
            last_box[label] = box_around_points(pts, 0.05)
        for label in list(last_box):
            if label not in seen_now and label not in lost:
                lost[label] = [last_box[label], 0]
        for label in list(lost):
            lost[label][1] += 1
            if lost[label][1] > LOST_HAND_TTL:
                del lost[label]
        if found:
            crop = tracking_crop(found, crop)
            misses = 0
        background.update(work, [box_around_points(h[2], 0.08) for h in found])
        return found, crop, misses, pose_model

    while not detect_stop.is_set():
        with detect_lock:
            index, work = detect_input["index"], detect_input["frame"]
            as_is, mirrored_now = detect_input["as_is"], detect_input["mirrored"]
        if work is None or index == done:
            time.sleep(0.001)
            continue
        if detect_reset.is_set():
            detect_reset.clear()
            crop, misses = None, 0
            lost.clear()
            last_box.clear()
            tracks.clear()
            phantom_spots.clear()
        ts_ms = int(index * 1000 / (video_fps or 30))
        phantom_spots[:] = [(p, f) for p, f in phantom_spots if index - f <= PHANTOM_TTL]
        try:
            found, crop, misses, pose_model = process_frame(index, work, as_is, mirrored_now, ts_ms, crop, misses)
        except Exception:
            import traceback
            print(f"detector failed on frame {index}:")
            traceback.print_exc()
            found, pose_model = [], "mediapipe"
        done = index
        with detect_lock:
            detect_output["index"], detect_output["hands"] = index, found
            detect_output["roi"], detect_output["pose_model"] = crop, pose_model
        continue

detector = threading.Thread(target=detector_worker, daemon=True)
detector.start()

frames = 0
start_time = time.time()
last_status = 0.0
last_fps_time = start_time
last_fps_frames = 0
live_fps = 0.0
hands_seen = []
# Offline files are processed frame by frame unless --realtime asks for live pacing.
lockstep = bool(args.video) and not args.realtime
tracks = []
capture = Capture()

try:
    while cap.isOpened():
        ok, raw_frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break

        if recorder is not None:
            # Saved un-mirrored and un-rotated, like a file recorded on the camera,
            # so --video replays it exactly as the live feed was processed.
            recorder.write(cv2.flip(raw_frame, 1) if record_flip else raw_frame)
        frame = rotate(raw_frame)
        frames += 1
        now = time.time()
        if args.realtime and video_fps:
            ahead = start_time + frames / video_fps - now
            if ahead > 0:
                time.sleep(ahead)
        stream_t = frames / video_fps if video_fps else now - start_time

        # Only flip when the camera's own orientation disagrees with the view we
        # want; flipping an already-mirrored feed would undo the mirror.
        if mirror_view != camera_already_mirrors:
            frame = cv2.flip(frame, 1)
        preview = cv2.resize(frame, preview_size)

        # Hand off this frame and take whatever the detector has finished.
        with detect_lock:
            detect_input["index"], detect_input["frame"] = frames, frame
            detect_input["as_is"], detect_input["mirrored"] = labels_as_is(), mirror_view
        if lockstep:
            # File processing: every frame goes through the detector, in order.
            # Live, the loop must keep the camera's pace and takes whatever the
            # worker has finished; offline that made results depend on timing.
            while detect_output["index"] != frames and not detect_stop.is_set():
                time.sleep(0.0005)
        with detect_lock:
            hands = list(detect_output["hands"])
            lock_box, frame_pose_model = detect_output["roi"], detect_output["pose_model"]

        # HaWoR-style stages: camera telemetry, camera-frame 3D hand, lift to world.
        world_record = pipeline.step(frames, stream_t, frame, hands, width, height, mirrored=mirror_view)
        # Which model actually produced this frame's joints, so a downstream
        # classifier can train on HaMeR frames alone if it wants to.
        world_record["pose_model"] = frame_pose_model
        if body_anchors:
            world_record["body_anchors"] = [[round(a[0], 3), round(a[1], 3)] for a in body_anchors]
        # Crops come from `frame`, which is never drawn on; overlays go on `preview`.
        capture.step(frames, frame, hands, mirrored=mirror_view)
        world_hands = {h["label"]: h for h in world_record["hands"]}
        cam_pose = world_record["camera"]

        if lock_box is not None:
            cv2.rectangle(preview, (int(lock_box[0] * preview_size[0]), int(lock_box[1] * preview_size[1])),
                          (int(lock_box[2] * preview_size[0]), int(lock_box[3] * preview_size[1])),
                          (90, 90, 90), 1)

        hands_seen = []
        for label, confidence, points, _ in hands:
            hands_seen.append(label)
            pixels = [(int(x * preview_size[0]), int(y * preview_size[1])) for x, y in points]
            color = HAND_COLORS.get(label, (0, 255, 0))
            for a, b in HAND_CONNECTIONS:
                cv2.line(preview, pixels[a], pixels[b], color, 2)
            for point in pixels:
                cv2.circle(preview, point, 3, (255, 255, 255), -1)
            # Label below the wrist, clear of the fingers, with an outline so it
            # reads over skin and bright backgrounds.
            lx = min(max(pixels[0][0] - 40, 4), preview_size[0] - 220)
            ly = min(pixels[0][1] + 30, preview_size[1] - 30)
            draw_label(preview, f"{label} {confidence:.0%}", (lx, ly), color, 0.75, 2)
            wh = world_hands.get(label)
            if wh:
                note = f"{wh['depth_m']:.2f} m" if "depth_m" in wh else "partial"
                if wh.get("fingers_hidden"):
                    note += "  fingers hidden"
                draw_label(preview, note, (lx, ly + 26), color, 0.6, 2)

        if now - last_fps_time >= 0.5:
            live_fps = (frames - last_fps_frames) / (now - last_fps_time)
            last_fps_time = now
            last_fps_frames = frames

        # No status text on the picture: the fps, mirror/rotation state and
        # camera telemetry all live in asl_output.json instead.
        if now - last_status >= STATUS_INTERVAL:
            last_status = now
            with open("asl_output.json", "w") as f:
                json.dump({
                    "timestamp": now,
                    "status": "live",
                    "camera_active": True,
                    "camera_index": cam_index,
                    "camera_name": cam_name,
                    "resolution": f"{width}x{height}",
                    "mirror_view": mirror_view,
                    "camera_mirrors": camera_already_mirrors,
                    "view": "egocentric" if egocentric else "facing",
                    "labels_as_is": labels_as_is(),
                    "rotation": rotation,
                    "live_fps": round(live_fps, 1),
                    "hands_detected": len(hands_seen),
                    "handedness": hands_seen,
                    "pose_model": frame_pose_model,
                    "pose_model_requested": "hamer" if hamer is not None else "mediapipe",
                    "telemetry_source": cam_pose["source"],
                    "camera_pose": cam_pose,
                    "hands_world": {label: {"center_image": h.get("center_image"), "center_world": h.get("center_world"),
                                            "depth_m": h.get("depth_m"), "partial": h.get("partial"),
                                            "fingers_hidden": h.get("fingers_hidden")}
                                    for label, h in world_hands.items()},
                    "world_output": args.out,
                    "world_records_written": pipeline.written,
                    "frames": frames,
                    "elapsed_seconds": round(now - start_time, 2),
                }, f, indent=2)

        # Capture mode is the one exception to a text-free picture: without it
        # there is no way to tell whether a burst is still recording.
        capture_text = capture.status()
        if capture_text:
            draw_label(preview, capture_text, (12, 32), (0, 0, 255) if capture.busy else (0, 220, 0), 0.7, 2)

        if args.no_preview:
            continue
        # Show live camera popup window
        cv2.imshow("Insta360 ASL Stream", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 9:                   # Tab
            capture.toggle()
            continue
        if capture.enabled:
            # Every letter is a label in capture mode, including q, m and r.
            if key == 27:              # Esc also leaves capture mode
                capture.toggle()
            elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                capture.start(chr(key))
            continue
        if key == ord('q'):
            break
        if key == ord('x'):
            # Forget which hands are the main ones and re-lock on the closest.
            detect_reset.set()
            print("forgot tracked hands; re-locking on the closest hands in view")
        if key == ord('m'):
            mirror_view = not mirror_view
            detect_reset.set()         # the tracked box and labels are for the old orientation
            pipeline.reset()           # PnP warm starts and the VO keyframe are too
            print(f"mirror view {'on' if mirror_view else 'off'}")
        if key == ord('h'):
            # The camera-mirrors assumption was wrong: swap Left/Right (and the
            # 3D geometry's chirality) while leaving the picture exactly as it is.
            camera_already_mirrors = not camera_already_mirrors
            mirror_view = not mirror_view
            detect_reset.set()
            pipeline.reset()
            print(f"camera mirrors: {'yes' if camera_already_mirrors else 'no'} -> handedness swapped")
            if recorder is not None:
                print("  note: the recording keeps the chirality it started with; replay it with "
                      f"--camera-mirrors {'yes' if record_flip else 'no'} if labels come out swapped")
        if key == ord('r'):
            rotation = ROTATIONS[(ROTATIONS.index(rotation) + 1) % len(ROTATIONS)]
            configure_frame_pipeline(raw_frame)
            width, height, preview_size, K = frame_geometry(rotate(raw_frame))
            pipeline.set_intrinsics(K)
            detect_reset.set()
            pipeline.reset()
            print(f"rotation {rotation}: {width}x{height}")
except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    detect_stop.set()
    detector.join(timeout=2)
    pipeline.close()
    if recorder is not None:
        recorder.release()
    landmarker.close()
    tracker.close()
    lost_finder.close()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"Stopped after {frames} frames ({elapsed:.1f}s, {frames / max(elapsed, 1e-6):.1f} fps average).")
