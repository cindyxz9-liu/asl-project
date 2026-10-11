# Labeled training-data capture for the ASL letter classifier.
#
# Press Tab in the preview to enter capture mode, then press a letter key while
# holding that sign. Static letters save a short burst of hand crops (move the
# hand a little during the burst for variety); motion letters (J, Z) record a
# short clip instead, since one still frame cannot tell them apart from I / D.
#
# Layout, ready to drag into a Roboflow classification project (folder = label):
#   dataset/images/<LETTER>/<LETTER>_<stamp>_<n>.jpg
#   dataset/sequences/<LETTER>/<stamp>/frame_<n>.jpg
#   dataset/landmarks.jsonl   one record per saved crop: label, file, hand,
#                             21 image points (relative to the crop) and the
#                             21 MediaPipe world landmarks (metres, hand-centred)
import os
import json
import time
import cv2

DATASET_DIR = "dataset"
MOTION_LETTERS = {"J", "Z"}
BURST_SAMPLES = 20        # crops saved per key press for a static letter
BURST_EVERY = 2           # frames between burst samples (~1.3 s at 30 fps)
SEQUENCE_FRAMES = 45      # frames recorded for a motion letter (~1.5 s at 30 fps)
CROP_PAD = 0.25           # padding around the hand, as a fraction of its size
CROP_SIZE = 256           # saved crops are square at this size


def hand_crop(frame, points, pad=CROP_PAD, size=CROP_SIZE):
    """Square crop around normalised landmark points, plus the points
    re-expressed relative to that crop. None if the hand is off the frame."""
    fh, fw = frame.shape[:2]
    xs = [x * fw for x, _ in points]
    ys = [y * fh for _, y in points]
    side = int(min(max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * pad), fw, fh))
    if side < 16:
        return None, None
    # Near an edge, slide the square back inside the frame rather than drop the sample.
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x0 = int(min(max(cx - side / 2, 0), fw - side))
    y0 = int(min(max(cy - side / 2, 0), fh - side))
    crop = frame[y0:y0 + side, x0:x0 + side]
    rel = [((x - x0) / side, (y - y0) / side) for x, y in zip(xs, ys)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), rel


def main_hand(hands):
    """The hand doing the sign: the largest one on screen."""
    def area(hand):
        pts = hand[2]
        xs = [x for x, _ in pts]; ys = [y for _, y in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    return max(hands, key=area) if hands else None


class Capture:
    def __init__(self, root=DATASET_DIR):
        self.root = root
        self.enabled = False
        self.label = None
        self.remaining = 0
        self.sequence_dir = None
        self.taken = 0
        self.counts = {}
        self.last_saved = None
        self._count_existing()

    def _count_existing(self):
        images = os.path.join(self.root, "images")
        if os.path.isdir(images):
            for letter in os.listdir(images):
                folder = os.path.join(images, letter)
                if os.path.isdir(folder):
                    self.counts[letter] = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        sequences = os.path.join(self.root, "sequences")
        if os.path.isdir(sequences):
            for letter in os.listdir(sequences):
                folder = os.path.join(sequences, letter)
                if os.path.isdir(folder):
                    self.counts[letter] = len(os.listdir(folder))

    @property
    def busy(self):
        return self.remaining > 0

    def toggle(self):
        self.enabled = not self.enabled
        self.remaining = 0
        print(f"capture mode {'on: hold a sign and press its letter (Tab to leave)' if self.enabled else 'off'}")

    def start(self, letter):
        """Begin a burst (static letter) or a clip (motion letter)."""
        if self.busy:
            return
        self.label = letter.upper()
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.taken = 0
        if self.label in MOTION_LETTERS:
            self.remaining = SEQUENCE_FRAMES
            self.sequence_dir = os.path.join(self.root, "sequences", self.label, self.stamp)
            os.makedirs(self.sequence_dir, exist_ok=True)
            self.sequence = []
            print(f"recording {self.label} motion: do the full movement now")
        else:
            self.remaining = BURST_SAMPLES
            self.sequence_dir = None
            os.makedirs(os.path.join(self.root, "images", self.label), exist_ok=True)
            print(f"capturing {self.label}: move the hand slightly while holding the sign")

    def step(self, frame_index, frame, hands, mirrored=False):
        """Call once per frame with the clean (un-drawn) frame.

        `mirrored` says the frame and landmarks are a mirror image of the
        scene. Everything is normalised to the physical image before saving
        (frame flipped, x -> 1-x, world x -> -x; labels are already physical),
        so a dataset recorded live and one replayed from a file agree.
        """
        if not self.busy:
            return
        if self.sequence_dir is None and frame_index % BURST_EVERY:
            return
        if mirrored and hands:
            frame = cv2.flip(frame, 1)
            hands = [(label, conf, [(1.0 - x, y) for x, y in pts], [(-x, y, z) for x, y, z in world])
                     for label, conf, pts, world in hands]
        hand = main_hand(hands)
        if self.sequence_dir is not None:
            self._record_sequence_frame(frame, hand)
        elif hand is not None:
            self._save_still(frame, hand)
        if self.remaining == 0:
            self._finish()

    def _save_still(self, frame, hand):
        label_hand, score, points, world = hand
        crop, rel = hand_crop(frame, points)
        if crop is None:
            return
        name = f"{self.label}_{self.stamp}_{self.taken:02d}.jpg"
        path = os.path.join(self.root, "images", self.label, name)
        cv2.imwrite(path, crop)
        self._log({"label": self.label, "kind": "still", "file": os.path.relpath(path, self.root),
                   "hand": label_hand, "points": rel, "world": world})
        self.taken += 1
        self.remaining -= 1

    def _record_sequence_frame(self, frame, hand):
        # Every frame counts toward the clip length, hand or not, so a clip is
        # always the same duration; missing hands are logged as null.
        record = {"index": SEQUENCE_FRAMES - self.remaining, "hand": None}
        if hand is not None:
            label_hand, score, points, world = hand
            crop, rel = hand_crop(frame, points)
            if crop is not None:
                name = f"frame_{record['index']:03d}.jpg"
                cv2.imwrite(os.path.join(self.sequence_dir, name), crop)
                # Full-frame points as well: the motion itself (the Z stroke)
                # is in where the hand travels, which crop-relative points hide.
                record.update({"hand": label_hand, "file": name, "points": rel,
                               "points_frame": points, "world": world})
                self.taken += 1
        self.sequence.append(record)
        self.remaining -= 1

    def _finish(self):
        if self.sequence_dir is not None:
            with open(os.path.join(self.sequence_dir, "landmarks.json"), "w") as f:
                json.dump({"label": self.label, "frames": self.sequence}, f)
            self._log({"label": self.label, "kind": "sequence",
                       "file": os.path.relpath(self.sequence_dir, self.root), "frames": len(self.sequence)})
            self.counts[self.label] = self.counts.get(self.label, 0) + 1
            print(f"saved {self.label} clip ({self.taken}/{SEQUENCE_FRAMES} frames had a hand); "
                  f"{self.counts[self.label]} clips total")
        else:
            self.counts[self.label] = self.counts.get(self.label, 0) + self.taken
            print(f"saved {self.taken} {self.label} images; {self.counts[self.label]} total")
        self.last_saved = self.label

    def _log(self, record):
        with open(os.path.join(self.root, "landmarks.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def status(self):
        """Short overlay text, or None when capture mode is off."""
        if not self.enabled:
            return None
        if self.busy:
            what = "REC" if self.sequence_dir is not None else "CAPTURE"
            return f"{what} {self.label}  {self.remaining}"
        if self.last_saved:
            return f"CAPTURE  {self.last_saved}: {self.counts.get(self.last_saved, 0)}  (press a letter)"
        return "CAPTURE  press a letter"
