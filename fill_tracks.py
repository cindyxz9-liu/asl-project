"""Give every frame a skeleton for both hands, honestly labelled.

Reads a world-space JSONL written by asl_stream.py and writes a copy in which
each hand has a record on every frame:

  observed      the detector saw it (unchanged records)
  infilled      asl_stream's own short-gap interpolation (unchanged)
  interpolated  this script: linear interpolation of the joints across a gap
                of up to --max-gap frames between two observations
  held          this script: the last observed pose, frozen, for frames where
                the hand was not seen for longer than that, or before its first
                / after its last observation

A model trained on these must read the `inferred` field: held and
interpolated poses are not observations of the hands.
"""
import argparse
import json
import numpy as np

JOINT_FIELDS = ("joints_cam", "joints_world", "world_landmarks", "image_points")


def lerp(a, b, t):
    return (np.asarray(a) * (1 - t) + np.asarray(b) * t).round(4).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--max-gap", type=int, default=60, help="longest gap (frames) to interpolate across")
    ap.add_argument("--hands", default="Left,Right")
    ap.add_argument("--max-hold", type=int, default=15,
                    help="hold a hand's last observed pose for at most this many frames after it was seen "
                         "(never before its first sighting); beyond that the hand is recorded as "
                         "out_of_view with no joints, rather than a frozen pose floating where no hand is")
    ap.add_argument("--max-jump", type=float, default=0.3,
                    help="do not interpolate between two sightings this far apart in the image (fraction of "
                         "the frame); hold instead, so a skeleton never slides across the picture")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.src)]
    labels = a.hands.split(",")

    # Body anchors learned during the run (the signer's chin, read as a hand by
    # a chest-mounted camera) are applied in hindsight to the whole clip: the
    # tracker only learns them after the chin has shown up a few times.
    anchors = next((r["body_anchors"] for r in reversed(recs) if r.get("body_anchors")), [])
    removed = 0
    if anchors:
        for r in recs:
            keep = []
            for h in r["hands"]:
                c = h.get("center_image")
                size = h.get("box_diag_f")
                near = c is not None and any(((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5 < 0.12 for x, y in anchors)   # ANCHOR_RADIUS
                if near and not h.get("infilled") and (size is None or size < 0.35):   # ANCHOR_KEEP_F
                    removed += 1
                    continue
                keep.append(h)
            r["hands"] = keep
    if anchors:
        print(f"body anchors {anchors}: removed {removed} detections on them")
    for label in labels:
        seen = [(i, h) for i, r in enumerate(recs) for h in r["hands"] if h["label"] == label and "joints_cam" in h]
        if not seen:
            continue
        idx = [i for i, _ in seen]
        for i, r in enumerate(recs):
            if any(h["label"] == label for h in r["hands"]):
                for h in r["hands"]:
                    if h["label"] == label:
                        h.setdefault("inferred", "infilled" if h.get("infilled") else "observed")
                continue
            k = np.searchsorted(idx, i)
            prev = seen[k - 1] if k > 0 else None
            nxt = seen[k] if k < len(seen) else None
            jump = None
            if prev and nxt and prev[1].get("center_image") and nxt[1].get("center_image"):
                jump = ((prev[1]["center_image"][0] - nxt[1]["center_image"][0]) ** 2 +
                        (prev[1]["center_image"][1] - nxt[1]["center_image"][1]) ** 2) ** 0.5
            if prev and nxt and nxt[0] - prev[0] <= a.max_gap and (jump is None or jump <= a.max_jump):
                t = (i - prev[0]) / (nxt[0] - prev[0])
                h = {"label": label, "confidence": 0.0, "infilled": True, "inferred": "interpolated"}
                for f in JOINT_FIELDS:
                    if prev[1].get(f) is not None and nxt[1].get(f) is not None:
                        h[f] = lerp(prev[1][f], nxt[1][f], t)
                h["wrist_world"] = h.get("joints_world", [[None]*3])[0]
            elif prev and i - prev[0] <= a.max_hold:
                h = {k2: v for k2, v in prev[1].items()
                     if k2 in JOINT_FIELDS or k2 in ("wrist_world", "center_world", "center_image", "depth_m")}
                h.update({"label": label, "confidence": 0.0, "infilled": True, "inferred": "held",
                          "held_frames": i - prev[0]})
            else:
                # Not in view: before its first sighting, or gone longer than
                # --max-hold. A record, so every frame has both hands, but no
                # invented pose.
                h = {"label": label, "confidence": 0.0, "infilled": True, "inferred": "out_of_view"}
            r["hands"].append(h)
        for r in recs:
            r["hands"].sort(key=lambda h: h["label"])
    with open(a.dst, "w") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    counts = {}
    for r in recs:
        for h in r["hands"]:
            counts[h.get("inferred", "observed")] = counts.get(h.get("inferred", "observed"), 0) + 1
    print(f"{len(recs)} frames; both hands present on every frame; records by kind: {counts}")


if __name__ == "__main__":
    main()
