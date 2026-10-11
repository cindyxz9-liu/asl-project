"""Render a JSONL from asl_stream.py (optionally filled by fill_tracks.py)
onto its source video: skeletons, labels, and the camera telemetry.
Observed hands are drawn in colour; inferred ones (interpolated / held) in grey."""
import argparse, json, math
import cv2
import numpy as np

CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
COL = {"Left": (255, 170, 0), "Right": (0, 170, 255)}
GREY = (160, 160, 160)


def label(img, text, org, color, s=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("jsonl"); ap.add_argument("out")
    ap.add_argument("--fisheye-hfov", type=float, default=78); ap.add_argument("--hfov", type=float, default=70)
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none",
                    help="same --rotate the clip was processed with")
    ap.add_argument("--max-long-edge", type=int, default=1920); ap.add_argument("--out-long-edge", type=int, default=960)
    ap.add_argument("--draw-held", type=int, default=3, metavar="FRAMES",
                    help="draw a held (frozen) skeleton only for this many frames after the hand was last seen; "
                         "after that the hand is out of view and only its label is shown, so no skeleton floats "
                         "where there is no hand. The records keep every held pose regardless.")
    a = ap.parse_args()
    import sys; sys.argv = [sys.argv[0], "--video", "x"]          # borrow enhance() from the stream script
    src_py = open("asl_stream.py").read(); ns = {"cv2": cv2, "np": np, "math": math}
    exec(src_py[src_py.index("EXPOSURE_TARGET"):src_py.index("undistort_maps = None")], ns)
    enhance = ns["enhance"]
    recs = {r["frame"]: r for r in map(json.loads, open(a.jsonl))}
    src = cv2.VideoCapture(a.video); W0, H0, fps = int(src.get(3)), int(src.get(4)), src.get(5)
    code = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[a.rotate]
    if code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        W0, H0 = H0, W0
    sc = min(1.0, a.max_long_edge / max(W0, H0)); W, H = int(W0 * sc), int(H0 * sc)
    fp = (W / 2) / math.tan(math.radians(a.hfov / 2))
    maps = None
    if a.fisheye_hfov:
        ff = (W / 2) / math.radians(a.fisheye_hfov / 2)
        Kf = np.array([[ff, 0, W / 2], [0, ff, H / 2], [0, 0, 1]]); Kp = np.array([[fp, 0, W / 2], [0, fp, H / 2], [0, 0, 1]])
        maps = cv2.fisheye.initUndistortRectifyMap(Kf, np.zeros(4), np.eye(3), Kp, (W, H), cv2.CV_16SC2)
    osc = a.out_long_edge / max(W, H); ow, oh = int(W * osc), int(H * osc)
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    i = 0
    while True:
        ok, f = src.read()
        if not ok:
            break
        i += 1
        if code is not None:
            f = cv2.rotate(f, code)
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        if maps is not None:
            f = cv2.remap(f, maps[0], maps[1], cv2.INTER_LINEAR)
        img = cv2.resize(enhance(f), (ow, oh))
        r = recs.get(i)
        if r:
            cam = r["camera"]; yaw, pitch, roll = cam["yaw_pitch_roll_deg"]
            label(img, f"frame {i}  camera yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f}  "
                       f"turning {math.degrees(np.linalg.norm(cam['gyro_rad_s'])):.0f} deg/s", (8, 22), (0, 255, 255), 0.5)
            y = 46
            for h in sorted(r["hands"], key=lambda h: h["label"]):
                kind = h.get("inferred", "infilled" if h.get("infilled") else "observed")
                if kind == "out_of_view" or (kind == "held" and (h.get("held_frames") or 0) > a.draw_held):
                    label(img, f"{h['label']}: out of view", (8, y), GREY); y += 20
                    continue
                if h.get("image_points"):
                    # Exactly where the detector saw the hand on this frame.
                    P = np.array(h["image_points"]); u = P[:, 0] * W * osc; v = P[:, 1] * H * osc
                elif "joints_cam" in h:
                    J = np.array(h["joints_cam"])
                    u = (fp * J[:, 0] / J[:, 2] + W / 2) * osc; v = (fp * J[:, 1] / J[:, 2] + H / 2) * osc
                else:
                    continue
                pts = [(int(x), int(yy)) for x, yy in zip(u, v)]
                col = COL[h["label"]] if kind == "observed" else GREY
                for p, q in CONN:
                    cv2.line(img, pts[p], pts[q], col, 2 if kind == "observed" else 1, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(img, p, 2, (255, 255, 255) if kind == "observed" else GREY, -1)
                note = kind if kind != "held" else f"held {h.get('held_frames', '')} f"
                extra = f"  {h['depth_m']:.2f} m" if kind == "observed" and h.get("depth_m") is not None else ""
                extra += "  fingers hidden" if h.get("fingers_hidden") else ""
                label(img, f"{h['label']}: {note}{extra}", (8, y), col); y += 20
        out.write(img)
    out.release(); print(f"wrote {a.out}: {i} frames")


if __name__ == "__main__":
    main()
