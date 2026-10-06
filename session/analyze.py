"""Automatic telemetry analysis for a session directory.

Runs without a human: telemetry stats, frame dedup, pseudo-labels with
the production model, borderline mining. Writes ANALYSIS.md + contact
sheets next to the data. Training/fine-tune decisions come later and
only through acceptance gates; this stage never modifies models.
"""

import csv
import os
from collections import Counter

import cv2
import numpy as np

BORDERLINE_LOW = 0.25
BORDERLINE_HIGH = 0.5


def telemetry_stats(csv_path):
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return {"rows": 0}
    states = Counter(r.get("state", "") for r in rows)
    confs = [float(r["confidence"]) for r in rows if r.get("confidence")]
    fps = [float(r["inference_fps"]) for r in rows if r.get("inference_fps")]
    backends = Counter(r.get("backend", "") for r in rows)
    detected = states.get("DETECTED", 0)
    return {
        "rows": len(rows),
        "detected": detected,
        "detection_rate": (detected / len(rows)) if rows else 0.0,
        "conf_mean": float(np.mean(confs)) if confs else 0.0,
        "conf_min": float(np.min(confs)) if confs else 0.0,
        "fps_mean": float(np.mean(fps)) if fps else 0.0,
        "backends": dict(backends),
    }


def dhash(image, size=9):
    gray = cv2.cvtColor(cv2.resize(image, (size, 8)), cv2.COLOR_BGR2GRAY)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return sum(int(b) << i for i, b in enumerate(bits))


def dedup(images, threshold=5):
    """Greedy near-duplicate removal. Returns kept paths."""
    kept = []
    hashes = []
    for path in sorted(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        digest = dhash(image)
        if any(bin(digest ^ known).count("1") <= threshold for known in hashes):
            continue
        hashes.append(digest)
        kept.append(path)
    return kept


def pseudo_label(detector, image_paths, conf=0.15):
    """Run production model over frames. Returns {name: max_conf}."""
    out = {}
    for path in image_paths:
        result = detector(str(path), verbose=False)[0]
        best = 0.0
        for box in result.boxes:
            best = max(best, float(box.conf.item()))
        out[os.path.basename(path)] = best
    return out


def contact_sheet(paths, dest, cols=5, cell=320):
    images = [cv2.imread(str(p)) for p in paths[:40]]
    images = [im for im in images if im is not None]
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * 205, cols * cell, 3), 32, dtype=np.uint8)
    for i, im in enumerate(images):
        thumb = cv2.resize(im, (cell, 180))
        canvas[(i // cols) * 205:(i // cols) * 205 + 180, (i % cols) * cell:(i % cols) * cell + cell] = thumb
    cv2.imwrite(str(dest), canvas)


def analyze_session(session_dir, model_path="yolo11n.pt", conf=0.15):
    """Analyze one session dir in place. Returns the report dict."""
    from ultralytics import YOLO

    session_dir = os.path.abspath(session_dir)
    stats = telemetry_stats(os.path.join(session_dir, "phone_telemetry.csv"))
    frames_dir = os.path.join(session_dir, "frames")
    frames = []
    if os.path.isdir(frames_dir):
        frames = [os.path.join(frames_dir, n) for n in os.listdir(frames_dir) if n.endswith(".jpg")]

    report = {
        "session": os.path.basename(session_dir.rstrip(os.sep)),
        "telemetry": stats,
        "frames_total": len(frames),
        "frames_kept": 0,
        "pseudo_labeled": 0,
        "borderline": [],
        "decision": "",
    }

    kept = dedup(frames)
    report["frames_kept"] = len(kept)
    if kept:
        detector = YOLO(model_path)
        scores = pseudo_label(detector, kept, conf=conf)
        report["pseudo_labeled"] = len(scores)
        borderline = sorted(
            [n for n, c in scores.items() if BORDERLINE_LOW <= c < BORDERLINE_HIGH]
        )
        report["borderline"] = borderline
        contact_sheet(
            [os.path.join(frames_dir, n) for n in borderline],
            os.path.join(session_dir, "borderline.jpg"),
        )
        if stats["rows"] and stats["detection_rate"] < 0.05 and not borderline:
            report["decision"] = "no cat activity worth training on"
        elif borderline:
            report["decision"] = (
                f"{len(borderline)} borderline frames need human review "
                "before any retraining"
            )
        else:
            report["decision"] = "clean session, nothing to mine"
    else:
        report["decision"] = "no frames captured"

    with open(os.path.join(session_dir, "ANALYSIS.md"), "w", encoding="utf-8") as f:
        t = stats
        f.write(f"# Analysis: {report['session']}\n\n")
        f.write(f"- telemetry rows: {t['rows']}, DETECTED: {t.get('detected', 0)} "
                f"({t.get('detection_rate', 0):.1%})\n")
        f.write(f"- mean conf: {t.get('conf_mean', 0):.2f}, mean fps: {t.get('fps_mean', 0):.1f}\n")
        f.write(f"- backends: {t.get('backends', {})}\n")
        f.write(f"- frames: {report['frames_total']} total, {report['frames_kept']} after dedup\n")
        f.write(f"- borderline [{BORDERLINE_LOW}, {BORDERLINE_HIGH}): {len(report['borderline'])}\n")
        for name in report["borderline"]:
            f.write(f"  - {name}\n")
        f.write(f"\nDecision: {report['decision']}\n")
    return report


if __name__ == "__main__":
    import sys

    print(analyze_session(sys.argv[1]) if len(sys.argv) > 1 else "usage: analyze.py <session_dir>")
