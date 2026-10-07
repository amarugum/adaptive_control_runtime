from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a review sheet comparing nearest-to-centroid representatives "
            "for several K values with exclusion-aware IoU-diverse candidates."
        )
    )
    p.add_argument("--cluster-run", required=True, type=Path)
    p.add_argument("--iou-csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--tile", type=int, default=230)
    p.add_argument("--cols", type=int, default=4)
    return p.parse_args()


def load_font(size: int):
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_mask(path: Path, channel: int, threshold: float, image_size: int = 256) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        if channel >= arr.shape[0]:
            raise ValueError(f"channel {channel} unavailable in {path}: shape={arr.shape}")
        arr = arr[channel]
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2-D or CxHxW target array, got {arr.shape} at {path}")

    mask = (arr >= threshold).astype(np.uint8) * 255
    im = Image.fromarray(mask, mode="L")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def fit_into_tile(mask: np.ndarray, tile: int) -> Image.Image:
    im = Image.fromarray(mask, mode="L").convert("RGB")
    im.thumbnail((tile, tile), Image.Resampling.NEAREST)
    bg = Image.new("RGB", (tile, tile), "black")
    bg.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return bg


def main() -> int:
    args = parse_args()
    run = args.cluster_run.resolve()
    metrics = pd.read_csv(run / "clustering_metrics.csv").set_index("k")
    iou = pd.read_csv(args.iou_csv)

    required_iou = {
        "target_id", "source_hole_uid", "source_target_path",
        "source_pulse", "area_fraction"
    }
    missing = required_iou - set(iou.columns)
    if missing:
        raise ValueError(f"IoU CSV missing columns: {sorted(missing)}")

    cluster_sections = []
    summary_rows = []

    for k in args.ks:
        kdir = run / f"K{k:02d}"
        assign = pd.read_csv(kdir / "cluster_assignments.csv")
        sizes = pd.read_csv(kdir / "cluster_sizes.csv").set_index("cluster_id")
        nearest = assign.loc[assign["is_nearest"].astype(int) == 1].sort_values("cluster_id")

        records = []
        for _, r in nearest.iterrows():
            cid = int(r["cluster_id"])
            rec = {
                "method": "cluster_nearest",
                "k": k,
                "cluster_id": cid,
                "candidate_id": f"K{k:02d}_C{cid:02d}_{r['hole_uid']}",
                "hole_uid": str(r["hole_uid"]),
                "pulse_number": int(r["pulse_number"]),
                "source_target_path": str(r["source_target_path"]),
                "cluster_size": int(sizes.loc[cid, "n_samples"]),
                "cluster_fraction": float(sizes.loc[cid, "fraction"]),
                "area_fraction": float(r["area_px"]) / (256 * 256),
            }
            records.append(rec)
            summary_rows.append(rec.copy())
        cluster_sections.append((k, records))

    iou_records = []
    for _, r in iou.iterrows():
        rec = {
            "method": "iou_diverse",
            "k": None,
            "cluster_id": None,
            "candidate_id": str(r["target_id"]),
            "hole_uid": str(r["source_hole_uid"]),
            "pulse_number": int(r["source_pulse"]),
            "source_target_path": str(r["source_target_path"]),
            "cluster_size": None,
            "cluster_fraction": None,
            "area_fraction": float(r["area_fraction"]),
        }
        iou_records.append(rec)
        summary_rows.append(rec.copy())

    tile = args.tile
    label_h = 58
    left_margin = 175
    right_margin = 25
    top_margin = 90
    section_gap = 30
    max_cluster_cols = max(args.ks)
    width = left_margin + max(max_cluster_cols, args.cols) * tile + right_margin

    cluster_height = sum(tile + label_h + 20 for _ in cluster_sections)
    iou_rows = math.ceil(len(iou_records) / args.cols)
    iou_height = 55 + iou_rows * (tile + label_h + 15)
    height = top_margin + cluster_height + section_gap + iou_height + 30

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(27)
    head_font = load_font(19)
    label_font = load_font(14)
    small_font = load_font(12)

    draw.text((20, 18), "Target candidate review: cleaned clustering vs cleaned IoU-diverse", fill="black", font=title_font)
    draw.text(
        (20, 55),
        "Clustering: nearest real sample to each centroid. IoU: exclusion-aware diverse observed masks.",
        fill="black",
        font=small_font,
    )

    y = top_margin
    for k, records in cluster_sections:
        mm = metrics.loc[k]
        draw.multiline_text(
            (15, y + 10),
            f"K={k}\nsil={mm['silhouette_score']:.3f}\nDBI={mm['davies_bouldin_index']:.3f}\nmin n={int(mm['min_cluster_size'])}",
            fill="black",
            font=head_font,
            spacing=4,
        )
        for j, rec in enumerate(records):
            x = left_margin + j * tile
            mask = load_mask(Path(rec["source_target_path"]), args.channel, args.threshold)
            canvas.paste(fit_into_tile(mask, tile - 8), (x + 4, y + 4))
            draw.multiline_text(
                (x + 5, y + tile + 4),
                f"C{rec['cluster_id']:02d}  {rec['hole_uid']}\n"
                f"n={rec['cluster_size']} ({100*rec['cluster_fraction']:.1f}%)  "
                f"area={100*rec['area_fraction']:.1f}%",
                fill="black",
                font=label_font,
                spacing=2,
            )
        y += tile + label_h + 20

    y += section_gap
    draw.text((15, y), "IoU-diverse candidates", fill="black", font=head_font)
    y += 42

    for j, rec in enumerate(iou_records):
        rr = j // args.cols
        cc = j % args.cols
        x = left_margin + cc * tile
        yy = y + rr * (tile + label_h + 15)
        mask = load_mask(Path(rec["source_target_path"]), args.channel, args.threshold)
        canvas.paste(fit_into_tile(mask, tile - 8), (x + 4, yy + 4))
        draw.multiline_text(
            (x + 5, yy + tile + 4),
            f"{rec['candidate_id']}\n{rec['hole_uid']}  area={100*rec['area_fraction']:.1f}%",
            fill="black",
            font=label_font,
            spacing=2,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(180, 180))

    summary_path = args.summary_csv or args.output.with_suffix(".csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved sheet   : {args.output}")
    print(f"Saved summary : {summary_path}")
    print("Displayed K   :", ", ".join(map(str, args.ks)))
    print(f"IoU candidates: {len(iou_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
