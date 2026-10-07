from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def resize_nearest(mask: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = mask.shape
    if (h, w) == (size, size):
        return mask
    yi = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
    xi = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
    return mask[np.ix_(yi, xi)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    u = np.logical_or(aa, bb).sum()
    return 1.0 if u == 0 else float(np.logical_and(aa, bb).sum()) / float(u)


def descriptors(mask: np.ndarray) -> dict[str, float]:
    y, x = np.nonzero(mask)
    area = int(mask.sum())
    out = {"area_px": area, "area_fraction": area / float(mask.size)}
    if area == 0:
        out.update({
            "centroid_x_px": np.nan,
            "centroid_y_px": np.nan,
            "width_px": 0,
            "height_px": 0,
        })
        return out
    out.update({
        "centroid_x_px": float(x.mean()),
        "centroid_y_px": float(y.mean()),
        "width_px": int(x.max() - x.min() + 1),
        "height_px": int(y.max() - y.min() + 1),
    })
    return out


def diverse_subset(items: list[dict], count: int) -> list[dict]:
    if len(items) <= count:
        return items

    # Start from a sample near median area, then greedily maximize
    # the minimum IoU-distance to already selected samples.
    areas = np.array([x["area_fraction"] for x in items], dtype=float)
    start = int(np.argmin(np.abs(areas - np.median(areas))))
    chosen = [start]
    remaining = set(range(len(items))) - {start}

    while remaining and len(chosen) < count:
        best_i = None
        best_dist = -1.0
        for i in remaining:
            min_dist = min(
                1.0 - iou(items[i]["mask"], items[j]["mask"])
                for j in chosen
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_i = i
        chosen.append(best_i)
        remaining.remove(best_i)

    return [items[i] for i in chosen]


def save_preview(path: Path, mask: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_exclusions(path: Path | None, column: str) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return set()
    if column not in rows[0]:
        raise ValueError(
            f"Exclusion CSV does not contain column {column!r}: {path}"
        )

    return {
        str(r[column]).strip()
        for r in rows
        if str(r.get(column, "")).strip()
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build IoU-diverse closed-loop target candidates from observed "
            "cumulative masks, with optional hole-level exclusions."
        )
    )
    ap.add_argument("--data-root", type=Path, default=Path("F:/260910_ml_data"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pulses", type=int, nargs="+", default=[50])
    ap.add_argument(
        "--pristine-channel",
        type=int,
        default=1,
        help="Cumulative processed-mask channel in target.npy",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--select-count", type=int, default=12)
    ap.add_argument("--min-area-fraction", type=float, default=0.001)
    ap.add_argument("--exclude-csv", type=Path, default=None)
    ap.add_argument("--exclude-hole-column", default="hole_uid")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    excluded = load_exclusions(args.exclude_csv, args.exclude_hole_column)

    items = []
    n_seen = 0
    n_excluded = 0

    for hole in sorted(p for p in data_root.glob("h*") if p.is_dir()):
        for pulse in args.pulses:
            p = hole / f"p{pulse:04d}" / "target.npy"
            if not p.exists():
                continue

            n_seen += 1
            if hole.name in excluded:
                n_excluded += 1
                continue

            arr = np.asarray(np.load(p), dtype=np.float32)
            if arr.ndim != 3 or not (0 <= args.pristine_channel < arr.shape[0]):
                raise ValueError(f"Unexpected target shape {arr.shape}: {p}")

            mask = resize_nearest(
                (arr[args.pristine_channel] >= args.threshold).astype(np.uint8),
                args.size,
            )
            d = descriptors(mask)
            if d["area_fraction"] < args.min_area_fraction:
                continue

            items.append({
                "hole_uid": hole.name,
                "pulse": pulse,
                "source": p,
                "mask": mask,
                **d,
            })

    if not items:
        raise RuntimeError("No candidate masks found")

    selected = diverse_subset(items, min(args.select_count, len(items)))

    rows = []
    for idx, item in enumerate(selected, 1):
        tid = f"T_DATA_{idx:02d}_{item['hole_uid']}_p{item['pulse']:04d}"
        fn = f"{tid}.npy"
        dest = out / fn
        np.save(dest, item["mask"].astype(np.float32), allow_pickle=False)
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()

        save_preview(
            out / f"{tid}.png",
            item["mask"],
            f"{tid} area={item['area_fraction']:.3f}",
        )

        rows.append({
            "target_id": tid,
            "file_name": fn,
            "source_hole_uid": item["hole_uid"],
            "source_pulse": item["pulse"],
            "source_target_path": str(item["source"]),
            "area_px": item["area_px"],
            "area_fraction": item["area_fraction"],
            "centroid_x_px": item["centroid_x_px"],
            "centroid_y_px": item["centroid_y_px"],
            "width_px": item["width_px"],
            "height_px": item["height_px"],
            "sha256": sha,
            "notes": (
                "Observed cumulative processed mask; exclusion-aware IoU-diverse "
                "candidate only, not yet approved for experiment"
            ),
        })

    csv_path = out / "target_candidates.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if excluded:
        audit_path = out / "excluded_holes_used_for_candidate_generation.csv"
        with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hole_uid"])
            for uid in sorted(excluded):
                w.writerow([uid])

    print(f"Observed masks found   : {n_seen}")
    print(f"Excluded by CSV        : {n_excluded}")
    print(f"Masks eligible for IoU : {len(items)}")
    print(f"Diverse candidates     : {len(rows)}")
    print(f"Output                 : {out}")
    print(f"Summary                : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
