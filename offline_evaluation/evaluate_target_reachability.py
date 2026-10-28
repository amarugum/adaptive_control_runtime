from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _looks_windows_absolute(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    text = str(value)
    if _looks_windows_absolute(text):
        # On Windows this is an ordinary absolute Path.  Keeping the drive form
        # unchanged also makes the config readable when inspected elsewhere.
        return Path(PureWindowsPath(text)) if Path(text).is_absolute() else Path(text)
    p = Path(text).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return obj


def write_csv(rows: Sequence[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unnamed"


def resize_binary(mask: np.ndarray, size: int) -> np.ndarray:
    """Match the ML dataset convention: binary mask + nearest-neighbor resize."""
    a = np.asarray(mask, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, None]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[None]
    else:
        raise ValueError(f"Expected [H,W] or [1,H,W], got {a.shape}")
    t = torch.from_numpy(a)
    out = F.interpolate(t, size=(size, size), mode="nearest")
    return (out[0, 0].numpy() >= 0.5).astype(np.float32)


def load_binary_target(path: Path) -> np.ndarray:
    a = np.asarray(np.load(path), dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Ideal target must be 2-D or [1,H,W], got {a.shape}: {path}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Non-finite values in target: {path}")
    # Ideal target files are expected to be exact binary masks.  Reject, rather
    # than silently threshold, because these files are the experiment definition.
    if not np.all((a == 0.0) | (a == 1.0)):
        vals = np.unique(a)
        raise ValueError(
            f"Ideal target must be exact binary 0/1: {path}; "
            f"example values={vals[:10]}"
        )
    if float(a.sum()) <= 0:
        raise ValueError(f"Empty target: {path}")
    return a


def load_dataset_mask(
    path: Path,
    pristine_channel: int,
    threshold: float,
    size: int,
) -> np.ndarray:
    a = np.asarray(np.load(path), dtype=np.float32)
    if a.ndim != 3 or not (0 <= pristine_channel < a.shape[0]):
        raise ValueError(
            f"Dataset target must be [C,H,W] and contain channel {pristine_channel}: "
            f"{path}, shape={a.shape}"
        )
    mask = (a[pristine_channel] >= threshold).astype(np.float32)
    return resize_binary(mask, size)


def rmse_binary(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum()) / float(union)


def soft_dice_loss_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    return float(1.0 - (2.0 * np.sum(p * t) + eps) / (np.sum(p) + np.sum(t) + eps))


def fractions_np(mask_binary: np.ndarray, target_binary: np.ndarray) -> Dict[str, float]:
    """Same convention as runtime.selector.fractions: 1 = processed region."""
    m = np.asarray(mask_binary, dtype=bool)
    t = np.asarray(target_binary, dtype=bool)
    denom = max(int(t.sum()), 1)
    overall = float(np.logical_and(m, t).sum()) / denom
    over = float(np.logical_and(m, ~t).sum()) / denom
    return {
        "overall_removal_fraction": overall,
        "under_removal_fraction": 1.0 - overall,
        "over_removal_fraction": over,
    }


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedTarget:
    target_id: str
    source_path: Path
    prepared_path: Path
    source_shape: Tuple[int, int]
    mask: np.ndarray
    source_sha256: str
    prepared_sha256: str


def discover_and_prepare_targets(
    target_root: Path,
    pattern: str,
    size: int,
    out_dir: Path,
    show_progress: bool = True,
) -> List[PreparedTarget]:
    files = sorted(target_root.glob(pattern))
    if not files:
        raise RuntimeError(f"No ideal target files matched {pattern!r} under {target_root}")

    prepared_root = out_dir / "targets_256"
    prepared_root.mkdir(parents=True, exist_ok=True)
    prepared: List[PreparedTarget] = []
    manifest_rows: List[Dict] = []

    for src in tqdm(
        files,
        desc="Prepare targets",
        unit="target",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        raw = load_binary_target(src)
        target_id = src.parent.name if src.name == "desired_target_mask.npy" else src.stem
        mask = resize_binary(raw, size)
        dst = prepared_root / f"{target_id}.npy"
        np.save(dst, mask.astype(np.float32), allow_pickle=False)

        p = PreparedTarget(
            target_id=target_id,
            source_path=src.resolve(),
            prepared_path=dst.resolve(),
            source_shape=tuple(raw.shape),
            mask=mask,
            source_sha256=sha256_file(src),
            prepared_sha256=sha256_file(dst),
        )
        prepared.append(p)
        manifest_rows.append({
            "target_id": target_id,
            "source_path": str(src.resolve()),
            "source_height": int(raw.shape[0]),
            "source_width": int(raw.shape[1]),
            "source_area_fraction": float(raw.mean()),
            "source_sha256": p.source_sha256,
            "prepared_path": str(dst.resolve()),
            "prepared_height": size,
            "prepared_width": size,
            "prepared_area_fraction": float(mask.mean()),
            "prepared_sha256": p.prepared_sha256,
            "resize_mode": "nearest",
        })

    write_csv(manifest_rows, out_dir / "target_conversion_manifest.csv")
    _save_target_contact_sheet(prepared, out_dir / "target_contact_sheet_256.png")
    return prepared


def _save_target_contact_sheet(targets: Sequence[PreparedTarget], path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    tile = 280
    label_h = 38
    cols = min(4, max(1, len(targets)))
    rows = math.ceil(len(targets) / cols)
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, t in enumerate(targets):
        r, c = divmod(i, cols)
        im = Image.fromarray((t.mask * 255).astype(np.uint8), mode="L").convert("RGB")
        im = im.resize((tile, tile), Image.Resampling.NEAREST)
        x = c * tile
        y = r * (tile + label_h)
        canvas.paste(im, (x, y))
        draw.text((x + 6, y + tile + 8), t.target_id, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(160, 160))


# ---------------------------------------------------------------------------
# Transition model loading
# ---------------------------------------------------------------------------

@dataclass
class ModelContext:
    name: str
    runtime_mode: str
    run_dir: Path
    model: torch.nn.Module
    cfg: Dict
    z_lookup: Optional[Dict[str, np.ndarray]]
    rows_by_key: Dict[Tuple[str, int, int], Dict]
    checkpoint_path: Path


def physical_key(row: Dict) -> Tuple[str, int, int]:
    return (
        str(row["hole_uid"]),
        int(row["current_pulse"]),
        int(row["next_pulse"]),
    )


def load_transition_rows(cfg: Dict, split_name: str) -> Dict[Tuple[str, int, int], Dict]:
    from transition.dataset import load_dataset_split, load_manifest

    d = cfg["data"]
    data_root = Path(d["root"])
    rows = load_manifest(data_root / d["manifest_name"])
    holes = sorted({str(r["hole_uid"]) for r in rows})
    splits = load_dataset_split(Path(d["split_csv"]), available_holes=holes)

    if split_name == "all":
        selected = rows
    else:
        if split_name not in splits:
            raise ValueError(f"Unknown split {split_name!r}; choose train/val/test/all")
        wanted = set(splits[split_name])
        selected = [r for r in rows if str(r["hole_uid"]) in wanted]

    out = {}
    for r in selected:
        k = physical_key(r)
        if k in out:
            raise ValueError(f"Duplicate physical transition key: {k}")
        out[k] = r
    return out


def load_transition_context(
    *,
    name: str,
    runtime_mode: str,
    run_dir: Path,
    ml_repo: Path,
    split_name: str,
    device: torch.device,
):
    if str(ml_repo) not in sys.path:
        sys.path.insert(0, str(ml_repo))

    from transition.models import TransitionUNet
    from transition.dataset import load_z_shock_npz

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Transition checkpoint not found: {ckpt}")

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    m = cfg["model"]
    variant = str(m["variant"]).upper()

    z_lookup = None
    shock_dim = None
    if variant == "B":
        z_path = Path(cfg["data"]["z_shock_npz"])
        if not z_path.exists():
            raise FileNotFoundError(f"z_shock cache not found for {name}: {z_path}")
        z_lookup = load_z_shock_npz(z_path)
        if not z_lookup:
            raise RuntimeError(f"Empty z_shock lookup: {z_path}")
        shock_dim = int(next(iter(z_lookup.values())).shape[0])

    model = TransitionUNet(
        variant=variant,
        encoder_channels=tuple(m["encoder_channels"]),
        bottleneck_channels=int(m["bottleneck_channels"]),
        action_hidden_dims=tuple(m["action_hidden_dims"]),
        condition_dim=int(m["condition_dim"]),
        shock_dim=shock_dim,
        group_norm_groups=int(m["group_norm_groups"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    rows_by_key = load_transition_rows(cfg, split_name)

    return ModelContext(
        name=name,
        runtime_mode=runtime_mode,
        run_dir=run_dir,
        model=model,
        cfg=cfg,
        z_lookup=z_lookup,
        rows_by_key=rows_by_key,
        checkpoint_path=ckpt,
    )


def load_context_metadata(
    *,
    name: str,
    runtime_mode: str,
    run_dir: Path,
    ml_repo: Path,
    split_name: str,
) -> Dict:
    """Load checkpoint config and row keys without constructing the GPU model."""
    if str(ml_repo) not in sys.path:
        sys.path.insert(0, str(ml_repo))
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Transition checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    rows_by_key = load_transition_rows(cfg, split_name)
    return {
        "name": name,
        "runtime_mode": runtime_mode,
        "run_dir": run_dir,
        "cfg": cfg,
        "rows_by_key": rows_by_key,
        "checkpoint_path": ckpt,
    }


# ---------------------------------------------------------------------------
# State selection
# ---------------------------------------------------------------------------

def select_common_state_keys(
    model_meta: Sequence[Dict],
    current_pulses: Optional[Sequence[int]],
    max_states_per_pulse: Optional[int],
    seed: int,
) -> List[Tuple[str, int, int]]:
    sets = [set(x["rows_by_key"]) for x in model_meta]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("No common physical transition states across requested models")

    keys = sorted(common, key=lambda x: (x[1], x[0], x[2]))
    if current_pulses:
        allowed = {int(x) for x in current_pulses}
        keys = [k for k in keys if int(k[1]) in allowed]

    if max_states_per_pulse is not None:
        rng = np.random.default_rng(seed)
        selected = []
        pulses = sorted({k[1] for k in keys})
        for p in pulses:
            group = [k for k in keys if k[1] == p]
            if len(group) > int(max_states_per_pulse):
                idx = rng.choice(len(group), size=int(max_states_per_pulse), replace=False)
                group = [group[int(i)] for i in sorted(idx)]
            selected.extend(group)
        keys = sorted(selected, key=lambda x: (x[1], x[0], x[2]))

    if not keys:
        raise RuntimeError("State selection produced zero physical transitions")
    return keys


# ---------------------------------------------------------------------------
# Empirical support from observed masks
# ---------------------------------------------------------------------------

def read_excluded_holes(path: Optional[Path], column: str = "hole_uid") -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return set()
    if column not in rows[0]:
        raise ValueError(f"{column!r} not present in exclusion CSV: {path}")
    return {str(r[column]).strip() for r in rows if str(r.get(column, "")).strip()}


def empirical_support_scan(
    *,
    targets: Sequence[PreparedTarget],
    data_root: Path,
    pristine_channel: int,
    threshold: float,
    image_size: int,
    pulses: Sequence[int],
    excluded_holes: set[str],
    top_k: int,
    iou_thresholds: Sequence[float],
    show_progress: bool = True,
) -> List[Dict]:
    pulse_set = {int(x) for x in pulses}
    # Keep a small descending list of best matches per target.
    best: Dict[str, List[Tuple[float, float, str]]] = {t.target_id: [] for t in targets}
    counts = {
        t.target_id: {float(th): 0 for th in iou_thresholds}
        for t in targets
    }
    n_masks = 0

    holes = sorted(p for p in data_root.glob("h*") if p.is_dir())
    valid_holes = [h for h in holes if h.name not in excluded_holes]
    for hole in tqdm(
        valid_holes,
        desc="Empirical support",
        unit="hole",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        for pulse in sorted(pulse_set):
            p = hole / f"p{pulse:04d}" / "target.npy"
            if not p.exists():
                continue
            obs = load_dataset_mask(p, pristine_channel, threshold, image_size)
            n_masks += 1
            label = f"{hole.name}/p{pulse:04d}"
            for t in targets:
                val_iou = iou_binary(obs, t.mask)
                val_rmse = rmse_binary(obs, t.mask)
                for th in iou_thresholds:
                    if val_iou >= float(th):
                        counts[t.target_id][float(th)] += 1
                arr = best[t.target_id]
                arr.append((val_iou, val_rmse, label))
                arr.sort(key=lambda x: (-x[0], x[1], x[2]))
                del arr[top_k:]

    if n_masks == 0:
        raise RuntimeError("Empirical support scan found zero observed masks")

    rows = []
    for t in targets:
        matches = best[t.target_id]
        top1 = matches[0]
        row = {
            "target_id": t.target_id,
            "n_observed_masks_scanned": n_masks,
            "nearest_observed_state": top1[2],
            "nearest_observed_iou": top1[0],
            "nearest_observed_rmse": top1[1],
            "top_k": len(matches),
            "top_k_median_iou": float(np.median([x[0] for x in matches])),
            "top_k_mean_iou": float(np.mean([x[0] for x in matches])),
        }
        for th in iou_thresholds:
            row[f"count_iou_ge_{str(float(th)).replace('.', 'p')}"] = counts[t.target_id][float(th)]
            row[f"fraction_iou_ge_{str(float(th)).replace('.', 'p')}"] = (
                counts[t.target_id][float(th)] / n_masks
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# GPU scoring helpers
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def normalize_actions_for_batch(
    *,
    actions,
    next_pulses: Sequence[int],
    norm_cfg: Dict,
) -> np.ndarray:
    from transition.dataset import normalize_metadata

    b = len(next_pulses)
    n = len(actions)
    out = np.empty((b, n, 4), dtype=np.float32)
    for i, pulse in enumerate(next_pulses):
        for j, a in enumerate(actions):
            raw = np.asarray(
                [a.energy_uJ, a.action_y_um, a.action_z_um, float(pulse)],
                dtype=np.float32,
            )
            out[i, j] = normalize_metadata(raw, norm_cfg)
    return out


def current_target_metrics_torch(
    current: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    dice_weight: float,
) -> Dict[str, np.ndarray]:
    """current [B,1,H,W], target [1,1,H,W]."""
    eps = 1e-6
    diff = current - target
    l1 = diff.abs().mean(dim=(1, 2, 3))
    rmse = torch.sqrt(diff.square().mean(dim=(1, 2, 3)))

    inter = (current * target).sum(dim=(1, 2, 3))
    denom = current.sum(dim=(1, 2, 3)) + target.sum()
    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
    cost = l1_weight * l1 + dice_weight * dice_loss

    cb = current >= 0.5
    tb = target >= 0.5
    target_area = max(int(tb.sum().item()), 1)
    overall = (cb & tb).sum(dim=(1, 2, 3)).float() / target_area
    over = (cb & (~tb)).sum(dim=(1, 2, 3)).float() / target_area
    under = 1.0 - overall
    excess_px = (cb & (~tb)).sum(dim=(1, 2, 3)).float()
    missing_px = ((~cb) & tb).sum(dim=(1, 2, 3)).float()

    return {
        "l1": l1.detach().cpu().numpy(),
        "rmse": rmse.detach().cpu().numpy(),
        "dice_loss": dice_loss.detach().cpu().numpy(),
        "cost": cost.detach().cpu().numpy(),
        "overall": overall.detach().cpu().numpy(),
        "over": over.detach().cpu().numpy(),
        "under": under.detach().cpu().numpy(),
        "excess_px": excess_px.detach().cpu().numpy(),
        "missing_px": missing_px.detach().cpu().numpy(),
    }


def score_candidate_chunk(
    next_chunk: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    l1_weight: float,
    dice_weight: float,
) -> Dict[str, np.ndarray]:
    """next_chunk [B,C,1,H,W], target [1,1,1,H,W]."""
    eps = 1e-6
    diff = next_chunk - target
    l1 = diff.abs().mean(dim=(2, 3, 4))
    rmse = torch.sqrt(diff.square().mean(dim=(2, 3, 4)))

    inter = (next_chunk * target).sum(dim=(2, 3, 4))
    denom = next_chunk.sum(dim=(2, 3, 4)) + target.sum()
    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
    cost = l1_weight * l1 + dice_weight * dice_loss

    pb = next_chunk >= threshold
    tb = target >= 0.5
    target_area = max(int(tb.sum().item()), 1)
    overall = (pb & tb).sum(dim=(2, 3, 4)).float() / target_area
    over = (pb & (~tb)).sum(dim=(2, 3, 4)).float() / target_area
    under = 1.0 - overall

    return {
        "l1": l1.detach().cpu().numpy(),
        "rmse": rmse.detach().cpu().numpy(),
        "dice_loss": dice_loss.detach().cpu().numpy(),
        "cost": cost.detach().cpu().numpy(),
        "overall": overall.detach().cpu().numpy(),
        "over": over.detach().cpu().numpy(),
        "under": under.detach().cpu().numpy(),
    }


def select_candidate(
    costs: np.ndarray,
    overs: np.ndarray,
    over_limit: float,
) -> Tuple[int, int, bool, float]:
    safe = np.asarray(overs < over_limit, dtype=bool)
    safe_idx = np.flatnonzero(safe)
    if len(safe_idx):
        ordered = safe_idx[np.argsort(costs[safe_idx], kind="stable")]
        selected = int(ordered[0])
        margin = float(costs[ordered[1]] - costs[ordered[0]]) if len(ordered) >= 2 else float("nan")
        return selected, int(len(safe_idx)), False, margin

    # Same fallback as the runtime policy: minimum over-removal, then cost.
    order = np.lexsort((costs, overs))
    selected = int(order[0])
    return selected, 0, True, float("nan")


# ---------------------------------------------------------------------------
# Main model evaluation
# ---------------------------------------------------------------------------

def load_current_batch(
    rows: Sequence[Dict],
    cfg: Dict,
    image_size: int,
) -> np.ndarray:
    d = cfg["data"]
    data_root = Path(d["root"])
    pristine_channel = int(d["pristine_channel"])
    threshold = float(d["mask_threshold"])
    arr = []
    for r in rows:
        p = data_root / r["current_target_path"]
        arr.append(load_dataset_mask(p, pristine_channel, threshold, image_size))
    return np.stack(arr, axis=0)[:, None].astype(np.float32)


def evaluate_model(
    *,
    context: ModelContext,
    keys: Sequence[Tuple[str, int, int]],
    targets: Sequence[PreparedTarget],
    actions,
    device: torch.device,
    batch_size: int,
    candidate_chunk_size: int,
    amp: bool,
    threshold: float,
    l1_weight: float,
    dice_weight: float,
    over_limit: float,
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    model = context.model
    cfg = context.cfg
    norm_cfg = cfg["data"]["metadata_normalization"]
    use_amp = bool(amp) and device.type == "cuda"
    n_actions = len(actions)
    t_masks = [
        torch.from_numpy(t.mask.astype(np.float32))
        .view(1, 1, 1, t.mask.shape[0], t.mask.shape[1])
        .to(device)
        for t in targets
    ]

    per_state_target: List[Dict] = []
    action_counts: Dict[Tuple[str, str, str], int] = {}
    by_pulse_acc: Dict[Tuple[str, int], List[Dict]] = {}

    batch_starts = list(range(0, len(keys), batch_size))
    batch_iter = tqdm(
        batch_starts,
        desc=f"{context.name}: 27-action sweep",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for b0 in batch_iter:
        batch_keys = keys[b0:b0 + batch_size]
        rows = [context.rows_by_key[k] for k in batch_keys]
        current_np = load_current_batch(rows, cfg, targets[0].mask.shape[0])
        current = torch.from_numpy(current_np).to(device)
        b = current.shape[0]

        next_pulses = [int(r["next_pulse"]) for r in rows]
        if show_progress and rows:
            current_batch_pulses = [int(r["current_pulse"]) for r in rows]
            if len(set(current_batch_pulses)) == 1:
                pulse_text = f"p{current_batch_pulses[0]:02d}"
            else:
                pulse_text = f"p{min(current_batch_pulses):02d}-{max(current_batch_pulses):02d}"
            batch_iter.set_postfix_str(
                f"{pulse_text}, states {min(b0 + batch_size, len(keys))}/{len(keys)}"
            )

        action_np = normalize_actions_for_batch(
            actions=actions,
            next_pulses=next_pulses,
            norm_cfg=norm_cfg,
        )
        actions_t = torch.from_numpy(action_np).to(device)

        z_base = None
        if model.variant == "B":
            if context.z_lookup is None:
                raise RuntimeError(f"{context.name}: Transition B has no z_shock lookup")
            zs = []
            for r in rows:
                key = str(r["current_g5_sample_id"])
                if key not in context.z_lookup:
                    raise KeyError(f"{context.name}: missing z_shock for {key}")
                zs.append(np.asarray(context.z_lookup[key], dtype=np.float32))
            z_base = torch.from_numpy(np.stack(zs, axis=0)).to(device)

        # Current-state metrics are target-specific but candidate-independent.
        current_metrics = []
        for t_tensor in t_masks:
            current_metrics.append(
                current_target_metrics_torch(
                    current,
                    t_tensor[0],  # [1,1,H,W]
                    l1_weight=l1_weight,
                    dice_weight=dice_weight,
                )
            )

        # [B,T,N] arrays for target-specific candidate metrics.
        shape = (b, len(targets), n_actions)
        cand = {
            "l1": np.empty(shape, dtype=np.float32),
            "rmse": np.empty(shape, dtype=np.float32),
            "dice_loss": np.empty(shape, dtype=np.float32),
            "cost": np.empty(shape, dtype=np.float32),
            "overall": np.empty(shape, dtype=np.float32),
            "over": np.empty(shape, dtype=np.float32),
            "under": np.empty(shape, dtype=np.float32),
        }

        # Target-independent action sensitivity.
        removal_area = np.empty((b, n_actions), dtype=np.float32)
        pred_sum = torch.zeros_like(current, dtype=torch.float32)
        pred_sq_sum = torch.zeros_like(current, dtype=torch.float32)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                encoded = model.encode_state(current)

            for j0 in range(0, n_actions, candidate_chunk_size):
                j1 = min(j0 + candidate_chunk_size, n_actions)
                c = j1 - j0
                action_chunk = actions_t[:, j0:j1].reshape(b * c, 4)

                z_chunk = None
                if z_base is not None:
                    z_chunk = (
                        z_base[:, None, :]
                        .expand(-1, c, -1)
                        .reshape(b * c, z_base.shape[-1])
                    )

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model.decode_from_encoded(
                        current_mask=current,
                        encoded_state=encoded,
                        action=action_chunk,
                        z_shock=z_chunk,
                        repeats=c,
                    )

                next_chunk = out["next_pred"].float().reshape(
                    b, c, *current.shape[1:]
                )
                removal_chunk = out["removal_prob"].float().reshape(
                    b, c, *current.shape[1:]
                )

                pred_sum += next_chunk.sum(dim=1)
                pred_sq_sum += next_chunk.square().sum(dim=1)
                removal_area[:, j0:j1] = (
                    (removal_chunk >= threshold)
                    .float()
                    .sum(dim=(2, 3, 4))
                    .detach()
                    .cpu()
                    .numpy()
                )

                for ti, t_tensor in enumerate(t_masks):
                    metrics = score_candidate_chunk(
                        next_chunk,
                        t_tensor,
                        threshold=threshold,
                        l1_weight=l1_weight,
                        dice_weight=dice_weight,
                    )
                    for name in cand:
                        cand[name][:, ti, j0:j1] = metrics[name]

                del out, next_chunk, removal_chunk

        if n_actions > 1:
            variance = (pred_sq_sum - pred_sum.square() / n_actions) / (n_actions - 1)
            variance = torch.clamp(variance, min=0.0)
            mean_pixel_std = (
                torch.sqrt(variance)
                .mean(dim=(1, 2, 3))
                .detach()
                .cpu()
                .numpy()
            )
        else:
            mean_pixel_std = np.zeros(b, dtype=np.float32)

        removal_range = removal_area.max(axis=1) - removal_area.min(axis=1)

        for bi, (pk, row) in enumerate(zip(batch_keys, rows)):
            for ti, target in enumerate(targets):
                costs = cand["cost"][bi, ti]
                overs = cand["over"][bi, ti]
                selected, safe_count, no_safe, second_margin = select_candidate(
                    costs, overs, over_limit
                )
                a = actions[selected]
                cm = current_metrics[ti]

                min_rmse_any = float(cand["rmse"][bi, ti].min())
                safe_idx = np.flatnonzero(overs < over_limit)
                min_rmse_safe = (
                    float(cand["rmse"][bi, ti, safe_idx].min())
                    if len(safe_idx)
                    else float("nan")
                )

                rec = {
                    "model": context.name,
                    "runtime_mode": context.runtime_mode,
                    "variant": model.variant,
                    "hole_uid": str(row["hole_uid"]),
                    "current_pulse": int(row["current_pulse"]),
                    "next_pulse": int(row["next_pulse"]),
                    "sample_id": str(row["sample_id"]),
                    "target_id": target.target_id,
                    "current_cost": float(cm["cost"][bi]),
                    "current_rmse": float(cm["rmse"][bi]),
                    "current_l1": float(cm["l1"][bi]),
                    "current_dice_loss": float(cm["dice_loss"][bi]),
                    "current_overall_fraction": float(cm["overall"][bi]),
                    "current_under_fraction": float(cm["under"][bi]),
                    "current_over_fraction": float(cm["over"][bi]),
                    "current_excess_processed_px": int(round(float(cm["excess_px"][bi]))),
                    "current_missing_target_px": int(round(float(cm["missing_px"][bi]))),
                    "monotonic_feasible_exact": int(float(cm["excess_px"][bi]) == 0.0),
                    "monotonic_feasible_with_runtime_over_limit": int(
                        float(cm["over"][bi]) < over_limit
                    ),
                    "safe_action_count": int(safe_count),
                    "no_safe_action": int(no_safe),
                    "selected_action_id": a.action_id,
                    "selected_energy_uJ": float(a.energy_uJ),
                    "selected_action_y_um": float(a.action_y_um),
                    "selected_action_z_um": float(a.action_z_um),
                    "selected_cost": float(cand["cost"][bi, ti, selected]),
                    "selected_rmse": float(cand["rmse"][bi, ti, selected]),
                    "selected_l1": float(cand["l1"][bi, ti, selected]),
                    "selected_dice_loss": float(cand["dice_loss"][bi, ti, selected]),
                    "selected_predicted_overall_fraction": float(
                        cand["overall"][bi, ti, selected]
                    ),
                    "selected_predicted_under_fraction": float(
                        cand["under"][bi, ti, selected]
                    ),
                    "selected_predicted_over_fraction": float(
                        cand["over"][bi, ti, selected]
                    ),
                    "cost_gain_vs_current": float(
                        cm["cost"][bi] - cand["cost"][bi, ti, selected]
                    ),
                    "rmse_gain_vs_current": float(
                        cm["rmse"][bi] - cand["rmse"][bi, ti, selected]
                    ),
                    "selected_improves_cost": int(
                        cand["cost"][bi, ti, selected] < cm["cost"][bi]
                    ),
                    "selected_improves_rmse": int(
                        cand["rmse"][bi, ti, selected] < cm["rmse"][bi]
                    ),
                    "min_rmse_any_action": min_rmse_any,
                    "min_rmse_safe_action": min_rmse_safe,
                    "candidate_cost_min": float(costs.min()),
                    "candidate_cost_max": float(costs.max()),
                    "candidate_cost_range": float(costs.max() - costs.min()),
                    "candidate_cost_std": float(costs.std()),
                    "selected_vs_second_safe_cost_margin": second_margin,
                    "removal_area_range_px_across_27": float(removal_range[bi]),
                    "mean_pixel_std_next_across_27": float(mean_pixel_std[bi]),
                }
                per_state_target.append(rec)
                by_pulse_acc.setdefault(
                    (target.target_id, int(row["current_pulse"])), []
                ).append(rec)
                action_counts[
                    (target.target_id, a.action_id, context.name)
                ] = action_counts.get(
                    (target.target_id, a.action_id, context.name), 0
                ) + 1

    summary = summarize_per_target(per_state_target, context.name)
    by_pulse = summarize_by_pulse(per_state_target, context.name)

    action_rows = []
    for target in targets:
        subset = [r for r in per_state_target if r["target_id"] == target.target_id]
        n = max(len(subset), 1)
        for a in actions:
            count = action_counts.get((target.target_id, a.action_id, context.name), 0)
            action_rows.append({
                "model": context.name,
                "target_id": target.target_id,
                "action_id": a.action_id,
                "energy_uJ": float(a.energy_uJ),
                "action_y_um": float(a.action_y_um),
                "action_z_um": float(a.action_z_um),
                "selected_count": int(count),
                "selected_fraction": float(count / n),
            })

    return per_state_target, summary, by_pulse + action_rows


def summarize_per_target(rows: Sequence[Dict], model_name: str) -> List[Dict]:
    out = []
    targets = sorted({str(r["target_id"]) for r in rows})
    for tid in targets:
        rs = [r for r in rows if str(r["target_id"]) == tid]
        arr = lambda k: np.asarray([float(r[k]) for r in rs], dtype=np.float64)
        energies = arr("selected_energy_uJ")
        ys = arr("selected_action_y_um")
        zs = arr("selected_action_z_um")
        out.append({
            "model": model_name,
            "target_id": tid,
            "n_states": len(rs),
            "exact_monotonic_feasible_fraction": float(
                np.mean(arr("monotonic_feasible_exact"))
            ),
            "runtime_tolerance_monotonic_feasible_fraction": float(
                np.mean(arr("monotonic_feasible_with_runtime_over_limit"))
            ),
            "no_safe_action_fraction": float(np.mean(arr("no_safe_action"))),
            "mean_safe_action_count": float(np.mean(arr("safe_action_count"))),
            "min_safe_action_count": int(np.min(arr("safe_action_count"))),
            "mean_current_cost": float(np.mean(arr("current_cost"))),
            "mean_selected_cost": float(np.mean(arr("selected_cost"))),
            "median_selected_cost": float(np.median(arr("selected_cost"))),
            "p10_selected_cost": float(np.percentile(arr("selected_cost"), 10)),
            "mean_cost_gain_vs_current": float(np.mean(arr("cost_gain_vs_current"))),
            "fraction_selected_improves_cost": float(
                np.mean(arr("selected_improves_cost"))
            ),
            "mean_current_rmse": float(np.mean(arr("current_rmse"))),
            "mean_selected_rmse": float(np.mean(arr("selected_rmse"))),
            "mean_rmse_gain_vs_current": float(np.mean(arr("rmse_gain_vs_current"))),
            "fraction_selected_improves_rmse": float(
                np.mean(arr("selected_improves_rmse"))
            ),
            "mean_selected_over_fraction": float(
                np.mean(arr("selected_predicted_over_fraction"))
            ),
            "p95_selected_over_fraction": float(
                np.percentile(arr("selected_predicted_over_fraction"), 95)
            ),
            "mean_selected_under_fraction": float(
                np.mean(arr("selected_predicted_under_fraction"))
            ),
            "mean_candidate_cost_range": float(
                np.mean(arr("candidate_cost_range"))
            ),
            "mean_candidate_cost_std": float(
                np.mean(arr("candidate_cost_std"))
            ),
            "mean_removal_area_range_px_across_27": float(
                np.mean(arr("removal_area_range_px_across_27"))
            ),
            "mean_pixel_std_next_across_27": float(
                np.mean(arr("mean_pixel_std_next_across_27"))
            ),
            "unique_selected_actions": len({r["selected_action_id"] for r in rs}),
            "selected_energy_10_fraction": float(np.mean(energies == 10)),
            "selected_energy_15_fraction": float(np.mean(energies == 15)),
            "selected_energy_20_fraction": float(np.mean(energies == 20)),
            "selected_y_minus3_fraction": float(np.mean(ys == -3)),
            "selected_y_0_fraction": float(np.mean(ys == 0)),
            "selected_y_plus3_fraction": float(np.mean(ys == 3)),
            "selected_z_0_fraction": float(np.mean(zs == 0)),
            "selected_z_minus5_fraction": float(np.mean(zs == -5)),
            "selected_z_minus10_fraction": float(np.mean(zs == -10)),
        })
    return out


def summarize_by_pulse(rows: Sequence[Dict], model_name: str) -> List[Dict]:
    out = []
    keys = sorted({
        (str(r["target_id"]), int(r["current_pulse"]))
        for r in rows
    })
    for tid, pulse in keys:
        rs = [
            r for r in rows
            if str(r["target_id"]) == tid and int(r["current_pulse"]) == pulse
        ]
        arr = lambda k: np.asarray([float(r[k]) for r in rs], dtype=np.float64)
        out.append({
            "model": model_name,
            "target_id": tid,
            "current_pulse": pulse,
            "next_pulse": int(rs[0]["next_pulse"]),
            "n_states": len(rs),
            "exact_monotonic_feasible_fraction": float(
                np.mean(arr("monotonic_feasible_exact"))
            ),
            "runtime_tolerance_monotonic_feasible_fraction": float(
                np.mean(arr("monotonic_feasible_with_runtime_over_limit"))
            ),
            "no_safe_action_fraction": float(np.mean(arr("no_safe_action"))),
            "mean_safe_action_count": float(np.mean(arr("safe_action_count"))),
            "mean_selected_cost": float(np.mean(arr("selected_cost"))),
            "mean_cost_gain_vs_current": float(np.mean(arr("cost_gain_vs_current"))),
            "fraction_selected_improves_cost": float(
                np.mean(arr("selected_improves_cost"))
            ),
            "mean_selected_rmse": float(np.mean(arr("selected_rmse"))),
            "mean_rmse_gain_vs_current": float(np.mean(arr("rmse_gain_vs_current"))),
            "mean_selected_over_fraction": float(
                np.mean(arr("selected_predicted_over_fraction"))
            ),
            "mean_candidate_cost_range": float(
                np.mean(arr("candidate_cost_range"))
            ),
            "mean_pixel_std_next_across_27": float(
                np.mean(arr("mean_pixel_std_next_across_27"))
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_summary_metric(
    rows: Sequence[Dict],
    metric: str,
    out_path: Path,
    ylabel: str,
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = sorted({str(r["model"]) for r in rows})
    targets = sorted({str(r["target_id"]) for r in rows})
    x = np.arange(len(targets), dtype=float)
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.2), 5))
    for mi, model in enumerate(models):
        vals = []
        for tid in targets:
            match = next(
                r for r in rows
                if str(r["model"]) == model and str(r["target_id"]) == tid
            )
            vals.append(float(match[metric]))
        ax.bar(x + (mi - (len(models)-1)/2) * width, vals, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(metric.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_support_metric(rows: Sequence[Dict], metric: str, out_path: Path, ylabel: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    targets = [r["target_id"] for r in rows]
    vals = [float(r[metric]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.2), 5))
    ax.bar(targets, vals)
    ax.set_ylabel(ylabel)
    ax.set_title(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Offline target reachability / controllability evaluation. "
            "Uses observed test states, sweeps all 27 actions with Transition A/B, "
            "and never controls real hardware."
        )
    )
    ap.add_argument(
        "--config",
        default="offline_evaluation/configs/target_reachability.yaml",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate targets, checkpoints, and common states without model inference.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    user_cfg = load_yaml(config_path)
    base = config_path.parent
    show_progress = not bool(args.no_progress)

    # Make runtime package importable regardless of current working directory.
    runtime_root = Path(__file__).resolve().parents[1]
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))

    from runtime.action_registry import ActionRegistry
    from runtime.config import load_runtime_config

    runtime_cfg_path = resolve_path(
        user_cfg.get("runtime_config", "../../adaptive_runtime_config.yaml"),
        base,
    )
    runtime_cfg = load_runtime_config(runtime_cfg_path)

    ml_repo = Path(runtime_cfg["paths"]["ml_repo"])
    if not ml_repo.exists():
        raise FileNotFoundError(f"ML repository not found: {ml_repo}")
    if str(ml_repo) not in sys.path:
        sys.path.insert(0, str(ml_repo))

    target_cfg = user_cfg.get("targets", {}) or {}
    target_root = resolve_path(target_cfg["root"], base)
    target_pattern = str(target_cfg.get("pattern", "T_*/desired_target_mask.npy"))
    image_size = int(target_cfg.get("output_size", 256))

    output_root = resolve_path(
        user_cfg.get(
            "output_root",
            "F:/261006_adaptive_control_data_test/offline_evaluation",
        ),
        base,
    )
    run_dir = output_root / datetime.now().strftime(
        "target_reachability_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    targets = discover_and_prepare_targets(
        target_root=target_root,
        pattern=target_pattern,
        size=image_size,
        out_dir=run_dir,
        show_progress=show_progress,
    )

    # Resolve actual action registry from the runtime storage convention.
    action_registry_path = Path(runtime_cfg["paths"]["action_registry_csv"])
    if not action_registry_path.exists():
        raise FileNotFoundError(
            f"Action registry not found: {action_registry_path}\n"
            "Initialize the current test storage root first with init_storage.py."
        )
    action_registry = ActionRegistry(action_registry_path)
    actions = action_registry.actions

    eval_cfg = user_cfg.get("evaluation", {}) or {}
    device = resolve_device(eval_cfg.get("device", "auto"))
    split_name = str((user_cfg.get("states", {}) or {}).get("split", "test"))
    state_cfg = user_cfg.get("states", {}) or {}
    current_pulses = state_cfg.get("current_pulses")
    max_states_per_pulse = state_cfg.get("max_states_per_pulse")
    seed = int(state_cfg.get("seed", 42))

    model_specs = user_cfg.get("models") or [
        {"name": "parameter_transition_A", "runtime_mode": "action_history_only"},
        {"name": "shock_transition_B", "runtime_mode": "shock_metadata"},
    ]

    model_meta = []
    for spec in model_specs:
        name = str(spec["name"])
        mode = str(spec["runtime_mode"])
        if mode not in runtime_cfg["models"]:
            raise KeyError(f"runtime mode not found in adaptive_runtime_config.yaml: {mode}")
        run_dir_model = Path(runtime_cfg["models"][mode]["transition_run"])
        model_meta.append(
            load_context_metadata(
                name=name,
                runtime_mode=mode,
                run_dir=run_dir_model,
                ml_repo=ml_repo,
                split_name=split_name,
            )
        )

    common_keys = select_common_state_keys(
        model_meta,
        current_pulses=current_pulses,
        max_states_per_pulse=max_states_per_pulse,
        seed=seed,
    )

    validation_report = {
        "runtime_config": str(runtime_cfg_path),
        "ml_repo": str(ml_repo),
        "target_root": str(target_root),
        "n_targets": len(targets),
        "target_ids": [t.target_id for t in targets],
        "action_registry": str(action_registry_path),
        "action_registry_sha256": action_registry.sha256,
        "n_actions": len(actions),
        "state_split": split_name,
        "n_common_physical_states": len(common_keys),
        "current_pulses": sorted({k[1] for k in common_keys}),
        "device": str(device),
        "models": [
            {
                "name": m["name"],
                "runtime_mode": m["runtime_mode"],
                "transition_run": str(m["run_dir"]),
                "checkpoint": str(m["checkpoint_path"]),
                "variant": str(m["cfg"]["model"]["variant"]),
                "n_states_before_intersection": len(m["rows_by_key"]),
            }
            for m in model_meta
        ],
    }
    write_json(validation_report, run_dir / "validation_report.json")

    if args.validate_only:
        print("Validation only: PASS")
        print(f"Targets             : {len(targets)}")
        print(f"Common states       : {len(common_keys)}")
        print(f"Current pulses      : {validation_report['current_pulses']}")
        print(f"Output              : {run_dir}")
        return 0

    # Runtime selection policy is deliberately reused rather than inventing a
    # separate offline threshold.
    threshold = float(runtime_cfg["action_selection"]["candidate_threshold"])
    l1_weight = float(runtime_cfg["action_selection"]["l1_weight"])
    dice_weight = float(runtime_cfg["action_selection"]["dice_weight"])
    over_limit = float(runtime_cfg["stop_policy"]["over_removal_limit"])

    # Empirical support is model-independent.
    support_rows: List[Dict] = []
    support_cfg = user_cfg.get("empirical_support", {}) or {}
    if bool(support_cfg.get("enabled", True)):
        d0 = model_meta[0]["cfg"]["data"]
        data_root = Path(d0["root"])
        exclude_path = support_cfg.get("exclude_csv")
        excluded = read_excluded_holes(
            resolve_path(exclude_path, base) if exclude_path else None,
            str(support_cfg.get("exclude_hole_column", "hole_uid")),
        )
        support_rows = empirical_support_scan(
            targets=targets,
            data_root=data_root,
            pristine_channel=int(d0["pristine_channel"]),
            threshold=float(d0["mask_threshold"]),
            image_size=image_size,
            pulses=[int(x) for x in support_cfg.get("pulses", [50])],
            excluded_holes=excluded,
            top_k=int(support_cfg.get("top_k", 10)),
            iou_thresholds=[
                float(x)
                for x in support_cfg.get("iou_thresholds", [0.5, 0.7, 0.8])
            ],
            show_progress=show_progress,
        )
        write_csv(support_rows, run_dir / "empirical_support_summary.csv")
        plot_support_metric(
            support_rows,
            "nearest_observed_iou",
            run_dir / "plots" / "nearest_observed_iou.png",
            "Nearest observed IoU",
        )

    # Model-based one-step local controllability.
    all_per_state: List[Dict] = []
    all_summary: List[Dict] = []
    all_by_pulse: List[Dict] = []
    all_action_freq: List[Dict] = []

    batch_size = int(eval_cfg.get("batch_size", 8))
    candidate_chunk_size = int(eval_cfg.get("candidate_chunk_size", 9))
    amp = bool(eval_cfg.get("amp", True))

    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be >= 1")
    candidate_chunk_size = min(candidate_chunk_size, len(actions))

    model_pairs = list(zip(model_specs, model_meta))
    for model_index, (spec, meta) in enumerate(model_pairs, start=1):
        if show_progress:
            tqdm.write(
                f"\n[{model_index}/{len(model_pairs)}] "
                f"{spec['name']} ({spec['runtime_mode']})"
            )
        context = load_transition_context(
            name=str(spec["name"]),
            runtime_mode=str(spec["runtime_mode"]),
            run_dir=Path(meta["run_dir"]),
            ml_repo=ml_repo,
            split_name=split_name,
            device=device,
        )

        # Restrict to exactly the common physical states selected above.
        missing = [k for k in common_keys if k not in context.rows_by_key]
        if missing:
            raise RuntimeError(
                f"{context.name}: common-state mismatch; first missing={missing[:3]}"
            )

        per_state, summary, mixed = evaluate_model(
            context=context,
            keys=common_keys,
            targets=targets,
            actions=actions,
            device=device,
            batch_size=batch_size,
            candidate_chunk_size=candidate_chunk_size,
            amp=amp,
            threshold=threshold,
            l1_weight=l1_weight,
            dice_weight=dice_weight,
            over_limit=over_limit,
            show_progress=show_progress,
        )

        # evaluate_model returns by-pulse rows followed by action-frequency rows.
        by_pulse = [r for r in mixed if "current_pulse" in r]
        action_freq = [r for r in mixed if "action_id" in r]

        all_per_state.extend(per_state)
        all_summary.extend(summary)
        all_by_pulse.extend(by_pulse)
        all_action_freq.extend(action_freq)

        # Free GPU memory before loading the next Transition variant.
        del context.model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(all_per_state, run_dir / "per_state_target.csv")
    write_csv(all_summary, run_dir / "target_model_summary.csv")
    write_csv(all_by_pulse, run_dir / "target_model_by_pulse.csv")
    write_csv(all_action_freq, run_dir / "selected_action_frequency.csv")

    # Merge model-independent empirical support into a convenient report.
    support_lookup = {r["target_id"]: r for r in support_rows}
    merged = []
    for r in all_summary:
        row = dict(r)
        s = support_lookup.get(str(r["target_id"]))
        if s:
            for k, v in s.items():
                if k != "target_id":
                    row[f"support_{k}"] = v
        merged.append(row)
    write_csv(merged, run_dir / "target_reachability_controllability_report.csv")

    plot_summary_metric(
        all_summary,
        "no_safe_action_fraction",
        run_dir / "plots" / "no_safe_action_fraction.png",
        "Fraction of states with no safe action",
    )
    plot_summary_metric(
        all_summary,
        "mean_safe_action_count",
        run_dir / "plots" / "mean_safe_action_count.png",
        "Mean safe actions out of 27",
    )
    plot_summary_metric(
        all_summary,
        "fraction_selected_improves_cost",
        run_dir / "plots" / "fraction_selected_improves_cost.png",
        "Fraction of states improved by selected action",
    )
    plot_summary_metric(
        all_summary,
        "mean_candidate_cost_range",
        run_dir / "plots" / "mean_candidate_cost_range.png",
        "Mean target-cost range across 27 actions",
    )
    plot_summary_metric(
        all_summary,
        "mean_pixel_std_next_across_27",
        run_dir / "plots" / "mean_pixel_std_next_across_27.png",
        "Mean pixelwise prediction std across 27 actions",
    )

    resolved = {
        "source_config": str(config_path),
        "runtime_config": str(runtime_cfg_path),
        "output_dir": str(run_dir),
        "evaluation_policy": {
            "candidate_threshold": threshold,
            "l1_weight": l1_weight,
            "dice_weight": dice_weight,
            "over_removal_limit": over_limit,
            "candidate_selection": (
                "safe min cost; if no safe action, minimum over-removal then cost"
            ),
        },
        "state_selection": {
            "split": split_name,
            "n_states": len(common_keys),
            "current_pulses": sorted({k[1] for k in common_keys}),
            "max_states_per_pulse": max_states_per_pulse,
            "seed": seed,
        },
        "models": validation_report["models"],
        "targets": [
            {
                "target_id": t.target_id,
                "source_path": str(t.source_path),
                "source_sha256": t.source_sha256,
                "prepared_path": str(t.prepared_path),
                "prepared_sha256": t.prepared_sha256,
            }
            for t in targets
        ],
        "important_interpretation": [
            (
                "This is a one-step local controllability evaluation from observed "
                "physical states, not a synthetic multi-step closed-loop rollout."
            ),
            (
                "Transition B is not rolled forward beyond one hypothetical step "
                "because future z_shock is not available without a future observation."
            ),
            (
                "monotonic_feasible_exact checks whether the current processed region "
                "is a subset of the desired target. Once processing exceeds the target, "
                "the transition model cannot restore material."
            ),
        ],
    }
    write_json(resolved, run_dir / "resolved_evaluation.json")

    print("Offline reachability / controllability evaluation complete.")
    print(f"Targets             : {len(targets)}")
    print(f"Common test states  : {len(common_keys)}")
    print(f"Models              : {', '.join(r['model'] for r in all_summary[::len(targets)]) if all_summary else ''}")
    print(f"Output              : {run_dir}")
    print(f"Main report         : {run_dir / 'target_reachability_controllability_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
