from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Any


# =============================================================================
# FIXED PARAMS
# =============================================================================
DEFAULT_FS_HZ = 25.0

# Envelope
RMS_WIN_SEC = 1.0

# Segmentation
SMOOTH_SEC = 10.0
LOW_THR = 0.06
MIN_BREAK_SEC = 3.0
MERGE_GAP_SEC = 8.0
MIN_ACTIVE_SEC = 20.0

# Warmup start logic
WARM_ON_THR = 0.10
WARM_HOLD_SEC = 10.0

# Main split logic
MAIN_THR = 0.75
MAIN_HOLD_SEC = 30.0

# Cleanup
MIN_PHASE_SEC = 5.0
SNAP_TOL_SEC = 1.0

EPS = 1e-9
PHASE_LABELS = ["warmup", "aerobic", "anaerobic", "main_test", "cooldown"]
EXERCISE_OPTIONS = ["Exercise 1", "Exercise 2"]
EX2_ACTIVE_THR = 0.30
EX2_BREAK_THR = 0.12
EX2_MIN_BREAK_SEC = 5.0
EX2_ACTIVE_GAP_SEC = 8.0
EX2_MIN_BOUT_SEC = 20.0
EX2_COOLDOWN_MIN_SEC = 180.0
EX2_SPRINT_CORE_THR = 0.45
EX2_SPRINT_PAD_SEC = 2.0
BRAND_YELLOW = "#FFDD00"
BRAND_YELLOW_SOFT = "#FFF3A6"
BRAND_YELLOW_PALE = "#FFF9D6"
BRAND_BLACK = "#121212"
BRAND_CHARCOAL = "#2A2A2A"
BRAND_WHITE = "#FFFDF7"
BRAND_PANEL = "#FFF7CC"
BRAND_GRID = "#E6D98B"
BRAND_BREAK = "rgba(18, 18, 18, 0.08)"
EXERCISE_CHANNELS = {
    "Exercise 1": [
        "Left Quadriceps Group / uV",
        "Right Quadriceps Group / uV",
        "Left Hamstrings / uV",
        "Right Hamstrings / uV",
        "Left Gluteus / uV",
        "Right Gluteus / uV",
    ],
    "Exercise 2": [
        "Left Trapezius / uV",
        "Right Trapezius / uV",
        "Left Pectoralis / uV",
        "Right Pectoralis / uV",
        "Left Latissimus dorsi / uV",
        "Right Latissimus dorsi / uV",
        "Left Deltoids / uV",
        "Left Triceps / uV",
        "Left Biceps / uV",
        "Left Wrist extensors / uV",
        "Left Wrist flexors / uV",
        "Right Deltoids / uV",
        "Right Triceps / uV",
        "Right Biceps / uV",
        "Right Wrist extensors / uV",
        "Right Wrist flexors / uV",
        "Left Quadriceps Group / uV",
        "Right Quadriceps Group / uV",
        "Left Hamstrings / uV",
        "Right Hamstrings / uV",
        "Left Gluteus / uV",
        "Right Gluteus / uV",
        "Left Gastrocnemius / uV",
        "Left Tibialis / uV",
        "Left Soleus / uV",
        "Right Gastrocnemius / uV",
        "Right Tibialis / uV",
        "Right Soleus / uV",
    ],
}


# =============================================================================
# 1) CSV loader 
# =============================================================================
def _is_number_like(s: str) -> bool:
    s = str(s).strip() if s else ""
    if not s:
        return False
    s = s.replace(",", ".")
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _detect_delimiter(header_line: str) -> str:
    candidates = ["\t", ";", ","]
    best_sep = candidates[0]
    best_cols = 0
    for sep in candidates:
        cols = header_line.split(sep)
        if len(cols) > best_cols:
            best_cols = len(cols)
            best_sep = sep
    return best_sep


def _clean_elapsed_time_value(s: str) -> str:
    s = str(s).strip() if s else ""
    if not s:
        return ""

    s = s.replace(",", ".").replace("\t", ".")
    filtered = "".join(ch for ch in s if ch.isdigit() or ch == ".")

    if filtered.count(".") > 1:
        first_dot = filtered.find(".")
        filtered = filtered[: first_dot + 1] + filtered[first_dot + 1 :].replace(".", "")

    return filtered


def _find_header_idx(lines: list[str]) -> int | None:
    for i, ln in enumerate(lines):
        normalized = ln.strip().lstrip("\ufeff")
        if "Time" in normalized and "Elapsed time" in normalized:
            return i
    return None


def _extract_sampling_rate_hz(header_line: str) -> float | None:
    marker = "Sampling frequency:"
    if marker not in header_line:
        return None

    tail = header_line.split(marker, 1)[1].strip()
    token = []
    for ch in tail:
        if ch.isdigit() or ch == ".":
            token.append(ch)
        elif token:
            break

    if not token:
        return None

    try:
        return float("".join(token))
    except ValueError:
        return None


def _format_hms(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _load_myontec_csv_exercise_1(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = _find_header_idx(lines)
    if header_idx is None:
        raise ValueError("Could not find header line containing 'Time' and 'Elapsed time'.")

    header_line = lines[header_idx]
    sep = _detect_delimiter(header_line)
    header = header_line.split(sep)

    data_rows = []
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue
        parts = ln.split(sep)
        if len(parts) < 2:
            continue
        if not _is_number_like(parts[1]):
            continue

        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        else:
            parts = parts[: len(header)]
        data_rows.append(parts)

    if not data_rows:
        raise ValueError("No data rows found after filtering. Check separator/format.")

    df = pd.DataFrame(data_rows, columns=header)

    df["elapsed_s"] = (
        df["Elapsed time"].astype(str).str.strip().str.replace(",", ".", regex=False).astype(float)
    )

    for c in df.columns:
        if c in ("Time", "Elapsed time", "elapsed_s"):
            continue
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.replace({"": np.nan, "None": np.nan})
        df[c] = pd.to_numeric(s, errors="coerce")

    df = df.sort_values("elapsed_s").reset_index(drop=True)
    return df


def _load_myontec_csv_exercise_2(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = _find_header_idx(lines)
    if header_idx is None:
        raise ValueError("Could not find header line containing 'Time' and 'Elapsed time'.")

    header_line = lines[header_idx]
    sampling_rate_hz = _extract_sampling_rate_hz(header_line)
    raw_header = header_line.split(";")
    if "Marker" in raw_header:
        marker_idx = raw_header.index("Marker")
        header = raw_header[: marker_idx + 1]
    else:
        header = raw_header

    data_rows = []
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue

        parts = ln.split(";")
        if len(parts) < 2:
            continue

        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        else:
            parts = parts[: len(header)]

        elapsed_clean = _clean_elapsed_time_value(parts[1])
        if not _is_number_like(elapsed_clean):
            continue
        parts[1] = elapsed_clean

        data_rows.append(parts)

    if not data_rows:
        raise ValueError("No data rows found after filtering. Check separator/format.")

    df = pd.DataFrame(data_rows, columns=header)
    df["elapsed_s"] = pd.to_numeric(df["Elapsed time"], errors="coerce")
    df = df.dropna(subset=["elapsed_s"]).copy()

    for c in df.columns:
        if c in ("Time", "Elapsed time", "elapsed_s", "Marker"):
            continue
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.replace({"": np.nan, "None": np.nan})
        df[c] = pd.to_numeric(s, errors="coerce")

    df = df.sort_values("elapsed_s").reset_index(drop=True)
    if sampling_rate_hz is not None:
        df.attrs["sampling_rate_hz"] = sampling_rate_hz
    return df


def load_myontec_csv_from_bytes(data: bytes, exercise_name: str = "Exercise 1") -> pd.DataFrame:
    if exercise_name == "Exercise 1":
        return _load_myontec_csv_exercise_1(data)
    if exercise_name == "Exercise 2":
        return _load_myontec_csv_exercise_2(data)
    raise ValueError(f"Unsupported exercise: {exercise_name}")


def load_myontec_file(uploaded_file: Any, exercise_name: str = "Exercise 1") -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "") or "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            raw_df = pd.read_excel(uploaded_file, header=None, dtype=str)
        except ImportError as e:
            raise ValueError(f"Excel support is unavailable: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}") from e

        # Rebuild the sheet as delimiter-separated text so both Excel and CSV
        # go through the same exercise-specific parsing rules.
        lines = []
        for row in raw_df.fillna("").itertuples(index=False, name=None):
            lines.append(";".join(str(cell).strip() for cell in row))
        data = "\n".join(lines).encode("utf-8")
        return load_myontec_csv_from_bytes(data, exercise_name=exercise_name)

    return load_myontec_csv_from_bytes(uploaded_file.getvalue(), exercise_name=exercise_name)


# =============================================================================
# 2) EMG intensity envelope
# =============================================================================
def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    win = max(1, int(win))
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    x2 = xp * xp
    kernel = np.ones(win, dtype=np.float32) / float(win)
    m = np.convolve(x2, kernel, mode="valid")
    return np.sqrt(m + 1e-8).astype(np.float32)


def robust_norm01(x: np.ndarray, lo_q: int = 5, hi_q: int = 95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_q), np.percentile(x, hi_q)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def compute_emg_intensity(df: pd.DataFrame, fs: float, win_sec: float, emg_cols: list[str]):
    missing = [c for c in emg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EMG columns: {missing}")

    t = df["elapsed_s"].to_numpy(dtype=np.float32)
    win = int(round(win_sec * fs))

    env = {}
    for c in emg_cols:
        x = df[c].to_numpy(dtype=np.float32)

        # Fill NaNs with per-channel median
        if np.isfinite(x).any():
            fill = float(np.nanmedian(x[np.isfinite(x)]))
        else:
            fill = 0.0
        x = np.nan_to_num(x, nan=fill)

        env[c] = moving_rms(x, win=win)

    emg_env_df = pd.DataFrame(env)
    intensity_raw = emg_env_df.to_numpy(dtype=np.float32).mean(axis=1)
    intensity = robust_norm01(intensity_raw, 5, 95)

    return t, emg_env_df, intensity


# =============================================================================
# 3) Segmentation helpers
# =============================================================================
def smooth_ma(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), k, mode="same").astype(np.float32)


def find_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if not v else i + 1
            regions.append((start, end))
            start = None
    return regions


def merge_nearby_regions(
    regions: list[tuple[int, int]],
    fs: float,
    gap_sec: float = 8.0,
    min_len_sec: float = 0.0
) -> list[tuple[int, int]]:
    if not regions:
        return []
    gap = int(round(gap_sec * fs))
    min_len = int(round(min_len_sec * fs))

    regions = sorted(regions, key=lambda r: r[0])
    merged = [regions[0]]
    for a, b in regions[1:]:
        pa, pb = merged[-1]
        if a <= pb + gap:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))

    if min_len > 0:
        merged = [(a, b) for (a, b) in merged if (b - a) >= min_len]
    return merged


def clip_phases_around_breaks(
    phases: list[dict[str, Any]],
    breaks: list[tuple[int, int]],
    fs: float,
    min_phase_sec: float = 5.0
) -> list[dict[str, Any]]:
    if not phases:
        return []
    breaks = sorted(breaks, key=lambda x: x[0])
    min_len = int(round(min_phase_sec * fs))

    clipped = []
    for p in phases:
        i0, i1 = int(p["i0"]), int(p["i1"])

        for (b0, b1) in breaks:
            if i1 <= b0 or i0 >= b1:
                continue

            if i0 >= b0 and i1 <= b1:
                i0, i1 = 0, 0
                break

            if i0 < b0 < i1 <= b1:
                i1 = b0
            elif b0 <= i0 < b1 < i1:
                i0 = b1
            elif i0 < b0 and i1 > b1:
                left_len = b0 - i0
                right_len = i1 - b1
                if right_len >= left_len:
                    i0 = b1
                else:
                    i1 = b0

        if (i1 - i0) >= min_len:
            clipped.append({**p, "i0": i0, "i1": i1})

    clipped.sort(key=lambda p: p["i0"])
    return clipped


def snap_phases_to_breaks_strict(
    phases: list[dict[str, Any]],
    breaks: list[tuple[int, int]],
    fs: float,
    tol_sec: float = 0.2,
    warmup_label: str = "warmup"
) -> list[dict[str, Any]]:
    if not phases or not breaks:
        return phases

    tol = int(round(tol_sec * fs))
    breaks = sorted(breaks, key=lambda x: x[0])

    out = []
    for p in phases:
        i0, i1 = int(p["i0"]), int(p["i1"])
        for (b0, b1) in breaks:
            if abs(i1 - b0) <= tol:
                i1 = b0
            if abs(i0 - b1) <= tol:
                i0 = b1
        out.append({**p, "i0": i0, "i1": i1})

    _, b1_first = breaks[0]
    for p in out:
        if p.get("label") == warmup_label:
            if abs(int(p["i0"]) - b1_first) <= tol:
                p["i0"] = int(b1_first)

    out = [p for p in out if int(p["i1"]) > int(p["i0"])]
    out.sort(key=lambda p: int(p["i0"]))
    return out


def attach_times_to_phases(phases: list[dict[str, Any]], t: np.ndarray, fs: float) -> list[dict[str, Any]]:
    t = np.asarray(t, dtype=np.float32)
    out = []
    n = len(t)

    for p in phases:
        i0, i1 = int(p["i0"]), int(p["i1"])
        if i0 < 0 or i1 <= i0 or i1 > n:
            continue

        t0 = float(t[i0])
        t1_last = float(t[i1 - 1])
        t1 = float(t[i1]) if i1 < n else float(t1_last + (1.0 / fs))  # end-exclusive boundary

        out.append({**p, "t0": t0, "t1": t1, "t1_last": t1_last})

    return out


def attach_times_to_breaks(breaks: list[tuple[int, int]], t: np.ndarray, fs: float) -> list[dict[str, Any]]:
    t = np.asarray(t, dtype=np.float32)
    out = []
    n = len(t)

    for (b0, b1) in breaks:
        b0, b1 = int(b0), int(b1)
        if b0 < 0 or b1 <= b0 or b1 > n:
            continue

        t0 = float(t[b0])
        t1_last = float(t[b1 - 1])
        t1 = float(t[b1]) if b1 < n else float(t1_last + (1.0 / fs))  # end-exclusive boundary

        out.append({"i0": b0, "i1": b1, "t0": t0, "t1": t1, "t1_last": t1_last})

    return out


def phase_time_defaults(phases_t: list[dict[str, Any]]) -> dict[str, dict[str, float | bool]]:
    defaults = {
        label: {"enabled": False, "t0": 0.0, "t1": 0.0}
        for label in PHASE_LABELS
    }
    for p in phases_t:
        label = str(p["label"])
        if label not in defaults:
            continue
        defaults[label] = {
            "enabled": True,
            "t0": round(float(p["t0"]), 2),
            "t1": round(float(p["t1"]), 2),
        }
    return defaults


def build_manual_phases(
    t: np.ndarray,
    fs: float,
    manual_specs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    t = np.asarray(t, dtype=np.float32)
    n = len(t)
    if n == 0:
        return [], ["No time axis is available for manual phase boundaries."]

    time_max = float(t[-1] + (1.0 / fs))
    phases = []
    errors = []

    for spec in manual_specs:
        label = str(spec["label"])
        if not bool(spec.get("enabled", False)):
            continue

        t0 = float(np.clip(spec.get("t0", 0.0), 0.0, time_max))
        t1 = float(np.clip(spec.get("t1", 0.0), 0.0, time_max))
        if t1 <= t0:
            errors.append(f"{label}: end time must be greater than start time.")
            continue

        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="left"))
        i1 = min(i1, n)

        if i0 >= n or i1 <= i0:
            errors.append(f"{label}: selected range does not map to a valid sample interval.")
            continue

        phases.append({"label": label, "i0": i0, "i1": i1})

    phases.sort(key=lambda p: (int(p["i0"]), PHASE_LABELS.index(str(p["label"])) if str(p["label"]) in PHASE_LABELS else 999))

    for prev, curr in zip(phases, phases[1:]):
        if int(prev["i1"]) > int(curr["i0"]):
            errors.append(
                f"{prev['label']} overlaps {curr['label']}. Manual phases must not overlap."
            )

    return phases, errors


def segment_phases_from_emg_exercise_1(
    t: np.ndarray,
    I: np.ndarray,
    fs: float = 25.0
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray]:
    t = np.asarray(t, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)

    I_s = smooth_ma(I, win=int(round(SMOOTH_SEC * fs)))

    # A) Breaks
    low_mask = I_s < LOW_THR
    low_regions = find_regions(low_mask)
    min_break_len = int(round(MIN_BREAK_SEC * fs))
    breaks = [(a, b) for (a, b) in low_regions if (b - a) >= min_break_len]
    breaks = merge_nearby_regions(breaks, fs=fs, gap_sec=MERGE_GAP_SEC, min_len_sec=MIN_BREAK_SEC)

    # B) Active regions
    active = np.ones_like(low_mask, dtype=bool)
    for (a, b) in breaks:
        active[a:b] = False
    active_regions = find_regions(active)

    min_active_len = int(round(MIN_ACTIVE_SEC * fs))
    active_regions = [(a, b) for (a, b) in active_regions if (b - a) >= min_active_len]
    if not active_regions:
        return [], breaks, I_s

    # Main = longest active region
    lengths = [b - a for (a, b) in active_regions]
    main_idx = int(np.argmax(lengths))
    am, bm = active_regions[main_idx]

    phases = []

    # C) Warmup candidate before main
    warm_candidates = [r for i, r in enumerate(active_regions) if i < main_idx]
    if warm_candidates:
        aw, bw = warm_candidates[0]
        hold_len = int(round(WARM_HOLD_SEC * fs))
        warm_seg = I_s[aw:bw]
        above = warm_seg >= WARM_ON_THR

        true_start = aw
        for i in range(0, max(0, len(above) - hold_len)):
            if above[i: i + hold_len].all():
                true_start = aw + i
                break

        phases.append({"label": "warmup", "i0": true_start, "i1": bw})

    # D) Cooldown candidate after main
    cool_candidates = [r for i, r in enumerate(active_regions) if i > main_idx]
    if cool_candidates:
        ac, bc = cool_candidates[-1]
        phases.append({"label": "cooldown", "i0": ac, "i1": bc})

    # E) Split main into aerobic/anaerobic
    Im = I_s[am:bm]
    above = Im >= MAIN_THR
    hold_len = int(round(MAIN_HOLD_SEC * fs))

    split_i = None
    for i in range(0, max(0, len(above) - hold_len)):
        if above[i: i + hold_len].all():
            split_i = am + i
            break

    if split_i is None:
        phases.append({"label": "main_test", "i0": am, "i1": bm})
    else:
        phases.append({"label": "aerobic", "i0": am, "i1": split_i})
        phases.append({"label": "anaerobic", "i0": split_i, "i1": bm})

    phases = sorted(phases, key=lambda p: p["i0"])
    return phases, breaks, I_s


def _label_sequential_regions(
    regions: list[tuple[int, int]],
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        {"label": f"{prefix}_{idx}", "i0": a, "i1": b}
        for idx, (a, b) in enumerate(regions, start=1)
    ]


def _trim_region_to_core(
    x: np.ndarray,
    region: tuple[int, int],
    thr: float,
    fs: float,
    pad_sec: float = 0.0,
) -> tuple[int, int]:
    a, b = region
    if b <= a:
        return region

    seg = np.asarray(x[a:b], dtype=np.float32)
    core_regions = find_regions(seg >= thr)
    if not core_regions:
        return region

    ca, cb = max(core_regions, key=lambda r: r[1] - r[0])
    pad = int(round(pad_sec * fs))
    na = max(a, a + ca - pad)
    nb = min(b, a + cb + pad)
    if nb <= na:
        return region
    return na, nb


def segment_phases_from_emg_exercise_2(
    t: np.ndarray,
    I: np.ndarray,
    fs: float = 100.0,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray]:
    t = np.asarray(t, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)

    I_s = smooth_ma(I, win=int(round(SMOOTH_SEC * fs)))

    low_regions = find_regions(I_s < EX2_BREAK_THR)
    breaks = merge_nearby_regions(
        low_regions,
        fs=fs,
        gap_sec=MERGE_GAP_SEC,
        min_len_sec=EX2_MIN_BREAK_SEC,
    )

    active_regions = find_regions(I_s >= EX2_ACTIVE_THR)
    active_regions = merge_nearby_regions(
        active_regions,
        fs=fs,
        gap_sec=EX2_ACTIVE_GAP_SEC,
        min_len_sec=EX2_MIN_BOUT_SEC,
    )
    if not active_regions:
        return [], breaks, I_s

    phases: list[dict[str, Any]] = []

    warmup_region = active_regions[0]
    phases.append({"label": "warmup", "i0": warmup_region[0], "i1": warmup_region[1]})

    remaining_regions = active_regions[1:]
    cooldown_region: tuple[int, int] | None = None
    if remaining_regions:
        last_region = remaining_regions[-1]
        last_duration_sec = (last_region[1] - last_region[0]) / fs
        starts_late = len(t) > 0 and float(t[last_region[0]]) >= (0.70 * float(t[-1]))
        if last_duration_sec >= EX2_COOLDOWN_MIN_SEC and starts_late:
            cooldown_region = last_region
            remaining_regions = remaining_regions[:-1]

    sprint_regions: list[tuple[int, int]] = []
    main_regions: list[tuple[int, int]] = []
    if remaining_regions:
        if len(remaining_regions) == 1:
            main_regions = remaining_regions
        else:
            gaps = [remaining_regions[i + 1][0] - remaining_regions[i][1] for i in range(len(remaining_regions) - 1)]
            split_idx = int(np.argmax(gaps)) + 1
            sprint_regions = remaining_regions[:split_idx]
            main_regions = remaining_regions[split_idx:]

            if not sprint_regions or not main_regions:
                durations = np.asarray([b - a for (a, b) in remaining_regions], dtype=np.int32)
                long_mask = durations >= int(round(90.0 * fs))
                first_long = int(np.argmax(long_mask)) if long_mask.any() else max(1, len(remaining_regions) // 2)
                sprint_regions = remaining_regions[:first_long]
                main_regions = remaining_regions[first_long:]

    sprint_regions = [
        _trim_region_to_core(
            I_s,
            region,
            thr=EX2_SPRINT_CORE_THR,
            fs=fs,
            pad_sec=EX2_SPRINT_PAD_SEC,
        )
        for region in sprint_regions
    ]

    phases.extend(_label_sequential_regions(sprint_regions, "sprint"))
    phases.extend(_label_sequential_regions(main_regions, "main"))

    if cooldown_region is not None:
        phases.append({"label": "cooldown", "i0": cooldown_region[0], "i1": cooldown_region[1]})

    phases = sorted(phases, key=lambda p: int(p["i0"]))
    return phases, breaks, I_s


def segment_phases_from_emg(
    t: np.ndarray,
    I: np.ndarray,
    fs: float = 25.0,
    exercise_name: str = "Exercise 1",
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray]:
    if exercise_name == "Exercise 1":
        return segment_phases_from_emg_exercise_1(t, I, fs=fs)
    if exercise_name == "Exercise 2":
        return segment_phases_from_emg_exercise_2(t, I, fs=fs)
    raise ValueError(f"Unsupported exercise: {exercise_name}")


# =============================================================================
# 4) Plotly figures 
# =============================================================================
def build_timeline_figure(
    t: np.ndarray,
    I: np.ndarray,
    I_s: np.ndarray,
    phases_t: list[dict[str, Any]],
    breaks_t: list[dict[str, Any]],
    exercise_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=I,
            mode="lines",
            name="EMG intensity (norm)",
            opacity=0.35,
            line=dict(color="rgba(255, 221, 0, 0.28)", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=I_s,
            mode="lines",
            name="EMG intensity (smoothed)",
            line=dict(color=BRAND_BLACK, width=2.5),
        )
    )
    main_start_x: list[float | None] = []
    main_start_y: list[float | None] = []
    main_start_text: list[str | None] = []
    main_end_x: list[float | None] = []
    main_end_y: list[float | None] = []
    main_end_text: list[str | None] = []

    for j, b in enumerate(breaks_t):
        fig.add_vrect(
            x0=b["t0"], x1=b["t1"],
            fillcolor="rgba(0, 0, 0, 0)",
            line_width=0,
        )

    phase_fill = {
        "warmup": "rgba(42, 42, 42, 0.08)",
        "aerobic": "rgba(255, 221, 0, 0.12)",
        "anaerobic": "rgba(18, 18, 18, 0.08)",
        "cooldown": "rgba(42, 42, 42, 0.08)",
        "main_test": "rgba(255, 221, 0, 0.10)",
        "sprint": "rgba(255, 221, 0, 0.10)",
        "main": "rgba(255, 221, 0, 0.10)",
    }

    for p in phases_t:
        label = str(p["label"])
        is_main = label.startswith("main_")
        should_shade = False
        fillcolor = "rgba(255, 221, 0, 0.07)"

        if exercise_name == "Exercise 1":
            if label in {"warmup", "aerobic", "anaerobic", "cooldown"}:
                should_shade = True
                fillcolor = phase_fill[label]
        elif exercise_name == "Exercise 2":
            if label in {"warmup", "cooldown"}:
                should_shade = True
                fillcolor = phase_fill[label]
            elif label.startswith("sprint_"):
                should_shade = True
                fillcolor = phase_fill["sprint"]
            elif is_main:
                should_shade = True
                fillcolor = phase_fill["main"]

        if should_shade:
            fig.add_vrect(
                x0=p["t0"], x1=p["t1"],
                fillcolor=fillcolor,
                line_width=0,
            )

        if is_main:
            main_start_x.extend([float(p["t0"]), float(p["t0"]), None])
            main_start_y.extend([0.0, 1.2, None])
            main_start_text.extend([f"{label} start: {float(p['t0']):.1f} s"] * 2 + [None])
            main_end_x.extend([float(p["t1"]), float(p["t1"]), None])
            main_end_y.extend([0.0, 1.2, None])
            main_end_text.extend([f"{label} end: {float(p['t1']):.1f} s"] * 2 + [None])
            fig.add_annotation(
                x=(float(p["t0"]) + float(p["t1"])) / 2.0,
                y=1.07,
                text=label,
                showarrow=False,
                xanchor="center",
            )
        else:
            fig.add_vline(x=p["t0"], line_width=1, line_dash="dash", opacity=0.6)
            fig.add_annotation(
                x=p["t0"], y=1.06, text=label,
                showarrow=False, textangle=-90, xanchor="left",
            )

    if main_start_x:
        fig.add_trace(
            go.Scatter(
                x=main_start_x,
                y=main_start_y,
                mode="lines",
                line=dict(color="rgba(18, 18, 18, 0.28)", width=1, dash="dash"),
                hovertext=main_start_text,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )
    if main_end_x:
        fig.add_trace(
            go.Scatter(
                x=main_end_x,
                y=main_end_y,
                mode="lines",
                line=dict(color="rgba(18, 18, 18, 0.28)", width=1, dash="dash"),
                hovertext=main_end_text,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=45, b=70),
        template="plotly_white",
        paper_bgcolor=BRAND_WHITE,
        plot_bgcolor=BRAND_WHITE,
        font=dict(color=BRAND_CHARCOAL),
        xaxis_title="Time (s)",
        yaxis_title="Intensity (0..1)",
        yaxis=dict(range=[0.0, 1.2]),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BRAND_GRID, zeroline=False, linecolor=BRAND_CHARCOAL, mirror=False)
    fig.update_yaxes(showgrid=True, gridcolor=BRAND_GRID, zeroline=False, linecolor=BRAND_CHARCOAL, mirror=False)
    return fig


def build_env_figure(t: np.ndarray, emg_env_df: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for c in selected:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=emg_env_df[c].to_numpy(),
                mode="lines",
                name=c,
                line=dict(width=2),
            )
        )
    fig.update_layout(
        height=500,
        template="plotly_white",
        paper_bgcolor=BRAND_WHITE,
        plot_bgcolor=BRAND_WHITE,
        font=dict(color=BRAND_CHARCOAL),
        margin=dict(l=20, r=20, t=35, b=70),
        xaxis_title="Time (s)",
        yaxis_title="RMS (uV)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BRAND_GRID, zeroline=False, linecolor=BRAND_CHARCOAL, mirror=False)
    fig.update_yaxes(showgrid=True, gridcolor=BRAND_GRID, zeroline=False, linecolor=BRAND_CHARCOAL, mirror=False)
    return fig


# =============================================================================
# 5) Compute Load Metrics
# =============================================================================
def compute_myontec_load_signals(emg_env_df: pd.DataFrame, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements load calculations based on the ENVELOPE channels (uV-like).

    Inputs:
      - emg_env_df: DataFrame with 6 envelope channels (uV-like)
      - fs: sampling rate (Hz)

    Outputs:
      - muscle_load: momentary muscle activity, per-sample (NaN for first ~1s)
      - sum_all_channels_x_dt: per-sample sum across channels of (VALUE * dt)
    """
    dt = 1.0 / float(fs)
    X = emg_env_df.to_numpy(dtype=np.float64)  # shape (N, CH)
    n, ch = X.shape

    # Per-sample contribution to Total Muscle Load numerator:
    # sum_ch( VALUE_ch,s * dt )
    sum_all_channels_x_dt = np.sum(X * dt, axis=1)  # shape (N,)

    # Muscle Load (momentary): sliding 1-second window AUC summed over channels, scaled by 0.6
    # window length in samples = fs (1 second)
    w = int(round(fs))
    if w < 1:
        w = 1

    # Compute sliding window sum for each channel, then sum channels.
    muscle_load = np.full(n, np.nan, dtype=np.float64)
    if n >= w:
        kernel = np.ones(w, dtype=np.float64)
        window_sum_all = np.zeros(n, dtype=np.float64)
        for j in range(ch):
            window_sum = np.convolve(X[:, j], kernel, mode="same")
            window_sum_all += window_sum

        # edges include partial windows for "same" convolution.
        # therefore we blank out the first w samples
        # so min/max are computed on stable full windows only.
        window_auc_all = window_sum_all * dt
        muscle_load[:] = window_auc_all * 0.6
        muscle_load[:w] = np.nan

    return muscle_load, sum_all_channels_x_dt


def phases_table_with_loads(
    phases_t: list[dict[str, Any]],
    muscle_load: np.ndarray,
    sum_all_channels_x_dt: np.ndarray,
    fs: float
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per phase, containing the following columns:
      label, t_start, t_end, min load, AVG load, max load, cumulative load

    Definitions:
      - min/avg/max load: min/mean/max of 'Muscle Load' within [i0, i1), ignoring NaNs
      - cumulative load: Total Muscle Load within [i0, i1) = sum(sum_all_channels_x_dt) / 100
    """
    rows = []
    for p in phases_t:
        label = str(p["label"])
        i0, i1 = int(p["i0"]), int(p["i1"])

        seg_ml = muscle_load[i0:i1]
        seg_contrib = sum_all_channels_x_dt[i0:i1]

        # min/max on stable values only
        seg_ml_f = seg_ml[np.isfinite(seg_ml)]
        min_load = float(np.min(seg_ml_f)) if seg_ml_f.size else np.nan
        avg_load = float(np.mean(seg_ml_f)) if seg_ml_f.size else np.nan
        max_load = float(np.max(seg_ml_f)) if seg_ml_f.size else np.nan

        cumulative_load = float(np.sum(seg_contrib) / 100.0) if np.isfinite(seg_contrib).any() else np.nan

        rows.append({
            "label": label,
            "t_start": round(float(p["t0"]), 2),
            "t_end": round(float(p["t1"]), 2),
            "min load": min_load,
            "AVG load": avg_load,
            "max load": max_load,
            "cumulative load": cumulative_load,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6) Streamlit UI
# =============================================================================
st.set_page_config(page_title="Myontec EMG Segmentation Analysis Dashboard", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --brand-yellow: #FFDD00;
        --brand-yellow-soft: #FFF3A6;
        --brand-yellow-pale: #FFF9D6;
        --brand-black: #121212;
        --brand-charcoal: #2A2A2A;
        --brand-white: #FFFDF7;
        --brand-grid: #E6D98B;
        --brand-panel: #FFF7CC;
      }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background:
          radial-gradient(circle at top right, rgba(255, 221, 0, 0.18), transparent 26rem),
          linear-gradient(180deg, #fffef7 0%, #fff9dc 100%);
        color: var(--brand-charcoal);
      }
      [data-testid="stSidebar"] {
        background:
          linear-gradient(180deg, rgba(255, 221, 0, 0.28), rgba(255, 255, 255, 0.95));
        border-right: 1px solid rgba(18, 18, 18, 0.08);
      }
      .block-container { padding-top: 2.0rem; margin-top: 2rem }
      .titlebar {
        padding: 18px 20px;
        border-radius: 18px;
        background:
          linear-gradient(120deg, rgba(255, 221, 0, 0.92), rgba(255, 243, 166, 0.88));
        border: 1px solid rgba(18, 18, 18, 0.10);
        box-shadow: 0 18px 40px rgba(18, 18, 18, 0.08);
        color: var(--brand-black);
      }
      .muted { opacity: 0.72; }
      h1, h2, h3, h4, label, .stMarkdown, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--brand-charcoal);
      }
      [data-testid="stMetric"] {
        background: rgba(255, 253, 247, 0.92);
        border: 1px solid rgba(18, 18, 18, 0.08);
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 10px 24px rgba(18, 18, 18, 0.05);
      }
      [data-testid="stFileUploader"], [data-testid="stNumberInput"], [data-baseweb="select"], .stExpander {
        background: rgba(255, 253, 247, 0.86);
        border-radius: 14px;
      }
      [data-baseweb="input"] input, [data-baseweb="select"] > div {
        background: rgba(255, 253, 247, 0.96) !important;
        color: var(--brand-black) !important;
        border-color: rgba(18, 18, 18, 0.14) !important;
      }
      .stButton > button, .stDownloadButton > button {
        background: var(--brand-black);
        color: var(--brand-yellow);
        border: 1px solid var(--brand-black);
        border-radius: 999px;
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        background: #000000;
        color: var(--brand-white);
        border-color: #000000;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid rgba(18, 18, 18, 0.08);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(255, 253, 247, 0.96);
      }
      .js-plotly-plot .plotly .modebar-btn {
        padding: 8px 10px;
      }
      .js-plotly-plot .plotly .modebar-btn svg {
        width: 20px;
        height: 20px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="titlebar">
      <div style="font-size: 20px; font-weight: 700;">EMG Phase Segmentation Dashboard</div>
      <div class="muted">Upload EMG data, review intensity trends, and inspect automated phase segmentation.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Controls")
fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=1.0, max_value=500.0, value=DEFAULT_FS_HZ, step=1.0)
exercise_name = st.sidebar.selectbox(
    "Exercise",
    options=EXERCISE_OPTIONS,
    index=0,
)

uploaded = st.file_uploader("Upload file", type=["csv", "txt", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a CSV, TXT, or Excel file to view the dashboard.")
    st.stop()

# Load
try:
    df = load_myontec_file(uploaded, exercise_name=exercise_name)
except (ValueError, UnicodeDecodeError) as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

effective_fs = float(df.attrs.get("sampling_rate_hz", fs))
if "sampling_rate_hz" in df.attrs:
    st.sidebar.caption(f"Using sampling rate from file: {effective_fs:.1f} Hz")

# Compute intensity/envelopes
try:
    emg_cols = EXERCISE_CHANNELS[exercise_name]
    t, emg_env_df, emg_intensity = compute_emg_intensity(df, fs=effective_fs, win_sec=RMS_WIN_SEC, emg_cols=emg_cols)
except (ValueError, KeyError) as e:
    st.error(f"Failed to compute EMG intensity: {e}")
    st.stop()

duration_s = float(t[-1]) if len(t) else 0.0
time_max = float(t[-1] + (1.0 / fs)) if len(t) else 0.0

# Segment phases
auto_phases, breaks, I_s = segment_phases_from_emg(
    t,
    emg_intensity,
    fs=effective_fs,
    exercise_name=exercise_name,
)

# Clip + snap
auto_phases = clip_phases_around_breaks(auto_phases, breaks, fs=effective_fs, min_phase_sec=MIN_PHASE_SEC)
auto_phases = snap_phases_to_breaks_strict(auto_phases, breaks, fs=effective_fs, tol_sec=SNAP_TOL_SEC)

# Attach times
auto_phases_t = attach_times_to_phases(auto_phases, t, fs=effective_fs)
breaks_t = attach_times_to_breaks(breaks, t, fs=effective_fs)

phase_mode = st.sidebar.radio("Phase boundaries", ["Auto", "Manual"], index=0)
phases = auto_phases

if phase_mode == "Manual":
    st.sidebar.caption("Set exact phase start and end times in seconds.")
    defaults = phase_time_defaults(auto_phases_t)
    manual_specs = []

    for label in PHASE_LABELS:
        default_enabled = bool(defaults[label]["enabled"])
        enabled = st.sidebar.checkbox(
            f"Use {label}",
            value=default_enabled,
            key=f"manual_phase_enabled_{label}",
        )
        t0_default = float(defaults[label]["t0"])
        t1_default = float(defaults[label]["t1"])
        if enabled:
            t0 = st.sidebar.number_input(
                f"{label} start (s)",
                min_value=0.0,
                max_value=time_max,
                value=min(t0_default, time_max),
                step=0.5,
                key=f"manual_phase_t0_{label}",
            )
            t1 = st.sidebar.number_input(
                f"{label} end (s)",
                min_value=0.0,
                max_value=time_max,
                value=min(t1_default if t1_default > 0 else time_max, time_max),
                step=0.5,
                key=f"manual_phase_t1_{label}",
            )
        else:
            t0 = t0_default
            t1 = t1_default

        manual_specs.append({"label": label, "enabled": enabled, "t0": t0, "t1": t1})

    manual_phases, manual_errors = build_manual_phases(t, fs=effective_fs, manual_specs=manual_specs)
    if manual_errors:
        st.sidebar.error("Manual phase boundaries are invalid.")
        for msg in manual_errors:
            st.sidebar.caption(msg)
    else:
        phases = manual_phases

phases_t = attach_times_to_phases(phases, t, fs=effective_fs)

# Breaks table 
def breaks_table(breaks_t: list[dict[str, Any]], fs: float) -> pd.DataFrame:
    rows = []
    for b in breaks_t:
        dur = (int(b["i1"]) - int(b["i0"])) / fs
        rows.append(
            {
                "t_start": round(b["t0"], 2),
                "t_end": round(b["t1"], 2),
                "duration_s": round(dur, 2),
                "i0": int(b["i0"]),
                "i1": int(b["i1"]),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Compute load signals using the ENVELOPES
# =============================================================================
muscle_load, sum_all_channels_x_dt = compute_myontec_load_signals(emg_env_df, fs=effective_fs)

# Phase table 
ph_df = phases_table_with_loads(phases_t, muscle_load, sum_all_channels_x_dt, fs=effective_fs)
br_df = breaks_table(breaks_t, fs=effective_fs)

# KPI load metrics excluding detected breaks
non_break_mask = np.ones(len(muscle_load), dtype=bool)
for b in breaks_t:
    i0 = max(0, min(int(b["i0"]), len(non_break_mask)))
    i1 = max(0, min(int(b["i1"]), len(non_break_mask)))
    non_break_mask[i0:i1] = False

avg_load_ex_breaks = float(np.nanmean(muscle_load[non_break_mask])) if non_break_mask.any() else np.nan
cumulative_load_ex_breaks = (
    float(np.nansum(sum_all_channels_x_dt[non_break_mask])) if non_break_mask.any() else np.nan
)

# KPI row
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Current Date/Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
c2.metric("Duration", _format_hms(duration_s))
c3.metric("Phases", f"{len(ph_df)}")
# c4.metric("Average Load", f"{float(np.nanmean(muscle_load)):.3f}")
c4.metric("Average Load", f"{avg_load_ex_breaks:.3f}")
c5.metric("Energy Expenditure (kcal)", "XXXX")
# c6.metric("Cumulative Load (µV·s)", f"{float(np.nansum(sum_all_channels_x_dt)):.3f}")
c6.metric("Cumulative Load (µV·s)", f"{cumulative_load_ex_breaks:.3f}")

# Layout: full-width plot, tables below
fig = build_timeline_figure(
    t=t,
    I=emg_intensity,
    I_s=I_s,
    phases_t=phases_t,
    breaks_t=breaks_t,
    exercise_name=exercise_name,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Per-channel RMS envelopes"):
    options = list(emg_env_df.columns)
    default_sel = options
    selected = st.multiselect("Select channel(s) to display", options=options, default=default_sel)

    if not selected:
        st.info("Select at least one channel to display its RMS envelope.")
    else:
        env_fig = build_env_figure(t, emg_env_df, selected)
        st.plotly_chart(env_fig, use_container_width=True)

st.subheader("Detected Phases (Load Summary)")
st.dataframe(ph_df, use_container_width=True, height=240)
st.download_button(
    "Download phases_loads.csv",
    data=ph_df.to_csv(index=False).encode("utf-8"),
    file_name="phases_loads.csv",
    mime="text/csv",
)

# st.subheader("Detected Breaks / Dropouts")
# st.dataframe(br_df, use_container_width=True, height=240)
# st.download_button(
#     "Download breaks.csv",
#     data=br_df.to_csv(index=False).encode("utf-8"),
#     file_name="breaks.csv",
#     mime="text/csv",
# )
