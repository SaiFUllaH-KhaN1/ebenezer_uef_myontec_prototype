from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Any



# -----------------------------------------------------------------------------
# FIXED PARAMS 
# -----------------------------------------------------------------------------
DEFAULT_FS_HZ = 25.0  # default sampling rate 

# Envelope
RMS_WIN_SEC = 1.0  # compute_emg_intensity(..., win_sec=1.0)

# Segmentation
SMOOTH_SEC = 2.0  # I_s = smooth_ma(I, win=int(round(2.0*fs)))
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
MIN_PHASE_SEC = 5.0          # clip_phases_around_breaks(..., min_phase_sec=5.0)
SNAP_TOL_SEC = 1.0           # snap_phases_to_breaks_strict(..., tol_sec=1.0)


# =============================================================================
# 1) Robust Myontec CSV loader (from uploaded bytes)
# =============================================================================
def _is_number_like(s: str) -> bool:
    """Return True if string looks like a number (supports comma decimals)."""
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


def load_myontec_csv_from_bytes(data: bytes) -> pd.DataFrame:
    """
    Load session CSV that may include:
      - metadata lines mixed with data
      - comma decimals (e.g., 0,04)
      - separators: ';' or '\\t', and sometimes ','

    We only keep rows where the 2nd column ("Elapsed time") is numeric.
    """
    text = data.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # Find header line
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Time") and "Elapsed time" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line containing 'Time' and 'Elapsed time'.")

    header_line = lines[header_idx]
    sep = _detect_delimiter(header_line)
    header = header_line.split(sep)

    data_rows = []
    for ln in lines[header_idx + 1 :]:
        if not ln.strip():
            continue
        parts = ln.split(sep)
        if len(parts) < 2:
            continue

        if not _is_number_like(parts[1]):  # elapsed time must be numeric
            continue

        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        else:
            parts = parts[: len(header)]
        data_rows.append(parts)

    if not data_rows:
        raise ValueError("No data rows found after filtering. Check separator/format.")

    df = pd.DataFrame(data_rows, columns=header)

    # elapsed time → float seconds
    df["elapsed_s"] = (
        df["Elapsed time"].astype(str).str.strip().str.replace(",", ".", regex=False).astype(float)
    )

    # numeric columns (comma -> dot)
    for c in df.columns:
        if c in ("Time", "Elapsed time", "elapsed_s"):
            continue
        s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.replace({"": np.nan, "None": np.nan})
        df[c] = pd.to_numeric(s, errors="coerce")

    df = df.sort_values("elapsed_s").reset_index(drop=True)
    return df


# =============================================================================
# 2) EMG intensity envelope (your logic)
# =============================================================================
def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Sliding RMS envelope with window 'win' samples."""
    x = np.asarray(x, dtype=np.float32)
    win = max(1, int(win))
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    x2 = xp * xp
    kernel = np.ones(win, dtype=np.float32) / float(win)
    m = np.convolve(x2, kernel, mode="valid")
    return np.sqrt(m + 1e-8).astype(np.float32)


def robust_norm01(x: np.ndarray, lo_q: int = 5, hi_q: int = 95) -> np.ndarray:
    """Robust normalize to [0,1] using percentiles."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_q), np.percentile(x, hi_q)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def compute_emg_intensity(df: pd.DataFrame, fs: float, win_sec: float):
    """
    Per-channel EMG RMS envelopes + combined intensity (mean across channels, norm [0,1]).
    """
    emg_cols = [
        "Left Quadriceps Group / uV",
        "Right Quadriceps Group / uV",
        "Left Hamstrings / uV",
        "Right Hamstrings / uV",
        "Left Gluteus / uV",
        "Right Gluteus / uV",
    ]
    missing = [c for c in emg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing EMG columns: {missing}")

    t = df["elapsed_s"].to_numpy(dtype=np.float32)
    win = int(round(win_sec * fs))

    env = {}
    for c in emg_cols:
        x = df[c].to_numpy(dtype=np.float32)

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


def merge_nearby_regions(regions: list[tuple[int, int]], fs: float, gap_sec: float = 8.0, min_len_sec: float = 0.0) -> list[tuple[int, int]]:
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


def clip_phases_around_breaks(phases: list[dict[str, Any]], breaks: list[tuple[int, int]], fs: float, min_phase_sec: float = 5.0) -> list[dict[str, Any]]:
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


def snap_phases_to_breaks_strict(phases: list[dict[str, Any]], breaks: list[tuple[int, int]], fs: float, tol_sec: float = 0.2, warmup_label: str = "warmup") -> list[dict[str, Any]]:
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
        if i1 < n:
            t1 = float(t[i1])
        else:
            t1 = float(t1_last + (1.0 / fs))

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
        if b1 < n:
            t1 = float(t[b1])
        else:
            t1 = float(t1_last + (1.0 / fs))

        out.append({"i0": b0, "i1": b1, "t0": t0, "t1": t1, "t1_last": t1_last})

    return out


def segment_phases_from_emg(t: np.ndarray, I: np.ndarray, fs: float = 25.0) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray]:
    t = np.asarray(t, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)

    I_s = smooth_ma(I, win=int(round(SMOOTH_SEC * fs)))  # 2s smoothing

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
            if above[i : i + hold_len].all():
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
        if above[i : i + hold_len].all():
            split_i = am + i
            break

    if split_i is None:
        phases.append({"label": "main_test", "i0": am, "i1": bm})
    else:
        phases.append({"label": "aerobic", "i0": am, "i1": split_i})
        phases.append({"label": "anaerobic", "i0": split_i, "i1": bm})

    phases = sorted(phases, key=lambda p: p["i0"])
    return phases, breaks, I_s


# =============================================================================
# 4) Plotly figures
# =============================================================================
def build_timeline_figure(t: np.ndarray, I: np.ndarray, I_s: np.ndarray, phases_t: list[dict[str, Any]], breaks_t: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=I, mode="lines", name="EMG intensity (norm)", opacity=0.25))
    fig.add_trace(go.Scatter(x=t, y=I_s, mode="lines", name="EMG intensity (smoothed)", line=dict(width=3)))

    # Breaks shaded
    for j, b in enumerate(breaks_t):
        fig.add_vrect(
            x0=b["t0"], x1=b["t1"],
            fillcolor="rgba(255,255,255,0.08)",
            line_width=0,
            annotation_text="break" if j == 0 else None,
            annotation_position="top left",
        )

    # Phase bands
    phase_fill = {
        "warmup": "rgba(0, 200, 255, 0.10)",
        "aerobic": "rgba(0, 255, 150, 0.10)",
        "anaerobic": "rgba(255, 120, 0, 0.12)",
        "cooldown": "rgba(180, 180, 255, 0.10)",
        "main_test": "rgba(255, 255, 0, 0.08)",
    }

    for p in phases_t:
        fig.add_vrect(
            x0=p["t0"], x1=p["t1"],
            fillcolor=phase_fill.get(p["label"], "rgba(200,200,200,0.08)"),
            line_width=0,
        )
        fig.add_vline(x=p["t0"], line_width=1, line_dash="dash", opacity=0.6)
        fig.add_annotation(
            x=p["t0"], y=1.06, text=p["label"],
            showarrow=False, textangle=-90, xanchor="left",
        )

    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=45, b=30),
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Intensity (0..1)",
        yaxis=dict(range=[0, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_env_figure(t: np.ndarray, emg_env_df: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for c in selected:
        fig.add_trace(go.Scatter(x=t, y=emg_env_df[c].to_numpy(), mode="lines", name=c))
    fig.update_layout(
        height=340,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=35, b=25),
        xaxis_title="Time (s)",
        yaxis_title="RMS (uV)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


# =============================================================================
# 5) Streamlit UI
# =============================================================================
st.set_page_config(page_title="Myontec EMG Segmentation Analysis Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.0rem; margin-top: 2rem }
      .titlebar {
        padding: 14px 16px; border-radius: 16px;
        background: linear-gradient(90deg, rgba(0,200,255,0.18), rgba(0,255,150,0.12));
        border: 1px solid rgba(255,255,255,0.08);
      }
      .muted { opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="titlebar">
      <div style="font-size: 20px; font-weight: 700;">EMG Phase Segmentation Dashboard</div>
      <div class="muted">CSV → EMG envelope → breaks + warmup/aerobic/anaerobic/cooldown</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: ONLY sampling rate control
st.sidebar.header("Controls")
fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=1.0, max_value=500.0, value=DEFAULT_FS_HZ, step=1.0)

uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
if not uploaded:
    st.info("Upload a CSV to view the dashboard.")
    st.stop()

# Load
try:
    df = load_myontec_csv_from_bytes(uploaded.getvalue())
except (ValueError, UnicodeDecodeError) as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

# Compute intensity/envelopes
try:
    t, emg_env_df, emg_intensity = compute_emg_intensity(df, fs=fs, win_sec=RMS_WIN_SEC)
except (ValueError, KeyError) as e:
    st.error(f"Failed to compute EMG intensity: {e}")
    st.stop()

# Segment phases
phases, breaks, I_s = segment_phases_from_emg(t, emg_intensity, fs=fs)

# Clip + snap 
phases = clip_phases_around_breaks(phases, breaks, fs=fs, min_phase_sec=MIN_PHASE_SEC)
phases = snap_phases_to_breaks_strict(phases, breaks, fs=fs, tol_sec=SNAP_TOL_SEC)

# Attach boundary times (end-exclusive)
phases_t = attach_times_to_phases(phases, t, fs=fs)
breaks_t = attach_times_to_breaks(breaks, t, fs=fs)

# Tables
def phases_table(phases_t: list[dict[str, Any]], fs: float) -> pd.DataFrame:
    rows = []
    for p in phases_t:
        dur = (int(p["i1"]) - int(p["i0"])) / fs
        rows.append(
            {
                "label": p["label"],
                "t_start": round(p["t0"], 2),
                "t_end": round(p["t1"], 2),
                "duration_s": round(dur, 2),
                "i0": int(p["i0"]),
                "i1": int(p["i1"]),
            }
        )
    return pd.DataFrame(rows)


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


ph_df = phases_table(phases_t, fs)
br_df = breaks_table(breaks_t, fs)
duration_s = float(t[-1]) if len(t) else 0.0

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df)}")
c2.metric("Duration", f"{duration_s:.1f}s")
c3.metric("Phases", f"{len(ph_df)}")
c4.metric("Breaks", f"{len(br_df)}")
c5.metric("Mean Intensity", f"{float(np.mean(I_s)):.3f}")

# Layout: plot + tables
left, right = st.columns([1.65, 1.0], gap="large")

with left:
    fig = build_timeline_figure(t=t, I=emg_intensity, I_s=I_s, phases_t=phases_t, breaks_t=breaks_t)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-channel RMS envelopes"):
        # User selects which channel(s) to show
        options = list(emg_env_df.columns)
        default_sel = options[:2] if len(options) >= 2 else options
        selected = st.multiselect(
            "Select channel(s) to display",
            options=options,
            default=default_sel,
        )

        if not selected:
            st.info("Select at least one channel to display its RMS envelope.")
        else:
            env_fig = build_env_figure(t, emg_env_df, selected)
            st.plotly_chart(env_fig, use_container_width=True)

with right:
    st.subheader("Detected Phases")
    st.dataframe(ph_df, use_container_width=True, height=240)
    st.download_button(
        "Download phases.csv",
        data=ph_df.to_csv(index=False).encode("utf-8"),
        file_name="phases.csv",
        mime="text/csv",
    )

    st.subheader("Detected Breaks / Dropouts")
    st.dataframe(br_df, use_container_width=True, height=240)
    st.download_button(
        "Download breaks.csv",
        data=br_df.to_csv(index=False).encode("utf-8"),
        file_name="breaks.csv",
        mime="text/csv",
    )