# Myontec EMG Segmentation Dashboard

This project is a Streamlit dashboard for analyzing Myontec EMG session CSV files.

The app in `app.py`:
- uploads and parses Myontec CSV exports (including mixed metadata rows and comma decimals),
- computes per-channel RMS envelopes,
- builds a normalized combined EMG intensity signal,
- segments the session into phases (`warmup`, `aerobic`, `anaerobic`, `cooldown`, or `main_test`),
- detects low-intensity breaks/dropouts,
- visualizes results with Plotly,
- lets you download detected phases and breaks as CSV files.

## Required EMG Columns

The uploaded file must contain these columns:
- `Left Quadriceps Group / uV`
- `Right Quadriceps Group / uV`
- `Left Hamstrings / uV`
- `Right Hamstrings / uV`
- `Left Gluteus / uV`
- `Right Gluteus / uV`

It must also include `Time` and `Elapsed time` columns.

## Getting Started

1. Create and activate a virtual environment.

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

We are using pip-tools now for managing dependencies. To modify the project’s dependencies:

Edit the requirements.in file to add or modify packages. So just add names of the packages without any
versions. Then, 
Run pip-compile to generate the requirements.txt file with the exact package versions.
Use pip-sync to install the dependencies as specified in requirements.txt.
This approach will save us  **MIGRAINE** across all environments.

3. Run the Streamlit app.

```bash
streamlit run streamlit_server.py
```

4. Open the local URL shown in the terminal (typically `http://localhost:8501`).

## How To Use

1. Set the sampling rate in the sidebar (default: `25 Hz`).
2. Upload a Myontec CSV file.
3. Review the timeline plot, phase table, and break table.
4. Download `phases.csv` and `breaks.csv` from the app.

## Main Dependencies

Dependencies are fully listed in `requirements.txt`. Core runtime packages include:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
