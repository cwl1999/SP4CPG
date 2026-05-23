# Windows Client Installation Guide

This document describes how to install and launch the SP4CPG Windows client. The client provides a graphical interface for configuring the repository path, dataset path, HGCN model file, detection workflow, result visualization, and report export.

[← Back to README](README.md)

## 1. Environment Requirements

Before launching the Windows client, prepare the following environment:

| Item | Requirement |
|---|---|
| Operating system | Windows 10/11 |
| Python | Python 3.8+ |
| Project code | SP4CPG repository |
| Model file | Trained HGCN `.pt` checkpoint, if running detection directly |
| Optional graph workflow | JDK + Joern, required only when generating CPG/HCPG files from source |

If you only use an existing HCPG dataset and a trained HGCN model, you do not need to run the full CPG/HCPG generation workflow.

## 2. Clone the Repository

```powershell
git clone https://github.com/DataAvailable/SP4CPG
cd SP4CPG
```

If you already have the repository locally, enter the repository root directly:

```powershell
cd E:\Projects\SP4CPG
```

## 3. Create a Virtual Environment for the Client

It is recommended to create a separate virtual environment for the client to avoid conflicts with other Python environments.

```powershell
cd E:\Projects\SP4CPG\client
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If the `.venv` directory was copied from another location, delete it and recreate it. A moved virtual environment may cause `pip` or `python` to point to an old path.

## 4. Install Client Dependencies

Install the dependencies required by the Windows client:

```powershell
python -m pip install -r requirements-client.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 --retries 10
```

If the client also executes HCPG preprocessing locally, ensure that the required preprocessing dependencies are installed:

```powershell
python -m pip install pydot pandas numpy scikit-learn tqdm tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you need to train or run HGCN models, install the deep learning dependencies according to your CUDA/PyTorch environment. For CPU-only testing, install the CPU version of PyTorch. For GPU acceleration, install the PyTorch version that matches your CUDA version.

## 5. Optional: Configure JDK and Joern

JDK and Joern are required only when the client needs to generate CPG/HCPG files from raw C/C++ functions.

### 5.1 Install JDK

Install JDK 21 or another JDK version compatible with your Joern version. `JAVA_HOME` must point to the JDK root directory, not to a JRE directory.

Example:

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-21"
$env:Path="$env:JAVA_HOME\bin;$env:Path"

java -version
javac -version
```

The `javac -version` command must work correctly.

### 5.2 Install and Test Joern

After installing Joern, make sure `joern`, `joern-parse`, and `joern-export` can be found from PowerShell:

```powershell
joern --help
joern-parse --help
joern-export --help
```

If these commands are not found, add the Joern directory to the `Path` environment variable, or configure the absolute Joern path in the client.

## 6. Launch the Windows Client

Start the client from the `client` directory:

```powershell
cd E:\Projects\SP4CPG\client
.\.venv\Scripts\activate
python -m sp4cpg_client.app
```

If a launcher script is provided, you can also run:

```powershell
run_client.bat
```

## 7. Configure a Detection Task in the Client

After the client starts, configure the task in the graphical interface.

### 7.1 Project and Data

Set the following fields:

| Field | Description |
|---|---|
| Repository root | The SP4CPG repository root, for example `E:\Projects\SP4CPG` |
| Python interpreter | The Python executable used to run backend scripts |
| Dataset file | Raw dataset such as `PrimeVul.json`, or HCPG dataset such as `hcpg_dataset.pkl` |
| Output directory | Directory for prediction results, metrics, and reports |

### 7.2 HGCN Detection

Set the HGCN-related fields:

| Field | Description |
|---|---|
| Model type | Select `HGCN` |
| Trained model file | Select the `.pt` checkpoint file |
| Training parameters | Configure epoch, batch size, learning rate, and patience if retraining is needed |

The client supports two common workflows:

1. Load an existing HGCN checkpoint and run detection directly.
2. Generate HCPG data and retrain the model before detection.

### 7.3 Optional HCPG Workflow

If you need to generate graph artifacts from the raw dataset, enable the corresponding workflow stages:

1. Generate CPG DOT files.
2. Generate HCPG DOT files.
3. Generate HCPG embedding data.
4. Run HGCN training or detection.

If you already have `hcpg_dataset.pkl`, you may skip CPG/HCPG generation and use the existing embedding file directly.

## 8. View Results and Export Reports

After the task finishes, the client displays the main metrics in the overview page, including:

- Accuracy
- Precision
- Recall
- F1
- High-risk alerts

The output directory may contain:

```text
model_predictions.json
model_predictions.csv
hgcn_metrics.json
final_report.json
final_report.csv
HTML report
```

Use the report export page to generate and open the HTML report.

## 9. Troubleshooting

### 9.1 `No module named 'pydot'`

Install `pydot` in the same Python environment used by the client:

```powershell
python -m pip install pydot -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Verify the installation:

```powershell
python -c "import pydot; print('pydot ok')"
```

### 9.2 `A Java JDK is not installed or can't be found`

This means `JAVA_HOME` points to a JRE or an invalid path. Install a JDK and set `JAVA_HOME` to the JDK root directory.

Correct example:

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-21"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
javac -version
```

### 9.3 `pip` points to an old virtual environment path

If `pip` reports a path from a previous location, recreate the virtual environment:

```powershell
cd E:\Projects\SP4CPG\client
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 9.4 Client launches but backend scripts use the wrong Python

In the client configuration page, set the Python interpreter explicitly to the intended environment, for example:

```text
E:\Projects\SP4CPG\client\.venv\Scripts\python.exe
```

This ensures that preprocessing and detection scripts use the same environment as the client.
