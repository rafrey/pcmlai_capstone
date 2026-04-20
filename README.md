# Redesigning the Federal Poverty Standard

[![Open this Notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafrey/pcmlai_capstone/blob/main/capstone.ipynb)

## Mission & Project Context

This project challenges and replaces the archaic 1960s Official Poverty Measure (OPM) with a modern, multidimensional "Capability Gap" machine learning model. The current OPM relies on a severely outdated heuristic that multiplies food costs by three, completely ignoring modern cost-of-living volatility such as rent, broadband, and healthcare.

Key concepts driving this research include:

* **The "Hidden Poor":** Families technically excluded by the federal poverty heuristic but who functionally have less than $0 in residual income after non-discretionary expenses.
* **Proportional Context:** Leveraging Symmetrical Mean Absolute Percentage Error (sMAPE) over absolute numbers to acknowledge that a $1,000 deficit is devastating to a $15,000 baseline, but mathematically negligible to a $150,000 baseline.
* **The "In-Kind Paradox" (Benefit Cliffs):** Financial traps where accepting a small income raise accidentally disqualifies a household from critical aid, pushing them deeper into poverty.

## Agile CRISP-DM & Rapid Prototyping

This project operates as a "Democratic ML" proof-of-concept (POC), demonstrating that an individual engineer can rapidly outperform legacy government heuristics using a newly conceptualized **CRISP-DM-F48 ("First 48")** framework.

**CRISP-DM-F48** is an accelerated variant of the CRISP-DM framework integrating Agile principles of rapid prototyping and fail-fast mentality. By time-boxing a single, narrowly scoped iteration of the full CRISP-DM lifecycle to a 48-hour sprint, we force a high-velocity proof-of-concept.

Borrowing from homicide investigations' "First 48" paradigm, this model posits that if actionable leads (data viability), a solid motive (business understanding/utility), and a clear weapon (viable ML architecture) cannot be established within the first 48 hours, the project "goes cold" and further investment is halted. This ruthless prioritization surfaces hidden systemic blockers ("where the bodies are buried") immediately, ensuring only highly viable, impact-driven data science initiatives proceed to extended development.

Instead of falling into the "analysis paralysis" common in massive government data projects, we restrict our initial scope specifically to the Colorado Front Range using 2022 Census data. This forces immediate, testable results and enforces a "Minimal Feature Set" to reduce administrative burden.

**Future Iterations:** This POC validates the core mathematical mechanics (like mapping benefit cliffs and enforcing Monotonic Constraints). Future iterations would expand geographic boundaries to a national level, incorporate multi-year temporal analysis for inflationary trends, and fully refine the federal policy feedback loop.

## Data Strategy & Pipeline

**What & Why:** We utilize the US Census Bureau's 2022 American Community Survey (ACS) 1-Year Public Use Microdata Sample (PUMS). This dataset provides the granular demographic and localized cost variables (rent, utilities, income) strictly necessary for multidimensional capability scoring.

**Fetching & Storage:** Raw dataset ZIP files are fetched programmatically via `urllib.request` during the data preparation phase, automatically extracted, and stored locally in `data/raw/` to ensure full data provenance and reproducibility.

**The Handoff Mechanism:** By aggressively filtering, transforming, and scaling the data in the initial phases, we export a fully sterilized 12-feature matrix (`tensor_baseline.csv`) to `data/clean/`. Subsequent phases ingest *only* this baseline tensor. This hard decoupling:

1. Completely isolates macroeconomic assumptions (like imputed health and transport costs).
2. Strictly prevents data leakage from exploration into the Keras models.
3. Eliminates costly data re-computation during iterative deep learning validation loops.

## Research Results

### EDA & Data Preparation Results
During the Data Understanding and Preparation phases (Section 5 of `capstone.ipynb`), several critical macroeconomic insights emerged that validate the necessity for a Deep Learning approach:

* **The "Hidden Poor":** Baseline evaluations against the legacy Official Poverty Measure (OPM) identified that over **25% of the impoverished population** in our dataset are technically classified as "above poverty." However, when accounting for localized housing, health, and broadband costs, these households functionally possess a **negative residual income**.
* **The "In-Kind Paradox" & Benefit Cliffs:** Bivariate analysis exposed severe welfare cliffs. SNAP (food stamp) utilization clusters tightly at the lowest incomes. As nominal income marginally exceeds aid thresholds, the automatic loss of benefits forces the entire burden of a rigid housing cost floor onto the household, resulting in an abrupt net-negative shift in true purchasing power.
* **Non-Linear Capability Gaps:** Statistical tests using Dummy (sMAPE > 312%) and Linear Regression (sMAPE > 87%) baselines failed structurally. Linear models fail because financial decay is not a straight line; it features abrupt cliffs tied to fixed costs and aid disqualification. This scientifically justifies the deployment of Keras Neural Networks to capture these non-linear capability boundaries.
* **The Minimal Feature Set:** Feature engineering proved that a strictly limited 12-variable tensor (including income, housing costs, broadband/health status, and OECD-modified family size) is sufficient for high-fidelity capability monitoring. This directly addresses policy concerns regarding "administrative burden" by proving that modernizing poverty measurement does not require sweeping data collection.

### Modeling & SHAP Explainability
To ensure the model is legally defensible and transparent, we utilized a monotonic neural network architecture and SHAP (Shapley Additive Explanations).
* **Mapping Benefit Cliffs:** SHAP successfully extracted the impact of each feature, visually mapping the non-linear "benefit cliffs" where marginally stable families are abruptly penalized by fixed costs or the abrupt loss of aid.
* **Predictive Dominance (sMAPE):** The Tier 2 Keras Monotonic MLP achieved a definitive sMAPE boundary score of **0.2404**. This represents a mathematically validated predictability improvement of **23.61%** over the standard linear / Official Poverty Measure baselines.
* **The Capability Formula:** We distilled the complex, multi-dimensional neural network weights into a transparent, flat mathematical function. This ensures that the "black box" model can be translated directly into a simple Capability Formula perfectly suited for legislative action and public policy text:

$$
\text{Capability}_{gap} = M_{phasein} \times \left( (\beta_1 \times \text{nominal\_income}) + (\beta_2 \times \text{housing\_cost}) + (\beta_3 \times \text{broadband\_status}) + (\beta_4 \times \text{health\_insurance}) + (\beta_5 \times \text{snap\_aid\_status}) + (\beta_6 \times \text{adults}) + (\beta_7 \times \text{children}) + (\beta_8 \times \text{oecd\_scale}) + (\beta_9 \times \text{puma\_code}) + (\beta_{10} \times \text{household\_weight}) \right)
$$

### Policy Validation & Deployment Strategy
* **The "In-Kind Paradox" in Colorado:** The framework empirically validated that true economic fragility in the Colorado Front Range is defined by steep capability cliffs triggered by regional cost spikes (like rent), not a uniform national income line. As households marginally increase income past rigid government thresholds, they abruptly lose access to critical in-kind subsidies, instantly plunging their true residual capability well below zero.
* **The Phase-In Multiplier:** To solve the primary barrier to entitlement reform, budgetary shock, we architected a "Phase-In Multiplier" deployment strategy. This allows lawmakers to pass the mathematically correct framework into law immediately, but scale the payouts to match current budget constraints. Legislatures can then incrementally increase the multiplier over a multi-year horizon to reach mathematical parity without triggering a single-cycle fiscal shock.

## Workspace Structure

This repository is strictly organized to map data flows cleanly across the CRISP-DM methodology:

* **`data/`**: (Ignored via `.gitignore`) Contains `raw/` Census downloads and `clean/` sterilized matrices.
* **`logs/`**: (Ignored via `.gitignore`) Stores TensorBoard execution logs and Hyperband tuning checkpoints from Phase 4.
* **`models/`**: (Ignored via `.gitignore`) Stores output weights and `.keras` architectures generated natively by Phase 4.
* **`scripts/`**: Directory containing cross-platform environment setup utilities (e.g., `install-dependencies.sh`, `install-dependencies.ps1`) and the TensorFlow CUDA repair script (`fix_tf_cuda_venv.sh`).
* **`requirements.txt`**: Configuration file for Python dependencies to ensure deterministic pip/uv installations.
* **`capstone.ipynb`**: Jupyter notebook representing the full CRISP-DM lifecycle of the first iteration of this research (prototype)

## Recommended Environment

This project was developed on WSL2 running Ubuntu using Python 3.13.7 for the Jupyter kernel. That is the reference environment for the notebook, TensorFlow, and the GPU repair workflow in this repository.

If Linux is not your native OS, use Ubuntu as the fallback environment. For Windows users, prefer WSL2 with Ubuntu for GPU work. For macOS users, prefer an Ubuntu VM if you want the closest match to the development environment used for this project.

## Installation & Setup

Ensure you have Python 3.10+ installed.

### Shared Prerequisites

1. **Clone the repository and enter the directory:**

   ```bash
   git clone <repository-url>
   cd pcmlai_capstone
   ```

2. **Install `uv`:**

   ```bash
   pip install uv
   uv python install 3.13.7
   ```

### Windows Native

Use this path if you want the simplest Windows-native setup and CPU execution is acceptable.

**Convenience shortcut (PowerShell):**

```powershell
.\scripts\install-dependencies.ps1
```

**Manual setup (PowerShell):**

```powershell
uv venv .venv --python 3.13.7
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Current TensorFlow pip packages on native Windows are CPU-only after TensorFlow 2.10. If you need GPU acceleration for this project, use WSL2 with Ubuntu instead of expecting a native Windows CUDA repair path.

### Linux / WSL2

Use this path for the reference development environment and for GPU acceleration.

**Convenience shortcut (bash/zsh):**

```bash
./scripts/install-dependencies.sh
```

The helper creates `.venv`, installs `requirements.txt`, probes whether TensorFlow can load its GPU libraries cleanly, and only runs `./scripts/fix_tf_cuda_venv.sh` if CUDA library repair is actually needed.

**Manual setup (bash/zsh):**

```bash
uv venv .venv --python 3.13.7
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Linux / WSL2 NVIDIA GPU fixup:**

If you are running Linux or Ubuntu on WSL2 with an NVIDIA GPU and TensorFlow cannot load its shared libraries cleanly, run:

```bash
./scripts/fix_tf_cuda_venv.sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Re-run the fixup script any time you recreate `.venv` or reinstall the TensorFlow CUDA packages.

### macOS

Use this path if you want a local macOS environment and CPU execution is acceptable.

**Manual setup (zsh/bash):**

```bash
uv venv .venv --python 3.13.7
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

CUDA installation is not applicable for this project on macOS. If you want the closest match to the development environment, use an Ubuntu VM instead of trying to reproduce the Linux CUDA flow on macOS.

### System Packages / GPU Prerequisites

#### Windows Native Prerequisites

* Install Python 3.10+ and `uv`.
* Install the Microsoft Visual C++ Redistributable if TensorFlow or its dependencies require standard Windows runtime DLLs.
* Keep NVIDIA drivers current if you also use WSL2 on the same machine.
* Do not expect native Windows TensorFlow GPU support for this repository's current setup; use WSL2 with Ubuntu for GPU work.

#### WSL2 / Ubuntu GPU Prerequisites

* Install current NVIDIA drivers on the Windows host before using WSL2 GPU passthrough.
* Confirm the GPU is visible with `nvidia-smi` before creating the virtual environment.
* Install any required CUDA toolkit or cuDNN system packages for your Ubuntu environment if your host setup does not already provide the necessary runtime pieces.
* After dependency installation, verify TensorFlow with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
* If the TensorFlow probe reports missing shared libraries, run `./scripts/install-dependencies.sh` or `./scripts/fix_tf_cuda_venv.sh`.

#### macOS Prerequisites

* Install Python 3.10+ and `uv`.
* CUDA is not supported or required for the current TensorFlow setup in this repository.
* Treat macOS as a CPU-only environment for this project unless you intentionally move to a different TensorFlow stack than the one documented here.

## Usage & Execution Order

To reproduce the Democratic ML proof-of-concept without data leakage, execute the capstone notebook sequentially

## Methodology & Technical Stack

We use **Keras / TensorFlow** to construct the non-linear boundaries capable of properly interpreting massive capability cliffs. Unsupervised Autoencoders act as our baseline threshold discovery mechanism, and a Sequential Multi-Layer Perceptron enforces the active logic visually supported by **SHAP (Shapley Additive Explanations)**.
