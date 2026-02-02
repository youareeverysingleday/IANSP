# IANSP

This repository provides the implementation of a two-stage pipeline for Intention Aware Next Stay Prediction on Continuous Trajectories with Temporal Contexts.

The codebase consists of:
1. Trajectory feature generation and intention-aware stay construction.
2. Next stay prediction models with single-GPU and multi-GPU support.

The repository is designed for reproducibility and experimental analysis
in trajectory prediction research.

---

## Overview

The overall workflow includes two main components:

**Stage 1: Trajectory Feature Generation**
- Preprocess raw trajectory data.
- Convert GPS trajectories into stay-based representations.
- Construct intention-aware stay records.
- Compute and export statistical features of trajectories.

**Stage 2: Trajectory Next Stay Prediction**
- Train and evaluate next stay prediction models.
- Support both single-GPU and multi-GPU (data-parallel) training.
- Produce prediction results consistent with the experimental protocol.

---

## Environment Setup

The recommended environment configuration is:

| Name   | Version |
|--------|---------|
| Ubuntu | 20.04   |
| Python | 3.12    |
| Conda  | 25.5.1  |

Create and activate a conda environment:
```bash
conda create -n iansp python=3.12
conda activate iansp
```

Install dependencies:
```bash
pip install -r requirements.txt
```
The provided requirements specify the minimal dependencies required to run the code. Exact package versions may vary across environments.

---

## Dataset

This project uses two datasets: one open-source dataset and one closed-source dataset.


1. [Microsoft GeoLife 1.3 Dataset](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
    The GeoLife GPS Trajectories dataset is released by Microsoft Research Asia
    and is publicly available.

    Official source: <https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/>

    Due to its size (~1.55GB), the raw dataset is not included in this repository.

    After downloading, extract the dataset and organize it as:
    ```text
    Data/Geolife Trajectories 1.3/
    ```

2. MoreUser Dataset (Private)

The MoreUser dataset is a private dataset and cannot be publicly released.
The repository provides the same preprocessing and modeling pipeline for this dataset,
assuming the same data schema as described in the configuration files.

---

## Dir Structure

```bash
./IANSP
|   .gitattributes
|   README.md
|   requirements.txt
|   
+---TrajectoryFeatureGeneration
|   |   1GeoLift_TrajectoryFeatureGeneration.ipynb
|   |   2GenerateContext.py
|   |   2GeoLift_GenerateContext.ipynb
|   |   3GeoLift_GenerateTrajectorySubsequence.ipynb
|   |   4DisplayStatistics.ipynb
|   |   5SampledData_MultiProcess.py
|   |   6DistributionofUserTimeIntervals.py
|   |   6DistributionofUserTimeIntervals_2datasets.py
|   |   
|   +---Data
|   |   \---Geolife Trajectories 1.3
|   |       |   User Guide-1.3.pdf
|   |       |   
|   |       \---Data
|   |           +---000
|   |           |   \---Trajectory
|   |           |           20081023025304.plt
|   |           |           20081024020959.plt
|   |           |           20081026134407.plt
|   |           |           
|   |           \---001
|   |               \---Trajectory
|   |                       20081023055305.plt
|   |                       20081023234104.plt
|   |                       20081024234405.plt
|   |                       
|   \---Utils
|           CalcGrid.py
|           OperJson.py
|           
\---TrajectoryNextStayPrediction
    |   GetInformationFromContext.py
    |   NextStayPrediction.py
    |   preprocess_cpu_cache_with_progress_patched.py
    |   train_gpu_from_cache_safe.py
    |   
    +---Common
    |       CalculateDistance.py
    |       CommonCode.py
    |       gpu_mem_track.py
    |       modelsize_estimate.py
    |       
    +---Data
    |       GeoLife_all.csv
    |       GeoLife_Nonroutine_top3.csv
    |       GeoLife_Routine_top3.csv
    |       
    \---runs_geolife
        \---unified_run_E1
            |   train_log.csv
            |   
            \---plots
                    curves_acc@1.png
                    curves_acc@10.png
                    curves_acc@5.png
                    curves_loss.png
                    curves_mrr@1.png
                    curves_mrr@10.png
                    curves_mrr@5.png
                    curves_ndcg@1.png
                    curves_ndcg@10.png
                    curves_ndcg@5.png
                    curves_rr_acc@1.png
                    curves_rr_acc@10.png
                    curves_rr_acc@5.png
```
---

## pipeline

### Trajectory Feature Generation

Trajectory feature preprocessing is handled by the 'TrajectoryFeatureGeneration' module.

This module contains a mixture of Jupyter notebooks and Python scripts. The files are designed to be executed sequentially according to the numeric prefix in their filenames (e.g., 0*.ipynb, 1*.py, etc.).

This stage:

- Parses raw trajectory data.
- Detects stay points.
- Constructs intention-aware stay representations.
- Computes trajectory-level and stay-level statistics.

To run trajectory feature generation:
```bash
cd TrajectoryFeatureGeneration
```
Then execute the files in ascending numerical order as indicated by their filenames.

The generated outputs are saved to:
```text
./Data/Output/
```

These output files serve as the input for the next stay prediction models.

### Trajectory Next Stay Prediction

The prediction models are implemented in the TrajectoryNextStayPrediction module.

####  Training and validation

```bash
for EXP in E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11
do
  echo "==== Preprocess $EXP ===="
  python preprocess_cpu_cache.py \
    --cache_dir ./cache_geolife \
    --experiment $EXP \
    --datachoose GeoLife
done
```

This module provides a multi-GPU parallel version using PyTorch distributed training.

Example (8 GPUs):

```bash
for EXP in E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11
do
  echo "==== Train $EXP ===="
  torchrun --standalone --nproc_per_node=8 train_gpu_from_cache.py \
    --cache_dir ./cache_moreuser \
    --experiment $EXP \
    --datachoose GeoLife
done
```
The number of GPUs can be adjusted according to available hardware.

---

## Notes on Reproducibility

- Random seeds are fixed where applicable.
- Minor numerical differences across platforms and hardware are expected.
- Training time depends on dataset size and hardware configuration.
- The same preprocessing and evaluation protocol is used across datasets.

---

## License

 MIT license.