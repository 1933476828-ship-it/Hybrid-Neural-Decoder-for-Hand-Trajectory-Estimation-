# Hybrid Neural Decoder for Hand Trajectory Estimation

A lightweight causal brain-computer interface (BCI) decoder for reconstructing continuous 2-D hand trajectories from motor-cortical spike trains.

This project was developed for a neural decoding competition, where the goal was to estimate hand position in real time using only neural activity available up to the current prediction step. The decoder combines direction classification, empirical trajectory priors, local displacement regression, temporal fusion, and geometric constraints to achieve stable and computationally efficient neural decoding.

---

## Overview

Brain-computer interfaces require decoders that can transform neural activity into continuous control signals for external devices such as prosthetic arms, robotic manipulators, cursors, or rehabilitation systems. In this project, the task is to estimate 2-D hand position from motor-cortical spike trains recorded during reaching movements.

The main challenge is **causality**: during online decoding, the model can only use spike activity observed up to the current time point. It cannot access future neural activity or future hand positions. Therefore, the decoder must make stable real-time predictions under limited information.

To address this, this repository implements a **hybrid direction-and-displacement decoder** that combines:

- time-bin-specific LDA direction classification,
- top-K weighted empirical trajectory priors,
- ridge-regression displacement estimation,
- sigmoid prior-evidence fusion,
- and angular geometric constraints.

The final decoder achieved an official hidden-test RMSE of **8.686** and ranked **1st in total training and prediction time** with a runtime of **0.602 s**.

---

## Key Features

- **Causal online decoding**
  - Predicts hand position using only neural activity available up to the current time step.

- **20 ms neural binning**
  - Converts 1 ms spike trains into 20 ms spike-count features to match the online evaluation interval.

- **Time-bin LDA direction inference**
  - Uses cumulative spike counts to infer the intended reaching direction at each time bin.

- **Top-K trajectory prior**
  - Avoids overcommitting to a single predicted direction by combining the top-K likely empirical mean trajectories.

- **Ridge-regression displacement model**
  - Estimates local hand-position updates using recent neural activity.

- **Sigmoid online fusion**
  - Relies more on the empirical trajectory prior early in the trial and gradually shifts toward neural displacement evidence.

- **Angular geometric constraint**
  - Suppresses implausible trajectory updates and improves decoding stability.

- **Low computational cost**
  - Designed for fast training and prediction, making it suitable for low-latency BCI applications.

---

## Method Summary

The decoder follows the pipeline below:

```text
Motor-Cortical Spike Trains
          ↓
20 ms Spike Binning
          ↓
Cumulative Spike Counts
          ↓
Time-Bin LDA Direction Classification
          ↓
Top-K Weighted Empirical Trajectory Prior
          ↓
Recent-Bin Spike Features
          ↓
Ridge-Regression Displacement Estimation
          ↓
Sigmoid Fusion
          ↓
Angular Constraint
          ↓
Final 2-D Hand Position Estimate
```

---

## Dataset

The project uses motor-cortical spike trains recorded during repeated reaching movements.

Each trial contains:

- neural spike trains from **98 units**,
- aligned hand-position trajectories,
- reaching movements toward **8 target directions**,
- spike data sampled at **1 ms resolution**.

For online decoding, spike trains are binned into **20 ms intervals**, and the decoder predicts the current 2-D hand position at each evaluation step.

---

## Decoder Architecture

### 1. Spike-Train Binning

Raw spike trains are converted into 20 ms spike-count bins. This reduces the dimensionality of the neural signal while matching the online prediction interval.

For each neuron and time bin, the decoder counts the number of spikes observed within the current 20 ms window.

---

### 2. Direction Classification with LDA

A separate Linear Discriminant Analysis classifier is trained for each time bin. The classifier uses cumulative spike-count features to estimate the most likely reaching direction.

Instead of using a single hard classification, the decoder selects the top-K most likely directions and converts their LDA scores into soft weights.

This helps reduce early decoding errors, especially when neural evidence is still limited.

---

### 3. Top-K Empirical Trajectory Prior

For each reaching direction, the model computes an empirical mean hand trajectory from the training set.

During online decoding, the top-K predicted directions are combined using softmax weights to form a weighted trajectory prior.

This prior provides a stable estimate of the expected hand position, especially during early movement periods when the spike prefix is short.

---

### 4. Ridge-Regression Displacement Estimation

The decoder also uses a ridge-regression model to estimate local 2-D displacement from recent neural activity.

The regression model uses square-root-transformed spike-count features from the current and previous 20 ms bins. This provides a trial-specific correction to the empirical trajectory prior.

---

### 5. Sigmoid Online Fusion

The final position estimate is generated by fusing:

- the top-K empirical trajectory prior, and
- the ridge-regression displacement estimate.

A sigmoid weighting schedule is used:

- early decoding relies more on the trajectory prior,
- later decoding relies more on recent neural evidence.

This design balances trajectory stability with trial-specific correction.

---

### 6. Angular Geometric Constraint

To reduce unstable updates, an angular constraint is applied after fusion.

The constraint compares the proposed displacement direction with the expected direction toward the estimated target. If the proposed update deviates too far, it is projected back into an admissible movement cone.

This helps prevent sudden jumps and implausible trajectory updates during online decoding.

---

## Official Performance

| Metric | Result |
|---|---:|
| Official hidden-test RMSE | **8.686** |
| Runtime | **0.602 s** |
| Runtime ranking | **1/30** |
| RMSE ranking | **9/30** |

The decoder was designed to prioritize both accuracy and computational efficiency. Its low runtime comes from a lightweight structure based on LDA scoring, ridge-regression matrix multiplication, and deterministic post-processing.

---

## Internal Validation

The model was also evaluated using an internal held-out split of the available training data. In this validation setting, the decoder achieved:

| Metric | Result |
|---|---:|
| Internal RMSE | **9.088 raw hand-position units** |
| Approximate converted error | **0.909 cm** |
| Internal runtime | **8.273 s** |

The decoded trajectories generally followed the eight reaching directions and remained close to the actual hand trajectories for most movements.

---

## Ablation Insights

Several design choices were tested during development.

### Direction Classifier

LDA was compared against Poisson Naive Bayes and SVM. LDA provided the best trade-off between accuracy, temporal stability, and computational cost.

### Top-K Prior

Using a top-K weighted trajectory prior improved robustness compared with using only the single most likely direction. The final model used **K = 3**.

### Fusion Strategy

Sigmoid fusion outperformed fixed and linear fusion strategies by allowing the model to rely on the trajectory prior early and neural displacement estimates later.

### Geometric Constraint

The angular constraint improved decoding stability by preventing implausible movement updates, outperforming both step-length-only constraints and no-constraint baselines.

---

## Repository Structure

```text
Hybrid-Neural-Decoder-for-Hand-Trajectory-Estimation/
│
├── README.md
├── main.m
├── positionEstimator.m
├── positionEstimatorTraining.m
├── testFunction_for_students_MTb.m
├── monkeydata_training.mat
├── report/
│   └── BMI_report.pdf
├── figures/
│   ├── decoder_pipeline.png
│   ├── internal_decoding_result.png
│   ├── classifier_comparison.png
│   ├── fusion_comparison.png
│   └── constraint_comparison.png
└── results/
    └── official_result_summary.txt
```

The actual file structure may vary depending on how the repository is organized.

---

## Installation

This project is implemented in MATLAB.

Recommended environment:

- MATLAB R2021a or later
- No external machine learning toolbox required for the core implementation
- Dataset file: `monkeydata_training.mat`

Clone the repository:

```bash
git clone https://github.com/1933476828-ship-it/Hybrid-Neural-Decoder-for-Hand-Trajectory-Estimation-.git
cd Hybrid-Neural-Decoder-for-Hand-Trajectory-Estimation-
```

---

## Usage

Open MATLAB and set the repository folder as the working directory.

Train the decoder:

```matlab
modelParameters = positionEstimatorTraining(monkeydata_training);
```

Run online position estimation:

```matlab
decodedHandPos = positionEstimator(test_data, modelParameters);
```

Evaluate the decoder using the provided testing script:

```matlab
testFunction_for_students_MTb
```

---

## Main Files

### `positionEstimatorTraining.m`

Trains the decoder parameters, including:

- time-bin LDA classifiers,
- empirical mean trajectories,
- ridge-regression displacement model,
- fusion and constraint parameters.

### `positionEstimator.m`

Performs causal online decoding. At each call, it estimates the current 2-D hand position using only the spike activity observed so far.

### `testFunction_for_students_MTb.m`

Runs the evaluation protocol and reports trajectory RMSE and runtime.

---

## Relevance

This project is relevant to:

- brain-computer interfaces,
- neural decoding,
- motor intention estimation,
- prosthetic control,
- assistive robotics,
- rehabilitation robotics,
- low-latency human-machine interfaces.

Although the current decoder was developed for an offline competition dataset, the same design principles are relevant to real-time neural interfaces where computational efficiency and causal prediction are essential.

---

## Limitations

The current decoder has several limitations:

- It relies on stereotyped direction-conditioned trajectory priors.
- The displacement regressor is global rather than direction-specific.
- The fusion schedule is manually designed rather than learned.
- The angular constraint is heuristic and does not explicitly model uncertainty.
- The model does not include a full probabilistic state-space formulation such as a Kalman filter.

---

## Future Work

Potential improvements include:

- direction-specific displacement regressors,
- mixture-of-experts decoding,
- learned temporal fusion schedules,
- probabilistic trajectory constraints,
- Kalman-filter or particle-filter extensions,
- uncertainty-aware target-direction estimation,
- real-time integration with prosthetic or robotic control systems.

---

## Authors

- Crist Lian  
- Yange Sun  
- Junmou Tang  
- Shiyue Yang  

All authors contributed to algorithm development, decoder refinement, result analysis, and report writing.

---

## Disclaimer

This repository is intended for research and educational purposes. It is not a clinically validated medical device or an approved prosthetic-control system.

---

## Keywords

`BCI` `BMI` `Neural Decoding` `Motor Cortex` `Spike Trains` `Hand Trajectory Estimation` `LDA` `Ridge Regression` `Causal Decoding` `Prosthetic Control` `Rehabilitation Robotics`
