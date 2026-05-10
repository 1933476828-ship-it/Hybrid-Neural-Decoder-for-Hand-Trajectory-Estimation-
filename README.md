# Hybrid Neural Decoder for Hand Trajectory Estimation

A lightweight causal brain-computer interface (BCI) decoder for continuous 2-D hand trajectory estimation from motor-cortical spike trains.

This project was developed for a neural decoding competition. The task was to reconstruct hand position during reaching movements using only neural spike activity available up to the current prediction time. The final method combines direction classification, empirical trajectory priors, displacement regression, online fusion, and geometric regularization to produce stable low-latency trajectory estimates.

In the official hidden-test evaluation, the decoder achieved an RMSE of **8.686**, ranked **9/30** in RMSE, and ranked **1/30** in total training and prediction time with a runtime of **0.602 s**.

---

## Overview

Brain-computer interfaces require causal decoders that translate neural activity into continuous control signals for external devices, such as prosthetic arms, robotic manipulators, cursors, or rehabilitation systems.

In this project, the decoder estimates the current 2-D hand position from motor-cortical spike trains recorded during reaching movements. The key constraint is that prediction must be **causal**: at each decoding step, the model can only use the spike activity observed up to the current time point.

The implemented decoder is designed to balance three goals:

- **Accuracy**: follow the actual 2-D hand trajectory as closely as possible.
- **Stability**: avoid sudden implausible jumps during online decoding.
- **Speed**: keep training and prediction computationally lightweight.

---

## Method Summary

The decoder uses a hybrid direction-and-displacement strategy.

```text
Motor-Cortical Spike Trains
        ↓
20 ms Spike Binning
        ↓
Cumulative Spike Counts
        ↓
Time-Bin LDA Direction Classification
        ↓
Top-K Weighted Mean-Trajectory Prior
        ↓
Recent-Bin Spike Features
        ↓
Ridge-Regression Displacement Estimate
        ↓
Sigmoid Prior-Regression Fusion
        ↓
Angular Geometric Constraint
        ↓
Final 2-D Hand Position Estimate
```

---

## Core Implementation

### `positionEstimatorTraining.m`

This function trains all model parameters from the training trials.

It performs:

- 20 ms spike binning,
- cumulative spike-count feature construction,
- empirical mean trajectory estimation for each reaching direction,
- final target-position estimation,
- time-bin-specific LDA training,
- global ridge-regression training for local displacement prediction,
- angular constraint parameter setup.

The main learned parameters include:

- `ldaW` and `ldaC`: LDA weights and intercepts for direction classification,
- `meanTraj`: empirical mean trajectory for each direction,
- `targetPos`: average final target positions,
- `regW`: ridge-regression weights for displacement estimation,
- `cosThresh` and `sinThresh`: angular constraint thresholds.

---

### `positionEstimator.m`

This function performs online causal decoding.

At each prediction call, it receives the spike prefix available so far and returns the estimated current hand position.

The online decoding procedure includes:

1. **Spike feature extraction**
   - Uses cumulative spike counts for direction classification.
   - Uses the two most recent 20 ms spike bins for displacement regression.

2. **LDA direction scoring**
   - Computes direction scores from cumulative spike counts.
   - Selects the top-3 most likely reaching directions.

3. **Top-K trajectory prior**
   - Combines the top-3 empirical mean trajectories using softmax weights.
   - Provides a stable prior estimate of hand position.

4. **Ridge-regression displacement prediction**
   - Uses square-root-transformed recent spike counts.
   - Predicts local 2-D displacement from the previously decoded position.

5. **Sigmoid fusion**
   - Early predictions rely more on the empirical trajectory prior.
   - Later predictions rely more on the regression-based displacement estimate.

6. **Angular constraint**
   - Suppresses implausible updates that deviate too far from the estimated target direction.
   - Helps reduce unstable jumps during online decoding.

The function uses persistent variables to maintain information across repeated calls within the same trial, including cumulative spike counts and recent spike-bin history.

---

### `testFunction_for_students_MTb.m`

This script evaluates the decoder.

It:

- loads the monkey reaching dataset,
- splits trials into training and testing sets,
- trains the decoder using `positionEstimatorTraining`,
- calls `positionEstimator` causally every 20 ms from 320 ms onward,
- plots decoded trajectories against actual hand trajectories,
- computes RMSE,
- reports total runtime.

---

## Dataset

The dataset contains neural and kinematic recordings from reaching movements.

Each trial includes:

- spike trains from **98 neural units**,
- hand-position trajectories,
- reaching movements toward **8 target directions**,
- spike data sampled at **1 ms resolution**.

The decoder bins spike trains into **20 ms intervals** and predicts hand position at the same temporal resolution.

---

## Decoder Details

### 1. Spike Binning

Raw spike trains are converted into 20 ms spike-count bins. This reduces the neural data dimensionality and matches the online evaluation interval.

### 2. Time-Bin LDA Direction Classification

For each time bin, a regularized Linear Discriminant Analysis classifier is trained using cumulative spike counts. This allows the decoder to estimate the intended reaching direction at each point in time.

### 3. Top-3 Weighted Trajectory Prior

Instead of trusting only the single most likely direction, the decoder selects the top-3 LDA directions and combines their empirical mean trajectories using softmax weights. This improves robustness when early neural evidence is uncertain.

### 4. Ridge-Regression Displacement Model

A global ridge-regression model predicts local 2-D displacement using square-root-transformed spike counts from the two most recent bins. This adds trial-specific correction beyond the average trajectory prior.

### 5. Sigmoid Online Fusion

The decoder fuses the empirical trajectory prior and the regression displacement estimate using a sigmoid weighting schedule.

This design reflects the idea that:

- early in the movement, the prior trajectory is more reliable;
- later in the movement, recent neural evidence provides useful trial-specific correction.

### 6. Angular Geometric Constraint

After fusion, an angular constraint checks whether the proposed movement update deviates too far from the estimated target direction.

If the update is implausible, it is projected back toward an admissible direction and capped with a smaller step length. Otherwise, only a broader maximum step constraint is applied.

This improves trajectory stability while keeping the decoder computationally simple.

---

## Performance

### Official Hidden-Test Evaluation

| Metric | Result |
|---|---:|
| RMSE | **8.686** |
| RMSE ranking | **9/30** |
| Total training and prediction time | **0.602 s** |
| Runtime ranking | **1/30** |

### Internal Validation

Using an internal train-test split, the decoder achieved:

| Metric | Result |
|---|---:|
| Internal RMSE | **9.088** |
| Approximate converted error | **0.909 cm** |

The decoded trajectories generally followed the main reaching directions and remained close to the actual hand trajectories for most trials.

---

## Why This Approach Works

The decoder performs well because it combines complementary sources of information:

- cumulative spike counts provide stable direction information;
- LDA gives fast and reliable reach-direction inference;
- empirical mean trajectories exploit the structured reaching task;
- ridge regression captures local trial-specific displacement;
- sigmoid fusion balances prior stability and neural correction;
- angular constraints reduce unstable online updates.

This makes the method lightweight, robust, and suitable for low-latency BCI-style decoding.

---

## Usage

Place the following files in the same folder named `BMI`:

```text
Hybrid-Neural-Decoder-for-Hand-Trajectory-Estimation/
├── testFunction_for_students_MTb.m
└── BMI/
    ├── positionEstimatorTraining.m
    ├── positionEstimator.m
    └── monkeydata_training.mat
```

Then open MATLAB, set the repository root folder as the current working directory, and run:


```matlab
testFunction_for_students_MTb
```

The script will automatically train and evaluate the decoder. Specifically, it will:

- load `monkeydata_training.mat`,
- train the decoder using `positionEstimatorTraining.m`,
- perform causal online decoding using `positionEstimator.m`,
- plot the decoded hand trajectories against the actual trajectories,
- report the final RMSE and runtime.

## Requirements

- MATLAB
- `monkeydata_training.mat`
- No external deep learning framework is required

The implementation is intentionally lightweight and relies on classical machine learning and regression methods rather than large neural networks.

---

## Relevance

This project is relevant to:

- brain-computer interfaces,
- brain-machine interfaces,
- neural decoding,
- motor intention estimation,
- prosthetic control,
- assistive robotics,
- rehabilitation robotics,
- low-latency human-machine interfaces.

Although developed for a competition dataset, the same ideas are applicable to neural interfaces where real-time prediction, low computational cost, and causal decoding are important.

---

## Limitations

The current method has several limitations:

- It relies on stereotyped direction-specific mean trajectories.
- The displacement regression model is global rather than direction-specific.
- The fusion schedule is manually designed instead of learned.
- The angular constraint is heuristic and does not explicitly model uncertainty.
- It does not include a full probabilistic state-space model such as a Kalman filter.

---

## Future Work

Potential improvements include:

- direction-specific displacement regressors,
- mixture-of-experts decoding,
- learned fusion schedules,
- uncertainty-aware target estimation,
- probabilistic trajectory constraints,
- Kalman-filter or particle-filter extensions,
- integration with real-time prosthetic or robotic control systems.

---

## Authors

- Crist Lian
- Yange Sun
- Junmou Tang
- Shiyue Yang


---

## Disclaimer

This repository is intended for research and educational purposes. It is not a clinically validated medical device or an approved prosthetic-control system.

The neural data have been generously provided by the laboratory of Prof. Krishna Shenoy at Stanford University. The data are to be used exclusively for educational purposes in the BIOE70011 Brain Machine Interfaces 2025--2026 course.

---

## Keywords

`BCI` `BMI` `Neural Decoding` `Motor Cortex` `Spike Trains` `Hand Trajectory Estimation` `LDA` `Ridge Regression` `Causal Decoding` `Prosthetic Control` `Rehabilitation Robotics`
