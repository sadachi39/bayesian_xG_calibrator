# Bayesian Inference of Soccer Expected Goals (xG)

This repository contains a Jupyter Notebook implementing a Bayesian framework for estimating soccer Expected Goals (xG) using shot-level event data from La Liga.

The project applies Bayesian logistic regression to quantify goal probability while explicitly capturing uncertainty through posterior distributions and posterior predictive sampling.

## Project Overview

Expected Goals (xG) measures the probability that a given shot results in a goal based on historical data. Instead of producing only point estimates (as in frequentist models), this project uses Bayesian inference to estimate full posterior distributions over model parameters and predicted xG values.

Two models are implemented:

- **Model 1 (Spatial Model):** Uses only shot distance and shot angle
- **Model 2 (Extended Model):** Adds a binary feature for whether the shot used a "special technique" (e.g., overhead kick, volley, backheel)

## Dataset

The dataset includes all recorded shots from **522 La Liga matches** across **five seasons (2014/15–2018/19)**.

- Total shots: **12,034**
- Goals: **1,349**
- Conversion rate: **~11.2%**
- Source: StatsBomb Open Data via the `statsbombpy` package

### Feature Engineering

From raw shot location coordinates, the following features are computed:

- **Distance to goal center** (Euclidean distance)
- **Shot angle** (visible width of the goal mouth, computed via law of cosines)
- **Technique indicator** (binary variable distinguishing normal vs. special technique shots)

Distance and angle are standardized to mean 0 and variance 1.

## Methodology

Both models use Bayesian logistic regression:

### Model 1
$$
\text{logit}(p_i) = \beta_0 + \beta_{dist}x_{dist} + \beta_{angle}x_{angle}
$$

### Model 2
$$
\text{logit}(p_i) = \beta_0 + \beta_{dist}x_{dist} + \beta_{angle}x_{angle} + \beta_{tech}x_{tech}
$$

### Priors

All coefficients use weakly informative priors:

$$
\beta_j \sim \mathcal{N}(0, 2)
$$

### Inference

Inference is performed using **PyMC** with the **No-U-Turn Sampler (NUTS)**:

- 4 chains
- 1,000 tuning steps
- 2,000 posterior draws per chain

Convergence is verified using:

- Trace plots
- R-hat (all parameters achieved **R̂ = 1.00**)
- Effective sample size (ESS > 3,000)

## Results Summary

Key findings from posterior estimates:

- Shot distance is the strongest predictor of scoring probability (negative effect)
- Shot angle has a positive effect
- Special technique has a slightly positive posterior mean but its 95% credible interval includes 0, meaning the effect is not statistically conclusive

## Posterior Predictive Scenario Analysis

The notebook evaluates iconic soccer shot scenarios such as:

- Back-post tap-in
- Penalty spot shot
- Gareth Bale’s 2018 overhead kick
- Edge-of-the-box winger shot
- David Beckham’s 2001 free kick vs Greece

Posterior predictive distributions show realistic uncertainty and correctly assign very low xG values to long-range efforts.

## Repository Contents

- `main.ipynb` — Jupyter Notebook containing:
  - data import and preprocessing
  - feature engineering (distance, angle, technique)
  - Bayesian model specification in PyMC
  - posterior sampling and diagnostics
  - posterior predictive simulation and scenario evaluation

## Installation

Install dependencies with:

```bash
pip install pymc arviz numpy pandas matplotlib statsbombpy
