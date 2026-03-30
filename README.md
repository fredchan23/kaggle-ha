# Kaggle-HA Approach Report: MLB Wins Prediction

**Best Kaggle Public Score: 2.98765 MAE** (50-50 Average: QR + ElasticNet)

## Dataset

- **Training:** 1,812 team-seasons (data.csv, 51 columns)
- **Prediction:** 453 team-seasons (predict.csv, 44 columns, no target `W`)
- **Target:** `W` (season wins, range 36–116, mean ~79)

## Approach Summary

```
Data Loading (1,812 x 51)
    |
Feature Engineering (24 domain features)
    |
Feature Selection (68 total: 44 base + 24 engineered)
    |
Train/Test Split (80/20, random_state=42)
    |
StandardScaler (continuous only, not one-hot)
    |
RidgeCV (alpha=0.9770, 5-fold CV)
    |
Retrain on all 1,812 samples
    |
Predict, round to integer --> Kaggle submission
```

## Feature Engineering (24 Engineered Features)

### Pythagorean Win Expectancy (4 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pythag_wp` | R^1.83 / (R^1.83 + RA^1.83) | Baseball-research optimal exponent (not the textbook 2.0) |
| `pyth_wins` | pythag_wp x G | Expected wins from run production |
| `pythagenport_wp` | R^exp / (R^exp + RA^exp), exp = ((R+RA)/G)^0.287 | Davenport's scoring-environment-adjusted formula |
| `pythagenport_wins` | pythagenport_wp x G | Environment-adjusted expected wins |

### Offensive Metrics (11 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `run_diff` | R - RA | Most direct predictor of wins |
| `run_diff_pg` | run_diff / G | Per-game normalisation |
| `batting_avg` | H / AB | Traditional batting measure |
| `obp_proxy` | (H + BB) / (AB + BB) | On-Base Percentage |
| `slg_proxy` | (1B + 2x2B + 3x3B + 4xHR) / AB | **Corrected SLG** — singles counted explicitly |
| `ops_proxy` | OBP + SLG | On-Base Plus Slugging |
| `iso` | SLG - BA | Isolated Power (extra-base hit contribution) |
| `hr_rate_off` | HR / AB | Home run rate |
| `bb_rate` | BB / PA | Walk rate |
| `so_rate` | SO / PA | Strikeout rate |

### Pitching / Defense Metrics (9 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `whip` | (BBA + HA) / IP | Walks + Hits per Inning Pitched |
| `k_bb_ratio` | SOA / BBA | Strikeout-to-walk ratio |
| `hr_rate_def` | HRA / IP | Home runs allowed rate |
| `fip_proxy` | (13xHRA + 3xBBA - 2xSOA) / IP + 3.2 | Fielding Independent Pitching |
| `era_vs_league` | ERA / mlb_rpg | ERA normalised against league average |
| `sv_rate` | SV / G | Save rate |
| `cg_rate` | CG / G | Complete game rate |
| `sho_rate` | SHO / G | Shutout rate |
| `r_vs_lg` / `ra_vs_lg` | R/G - mlb_rpg, RA/G - mlb_rpg | Offence/defence vs league average |

### Base Features (44)

- **Counting stats (10):** G, R, AB, H, 2B, 3B, HR, BB, SO, SB
- **Pitching stats (11):** RA, ER, ERA, CG, SHO, SV, IPouts, HA, HRA, BBA, SOA
- **Fielding (3):** E, DP, FP
- **League context (1):** mlb_rpg
- **Era indicators (8):** era_1 through era_8
- **Decade indicators (11):** decade_1910 through decade_2010

## Data Preprocessing

1. **Feature parity enforced:** Only features present in both data.csv and predict.csv are used
2. **Selective StandardScaler:** Continuous features only. One-hot encoded era/decade columns are NOT scaled (scaling binary indicators breaks interpretability)
3. **Division-by-zero protection:** Epsilon (1e-9) in denominators; `.replace(0, NaN)` then `.fillna(0)` for ratio features

## Models

### 1. Ridge Regression (L2 Regularisation) — Public 3.03292

- Alpha search: `np.logspace(-3, 5, 100)` — 100 values from 0.001 to 100,000
- Selection: RidgeCV with 5-fold cross-validation
- **Selected alpha: 0.9770**
- Final model: Retrained on all 1,812 samples with the selected alpha

**Why Ridge?** Multicollinearity is inherent in baseball stats (run_diff, Pythagorean expectancy, and ERA all correlate). L2 penalty distributes coefficient weight across correlated features instead of arbitrarily picking one. With 1,812 samples and 68 features, regularisation prevents overfitting.

| Metric | Value |
|--------|-------|
| 80/20 split test MAE | 2.8057 |
| 80/20 split test R² | 0.9225 |
| 10-fold CV MAE (full data) | 2.7126 +/- 0.1565 |
| **Kaggle Public Score** | **3.03292** |
| Local-to-public gap | ~0.23 MAE |

### 2. ElasticNet (L1 + L2) — Public 3.02880 (Current Best)

- Alpha/l1_ratio search: `ElasticNetCV` with 10 l1_ratios (0.01–0.99) x 60 alphas (1e-4 to 100) = 600 combinations, 5-fold CV
- **Selected alpha: 0.005356, l1_ratio: 0.99** (nearly pure Lasso)
- Feature selection: **32/68 coefficients zeroed out**, 36 active features retained
- Final model: Retrained on all 1,812 samples with the selected hyperparameters

**Why ElasticNet improved on Ridge:** While Ridge distributes weight across all 68 features (including redundant ones), ElasticNet's L1 component aggressively pruned nearly half the features. At l1_ratio=0.99, the model is almost pure Lasso — the data strongly preferred sparsity over weight-sharing. The small L2 residual (0.01) stabilises correlated survivors without keeping noise contributors.

| Metric | Value |
|--------|-------|
| 80/20 split test MAE | 2.8015 |
| 80/20 split test R² | 0.9226 |
| 10-fold CV MAE (full data) | 2.7055 +/- 0.1566 |
| **Kaggle Public Score** | **3.02880** |
| Local-to-public gap | ~0.21 MAE |

### 3. HuberRegressor (Robust Loss + L2) — Public 3.04526

- Tuning: `GridSearchCV` with 8 epsilon values (1.1–3.0) x 10 alpha values (1e-4–5.0) = 80 combos, 5-fold CV
- Tested two variants: all 68 features and ElasticNet-selected 36 features

**Why Huber did NOT improve:** The hypothesis was that robust loss (linear penalty for large residuals) would stabilise coefficients against outlier team-seasons. In practice:
1. With 1,812 samples, individual outliers don't distort MSE-trained coefficients enough to matter
2. HuberRegressor only supports L2 penalty — it lost ElasticNet's L1 feature selection, reintroducing the noise from redundant features
3. The "outlier" seasons in baseball are not extreme enough (in a statistical sense) to make the MSE-to-Huber switch pay off

| Metric | Value |
|--------|-------|
| **Kaggle Public Score** | **3.04526** |

### 4. PoissonRegressor (Count-Data GLM + L2) — Not Submitted

- Tuning: `GridSearchCV` with 13 alpha values (1e-5 to 10.0), 5-fold CV
- Tested two variants: all 68 features and ElasticNet-selected 36 features
- Best variant: Poisson on all features, test MAE = 3.0248

**Why Poisson did NOT improve:** The hypothesis was that a log-link GLM (correct for count data) would better model wins. In practice:
1. MLB wins cluster around 70–90, far from zero — the Poisson count-data assumption adds no value in this range. The target is effectively normally distributed.
2. The log-link `pred = exp(X @ coef)` introduces unnecessary nonlinearity for a near-linear relationship
3. Like Huber, only L2 penalty available — no feature selection
4. Very slow convergence with `lbfgs` solver on scaled features (10+ minutes for 10-fold CV)

| Metric | Value |
|--------|-------|
| 80/20 split test MAE | 3.0248 |
| Status | Not submitted (above 2.81 threshold) |

### 5. Two-Stage: ElasticNet Selection → RidgeCV Refit — Public 3.03703

- Stage 1: ElasticNet (same as Model 2) selects 36 features
- Stage 2: RidgeCV refitted on only those 36 features (unbiased L2 coefficients)
- Also tested OLS (no regularisation) as Stage 2 — Ridge won locally

**Why two-stage did NOT improve:** The hypothesis was that L1's coefficient shrinkage bias hurts prediction accuracy. In practice, ElasticNet's L1 shrinkage acts as *additional regularisation* that generalises better to the public set. Removing it (via Ridge/OLS refit) produced locally similar MAE but worse public score. The "bias" is beneficial — it prevents the model from over-committing to coefficient magnitudes that don't transfer.

**Key insight:** More regularisation + sparsity = better public score on this dataset. The local-to-public gap is the binding constraint, not coefficient bias.

| Metric | Value |
|--------|-------|
| 80/20 split test MAE | ~2.80 |
| **Kaggle Public Score** | **3.03703** |

### 6. Quantile Regression (Median, L1) — Public 3.01646 (Current Best)

- `QuantileRegressor(quantile=0.5)` — directly optimises MAE (median absolute deviation)
- Tuning: `GridSearchCV` with 13 alpha values (0 to 10.0), 5-fold CV
- Tested both: ElasticNet-selected 36 features and all 68 features
- `solver='highs'` — linear programming solver, fast and exact

**Why Quantile Regression improved:** All previous models (Ridge, ElasticNet, Huber, Poisson) optimise MSE or variants, but Kaggle scores MAE. MSE penalises large errors quadratically, pulling coefficients toward minimising big misses at the expense of typical predictions. Median regression (quantile=0.5) optimises the exact metric being evaluated — it finds coefficients that minimise the sum of absolute deviations. This loss-function alignment gave the largest single improvement in the entire exploration (+0.012 over ElasticNet).

The L1 penalty in `QuantileRegressor` also provides sparsity, consistent with the finding that aggressive feature pruning helps on this dataset.

| Metric | Value |
|--------|-------|
| **Kaggle Public Score** | **3.01646** |

## Validation Results (Summary)

| Model | Test MAE | 10-fold CV MAE | Public Score | Gap |
|-------|----------|----------------|--------------|-----|
| Ridge (68 feat) | 2.8057 | 2.7126 | 3.03292 | 0.23 |
| ElasticNet (68→36 feat) | 2.8015 | 2.7055 | 3.02880 | 0.21 |
| ElasticNet fine grid | ~2.80 | — | 3.03292 | — |
| Two-stage Ridge (36 feat) | ~2.80 | — | 3.03703 | — |
| Huber | — | — | 3.04526 | — |
| Poisson | 3.0248 | — | Not submitted | — |
| Quantile Regression | — | — | 3.01646 | — |
| Avg 60-40 QR+EN | — | — | 3.01646 | — |
| **Avg 50-50 QR+EN** | — | — | **2.98765** | — |

## Key Decisions That Led to 3.03 → 2.98765

### 1. Pythagorean exponent of 1.83

The textbook Pythagorean formula uses exponent 2.0. Baseball research (Davenport, Miller) found 1.83 is empirically optimal for MLB. This single change improved Kaggle score vs the exponent-2 version.

### 2. Correct SLG formula

The original SLG calculation was incorrect. The fix explicitly counts singles:
```
singles = H - 2B - 3B - HR
SLG = (singles + 2*2B + 3*3B + 4*HR) / AB
```
This leaves less signal on the table compared to incorrect approximations.

### 3. Training on all 1,812 samples for submission

Rather than submitting predictions from a model trained on 80% of data, the final model is retrained on all available data. This gives the model ~20% more training examples, which matters at this sample size.

### 4. Rounding predictions to integers

MLB wins are whole numbers. Rounding predictions to the nearest integer reduces MAE on a target that is inherently discrete.

### 5. ElasticNet feature selection (l1_ratio=0.99)

The data chose near-pure Lasso over Ridge. Zeroing out 32 of 68 features removed redundant signals (e.g., keeping `pyth_wins` but dropping `run_diff_pg`) and reduced noise on the unseen test partition. This narrowed the local-to-public gap from 0.23 to 0.21.

### 6. Loss function alignment — Quantile Regression

All MSE-based models (Ridge, ElasticNet) optimise the wrong objective for an MAE-scored competition. Switching to `QuantileRegressor(quantile=0.5)` directly minimises MAE. This was the largest single improvement: 3.029 → 3.016 (+0.012).

### 7. Prediction averaging (QR + ElasticNet, 50-50)

The largest single improvement in the entire exploration: 3.016 → **2.988** (+0.029). Simple arithmetic mean of QR and ElasticNet submission predictions, rounded to integers. No retraining, no stacking meta-learner, no risk of overfitting.

**Why it works:** QR (MAE loss) and ElasticNet (MSE loss) optimise different objectives, producing genuinely complementary errors. Where one overshoots, the other tends to undershoot. Averaging cancels these independent errors. The 50-50 split outperformed 60-40 QR-weighted (3.016) — the ElasticNet component carries equal weight because it provides orthogonal error correction, not just a weaker version of QR.

**Why 60-40 failed:** Tilting toward QR produced the same score as pure QR (3.016). The improvement comes specifically from equal blending of two different loss functions, not from weighting toward the locally "better" model.

## What Did NOT Work (Lessons from kaggle-lr and kaggle-automl)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Tree-based models (RF, XGBoost, LightGBM, CatBoost) | Test MAE 3.1–3.2+ | Overfit on 1,812 samples; engineered features already encode the nonlinearity trees would learn |
| Stacking ensembles | Test MAE ~2.84, public ~3.07 | Base learners (GBMs) overfit, so stacking amplifies noise |
| More features (79 in kaggle-automl) | Public ~3.06 | Extra features add noise with only 1,812 samples |
| FLAML / H2O AutoML | Public ~3.05 | AutoML converges to Ridge/GLM anyway; tree models it tries overfit |
| Longer H2O training (600s vs 60s) | No change | Leader model (GLM) converges in seconds |
| Park-adjusted Pythagorean (extra variants) | No improvement | Marginal signal drowned by added dimensionality |
| HuberRegressor (robust loss) | Public 3.04526 | Outlier team-seasons not extreme enough to distort MSE; L2-only penalty lost ElasticNet's feature selection benefit |
| PoissonRegressor (count GLM) | Test MAE 3.0248 | Wins ~70–90 are effectively normal, not Poisson; log-link adds unnecessary nonlinearity; L2-only; very slow |
| Two-stage (ElasticNet→Ridge) | Public 3.03703 | Removing L1 shrinkage bias also removed beneficial regularisation; model over-commits to coefficient magnitudes |
| ElasticNet fine grid (zoom on 0.99/0.005) | Public 3.03292 | Original coarse grid already near-optimal; finer tuning just fits CV noise |

## The Local-to-Public Gap (Structural, Not Fixable)

The ~0.23 MAE gap between local CV (2.71) and Kaggle public (3.03) is consistent across all models and approaches. Adversarial validation (AUC = 0.48) confirms there is no feature-level distribution shift between train and test.

The gap is likely caused by:
- Year-on-year variance in baseball scoring environments within eras
- Random split giving a slightly "easier" local test set than the Kaggle partition
- Irreducible noise — some teams deviate from their stats in ways no model can predict

## Conclusion

The **2.988** score was achieved through **domain knowledge + loss diversity + ensemble**:
- Baseball-research-informed features (1.83 exponent, correct SLG, rate stats, FIP)
- ElasticNet with L1 sparsity (l1_ratio=0.99, pruning 32/68 features)
- Quantile regression directly optimising the competition's MAE metric
- 50-50 prediction averaging of QR + ElasticNet for complementary error cancellation
- Careful preprocessing (selective scaling, feature parity)
- Full-data retraining for both component models

The ceiling for this dataset appears to be approximately **2.95–2.99** on the public leaderboard.

## Explored & Exhausted Avenues

- ~~Prediction averaging (QR+EN)~~ — **Public 2.988**, now the best score
- ~~Quantile regression~~ — **Public 3.016**, second best individually
- ~~Finer l1_ratio/alpha grid~~ — Public 3.033, original coarse grid was already optimal
- ~~Two-stage (ElasticNet→Ridge)~~ — Public 3.037, L1 shrinkage is beneficial not harmful
- ~~HuberRegressor~~ — Public 3.045, outliers not extreme enough
- ~~PoissonRegressor~~ — Test 3.025, wrong distributional assumption
- ~~Tree-based / AutoML~~ — Public 3.05+, overfit on 1,812 samples
- ~~60-40 QR-weighted average~~ — Public 3.016, equal weight is critical

## Remaining Plausible Directions

1. **Three-model average (QR + ElasticNet + Ridge)** — Ridge makes different errors from both QR and ElasticNet. A three-way equal average could cancel more noise. Free to try — all submission files exist.

2. **Target transformation** — Predict win percentage (W/G) then multiply back by G. Normalises across season lengths and eras.

3. **Stability selection** — Bootstrap ElasticNet many times, keep features selected in >80% of runs. More robust feature selection than a single L1 fit.

4. **Linear-only stacking** — Ridge + ElasticNet + QR base learners with a Ridge meta-learner. Low overfitting risk, might learn optimal blending weights beyond 50-50.
