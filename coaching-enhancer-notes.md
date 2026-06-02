Here's the comparison. The coaching session lists 5 enhancer categories. I'll mark what's been applied and what hasn't.

**1. Feature Selection** -- Partially applied.
You used ElasticNet's embedded L1 selection (32/68 features zeroed). But the coaching session also suggests mutual information filter (`SelectKBest`) and RFE wrapper (`RFE` with Ridge). Neither of those was tried.

**2. PCA on Selected Features** -- Not applied.
No PCA anywhere in the README. The coaching suggests PCA (95% variance threshold) on the selected feature subset to compress correlated features into orthogonal components.

**3. Feature Engineering** -- Partially applied. Specifically, you have NOT done:
- **Interaction features** (e.g., WHIP x FP, OPS x offense_index)
- **Group-by aggregation** (franchise-level mean/std/min/max of R, RA, OPS merged back)
- **Delta / year-over-year change** features (e.g., OPS minus prior year OPS)
- **Normalization vs franchise historical average** (e.g., R / franchise_mean_R)
- **Polynomial features** via `PolynomialFeatures(degree=2)` on key features

You did domain features (Pythagorean, FIP, OBP, SLG, ISO, etc.), league-average normalization, and rate stats. But the five sub-categories above are untouched.

**4. Cluster Analysis** -- Not applied.
Neither K-Means distance features nor cluster-level aggregates.

**5. Advanced Workflows** -- Partially applied.
- **VotingRegressor**: Not used. You did manual 50-50 averaging of QR+EN predictions, which is conceptually similar but not via `VotingRegressor`.
- **Stacking**: Tried and failed (public 3.07 with GBM base learners). The coaching suggests linear base learners, which aligns with your own "Remaining Plausible Directions" note about linear-only stacking.
- **Optuna**: Not used. You used `GridSearchCV` and `ElasticNetCV`. Optuna's Bayesian search could be more efficient, especially for LightGBM tuning.

**Realistic assessment for your dataset**: Several of these (franchise group-by, YoY deltas, cluster features) require `franchID` or temporal ordering that may or may not be in your feature set, and they risk overfitting on 1,812 rows. PCA and polynomial features are the most directly applicable to your current pipeline with minimal leakage risk. Mutual information filtering is also cheap to try as a sanity check against your ElasticNet selection.