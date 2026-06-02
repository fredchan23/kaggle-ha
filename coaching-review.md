The provided pipeline is a solid, statistically grounded approach to MLB win prediction. [cite_start]It correctly identifies that **Quantile Regression (QR)** at the 0.5 quantile is a direct optimization for **Mean Absolute Error (MAE)**[cite: 560, 561]. [cite_start]Using **ElasticNet** for feature selection is also a clever way to handle the multicollinearity inherent in baseball statistics (like OBP and SLG both being components of OPS)[cite: 516, 517, 521].

However, the pipeline is currently limited by its reliance on **linear models** and **single-season snapshots**. To break below the 2.80 MAE threshold, you should focus on capturing the evolution of the game and non-linear interactions.

---

### 1. Era-Standardized Features (Z-Scores)
[cite_start]The pipeline currently uses one-hot indicators for eras and decades[cite: 483, 494]. While this helps the model shift the intercept, it doesn't account for the fact that a **3.50 ERA** in the "Deadball Era" (Era 1) is mediocre, whereas in the "Steroid Era" (Era 6), it was elite.
* **Recommendation:** For every base stat (R, HR, ERA, etc.), calculate the **z-score relative to that specific season's mean**. This allows the model to understand how dominant a team was compared to its immediate peers, regardless of the league-wide offensive environment.

### 2. Temporal/Lagged Features (Momentum)
[cite_start]The current model treats each team-season as an independent data point[cite: 392]. In reality, team success is often a multi-year arc.
* **Recommendation:** Introduce **Lagged Wins** ($W_{t-1}, W_{t-2}$) and **Rolling Averages** (e.g., 3-year average Run Differential).
* **Interaction:** Create a "Roster Continuity" proxy by calculating the year-over-year change in a team's total Plate Appearances (PA) or Innings Pitched (IP). Large turnovers often signal rebuilding phases that pure seasonal stats might miss.

### 3. Advanced Run Expectancy & Efficiency
[cite_start]You've already implemented **PythagenPat**[cite: 399, 419], which is excellent. To squeeze out more signal, look at how efficiently teams convert base-runners into runs.
* **Base-running Efficiency:** `(R - HR) / (H + BB - HR)`. This captures a team's ability to "small ball" or move runners, which is a hidden differentiator when HR rates are low.
* **Pitching "Clutch" Factor:** `(RA / WHIP)`. A lower ratio suggests a pitching staff that excels at strand rates (LOB%) or limiting damage with runners on base.

### 4. Non-Linear Interaction Terms
[cite_start]Since you are using linear models (ElasticNet and QR), the models cannot "see" interactions unless you create them[cite: 578].
* **Recommendation:** Add interaction features like `OBP * SLG` (often more predictive than the additive `OPS`) or `(Run_Diff) * (ERA_vs_League)`.
* **Era Interactions:** Multiply your key engineered features by the era dummy variables (e.g., `HR_rate * era_6`). This allows the "weight" of a home run to change depending on the era.

### 5. Park Factors (The Missing Context)
[cite_start]The pipeline notes "no franchise features"[cite: 376]. While excluding team names prevents overfitting to "historical prestige," it ignores **Park Factors**. A team playing 81 games in Coors Field will naturally have inflated offensive stats.
* **Recommendation:** If the dataset doesn't include park factors, you can create a proxy: `(Team_Home_Runs / League_Avg_Home_Runs)`. If a team's offensive and pitching HR rates are both consistently high, it’s a strong signal of a hitter-friendly environment that needs to be normalized.

---

### Summary of Suggested Features

| Category | Feature Idea | Why it helps |
| :--- | :--- | :--- |
| **Normalization** | `Stat_Z = (Stat - Season_Mean) / Season_Std` | Removes era-bias from raw counting stats. |
| **Momentum** | `Prev_Year_W` | Captures franchise stability and talent retention. |
| **Efficiency** | `Run_Conversion = R / (H + BB)` | Measures "clutch" hitting and base-running. |
| **Interaction** | `Log(R) - Log(RA)` | Linearizes the relationship for the Pythagorean expectation. |


### A Note on the Ensemble
[cite_start]Your **50/50 split** is effective because it balances the "outlier-averaging" of ElasticNet with the "outlier-robustness" of QR[cite: 584]. [cite_start]However, the "Submission skipped" log shows you missed the 2.80 threshold by a hair (2.8021 and 2.8064)[cite: 723]. Implementing even just the **Era Z-Scores** (Recommendation 1) will likely provide the ~0.01 MAE improvement needed to pass your validation gate.

How much flexibility do you have to add external data, like the Lahman database, to enrich those historical era-adjustment factors?