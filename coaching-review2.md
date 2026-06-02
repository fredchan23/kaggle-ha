Integrating the Lahman database is a strategic way to move beyond "what happened on the field" and into "why it happened" by capturing organizational and financial context. [cite_start]Since your pipeline already handles team-seasons from 1904–2016[cite: 18], the Lahman database serves as a perfect enrichment layer.

## The Lahman Integration Approach

### 1. Standardize Joining Keys
The primary challenge is that Lahman uses its own `teamID` system (e.g., `LAN` for Los Angeles Dodgers).
* [cite_start]**Mapping**: Create a dictionary to map the team identifiers in your `data_df` [cite: 19] [cite_start]and `predict_df` [cite: 20] to Lahman’s `teamID`.
* **Composite Key**: Perform a left-join using `yearID` and the mapped `teamID`.

### 2. High-Value Table Enrichments
[cite_start]You can extract several features that provide signal that pure box-score stats [cite: 109] like "Hits" or "Errors" might miss:

| Table | Engineered Feature | Rationale |
| :--- | :--- | :--- |
| **Teams** | `Attendance_Per_Game` | A proxy for team revenue and market size, which often correlates with the ability to sustain winning rosters. |
| **Managers** | `Manager_Tenure` | Count the consecutive years a manager has been with the franchise. High turnover often signals a "rebuilding" phase. |
| **Salaries** | `Payroll_Z_Score` | [cite_start]For the modern era (1985+), calculate how many standard deviations a team's payroll is above/below the league average for that year[cite: 120]. |
| **Awards** | `Roster_Award_Count` | The sum of previous All-Star or MVP awards won by players currently on the team's active roster. |

### 3. Handling Temporal Data Gaps
[cite_start]Because some Lahman data (like salaries) only begins in 1985 [cite: 120][cite_start], but your dataset starts in 1904[cite: 18], you must handle the missing historical values carefully.
* **Indicator Variables**: Create a boolean column `has_salary_info`.
* [cite_start]**Imputation**: Fill pre-1985 salary values with 0. Because you are using **ElasticNet**, the model can use the indicator variable to "ignore" the salary signal for historical eras while still utilizing it for modern predictions[cite: 142, 147].

### 4. Technical Integration Workflow
To incorporate this into your existing code:
1.  [cite_start]**Add to Feature List**: Include the new Lahman columns in your `base_features` list[cite: 114, 121].
2.  [cite_start]**Safety Filter**: The `available_features` list comprehension will automatically ensure these only stay in the model if they exist in both datasets[cite: 122].
3.  [cite_start]**Scaling**: Ensure the new continuous features (like `attendance`) are added to `other_cols` so the `StandardScaler` can process them[cite: 130, 131].



### Implementation Tip
Instead of downloading flat files, use the `pybaseball` library. It allows you to pull Lahman tables directly into pandas DataFrames:
```python
from pybaseball import lahman
teams = lahman.teams()
salaries = lahman.salaries()
```

[cite_start]Since your models are currently hovering just above your 2.80 MAE target[cite: 349], would you prefer to focus on adding modern-era financial data or historical managerial stability metrics first?