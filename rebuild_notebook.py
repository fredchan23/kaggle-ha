import json

with open('moneyball-starter-code-share.ipynb') as f:
    nb = json.load(f)

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

# Grab original cells 0 (setup/imports) and 1 (load data)
original_setup    = nb['cells'][0]
original_loaddata = nb['cells'][1]

CELL_ENG = """def add_engineered_features(df):
    d = df.copy()
    G  = d['G']
    IP = d['IPouts'] / 3
    PA = d['AB'] + d['BB']

    d['run_diff']        = d['R'] - d['RA']
    d['run_diff_pg']     = d['run_diff'] / G
    # Fixed 1.83 exponent: baseball-research optimal for MLB
    d['pythag_wp']       = d['R']**1.83 / (d['R']**1.83 + d['RA']**1.83 + 1e-9)
    d['pyth_wins']       = d['pythag_wp'] * G
    # PythagenPat: dynamic exponent = ((R+RA)/G)^0.287
    dyn_exp = ((d['R'] + d['RA']) / (G + 1e-9)) ** 0.287
    d['pythagenport_wp'] = d['R']**dyn_exp / (d['R']**dyn_exp + d['RA']**dyn_exp + 1e-9)
    d['pythagenport_wins'] = d['pythagenport_wp'] * G

    d['batting_avg']     = d['H'] / d['AB']
    d['obp_proxy']       = (d['H'] + d['BB']) / PA
    # Correct SLG: singles counted explicitly
    singles              = d['H'] - d['2B'] - d['3B'] - d['HR']
    d['slg_proxy']       = (singles + 2*d['2B'] + 3*d['3B'] + 4*d['HR']) / (d['AB'] + 1e-9)
    d['ops_proxy']       = d['obp_proxy'] + d['slg_proxy']
    d['iso']             = d['slg_proxy'] - d['batting_avg']  # isolated power
    d['hr_rate_off']     = d['HR'] / d['AB']
    d['bb_rate']         = d['BB'] / PA
    d['so_rate']         = d['SO'] / PA

    d['whip']            = (d['BBA'] + d['HA']) / IP
    d['k_bb_ratio']      = d['SOA'] / d['BBA'].replace(0, float('nan'))
    d['hr_rate_def']     = d['HRA'] / IP
    d['fip_proxy']       = (13*d['HRA'] + 3*d['BBA'] - 2*d['SOA']) / IP + 3.2
    d['era_vs_league']   = d['ERA'] / d['mlb_rpg'].replace(0, float('nan'))

    d['sv_rate']         = d['SV'] / G
    d['cg_rate']         = d['CG'] / G
    d['sho_rate']        = d['SHO'] / G
    d['r_vs_lg']         = d['R'] / G - d['mlb_rpg']
    d['ra_vs_lg']        = d['RA'] / G - d['mlb_rpg']

    return d.fillna(0)

data_df    = add_engineered_features(data_df)
predict_df = add_engineered_features(predict_df)

eng_cols = [
    'run_diff','run_diff_pg',
    'pythag_wp','pyth_wins','pythagenport_wp','pythagenport_wins',
    'batting_avg','obp_proxy','slg_proxy','ops_proxy','iso',
    'hr_rate_off','bb_rate','so_rate',
    'whip','k_bb_ratio','hr_rate_def','fip_proxy','era_vs_league',
    'sv_rate','cg_rate','sho_rate','r_vs_lg','ra_vs_lg'
]
print(f"Engineered {len(eng_cols)} new features.")
"""

CELL_FEATURES = """import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score

base_features = [
    'G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
    'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA', 'SOA',
    'E', 'DP', 'FP', 'mlb_rpg',
    'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8',
    'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
    'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990', 'decade_2000', 'decade_2010',
] + eng_cols

available_features = [c for c in base_features if c in data_df.columns and c in predict_df.columns]
print(f"Total features: {len(available_features)}")

X = data_df[available_features]
y = data_df['W']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
"""

CELL_SCALE = """from sklearn.preprocessing import StandardScaler

one_hot_cols = [col for col in X_train.columns if col.startswith(('era_', 'decade_'))]
other_cols   = [col for col in X_train.columns if col not in one_hot_cols]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[other_cols] = scaler.fit_transform(X_train[other_cols])
X_test_scaled[other_cols]  = scaler.transform(X_test[other_cols])
print(f"Scaling {len(other_cols)} continuous features; {len(one_hot_cols)} binary cols left unscaled.")
"""

CELL_FIT = """alphas = np.logspace(-3, 5, 100)

ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
ridge.fit(X_train_scaled, y_train)
print(f"Best alpha: {ridge.alpha_:.4f}")
"""

CELL_EVAL = """from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_preds = ridge.predict(X_train_scaled)
test_preds  = ridge.predict(X_test_scaled)

print("Ridge Performance (80/20 split):")
print(f"  Train MAE:  {mean_absolute_error(y_train, train_preds):.4f}")
print(f"  Test  MAE:  {mean_absolute_error(y_test, test_preds):.4f}")
print(f"  Test  RMSE: {mean_squared_error(y_test, test_preds)**0.5:.4f}")
print(f"  Test  R2:   {r2_score(y_test, test_preds):.4f}")

# 10-fold CV on full data — more reliable estimate
X_all_scaled = X.copy()
X_all_scaled[other_cols] = scaler.transform(X[other_cols])

kf = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(
    RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error'),
    X_all_scaled, y,
    cv=kf, scoring='neg_mean_absolute_error'
)
cv_mae = -cv_scores.mean()
print(f"\\n10-fold CV MAE (full data): {cv_mae:.4f} +/- {cv_scores.std():.4f}")
print(f"Baseline LR public score:   3.0800")
print(f"Est. public (CV + ~0.33 gap): {cv_mae + 0.33:.4f}")
"""

CELL_SUBMIT = """from sklearn.linear_model import Ridge

# Refit scaler + model on ALL training data
scaler_final = StandardScaler()
X_all_final  = X.copy()
X_all_final[other_cols] = scaler_final.fit_transform(X[other_cols])

final_model = Ridge(alpha=ridge.alpha_)
final_model.fit(X_all_final, y)

predict_features = predict_df[available_features].copy()
predict_features[other_cols] = scaler_final.transform(predict_features[other_cols])
predict_preds = final_model.predict(predict_features)

print(f"Predictions: min={predict_preds.min():.1f}  mean={predict_preds.mean():.1f}  max={predict_preds.max():.1f}")

import pandas as pd
submission_df = pd.DataFrame({'ID': predict_df['ID'], 'W': predict_preds.round().astype(int)})
submission_df.to_csv('submission_predict.csv', index=False)
print("Saved: submission_predict.csv")
submission_df.head()
"""

nb['cells'] = [
    original_setup,
    original_loaddata,
    code_cell(CELL_ENG),
    code_cell(CELL_FEATURES),
    code_cell(CELL_SCALE),
    code_cell(CELL_FIT),
    code_cell(CELL_EVAL),
    code_cell(CELL_SUBMIT),
]

with open('moneyball-starter-code-share.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Rebuilt notebook with {len(nb['cells'])} cells.")
