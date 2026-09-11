import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import warnings, zipfile
warnings.filterwarnings('ignore')

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')
train_df = train_df.dropna(subset=['price'])
y = train_df['price'].values.copy()
feature_cols = [c for c in train_df.columns if c != 'price']
n_train = len(train_df)

all_feat = pd.concat([train_df[feature_cols], test_df[feature_cols]], ignore_index=True)

for col in all_feat.columns:
    if all_feat[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        all_feat[col] = all_feat[col].fillna(all_feat[col].median())
    else:
        mv = all_feat[col].mode()
        if len(mv) > 0:
            all_feat[col] = all_feat[col].fillna(mv[0])

all_feat['parking_type'] = all_feat['parking_type'].replace("Doesn't have", "Doesnt")
all_feat['parking_type'] = all_feat['parking_type'].replace("Doesn't Have", "Doesnt")

# FEATURES
all_feat['area_per_room'] = all_feat['area'] / (all_feat['rooms'] + 1)
all_feat['area_rooms'] = all_feat['area'] * all_feat['rooms']
all_feat['floor_ratio'] = all_feat['floor_number'] / (all_feat['total_floors'] + 1)
all_feat['age_per_floor'] = all_feat['age'] / (all_feat['total_floors'] + 1)
all_feat['levy_per_area'] = all_feat['monthly_levy'] / (all_feat['area'] + 1)
all_feat['levy_per_room'] = all_feat['monthly_levy'] / (all_feat['rooms'] + 1)
all_feat['area_sq'] = all_feat['area'] ** 2
all_feat['rooms_sq'] = all_feat['rooms'] ** 2
all_feat['age_sq'] = all_feat['age'] ** 2
all_feat['log_area'] = np.log1p(all_feat['area'])
all_feat['log_levy'] = np.log1p(all_feat['monthly_levy'])
all_feat['log_age'] = np.log1p(all_feat['age'])
all_feat['area_x_age'] = all_feat['area'] * all_feat['age']
all_feat['area_x_master'] = all_feat['area'] * all_feat['has_master_bedroom']
all_feat['rooms_x_master'] = all_feat['rooms'] * all_feat['has_master_bedroom']
all_feat['area_x_smart'] = all_feat['area'] * all_feat['has_smart_home']
all_feat['levy_x_master'] = all_feat['monthly_levy'] * all_feat['has_master_bedroom']
all_feat['area_x_tf'] = all_feat['area'] * all_feat['total_floors']
all_feat['area_x_rooms_x_master'] = all_feat['area'] * all_feat['rooms'] * all_feat['has_master_bedroom']
all_feat['levy_area_ratio'] = all_feat['monthly_levy'] / (all_feat['area'] * all_feat['rooms'] + 1)
all_feat['dist_hosp_x_area'] = all_feat['distance_to_hospital_km'] * all_feat['area']
all_feat['area_div_age'] = all_feat['area'] / (all_feat['age'] + 1)
all_feat['age_x_rooms'] = all_feat['age'] * all_feat['rooms']

tehran_lat, tehran_long = 35.6892, 51.3890
all_feat['dist_center'] = np.sqrt((all_feat['lat'] - tehran_lat)**2 + (all_feat['long'] - tehran_long)**2)
all_feat['dist_center_sq'] = all_feat['dist_center'] ** 2
all_feat['lat_long'] = all_feat['lat'] * all_feat['long']

coords = all_feat[['lat', 'long']].values
for nc in [10, 20]:
    km = KMeans(n_clusters=nc, random_state=42, n_init=10)
    all_feat[f'geo_c{nc}'] = km.fit_predict(coords)

# TARGET ENCODING
cat_cols = ['document_type', 'cooling_type', 'parking_type', 'neighbor_noise_level', 'water_pressure', 'exterior_style', 'window_type']

# Save label encodings first
for col in cat_cols:
    le = LabelEncoder()
    all_feat[col] = le.fit_transform(all_feat[col].astype(str))

# Target encoding on train portion, then map to all
y_series = pd.Series(y)
global_mean = y.mean()

for col in cat_cols:
    means = y_series.groupby(all_feat.iloc[:n_train][col]).agg(['mean', 'count'])
    smooth = (means['count'] * means['mean'] + 20 * global_mean) / (means['count'] + 20)
    all_feat[col + '_te'] = all_feat[col].map(smooth).fillna(global_mean).values

# Geo cluster mean area/levy
for gc in ['geo_c10', 'geo_c20']:
    grp = all_feat.iloc[:n_train].groupby(gc)['area'].agg(['mean', 'std']).fillna(0)
    all_feat[gc + '_area_mean'] = all_feat[gc].map(grp['mean']).fillna(0)
    all_feat[gc + '_area_std'] = all_feat[gc].map(grp['std']).fillna(0)
    grp2 = all_feat.iloc[:n_train].groupby(gc)['monthly_levy'].agg(['mean', 'std']).fillna(0)
    all_feat[gc + '_levy_mean'] = all_feat[gc].map(grp2['mean']).fillna(0)

all_feat = all_feat.fillna(0)
X = all_feat.iloc[:n_train]
X_test = all_feat.iloc[n_train:]
print(f'X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}')

# 5-FOLD CV with tuned params
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_xgb = np.zeros(n_train)
oof_lgbm = np.zeros(n_train)
oof_cat = np.zeros(n_train)
test_xgb = np.zeros(len(X_test))
test_lgbm = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]
    
    m_xgb = XGBRegressor(n_estimators=3000, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=5.0, min_child_weight=10, gamma=0.1, random_state=42, verbosity=0)
    m_xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_val)
    test_xgb += m_xgb.predict(X_test) / n_folds
    
    m_lgbm = LGBMRegressor(n_estimators=3000, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=5.0, min_child_weight=10, random_state=42, verbose=-1)
    m_lgbm.fit(X_trn, y_trn, eval_set=[(X_val, y_val)])
    oof_lgbm[val_idx] = m_lgbm.predict(X_val)
    test_lgbm += m_lgbm.predict(X_test) / n_folds
    
    m_cat = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.02, l2_leaf_reg=5, random_seed=42, verbose=0)
    m_cat.fit(X_trn, y_trn, eval_set=(X_val, y_val))
    oof_cat[val_idx] = m_cat.predict(X_val)
    test_cat += m_cat.predict(X_test) / n_folds
    
    ens = 0.35*oof_xgb[val_idx] + 0.35*oof_lgbm[val_idx] + 0.30*oof_cat[val_idx]
    r2 = r2_score(y_val, ens)
    s = max(0, 100 * (1 - (1 - r2) / 0.04))
    print(f'Fold {fold+1}: R2={r2:.6f}, Score={s:.2f}')

# Individual scores
for name, oof in [('XGB', oof_xgb), ('LGBM', oof_lgbm), ('CAT', oof_cat)]:
    r2 = r2_score(y, oof)
    s = max(0, 100 * (1 - (1 - r2) / 0.04))
    print(f'{name} OOF: R2={r2:.6f}, Score={s:.2f}')

# Ridge meta
stack = np.column_stack([oof_xgb, oof_lgbm, oof_cat])
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    meta = Ridge(alpha=alpha)
    meta.fit(stack, y)
    mp = meta.predict(stack)
    r2 = r2_score(y, mp)
    s = max(0, 100 * (1 - (1 - r2) / 0.04))
    print(f'Ridge(a={alpha}): R2={r2:.6f}, Score={s:.2f}, w={meta.coef_}')

# Best weighted ensemble search
best_score = 0
best_w = None
for w1 in np.arange(0.1, 0.6, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05 or w3 > 0.6:
            continue
        ens = w1*oof_xgb + w2*oof_lgbm + w3*oof_cat
        r2 = r2_score(y, ens)
        sc = max(0, 100 * (1 - (1 - r2) / 0.04))
        if sc > best_score:
            best_score = sc
            best_w = (w1, w2, w3)

print(f'\nBest weights: {best_w}, Score={best_score:.2f}')

# Use best ensemble for final prediction
w1, w2, w3 = best_w
test_preds = w1*test_xgb + w2*test_lgbm + w3*test_cat

submission = pd.DataFrame({'price': test_preds})
submission.to_csv('submission.csv', index=False)
print(f'\nSubmission shape: {submission.shape}')
print(submission.head(10))

with zipfile.ZipFile('result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('house_prediction.ipynb')
    zf.write('submission.csv')
print('result.zip created! DONE!')
