import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

pd.set_option('display.max_columns', None)
print('All imports successful')

# ============ LOAD DATA ============
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

target_col = 'gross_revenue_usd'
target_cls = 'sales_channel'

train['is_revenue_missing'] = (train[target_col] == 'Lost').astype(int)
train[target_col] = train[target_col].replace('Lost', np.nan)
train[target_col] = pd.to_numeric(train[target_col], errors='coerce')

print(f'Revenue missing: {train[target_col].isna().sum()} / {len(train)} ({train[target_col].isna().mean()*100:.1f}%)')

# ============ COMBINE DATA ============
all_data = pd.concat([train, test], axis=0, ignore_index=True)
print(f'Combined shape: {all_data.shape}')

drop_cols = ['transaction_id', target_col, target_cls]

cat_cols_raw = all_data.select_dtypes(include='object').columns.tolist()
cat_cols_raw = [c for c in cat_cols_raw if c not in drop_cols]

num_cols = all_data.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c not in drop_cols]

print(f'Categorical columns ({len(cat_cols_raw)}): {cat_cols_raw}')
print(f'Numerical columns ({len(num_cols)}): {num_cols}')

# ============ FEATURE ENGINEERING ============
for col in ['holiday_season', 'marketing_campaign']:
    if col in all_data.columns:
        all_data[col] = all_data[col].astype(int)

all_data['customer_ratio'] = all_data['new_customers'] / (all_data['new_customers'] + all_data['returning_customers']).replace(0, np.nan)
all_data['online_ratio'] = all_data['online_players'] / all_data['estimated_active_players'].replace(0, np.nan)
all_data['story_ratio'] = all_data['story_mode_players'] / all_data['estimated_active_players'].replace(0, np.nan)
all_data['gta_online_ratio'] = all_data['gta_online_players'] / all_data['online_players'].replace(0, np.nan)
all_data['peak_per_active'] = all_data['peak_concurrent_players'] / all_data['estimated_active_players'].replace(0, np.nan)
all_data['units_per_active'] = all_data['units_sold'] / all_data['estimated_active_players'].replace(0, np.nan)
all_data['market_gdp_interaction'] = all_data['gaming_market_size'] * all_data['gdp_per_capita_usd']
all_data['population_market'] = all_data['population_millions'] * all_data['gaming_market_size']
all_data['internet_gdp'] = all_data['internet_penetration_percentage'] * all_data['gdp_per_capita_usd']
all_data['playtime_session'] = all_data['average_playtime_hours'] * all_data['average_session_length_minutes']
all_data['refund_impact'] = all_data['refund_rate_percentage'] * all_data['units_sold']
all_data['total_players'] = all_data['online_players'] + all_data['story_mode_players']
all_data['weekend_weekday_ratio'] = all_data['weekend_sales_percentage'] / all_data['weekday_sales_percentage'].replace(0, np.nan)

print(f'Total features now: {all_data.shape[1]}')

# ============ ENCODE CATEGORICALS ============
label_encoders = {}
for col in cat_cols_raw:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    label_encoders[col] = le

train_processed = all_data.iloc[:len(train)].copy()
test_processed = all_data.iloc[len(train):].copy()

feature_cols = [c for c in all_data.columns if c not in ['transaction_id', target_col, target_cls, 'is_revenue_missing']]

cat_feature_indices = [feature_cols.index(c) for c in cat_cols_raw if c in feature_cols]

train_features = train_processed[feature_cols].values
test_features = test_processed[feature_cols].values

train_features = np.nan_to_num(train_features, nan=-1.0).astype(np.float32)
test_features = np.nan_to_num(test_features, nan=-1.0).astype(np.float32)

y_cls = np.array(train_processed[target_cls].astype(str).values).flatten()
y_reg = np.array(train_processed[target_col].values, dtype=float)

has_revenue = ~np.isnan(y_reg)

print(f'Features shape: {train_features.shape}')
print(f'Train rows with revenue: {has_revenue.sum()} / {len(has_revenue)}')
print(f'Classification target: Physical={sum(y_cls=="Physical")}, Digital={sum(y_cls=="Digital")}')

# ============ CLASSIFICATION ============
print('\n=== CLASSIFICATION MODEL (Sales Channel) ===')

N_SPLITS_CLS = 5
cls_oof = np.zeros((len(train_features), 2))
cls_test_preds = np.zeros((len(test_features), 2))

skf_cls = StratifiedKFold(n_splits=N_SPLITS_CLS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf_cls.split(train_features, y_cls)):
    print(f'Classification fold {fold+1}/{N_SPLITS_CLS}...')

    X_tr, X_val = train_features[tr_idx], train_features[val_idx]
    y_tr, y_val = y_cls[tr_idx], y_cls[val_idx]

    cat_cls = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3, min_data_in_leaf=10,
        random_seed=42 + fold, verbose=0,
        early_stopping_rounds=100, eval_metric='Logloss'
    )
    cat_cls.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)

    xgb_cls = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3,
        random_state=42 + fold, eval_metric='logloss',
        early_stopping_rounds=100, use_label_encoder=False, verbosity=0
    )
    xgb_cls.fit(X_tr, (y_tr == 'Physical').astype(int),
                eval_set=[(X_val, (y_val == 'Physical').astype(int))], verbose=False)

    lgb_cls = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3,
        random_state=42 + fold, verbose=-1
    )
    lgb_cls.fit(X_tr, (y_tr == 'Physical').astype(int),
                eval_set=[(X_val, (y_val == 'Physical').astype(int))],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    cat_val = cat_cls.predict_proba(X_val)
    xgb_val = xgb_cls.predict_proba(X_val)
    lgb_val = lgb_cls.predict_proba(X_val)

    ensemble_val = 0.4 * cat_val + 0.3 * xgb_val + 0.3 * lgb_val
    cls_oof[val_idx] = ensemble_val

    cat_test = cat_cls.predict_proba(test_features)
    xgb_test = xgb_cls.predict_proba(test_features)
    lgb_test = lgb_cls.predict_proba(test_features)

    cls_test_preds += (0.4 * cat_test + 0.3 * xgb_test + 0.3 * lgb_test) / N_SPLITS_CLS

    fold_ll = log_loss(y_val, ensemble_val)
    print(f'  Fold {fold+1} LogLoss: {fold_ll:.5f}')

oof_ll = log_loss(y_cls, cls_oof)
print(f'\nOverall OOF LogLoss: {oof_ll:.5f}')

# ============ REGRESSION ============
print('\n=== REGRESSION MODEL (Revenue) ===')

N_SPLITS_REG = 5
X_reg_train = train_features[has_revenue]
y_reg_train = y_reg[has_revenue]
y_reg_log = np.log1p(y_reg_train)

reg_oof = np.zeros(len(X_reg_train))
reg_test_preds = np.zeros(len(test_features))

kf_reg = KFold(n_splits=N_SPLITS_REG, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(kf_reg.split(X_reg_train, y_reg_train)):
    print(f'Regression fold {fold+1}/{N_SPLITS_REG}...')

    X_tr, X_val = X_reg_train[tr_idx], X_reg_train[val_idx]
    y_tr_log, y_val_log = y_reg_log[tr_idx], y_reg_log[val_idx]
    y_val_actual = y_reg_train[val_idx]

    cat_reg = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3, min_data_in_leaf=10,
        random_seed=42 + fold, verbose=0,
        early_stopping_rounds=100, eval_metric='RMSE'
    )
    cat_reg.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), verbose=0)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3,
        random_state=42 + fold, early_stopping_rounds=100, verbosity=0
    )
    xgb_reg.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=3,
        random_state=42 + fold, verbose=-1
    )
    lgb_reg.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    cat_val_log = cat_reg.predict(X_val)
    xgb_val_log = xgb_reg.predict(X_val)
    lgb_val_log = lgb_reg.predict(X_val)

    ensemble_val_log = 0.4 * cat_val_log + 0.3 * xgb_val_log + 0.3 * lgb_val_log
    ensemble_val_actual = np.expm1(ensemble_val_log)
    reg_oof[val_idx] = ensemble_val_actual

    cat_test_log = cat_reg.predict(test_features)
    xgb_test_log = xgb_reg.predict(test_features)
    lgb_test_log = lgb_reg.predict(test_features)

    reg_test_preds += np.expm1(0.4 * cat_test_log + 0.3 * xgb_test_log + 0.3 * lgb_test_log) / N_SPLITS_REG

    fold_rmsle = np.sqrt(mean_squared_log_error(y_val_actual, ensemble_val_actual))
    print(f'  Fold {fold+1} RMSLE: {fold_rmsle:.5f}')

oof_rmsle = np.sqrt(mean_squared_log_error(y_reg_train, reg_oof))
print(f'\nOverall OOF RMSLE: {oof_rmsle:.5f}')

# ============ SCORE ============
total_score = 100 * np.exp(-1.3 * (0.35 * oof_ll + 0.65 * oof_rmsle))
print(f'\n===== SIMULATED SCORE =====')
print(f'LogLoss: {oof_ll:.5f}')
print(f'RMSLE: {oof_rmsle:.5f}')
print(f'Score: {total_score:.2f} / 100')

# ============ SUBMISSION ============
test_pred_cls = ['Digital' if p[0] > p[1] else 'Physical' for p in cls_test_preds]
test_pred_rev = np.maximum(reg_test_preds, 0)

submission = pd.DataFrame({
    'transaction_id': test['transaction_id'],
    'sales_channel_pred': test_pred_cls,
    'revenue_pred': test_pred_rev
})

print(f'\nSubmission shape: {submission.shape}')
print(submission.head(10))

submission.to_csv('submission.csv', index=False)
print('\nsubmission.csv saved!')

# ============ ZIP ============
import zipfile

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('./' + file_name, file_name, compress_type=compression)

file_names = ['submission.csv', 'GTA.ipynb']
compress(file_names)
print('\nresult.zip generated!')
