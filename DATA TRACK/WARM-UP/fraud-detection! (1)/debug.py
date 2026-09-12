import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r'C:\Users\arpam\Downloads\fraud-detection! (1)\Data\train.csv')
test = pd.read_csv(r'C:\Users\arpam\Downloads\fraud-detection! (1)\Data\test.csv')

# Check if there's a pattern between train and test
print('=== TRAIN UNIQUE VALUES ===')
for col in train.columns:
    print(f'{col}: {train[col].nunique()} unique')

print('\n=== TEST UNIQUE VALUES ===')
for col in test.columns:
    print(f'{col}: {test[col].nunique()} unique')

print('\n=== TRAIN BATCH RANGE ===')
print(f"Train batch: {train['processing_batch_id'].min()} - {train['processing_batch_id'].max()}")
print(f"Test batch: {test['processing_batch_id'].min()} - {test['processing_batch_id'].max()}")

print('\n=== BATCH OVERLAP ===')
train_batches = set(train['processing_batch_id'].unique())
test_batches = set(test['processing_batch_id'].unique())
print(f"Common: {len(train_batches & test_batches)}")
print(f"Only train: {len(train_batches - test_batches)}")
print(f"Only test: {len(test_batches - train_batches)}")

print('\n=== USER OVERLAP ===')
train_users = set(train['user_id'].unique())
test_users = set(test['user_id'].unique())
print(f"Common: {len(train_users & test_users)}")

# Check if test labels might be flipped or different
print('\n=== BASIC FEATURE DISTRIBUTIONS ===')
for col in ['transaction_amount', 'ip_risk_score', 'transaction_speed_seconds', 'user_age_days']:
    print(f'\n{col}:')
    print(f'  Train fraud=0: mean={train[train.is_fraud==0][col].mean():.2f}, std={train[train.is_fraud==0][col].std():.2f}')
    print(f'  Train fraud=1: mean={train[train.is_fraud==1][col].mean():.2f}, std={train[train.is_fraud==1][col].std():.2f}')
    print(f'  Test:          mean={test[col].mean():.2f}, std={test[col].std():.2f}')

print('\n=== TIMESTAMP ANALYSIS ===')
def parse_ts(ts):
    parts = ts.split(':')
    mins = int(parts[0])
    sec_parts = parts[1].split('.')
    secs = int(sec_parts[0])
    frac = int(sec_parts[1])
    return mins * 60 + secs + frac / 10.0

train['ts_sec'] = train['timestamp'].apply(parse_ts)
test['ts_sec'] = test['timestamp'].apply(parse_ts)
print(f"Train ts: {train['ts_sec'].min():.1f} - {train['ts_sec'].max():.1f}")
print(f"Test ts: {test['ts_sec'].min():.1f} - {test['ts_sec'].max():.1f}")

# Test without batch_fraud_rate (target leakage)
print('\n=== TEST WITHOUT BATCH_FRAUD_RATE ===')

def parse_timestamp(ts):
    parts = ts.split(':')
    mins = int(parts[0])
    sec_parts = parts[1].split('.')
    secs = int(sec_parts[0])
    frac = int(sec_parts[1])
    return mins * 60 + secs + frac / 10.0

def build_features_no_leak(df, is_train=True):
    df = df.copy()
    df['timestamp_seconds'] = df['timestamp'].apply(parse_timestamp)
    df['ts_minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['ts_second'] = df['timestamp'].apply(lambda x: float(x.split(':')[1]))
    df['ts_sin_min'] = np.sin(2 * np.pi * df['ts_minute'] / 60)
    df['ts_cos_min'] = np.cos(2 * np.pi * df['ts_minute'] / 60)

    batch_agg = df.groupby('processing_batch_id').agg(
        batch_size=('user_id', 'count'),
        batch_amount_mean=('transaction_amount', 'mean'),
        batch_speed_mean=('transaction_speed_seconds', 'mean'),
        batch_ip_mean=('ip_risk_score', 'mean'),
        batch_age_mean=('user_age_days', 'mean'),
        batch_amount_median=('transaction_amount', 'median'),
        batch_speed_median=('transaction_speed_seconds', 'median'),
        batch_ip_median=('ip_risk_score', 'median'),
    ).reset_index()
    batch_std = df.groupby('processing_batch_id')['transaction_amount'].std().fillna(0).reset_index()
    batch_std.columns = ['processing_batch_id', 'batch_amount_std']
    batch_agg = batch_agg.merge(batch_std, on='processing_batch_id', how='left')
    df = df.merge(batch_agg, on='processing_batch_id', how='left')

    df['amount_dev_from_batch'] = df['transaction_amount'] - df['batch_amount_mean']
    df['speed_dev_from_batch'] = df['transaction_speed_seconds'] - df['batch_speed_mean']
    df['ip_dev_from_batch'] = df['ip_risk_score'] - df['batch_ip_mean']
    df['age_dev_from_batch'] = df['user_age_days'] - df['batch_age_mean']

    df['amount_batch_rank'] = df.groupby('processing_batch_id')['transaction_amount'].rank(pct=True)
    df['speed_batch_rank'] = df.groupby('processing_batch_id')['transaction_speed_seconds'].rank(pct=True)
    df['ip_batch_rank'] = df.groupby('processing_batch_id')['ip_risk_score'].rank(pct=True)
    df['age_batch_rank'] = df.groupby('processing_batch_id')['user_age_days'].rank(pct=True)

    df['amount_per_age'] = df['transaction_amount'] / (df['user_age_days'] + 1)
    df['ip_per_speed'] = df['ip_risk_score'] / (df['transaction_speed_seconds'] + 0.1)
    df['amount_per_speed'] = df['transaction_amount'] / (df['transaction_speed_seconds'] + 0.1)

    df['ip_x_amount'] = df['ip_risk_score'] * df['transaction_amount']
    df['ip_x_speed'] = df['ip_risk_score'] * df['transaction_speed_seconds']
    df['amount_x_speed'] = df['transaction_amount'] * df['transaction_speed_seconds']
    df['ip_x_age'] = df['ip_risk_score'] * df['user_age_days']
    df['speed_x_age'] = df['transaction_speed_seconds'] * df['user_age_days']

    df['ip_risk_score_sq'] = df['ip_risk_score'] ** 2
    df['user_age_days_log'] = np.log1p(df['user_age_days'])
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
    df['transaction_speed_log'] = np.log1p(df['transaction_speed_seconds'])

    df['ip_risk_bin'] = pd.cut(df['ip_risk_score'], bins=[0, 30, 50, 70, 85, 100], labels=False)
    df['speed_bin'] = pd.cut(df['transaction_speed_seconds'], bins=[0, 5, 10, 15, 20, 25], labels=False)
    df['age_bin'] = pd.cut(df['user_age_days'], bins=[0, 30, 90, 200, 500, 1000], labels=False)
    df['amount_bin'] = pd.cut(df['transaction_amount'], bins=[0, 50, 200, 500, 1000, 12000], labels=False)

    for col in ['product_category', 'payment_method']:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)

    df = pd.get_dummies(df, columns=['product_category', 'payment_method'], drop_first=False)
    return df

train_feat = build_features_no_leak(train, is_train=True)
test_feat = build_features_no_leak(test, is_train=False)

train_cols = set(train_feat.columns)
test_cols = set(test_feat.columns)
for c in train_cols - test_cols:
    if c != 'is_fraud':
        test_feat[c] = 0
for c in test_cols - train_cols:
    train_feat[c] = 0

drop_cols = ['timestamp', 'user_id', 'is_fraud']
feature_cols = [c for c in train_feat.columns if c not in drop_cols]
print(f'Features: {len(feature_cols)}')

X = train_feat[feature_cols].values
y = train_feat['is_fraud'].values
X_test = test_feat[feature_cols].values

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_accs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos, random_state=42+fold, eval_metric='logloss'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    fold_acc = accuracy_score(y_val, val_pred)
    fold_accs.append(fold_acc)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / n_folds
    print(f'Fold {fold+1}: {fold_acc:.6f}')

print(f'\nMean CV: {np.mean(fold_accs):.6f}')
print('OOF:', accuracy_score(y, oof_preds))

# Feature importance
importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
print('\nTop 20 features:')
print(importance.sort_values('importance', ascending=False).head(20).to_string())
