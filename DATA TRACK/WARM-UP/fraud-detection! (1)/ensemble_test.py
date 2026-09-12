import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r'C:\Users\arpam\Downloads\fraud-detection! (1)\Data\train.csv')
test = pd.read_csv(r'C:\Users\arpam\Downloads\fraud-detection! (1)\Data\test.csv')

def parse_timestamp(ts):
    parts = ts.split(':')
    mins = int(parts[0])
    sec_parts = parts[1].split('.')
    secs = int(sec_parts[0])
    frac = int(sec_parts[1])
    return mins * 60 + secs + frac / 10.0

def build_features(df, batch_stats=None, is_train=True):
    df = df.copy()
    df['timestamp_seconds'] = df['timestamp'].apply(parse_timestamp)
    df['ts_minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['ts_second'] = df['timestamp'].apply(lambda x: float(x.split(':')[1]))
    df['ts_sin_min'] = np.sin(2 * np.pi * df['ts_minute'] / 60)
    df['ts_cos_min'] = np.cos(2 * np.pi * df['ts_minute'] / 60)
    df['ts_sin_sec'] = np.sin(2 * np.pi * df['timestamp_seconds'] / 3600)
    df['ts_cos_sec'] = np.cos(2 * np.pi * df['timestamp_seconds'] / 3600)

    if is_train:
        batch_stats = df.groupby('processing_batch_id').agg(
            batch_size=('user_id', 'count'),
            batch_amount_mean=('transaction_amount', 'mean'),
            batch_amount_std=('transaction_amount', 'std'),
            batch_speed_mean=('transaction_speed_seconds', 'mean'),
            batch_speed_std=('transaction_speed_seconds', 'std'),
            batch_ip_mean=('ip_risk_score', 'mean'),
            batch_ip_std=('ip_risk_score', 'std'),
            batch_age_mean=('user_age_days', 'mean'),
            batch_age_std=('user_age_days', 'std'),
            batch_amount_median=('transaction_amount', 'median'),
            batch_speed_median=('transaction_speed_seconds', 'median'),
            batch_ip_median=('ip_risk_score', 'median'),
            batch_age_median=('user_age_days', 'median'),
            batch_amount_min=('transaction_amount', 'min'),
            batch_amount_max=('transaction_amount', 'max'),
            batch_speed_min=('transaction_speed_seconds', 'min'),
            batch_speed_max=('transaction_speed_seconds', 'max'),
            batch_ip_min=('ip_risk_score', 'min'),
            batch_ip_max=('ip_risk_score', 'max'),
        ).reset_index()
        batch_stats['batch_amount_std'] = batch_stats['batch_amount_std'].fillna(0)
        batch_stats['batch_speed_std'] = batch_stats['batch_speed_std'].fillna(0)
        batch_stats['batch_ip_std'] = batch_stats['batch_ip_std'].fillna(0)
        batch_stats['batch_age_std'] = batch_stats['batch_age_std'].fillna(0)
    df = df.merge(batch_stats, on='processing_batch_id', how='left')

    df['amount_dev_from_batch'] = df['transaction_amount'] - df['batch_amount_mean']
    df['speed_dev_from_batch'] = df['transaction_speed_seconds'] - df['batch_speed_mean']
    df['ip_dev_from_batch'] = df['ip_risk_score'] - df['batch_ip_mean']
    df['age_dev_from_batch'] = df['user_age_days'] - df['batch_age_mean']

    df['amount_zscore'] = df['amount_dev_from_batch'] / (df['batch_amount_std'] + 0.001)
    df['speed_zscore'] = df['speed_dev_from_batch'] / (df['batch_speed_std'] + 0.001)
    df['ip_zscore'] = df['ip_dev_from_batch'] / (df['batch_ip_std'] + 0.001)
    df['age_zscore'] = df['age_dev_from_batch'] / (df['batch_age_std'] + 0.001)

    df['amount_batch_rank'] = df.groupby('processing_batch_id')['transaction_amount'].rank(pct=True)
    df['speed_batch_rank'] = df.groupby('processing_batch_id')['transaction_speed_seconds'].rank(pct=True)
    df['ip_batch_rank'] = df.groupby('processing_batch_id')['ip_risk_score'].rank(pct=True)
    df['age_batch_rank'] = df.groupby('processing_batch_id')['user_age_days'].rank(pct=True)

    df['amount_range_in_batch'] = (df['transaction_amount'] - df['batch_amount_min']) / (df['batch_amount_max'] - df['batch_amount_min'] + 0.001)
    df['speed_range_in_batch'] = (df['transaction_speed_seconds'] - df['batch_speed_min']) / (df['batch_speed_max'] - df['batch_speed_min'] + 0.001)
    df['ip_range_in_batch'] = (df['ip_risk_score'] - df['batch_ip_min']) / (df['batch_ip_max'] - df['batch_ip_min'] + 0.001)

    df['amount_per_age'] = df['transaction_amount'] / (df['user_age_days'] + 1)
    df['ip_per_speed'] = df['ip_risk_score'] / (df['transaction_speed_seconds'] + 0.1)
    df['amount_per_speed'] = df['transaction_amount'] / (df['transaction_speed_seconds'] + 0.1)
    df['age_per_amount'] = df['user_age_days'] / (df['transaction_amount'] + 1)
    df['speed_per_amount'] = df['transaction_speed_seconds'] / (df['transaction_amount'] + 1)

    df['ip_x_amount'] = df['ip_risk_score'] * df['transaction_amount']
    df['ip_x_speed'] = df['ip_risk_score'] * df['transaction_speed_seconds']
    df['amount_x_speed'] = df['transaction_amount'] * df['transaction_speed_seconds']
    df['ip_x_age'] = df['ip_risk_score'] * df['user_age_days']
    df['speed_x_age'] = df['transaction_speed_seconds'] * df['user_age_days']
    df['amount_x_age'] = df['transaction_amount'] * df['user_age_days']

    df['ip_risk_score_sq'] = df['ip_risk_score'] ** 2
    df['ip_risk_score_cb'] = df['ip_risk_score'] ** 3
    df['user_age_days_log'] = np.log1p(df['user_age_days'])
    df['transaction_amount_log'] = np.log1p(df['transaction_amount'])
    df['transaction_speed_log'] = np.log1p(df['transaction_speed_seconds'])
    df['ip_sqrt'] = np.sqrt(df['ip_risk_score'])
    df['amount_sqrt'] = np.sqrt(df['transaction_amount'])

    df['ip_risk_bin'] = pd.cut(df['ip_risk_score'], bins=[0, 30, 50, 70, 85, 100], labels=False)
    df['speed_bin'] = pd.cut(df['transaction_speed_seconds'], bins=[0, 5, 10, 15, 20, 25], labels=False)
    df['age_bin'] = pd.cut(df['user_age_days'], bins=[0, 30, 90, 200, 500, 1000], labels=False)
    df['amount_bin'] = pd.cut(df['transaction_amount'], bins=[0, 50, 200, 500, 1000, 12000], labels=False)

    df['ts_hour_bin'] = pd.cut(df['ts_minute'], bins=12, labels=False)

    for col in ['product_category', 'payment_method']:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)

    df = pd.get_dummies(df, columns=['product_category', 'payment_method'], drop_first=False)

    return df, batch_stats

train_feat, batch_stats = build_features(train, is_train=True)
test_feat, _ = build_features(test, batch_stats=batch_stats, is_train=False)

train_cols = set(train_feat.columns)
test_cols = set(test_feat.columns)
for c in train_cols - test_cols:
    if c != 'is_fraud':
        test_feat[c] = 0
for c in test_cols - train_cols:
    train_feat[c] = 0

drop_cols = ['timestamp', 'user_id', 'is_fraud']
feature_cols = [c for c in train_feat.columns if c not in drop_cols]

X = train_feat[feature_cols].values
y = train_feat['is_fraud'].values
X_test = test_feat[feature_cols].values

print(f'Features: {len(feature_cols)}')

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

models = {
    'xgb1': XGBClassifier(n_estimators=1500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss'),
    'xgb2': XGBClassifier(n_estimators=1500, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.75, min_child_weight=3, gamma=0.2, reg_alpha=0.05, reg_lambda=0.5, random_state=43, eval_metric='logloss'),
    'xgb3': XGBClassifier(n_estimators=1500, max_depth=7, learning_rate=0.03, subsample=0.75, colsample_bytree=0.7, min_child_weight=7, gamma=0.3, reg_alpha=0.2, reg_lambda=2.0, random_state=44, eval_metric='logloss'),
    'rf': RandomForestClassifier(n_estimators=1000, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'et': ExtraTreesClassifier(n_estimators=1000, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'gb': GradientBoostingClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, min_samples_split=5, random_state=42),
}

test_preds_all = np.zeros((len(X_test), len(models)))
oof_preds_all = np.zeros((len(X), len(models)))
fold_accs_all = {}

for m_idx, (name, clf) in enumerate(models.items()):
    print(f'\n=== Training {name} ===')
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_accs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if hasattr(clf, 'scale_pos_weight') or 'xgb' in name:
            scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model = type(clf)(**{**clf.get_params(), 'scale_pos_weight': scale_pos})
        else:
            model = type(clf)(**clf.get_params())

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        fold_acc = accuracy_score(y_val, val_pred)
        fold_accs.append(fold_acc)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds

    print(f'  Mean CV: {np.mean(fold_accs):.6f} (+/- {np.std(fold_accs):.6f})')
    fold_accs_all[name] = np.mean(fold_accs)
    test_preds_all[:, m_idx] = test_preds
    oof_preds_all[:, m_idx] = oof_preds

# Try different ensemble strategies
print('\n=== ENSEMBLE RESULTS ===')

# Strategy 1: Majority vote
ensemble_vote = (test_preds_all >= 0.5).sum(axis=1)
ensemble_vote_pred = (ensemble_vote > len(models) / 2).astype(int)
print(f'Majority vote fraud rate: {ensemble_vote_pred.mean():.4f}')

# Strategy 2: Average probability
ensemble_avg = test_preds_all.mean(axis=1)
for thresh in [0.3, 0.4, 0.5, 0.6]:
    pred = (ensemble_avg >= thresh).astype(int)
    print(f'Avg prob thresh={thresh}: fraud_rate={pred.mean():.4f}')

# Strategy 3: Weighted average (by CV accuracy)
weights = np.array([fold_accs_all[m] for m in models.keys()])
weights = weights / weights.sum()
ensemble_weighted = (test_preds_all * weights).sum(axis=1)
for thresh in [0.3, 0.4, 0.5, 0.6]:
    pred = (ensemble_weighted >= thresh).astype(int)
    print(f'Weighted avg thresh={thresh}: fraud_rate={pred.mean():.4f}')

# OOF accuracy for each strategy
print('\n=== OOF ACCURACY ===')
for name in models.keys():
    print(f'  {name}: {accuracy_score(y, oof_preds_all[:, models.keys().index(name) if hasattr(models.keys(), "index") else list(models.keys()).index(name)]):.6f}')

# Try optimizing threshold on OOF
best_thresh = 0.5
best_acc = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    oof_avg = oof_preds_all.mean(axis=1)
    pred = (oof_avg >= thresh).astype(int)
    acc = accuracy_score(y, pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh
print(f'\nBest OOF threshold: {best_thresh:.2f} -> accuracy: {best_acc:.6f}')
