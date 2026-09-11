import pandas as pd
import numpy as np

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

print('=== TRAIN INFO ===')
print(train_df.dtypes)
print()
print('=== PRICE STATS ===')
p = train_df['price']
print(f'Min: {p.min()}, Max: {p.max()}, Mean: {p.mean()}, Median: {p.median()}, Std: {p.std()}')
print(f'Skew: {p.skew()}, Kurt: {p.kurt()}')
print()
print('=== MISSING VALUES (train) ===')
print(train_df.isnull().sum())
print()
print('=== MISSING VALUES (test) ===')
print(test_df.isnull().sum())
print()
print('=== CATEGORICAL UNIQUE COUNTS ===')
for col in train_df.select_dtypes(include='object').columns:
    all_vals = pd.concat([train_df[col], test_df[col]]).unique()
    print(f'{col}: {len(all_vals)} unique -> {all_vals}')
print()
print('=== NUMERIC CORRELATIONS WITH PRICE ===')
num_cols = train_df.select_dtypes(include=[np.number]).columns
corr = train_df[num_cols].corr()['price'].sort_values(ascending=False)
print(corr)
print()
print('=== TRAIN HEAD ===')
print(train_df.head(3).to_string())
print()
print('=== TEST HEAD ===')
print(test_df.head(3).to_string())
