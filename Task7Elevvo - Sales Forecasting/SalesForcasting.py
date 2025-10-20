# Sales Forecasting - end-to-end script for Walmart-like dataset
# Files expected in working directory: train.csv, test.csv, features.csv, stores.csv
# Outputs: model (pickle), validation metrics, actual_vs_pred.png, submission.csv
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# ---------- 1) Load data ----------
print('\nLoading CSV files...')
for fname in ['train.csv','test.csv','features.csv','stores.csv']:
    if not os.path.exists(fname):
        print(f"WARNING: {fname} not found in working dir. Make sure you placed it here.")

train = pd.read_csv('WalmartSalesForcast\\train.csv', parse_dates=['Date'])
test = pd.read_csv('WalmartSalesForcast\\test.csv', parse_dates=['Date'])
features = pd.read_csv('WalmartSalesForcast\\features.csv', parse_dates=['Date'])
stores = pd.read_csv('WalmartSalesForcast\\stores.csv')

print('Shapes: ', train.shape, test.shape, features.shape, stores.shape)

# ---------- 2) Basic cleanup & merging datasets ----------
train = train.merge(features, on=['Store','Date'], how='left')
train = train.merge(stores, on='Store', how='left')

test = test.merge(features, on=['Store','Date'], how='left')
test = test.merge(stores, on='Store', how='left')

def create_time_features(df):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    return df

train = create_time_features(train)
test = create_time_features(test)

if 'Type' in stores.columns:
    le = LabelEncoder()
    train['StoreType'] = le.fit_transform(train['Type'].fillna('M'))
    test['StoreType'] = le.transform(test['Type'].fillna('M'))
else:
    train['StoreType'] = 0
    test['StoreType'] = 0

# Filling NaN's in temp, FP, CPI, unemp
num_cols = ['Temperature','Fuel_Price','CPI','Unemployment']
for c in num_cols:
    if c in train.columns:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

for c in train.columns:
    if 'MarkDown' in str(c):
        train[c] = train[c].fillna(0)
        test[c] = test[c].fillna(0)

#introducing lag for better predictions
group_cols = ['Store']
if 'Dept' in train.columns:
    group_cols = ['Store','Dept']

all_df = pd.concat([train, test], sort=False).reset_index(drop=True)
all_df = all_df.sort_values(group_cols + ['Date'])

if 'Weekly_Sales' in train.columns:
    all_df['Weekly_Sales'] = all_df['Weekly_Sales'] if 'Weekly_Sales' in all_df.columns else np.nan


    for lag in [1, 2, 3, 4, 8, 12, 26, 52, 78]:
        all_df[f'lag_{lag}'] = all_df.groupby(group_cols)['Weekly_Sales'].shift(lag)

    for window in [4, 12, 26]:
        grp = all_df.groupby(group_cols)['Weekly_Sales']
        all_df[f'roll_slope_{window}'] = grp.transform(
            lambda x: x.shift(1).rolling(window).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == window else np.nan,
                raw=False
            )
        )

train = all_df[all_df['Weekly_Sales'].notnull()].copy()
test = all_df[all_df['Weekly_Sales'].isnull()].copy()
lag_cols = [c for c in train.columns if c.startswith('lag_') or c.startswith('roll_mean_')]
for c in lag_cols:
    train[c] = train[c].fillna(-1)
    test[c] = test[c].fillna(-1)

for df in [train,test]:
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)


print('\nCreating time-based validation split...')
# used last 12 weeks for validation

max_date = train['Date'].max()
val_cutoff = max_date - pd.Timedelta(days=7*12)  # last 12 weeks

train_idx = train[train['Date'] <= val_cutoff].index
val_idx = train[train['Date'] > val_cutoff].index

print('Train rows:', len(train_idx), 'Val rows:', len(val_idx))

#lightGBM
features_to_drop = ['Weekly_Sales','Date','Type','Locale','LocaleName','Size']
features_to_drop = [c for c in features_to_drop if c in train.columns]

feature_cols = [c for c in train.columns if c not in features_to_drop and c not in ['Id','IsHoliday']]

if 'IsHoliday' in train.columns:
    if 'IsHoliday' not in feature_cols:
        feature_cols.append('IsHoliday')

print('\nNumber of features used:', len(feature_cols))

X_train = train.loc[train_idx, feature_cols]
y_train = train.loc[train_idx, 'Weekly_Sales']
X_val = train.loc[val_idx, feature_cols]
y_val = train.loc[val_idx, 'Weekly_Sales']

#LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

print('\nTraining LightGBM...')
params = {
    'learning_rate': 0.02,
'num_leaves': 31,
'feature_fraction': 0.7,
'bagging_fraction': 0.7,
'lambda_l1': 0.3,
'lambda_l2': 0.3

}

bst = lgb.train(
    params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(100)
    ]
)


best_iter = bst.best_iteration or 1
print('Best iteration:', best_iter)

val_pred = bst.predict(X_val, num_iteration=best_iter)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
mae = mean_absolute_error(y_val, val_pred)
print(f'Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}')


with open('val_metrics.txt','w') as f:
    f.write(f'RMSE: {rmse}\nMAE: {mae}\n')
# aggregate per date
val_dates = train.loc[val_idx, ['Date']].copy()
val_dates = val_dates.assign(actual=y_val.values, pred=val_pred)
agg = val_dates.groupby('Date').sum()

plt.figure(figsize=(10,5))
plt.plot(agg.index, agg['actual'], label='Actual')
plt.plot(agg.index, agg['pred'], label='Predicted')
plt.title('Aggregate Actual vs Predicted on Validation')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_pred.png')
plt.close()

residual = agg['actual'] - agg['pred']
percent_error = residual / agg['actual'] * 100

plt.figure(figsize=(12,5))
plt.plot(agg.index, percent_error, color='green')
plt.axhline(0, color='black', linestyle='--')
plt.title('Percentage Error Over Time')
plt.ylabel('Error (%)')
plt.tight_layout()
plt.savefig('percent_error.png')
plt.close()


# training final model on full train
full_X = train[feature_cols]
full_y = train['Weekly_Sales']
full_data = lgb.Dataset(full_X, full_y)
final_params = params.copy()
final_params['learning_rate'] = 0.03
final_bst = lgb.train(final_params, full_data, num_boost_round=best_iter)


with open('lgb_model.pkl','wb') as f:
    pickle.dump(final_bst, f)

# uses test data set to compare predicitions
print('\nPredicting on test set...')
id_col = 'Id' if 'Id' in test.columns else None
X_test = test[feature_cols]
test_preds = final_bst.predict(X_test)

submission = pd.DataFrame()
if id_col:
    submission['Id'] = test[id_col].values
else:
    submission['Id'] = range(len(test_preds))
submission['Weekly_Sales'] = test_preds

submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')

# Clean up
gc.collect()

