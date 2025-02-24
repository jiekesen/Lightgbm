import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pandas as pd

data_label_tanghai = pd.read_csv("./test_data.csv")
data = data_label_tanghai.iloc[:, 1:]
label = data_label_tanghai["label"]

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
if isinstance(X_train, pd.DataFrame):
    X_train = X_train.values
if isinstance(y_train, pd.Series):
    y_train = y_train.values



def pearson_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]



lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


cv = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=cv,
    verbose=0,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print("Best parameters from GridSearchCV:", best_params)

lgb_model = lgb.LGBMRegressor(
    random_state=42,
    verbose=-1,
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    colsample_bytree=best_params['colsample_bytree'],
    subsample=best_params['subsample']
)



n_runs = 100
correlation_scores = []
for run in range(n_runs):
    cv = KFold(n_splits=10, shuffle=True, random_state=42 + run)
    scores = []
    for train_idx, val_idx in cv.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        lgb_model.fit(X_train_fold, y_train_fold)
        y_pred = lgb_model.predict(X_val_fold)
        score = pearson_correlation(y_val_fold, y_pred)
        scores.append(score)
    correlation_scores.append(np.mean(scores))

mean_corr = np.mean(correlation_scores)
std_error = np.std(correlation_scores, ddof=1) / np.sqrt(len(correlation_scores))

print(f"Mean Pearson correlation across {n_runs} runs: {mean_corr}")
print(f"Standard error of Pearson correlation: {std_error}")


test_pre = lgb_model.fit(X_train, y_train).predict(X_test)
test_correlation = np.corrcoef(y_test, test_pre)[0, 1]
print(f"Test data Pearson correlation: {test_correlation}")