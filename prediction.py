#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix,roc_curve
import patsy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import re
warnings.filterwarnings("ignore")

#%%
train_df = pd.read_csv("data/transformed_train.csv")
test_df = pd.read_csv("data/transformed_test.csv")
train_df
#%% Sample Design
plt.scatter(train_df["GrLivArea"], train_df["SalePrice"])
plt.xlabel("Living Area")
plt.ylabel("Price")
#%% Lets drop some extremes
train_df = train_df[(train_df['GrLivArea'] < 4000) & (train_df['SalePrice'] < 650_000)]
train_df

#%%
X_train = train_df.drop(columns=["SalePrice", "Id"])
Y_train = train_df["SalePrice"]
Y_train_log = np.log1p(Y_train)
# Sanitize column names for XGBoost
X_train.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_') for c in X_train.columns]
test_df.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_') for c in test_df.columns]
test_set_id = test_df["Id"]
test_df.drop(columns="Id", inplace=True)

# %% Random Forest
# Define Grid & Model
grid = {
    'max_features': ["sqrt", 0.1, 0.2, 20],
    'max_depth': [None, 10, 15, 20],
    'criterion': ['squared_error'],
    'min_samples_split': [3, 5, 6, 10],
    'n_estimators': [500]
}

prob_forest = RandomForestRegressor(
    random_state=42, 
    n_estimators=100,
    oob_score=True
)

# Run Grid Search
prob_forest_grid = GridSearchCV(
    prob_forest, 
    grid, 
    cv=5, 
    refit='neg_mean_squared_error',  
    scoring=['neg_mean_squared_error'], 
    n_jobs=-1
)

print("Running Random Forest Grid Search...")
prob_forest_grid.fit(X_train, Y_train_log)

# --- TABLE 1: DETAILED RESULTS (Every Combination) ---

# Extract results into a DataFrame
cv_results = pd.DataFrame(prob_forest_grid.cv_results_)

# Keep only the columns we care about
cols_to_keep = ['param_max_features', 'param_min_samples_split', 'mean_test_neg_mean_squared_error']
summary_table = cv_results[cols_to_keep].copy()

# Rename columns for readability
summary_table.columns = ['Max_Features', 'Min_Samples_Split', 'Neg_Brier']

# Calculate RMSE from Neg_Brier
summary_table['RMSE'] = np.sqrt(-1 * summary_table['Neg_Brier'])

# Drop the confusing 'Neg_Brier' column now that we have RMSE
summary_table = summary_table.drop(columns=['Neg_Brier'])

# Sort by Lowest RMSE (Best Probabilities)
summary_table = summary_table.sort_values(by='RMSE', ascending=True)

print("\n--- Detailed Results (All Combinations) ---")
summary_table.round(4)

#%% XGBoost
xgb_grid = {
    'colsample_bytree': [0.1, 0.2, 0.3, 0.5], # Equivalent to max_features
    'max_depth': [3, 5, 10, 15],              # XGBoost prefers shallower trees than RF
    'min_child_weight': [1, 3, 5, 10],        # Equivalent to min_samples_split/leaf
    'learning_rate': [0.05, 0.1],             # Crucial for boosting
    'n_estimators': [500]
}

xgb_model = XGBRegressor(
    random_state=42,
    objective='reg:squarederror',
    tree_method='hist' # Faster for 150 features
)

# Run Grid Search
xgb_grid_search = GridSearchCV(
    xgb_model, 
    xgb_grid, 
    cv=5, 
    refit='neg_mean_squared_error',  
    scoring=['neg_mean_squared_error'], 
    n_jobs=-1
)

print("Running XGBoost Grid Search...")
# Ensure you use Y_train_log as discussed!
xgb_grid_search.fit(X_train, Y_train_log)

# --- TABLE 1: DETAILED RESULTS ---
cv_results_xgb = pd.DataFrame(xgb_grid_search.cv_results_)

# Adjust columns to match XGBoost param names
cols_to_keep = ['param_colsample_bytree', 'param_max_depth', 'param_min_child_weight', 'mean_test_neg_mean_squared_error']
summary_table_xgb = cv_results_xgb[cols_to_keep].copy()

summary_table_xgb.columns = ['ColSample', 'Max_Depth', 'Min_Weight', 'Neg_MSE']
summary_table_xgb['RMSE'] = np.sqrt(-1 * summary_table_xgb['Neg_MSE'])

# Sort and display
summary_table_xgb = summary_table_xgb.drop(columns=['Neg_MSE']).sort_values(by='RMSE')

print("\n--- XGBoost Detailed Results ---")
print(summary_table_xgb.round(4).head(10))
#%%
# This regex replaces ANY non-alphanumeric character with an underscore
X_train.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in X_train.columns]

test_df.columns = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in test_df.columns]
#Define the Grid
lgbm_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [15, 31],            # Main complexity control for LGBM
    'feature_fraction': [0.2, 0.4],    # Same as colsample_bytree
    'bagging_fraction': [0.7, 0.8],    # Same as subsample
    'bagging_freq': [5],               # Perform bagging every 5 iterations
    'min_child_samples': [10, 20]      # Prevents overfitting on small data
}

lgbm_model = LGBMRegressor(
    random_state=42,
    objective='regression',
    verbose=-1 # Silences the iteration logs
)

# 3. Run Grid Search
lgbm_grid_search = GridSearchCV(
    lgbm_model, 
    lgbm_grid, 
    cv=5, 
    refit='neg_mean_squared_error',  
    scoring='neg_mean_squared_error', 
    n_jobs=-1
)

print("Running LightGBM Grid Search...")
lgbm_grid_search.fit(X_train, Y_train_log)

# 4. Extract Results
cv_results_lgbm = pd.DataFrame(lgbm_grid_search.cv_results_)
cv_results_lgbm['RMSE'] = np.sqrt(-1 * cv_results_lgbm['mean_test_score'])

# Keep relevant columns for display
summary_lgbm = cv_results_lgbm[['param_num_leaves', 'param_learning_rate', 'param_feature_fraction', 'RMSE']]
print("\n--- LightGBM Top Results ---")
print(summary_lgbm.sort_values('RMSE').head(5))

# %% Try out the models. It is important that altough XGBoost gave me a better RMSE even with CV, LightGBM ended up performing better on the live data
log_preds = lgbm_grid_search.best_estimator_.predict(test_df)

# Convert from log-space back to actual dollars
# np.expm1 is the inverse of np.log1p
final_preds = np.expm1(log_preds)
# %%
submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": final_preds
})

# Save to CSV
submission.to_csv("data/submission4.csv", index=False)
# %%
submission

# %%
