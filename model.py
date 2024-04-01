import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import joblib
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['SalePrice'] = np.log(train_data['SalePrice']) #log transform of the data

for dataset in [train_data, test_data]:
    missing_values = dataset.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index
    for column in missing_columns:
        if dataset[column].dtype == 'object':
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])
        else:
            dataset[column] = dataset[column].fillna(dataset[column].median())

X = pd.get_dummies(train_data.drop(['SalePrice', 'Id'], axis=1), drop_first=True)
y = train_data['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = pd.get_dummies(test_data.drop('Id', axis=1), drop_first=True)
X_test = X_test.reindex(columns = X.columns, fill_value=0)

#Here we create a stacked model 
base_models = [
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(random_state=42)),
    ('Extra Trees', ExtraTreesRegressor(random_state=42)),
    ('SVR', make_pipeline(StandardScaler(), SVR()))
]

params = {
    'Random Forest__n_estimators': [100, 200],
    'Gradient Boosting__n_estimators': [100, 200],
    'Gradient Boosting__learning_rate': [0.05, 0.1],
    'XGBoost__n_estimators': [100, 200],
    'Extra Trees__n_estimators': [100, 200],
    'SVR__svr__C': [0.1, 1, 10]
}

stacked_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(random_state=42), cv=5)

#We should also do a grid search to ensure that the parameter values that we pick are the best 
grid_search = GridSearchCV(estimator=stacked_model, param_grid=params, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best score: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_val = best_model.predict(X_val)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_score_train = r2_score(y_train, y_pred_train)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_score_val = r2_score(y_val, y_pred_val)

print(f"Train RMSE: {rmse_train}, Train R²: {r2_score_train}")
print(f"Validation RMSE: {rmse_val}, Validation R²: {r2_score_val}")

y_pred_test = best_model.predict(X_test)
submission = pd.DataFrame({
    "Id": test_data['Id'],
    "SalePrice": np.exp(y_pred_test)  #reverse log transform
})
#Saving submission file for Kaggle submission 
submission.to_csv('house_prices_submission.csv', index=False)
print("Submission file has been created.")

# Saving the best model
joblib.dump(best_model, 'best_stacked_model.joblib')
print("Best model has been saved.")
