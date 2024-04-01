# House Prices Prediction

## Project Overview
This project uses advanced regression techniques on the Kaggle House Prices dataset to predict house prices. We explore various machine learning models and employ a stacked regression approach to achieve high accuracy. The goal is to predict the sales price for each house based on a wide range of features.

## Dataset
The dataset used in this project is the "House Prices - Advanced Regression Techniques" dataset from Kaggle. It contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The dataset is split into training and test sets, with the training set including the sale price (target variable).

## Methodology
- **Data Preprocessing:** Handled missing values, encoded categorical variables, and performed log transformation on the target variable to normalize its distribution.
- **Feature Engineering:** Utilized one-hot encoding to transform categorical variables into a form that could be provided to ML algorithms.
- **Model Selection:** Explored multiple regression models including RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, ExtraTreesRegressor, and SVR.
- **Model Tuning:** Applied GridSearchCV for hyperparameter tuning to find the optimal settings for our models.
- **Model Stacking:** Created a stacked model using the regressors above as base models and Ridge regression as the final estimator.
- **Evaluation:** Evaluated the model on a validation set and reported the RMSE and R² score.

## Results
The stacked model achieved the following performance metrics:
- **Train RMSE:** 0.0482
- **Train R²:** 0.9847
- **Validation RMSE:** 0.1367
- **Validation R²:** 0.8998

Best parameters from GridSearchCV were:
- `Extra Trees__n_estimators`: 200
- `Gradient Boosting__learning_rate`: 0.05
- `Gradient Boosting__n_estimators`: 200
- `Random Forest__n_estimators`: 200
- `SVR__svr__C`: 1
- `XGBoost__n_estimators`: 200

## Running the Code
Ensure you have Python installed and the necessary libraries: pandas, numpy, scikit-learn, xgboost, and joblib. Clone the repository, navigate to the project directory, and run:

```bash
python house_prices_prediction.py
