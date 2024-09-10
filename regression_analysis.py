import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

def t_test_analysis(X_train, y_train):
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())
    print("\nT-test Analysis:")
    print(model.t_test(np.identity(len(X_train.columns))))

def association_analysis(X_train, y_train):
    f_values, p_values = f_regression(X_train, y_train)
    print("\nAssociation Analysis (F-test):")
    print("F-values:", f_values)
    print("p-values:", p_values)

def final_regression_model(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nFinal Regression Model:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def confidence_interval_analysis(X_train, y_train):
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    print("\nConfidence Interval Analysis:")
    print(model.conf_int())

def stepwise_regression_analysis(X_train, y_train):
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()
    print("\nStepwise Regression and Adjusted R-square Analysis:")
    print("AIC:", model.aic)
    print("BIC:", model.bic)
    print("Adjusted R-square:", model.rsquared_adj)

def collinearity_analysis(X_train):
    vif = pd.DataFrame()
    vif["Features"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    print("\nCollinearity Analysis (VIF Method):")
    print(vif)

def perform_regression(X_train, y_train, X_test, y_test):
    t_test_analysis(X_train, y_train)
    association_analysis(X_train, y_train)
    final_regression_model(X_train, y_train, X_test, y_test)
    confidence_interval_analysis(X_train, y_train)
    stepwise_regression_analysis(X_train, y_train)
    collinearity_analysis(X_train)

    # Add polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Lasso regression with cross-validated alpha selection
    lasso = Lasso(random_state=42, max_iter=10000)
    param_grid = {'alpha': np.logspace(-4, 0, 20)}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
    lasso_cv.fit(X_train_poly, y_train)

    # Print best alpha and R-squared
    print(f"Best alpha: {lasso_cv.best_params_['alpha']}")
    print(f"R-squared (train): {lasso_cv.score(X_train_poly, y_train)}")
    print(f"R-squared (test): {lasso_cv.score(X_test_poly, y_test)}")

    # Train Lasso with the best alpha
    lasso_best = Lasso(alpha=lasso_cv.best_params_['alpha'], random_state=42, max_iter=10000)
    lasso_best.fit(X_train_poly, y_train)