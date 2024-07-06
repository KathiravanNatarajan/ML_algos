from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

# Create a dictionary of regression models with baseline parameters
regression_models = {
    'Linear Regression': LinearRegression(fit_intercept=True, normalize=False),
    'Ridge Regression': Ridge(alpha=1.0, fit_intercept=True, normalize=False),
    'Lasso Regression': Lasso(alpha=1.0, fit_intercept=True, normalize=False),
    'ElasticNet Regression': ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=None, min_samples_split=2),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
    'AdaBoost Regressor': AdaBoostRegressor(n_estimators=50, learning_rate=1.0),
    'Support Vector Regressor': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=5, weights='uniform'),
    'CatBoost Regressor': CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, silent=True)
}

# Example usage: Print the models with their parameters
for model_name, model in regression_models.items():
    print(f"{model_name}: {model}")
