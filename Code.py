import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from pyswarm import pso

# Read the data
data = pd.read_csv('training data.csv')

# Separate features and target variables
X = data.drop(['W', 'Sa'], axis=1)
y_w = data['W']
y_sa = data['Sa']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the list of models
models = {
    'ANN': MLPRegressor(),
    'SVM': SVR(),
    'RF': RandomForestRegressor(),
    'GBDT': GradientBoostingRegressor()
}


# Define the objective function for particle swarm optimization
def objective_function(params, model, X, y):
    # Set parameters based on the model type
    if isinstance(model, MLPRegressor):
        model.set_params(hidden_layer_sizes=tuple(map(int, params[:2])),
                         alpha=params[2], learning_rate_init=params[3])
    elif isinstance(model, SVR):
        model.set_params(C=params[0], epsilon=params[1], gamma=params[2])
    elif isinstance(model, RandomForestRegressor):
        model.set_params(n_estimators=int(params[0]), max_depth=int(params[1]),
                         min_samples_split=int(params[2]))
    elif isinstance(model, GradientBoostingRegressor):
        model.set_params(n_estimators=int(params[0]), learning_rate=params[1],
                         max_depth=int(params[2]))

    # Train the model and calculate R2 score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return -r2_score(y_test, y_pred)  # Return negative R2 score as PSO minimizes the objective


# Predict W
best_model_w = None
best_r2_w = float('inf')
best_features_w = []

for name, model in models.items():
    # Set parameter ranges
    if name == 'ANN':
        lb = [10, 10, 0.0001, 0.0001]
        ub = [100, 100, 0.1, 0.1]
    elif name == 'SVM':
        lb = [0.1, 0.01, 0.01]
        ub = [10, 1, 1]
    elif name == 'RF':
        lb = [10, 1, 2]
        ub = [100, 20, 10]
    elif name == 'GBDT':
        lb = [10, 0.01, 1]
        ub = [100, 1, 10]

    # Run PSO
    best_params, _ = pso(objective_function, lb, ub, args=(model, X_scaled, y_w))

    # Train the model with best parameters
    if name == 'ANN':
        model.set_params(hidden_layer_sizes=tuple(map(int, best_params[:2])),
                         alpha=best_params[2], learning_rate_init=best_params[3])
    elif name == 'SVM':
        model.set_params(C=best_params[0], epsilon=best_params[1], gamma=best_params[2])
    elif name == 'RF':
        model.set_params(n_estimators=int(best_params[0]), max_depth=int(best_params[1]),
                         min_samples_split=int(best_params[2]))
    elif name == 'GBDT':
        model.set_params(n_estimators=int(best_params[0]), learning_rate=best_params[1],
                         max_depth=int(best_params[2]))

    # Evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_w, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    if r2 < best_r2_w:
        best_model_w = model
        best_r2_w = r2

# Select the top three features for W
if hasattr(best_model_w, 'feature_importances_'):
    importances = best_model_w.feature_importances_
else:
    importances = np.abs(best_model_w.coef_[0])
best_features_w = X.columns[np.argsort(importances)[-3:]].tolist()

# Predict Sa
remaining_features = [col for col in X.columns if col not in best_features_w]
X_sa = X[remaining_features]
X_sa_scaled = scaler.fit_transform(X_sa)

best_model_sa = None
best_r2_sa = float('inf')

for name, model in models.items():
    # Set parameter ranges (same as for W)
    if name == 'ANN':
        lb = [10, 10, 0.0001, 0.0001]
        ub = [100, 100, 0.1, 0.1]
    elif name == 'SVM':
        lb = [0.1, 0.01, 0.01]
        ub = [10, 1, 1]
    elif name == 'RF':
        lb = [10, 1, 2]
        ub = [100, 20, 10]
    elif name == 'GBDT':
        lb = [10, 0.01, 1]
        ub = [100, 1, 10]

    # Run PSO
    best_params, _ = pso(objective_function, lb, ub, args=(model, X_sa_scaled, y_sa))

    # Train the model with best parameters
    if name == 'ANN':
        model.set_params(hidden_layer_sizes=tuple(map(int, best_params[:2])),
                         alpha=best_params[2], learning_rate_init=best_params[3])
    elif name == 'SVM':
        model.set_params(C=best_params[0], epsilon=best_params[1], gamma=best_params[2])
    elif name == 'RF':
        model.set_params(n_estimators=int(best_params[0]), max_depth=int(best_params[1]),
                         min_samples_split=int(best_params[2]))
    elif name == 'GBDT':
        model.set_params(n_estimators=int(best_params[0]), learning_rate=best_params[1],
                         max_depth=int(best_params[2]))

    # Evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(X_sa_scaled, y_sa, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    if r2 < best_r2_sa:
        best_model_sa = model
        best_r2_sa = r2

# Print results
print(f"Best model for W: {type(best_model_w).__name__}")
print(f"R2 score for W: {1 - best_r2_w}")
print(f"Best features for W: {best_features_w}")

print(f"\nBest model for Sa: {type(best_model_sa).__name__}")
print(f"R2 score for Sa: {1 - best_r2_sa}")
print(f"Features for Sa: {remaining_features}")