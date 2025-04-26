# pip install mlflow
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score

mlflow.set_tracking_uri('http://127.0.0.1:5005')
df = pd.read_csv('Credit.csv')
df=df.drop(columns=["Unnamed: 0"])

# Load dataset (Assuming df is already loaded)
categorical_cols = ['Gender', 'Student', 'Married', 'Ethnicity']
numerical_cols = ['Income', 'Limit', 'Cards', 'Age', 'Education', 'Balance']  # Adjust based on dataset

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Student'] = df['Student'].map({'Yes': 1, 'No': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Ethnicity'] = df['Ethnicity'].astype('category').cat.codes

# Handling missing values
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

imputer_num = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

# Define features and target
X = df.drop(columns=['Rating'])  # Replace 'Rating' with the actual target column
y = df['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLFlow Experiment
mlflow.set_experiment("credit_experiment")

# Define Hyperparameter Search Space
search_space = hp.choice("classifier_type", [
    {
        "type": "dt",
        "criterion": hp.choice("dtree_criterion", ["gini", "entropy"]),
        "max_depth": hp.choice("dtree_max_depth", [None, hp.randint("dtree_max_depth_int", 1, 10)]),
        "min_samples_split": hp.randint("dtree_min_samples_split", 2, 10)
    },
    {
        "type": "rf",
        "n_estimators": hp.randint("rf_n_estimators", 20, 500),
        "max_features": hp.randint("rf_max_features", 2, X.shape[1]),
        "criterion": hp.choice("rf_criterion", ["gini", "entropy"])
    },
    {
        "type": "linear_reg",
        "fit_intercept": hp.choice("linear_fit_intercept", [True, False])
    }
])

# Define Objective Function for Hyperparameter Tuning
def objective(params):
    with mlflow.start_run():
        model_type = params.pop("type")
        
        # Initialize model based on type
        if model_type == "dt":
            model = DecisionTreeClassifier(**params)
        elif model_type == "rf":
            model = RandomForestClassifier(**params)
        elif model_type == "linear_reg":
            model = LinearRegression(**params)
        else:
            return {"loss": 1, "status": STATUS_OK}

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model
        if model_type == "linear_reg":
            metric = mean_squared_error(y_test, y_pred)  # MSE for Regression
            mlflow.log_metric("MSE", metric)
        else:
            metric = accuracy_score(y_test, y_pred)  # Accuracy for Classifiers
            mlflow.log_metric("accuracy", metric)

        # Log parameters and model
        mlflow.set_tag("Model", model_type)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, artifact_path="models")

        return {"loss": metric if model_type == "linear_reg" else -metric, "status": STATUS_OK}

# Perform Hyperparameter Tuning
trials = Trials()
best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=tpe.suggest,
    max_evals=32,
    trials=trials
)

print("Best Hyperparameters:", best_result)

# Feature Selection: Selecting Important Features from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = feature_importances.nlargest(5).index.tolist()  # Selecting top 5 important features
print("Selected Features:", top_features)

# Re-run the experiment with selected features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Retrain and log models with selected features
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

best_model = None
best_metric = float("inf")  # For MSE comparison
best_model_name = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name + " (selected features)"):
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        
        if model_name == "Linear Regression":
            metric = mean_squared_error(y_test, y_pred)  # MSE for Regression
            mlflow.log_metric("MSE", metric)
        else:
            metric = accuracy_score(y_test, y_pred)  # Accuracy for Classifiers
            mlflow.log_metric("accuracy", metric)

        # Log model
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, artifact_path=f'models/{model_name}')
        print(f"{model_name} Performance: {metric}")

        # Track the best model
        if model_name == "Linear Regression":
            if metric < best_metric:  # Lower MSE is better
                best_metric = metric
                best_model = model
                best_model_name = model_name
        else:
            if metric > best_metric:  # Higher accuracy is better
                best_metric = metric
                best_model = model
                best_model_name = model_name

# Log best model and register it
if best_model:
    with mlflow.start_run(run_name="Best Model"):
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        mlflow.log_metric("best_metric", best_metric)
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/best_model", best_model_name)

print(f"Best Model: {best_model_name} with Performance Metric: {best_metric}")
