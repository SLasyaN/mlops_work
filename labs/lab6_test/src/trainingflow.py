from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn

class CreditTrainFlow(FlowSpec):

    cv_folds = Parameter(
        "cv_folds",
        help="Number of folds for cross-validation",
        default=3
    )

    @step
    def start(self):
        # Set MLflow tracking configuration
        mlflow.set_tracking_uri("http://127.0.0.1:5006") 
        mlflow.set_experiment("credit-training")

        # Load data
        df = pd.read_csv('data/Credit.csv', index_col=0)
        self.df = df
        self.next(self.preprocess)

    @step
    def preprocess(self):
        from sklearn.model_selection import train_test_split

        df = self.df.copy()
        df.dropna(inplace=True)

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)

        X = df_encoded.drop(columns=['Balance'])
        y = df_encoded['Balance']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save test data for later use
        X_test.to_csv("data/X_test.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)

        # Assign to self for next step
        self.X_train = X_train
        self.y_train = y_train

        self.next(self.train_model)

    @step
    def train_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Cross-validation
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Train on full training data after cross-validation
        model.fit(self.X_train, self.y_train)

        # Save to self for next step
        self.model = model
        self.cv_scores = scores.tolist()
        self.mean_score = mean_score
        self.std_score = std_score

        self.next(self.register_model)

    @step
    def register_model(self):
        # Start a new MLflow run
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(self.model, artifact_path='model', registered_model_name='CreditModel')

            # Log metrics
            mlflow.log_metric("cv_mean_neg_mse", self.mean_score)
            mlflow.log_metric("cv_std_neg_mse", self.std_score)

        self.next(self.end)

    @step
    def end(self):
        print(f"Training complete and model registered with {self.cv_folds}-fold cross-validation.")
        print(f"Mean CV Negative MSE: {self.mean_score:.4f}")
        print(f"Std CV Negative MSE: {self.std_score:.4f}")

if __name__ == '__main__':
    CreditTrainFlow()
