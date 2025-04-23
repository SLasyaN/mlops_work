from metaflow import FlowSpec, step, Parameter
from metaflow import kubernetes, resources, conda_base, retry, catch, timeout
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

@conda_base(
    libraries={
        'numpy': '1.23.5',
        'scikit-learn': '1.2.2',
        'mlflow': '2.15.1',
        'gcsfs': '2023.6.0',
        'pandas': '1.3.3'
    },
    python='3.9.16'
)
class CreditTrainFlow(FlowSpec):

    cv_folds = Parameter(
        "cv_folds",
        help="Number of folds for cross-validation",
        default=3
    )

    @kubernetes()
    @catch(var="error")
    @retry(times=2)
    @step
    def start(self):
        mlflow.set_tracking_uri("https://mlflow-gcp-server-671261532927.us-west2.run.app")
        mlflow.set_experiment("credit-training")

        df = pd.read_csv('gs://mlops_lasya_bucket/data/Credit.csv', index_col=0)
        self.df = df
        self.next(self.preprocess)

    @kubernetes()
    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def preprocess(self):
        from sklearn.model_selection import train_test_split

        df = self.df.copy()
        df.dropna(inplace=True)
        df_encoded = pd.get_dummies(df, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)

        X = df_encoded.drop(columns=['Balance'])
        y = df_encoded['Balance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_test.to_csv("gs://mlops_lasya_bucket/data/X_test.csv", index=False)
        y_test.to_csv("gs://mlops_lasya_bucket/data/y_test.csv", index=False)

        self.X_train = X_train
        self.y_train = y_train
        self.next(self.train_model)

    @kubernetes()
    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=600)
    @step
    def train_model(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        model.fit(self.X_train, self.y_train)

        self.model = model
        self.cv_scores = scores.tolist()
        self.mean_score = mean_score
        self.std_score = std_score
        self.next(self.register_model)

    @kubernetes()
    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def register_model(self):
        mean_score_clean = float(str(self.mean_score).replace(',', '.'))
        std_score_clean = float(str(self.std_score).replace(',', '.'))

        print(f"[DEBUG] Logging cv_mean_neg_mse: {mean_score_clean}")
        print(f"[DEBUG] Logging cv_std_neg_mse: {std_score_clean}")

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                self.model,
                artifact_path='better_models',
                registered_model_name='CreditModel'
            )
            mlflow.log_metric("cv_mean_neg_mse", mean_score_clean)
            mlflow.log_metric("cv_std_neg_mse", std_score_clean)
            # Capture and print model URI
            model_uri = f"runs:/{run.info.run_id}/better_models"
            print(f"📍 Model URI: {model_uri}")
            self.model_uri = model_uri

        self.next(self.end)

    @step
    def end(self):
        print(f"✅ Training complete with {self.cv_folds}-fold cross-validation.")
        print(f"Mean CV Negative MSE: {self.mean_score:.4f}")
        print(f"Std CV Negative MSE: {self.std_score:.4f}")
        print(f"🔗 MLflow model URI: {self.model_uri}")

if __name__ == '__main__':
    CreditTrainFlow()






















