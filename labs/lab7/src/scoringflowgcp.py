from metaflow import FlowSpec, step, Parameter
from metaflow import conda_base, catch, retry, timeout
import pandas as pd
import mlflow
import mlflow.sklearn

@conda_base(
    python='3.9.16',
    libraries={
        'pandas': '1.3.3',
        'scikit-learn': '1.2.2',
        'mlflow': '2.18.0',
        'gcsfs': '2023.6.0',
        'numpy': '1.23.5'
    }
)
class CreditScoringFlow(FlowSpec):

    model_run_id = Parameter(
        'model_run_id',
        help='The MLflow run ID from which to load the trained model',
        default='8e44bc83298f4476a31fb92939083859'  
    )

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def start(self):
        print("Loading test data from GCS...")
        self.X_new = pd.read_csv("gs://mlops_lasya_bucket/data/X_test.csv")
        self.true_labels = pd.read_csv("gs://mlops_lasya_bucket/data/y_test.csv")
        self.next(self.load_model)

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def load_model(self):
        print(f"Loading model from MLflow run ID: {self.model_run_id}")
        mlflow.set_tracking_uri("https://mlflow-gcp-server-671261532927.us-west2.run.app")
        model_uri = 'runs:/8e44bc83298f4476a31fb92939083859/model'
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def predict(self):
        print("Running predictions...")
        self.preds = self.model.predict(self.X_new)
        self.next(self.evaluate)

    @catch(var="error")
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def evaluate(self):
        from sklearn.metrics import r2_score
        mlflow.set_tracking_uri("https://mlflow-gcp-server-671261532927.us-west2.run.app")
        mlflow.set_experiment("credit-training")

        r2 = r2_score(self.true_labels, self.preds)
        print("Prediction R2 score:", r2)
        print("Sample predictions:", self.preds[:5])

        # Optionally log R2 to the same run (or a new one if preferred)
        with mlflow.start_run(run_id=self.model_run_id):
            mlflow.log_metric("r2_score", r2)

        self.r2_score = r2
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")
        print(f"Logged R2 score: {self.r2_score:.4f}")

if __name__ == '__main__':
    CreditScoringFlow()
