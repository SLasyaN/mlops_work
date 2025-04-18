from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5006")

class CreditScoringFlow(FlowSpec):

    model_run_id = Parameter(
        'model_run_id',
        help='The MLflow run ID from which to load the trained model',
        default='9511d7dafa354bab83041fceaa4261ef'  # Replace as needed
    )

    @step
    def start(self):
        # Load test data for prediction
        self.X_new = pd.read_csv("data/X_test.csv")
        self.true_labels = pd.read_csv("data/y_test.csv")
        self.next(self.load_model)

    @step
    def load_model(self):
        model_uri = f'runs:/{self.model_run_id}/model'
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        self.preds = self.model.predict(self.X_new)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        from sklearn.metrics import r2_score

        r2 = r2_score(self.true_labels, self.preds)
        mlflow.set_experiment("credit-training")
        with mlflow.start_run(run_id=self.model_run_id):
            mlflow.log_metric("r2_score", r2)

        print("Prediction R2 score:", r2)
        print("Sample predictions:", self.preds[:5])
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")

if __name__ == '__main__':
    CreditScoringFlow()
