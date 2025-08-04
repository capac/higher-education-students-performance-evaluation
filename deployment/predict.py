import pickle
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

# best XGBoost model run saved in MLflow
RUN_ID = '654e9d9b4e034499a3cbc9cd18d45951'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_uri = f'runs:/{RUN_ID}/artifacts'
model = mlflow.pyfunc.load_model(model_uri)

path = client.download_artifacts(
    run_id=RUN_ID,
    path='best_model/preprocessor.bin'
    )
with open(path, 'rb') as f_out:
    preprocessor = pickle.load(f_out)


def predict(X):
    sr = pd.DataFrame([X])
    features = preprocessor.transform(sr)
    pred = model.predict(features)
    return float(pred[0])


app = Flask('higher-education-students-performance-evaluation')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    evaluation = request.get_json()
    prediction = predict(evaluation)

    result = {'grade': prediction}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
