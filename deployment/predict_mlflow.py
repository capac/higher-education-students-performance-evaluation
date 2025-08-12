import pickle
from flask import Flask, request, jsonify
import mlflow
import pandas as pd


# MLflow parameters
RUN_ID = '654e9d9b4e034499a3cbc9cd18d45951'
EXPERIMENT_NAME = 'higher-education-students-performance-evaluation'
MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
loaded_model = None

app = Flask(EXPERIMENT_NAME)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def retrieve_model_from_mlflow():
    """Load the model from MLflow registry"""
    global loaded_model
    global preprocessor
    try:
        # Load from a specific run
        model_uri = f'runs:/{RUN_ID}/artifacts'
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Load preprocessor model from registry
        client = mlflow.tracking.MlflowClient()
        path = client.download_artifacts(
            run_id=RUN_ID,
            path='best_model/preprocessor.bin'
            )
        with open(path, 'rb') as f_out:
            preprocessor = pickle.load(f_out)

    except Exception as e:
        print(f"Error loading model: {e}")
        loaded_model = None


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify(
                {'error': 'No data provided'}
                ), 400

        if loaded_model is None:
            return jsonify(
                {"error": "Model not loaded. Please check MLflow setup."}
                ), 500

        # Make prediction
        sr = pd.DataFrame([data])
        features = preprocessor.transform(sr)
        prediction = loaded_model.predict(features)

        # Return prediction
        result = {
            'prediction': float(prediction[0]),
            'input_features': len(data),
        }

        return jsonify(result)

    except Exception as e:
        print(f'Prediction error: {e}')
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Load the model from MLflow
    retrieve_model_from_mlflow()

    # Start Flask app
    app.run(host='0.0.0.0', port=9696, debug=False)
