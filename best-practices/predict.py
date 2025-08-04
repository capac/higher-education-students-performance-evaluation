import pickle
import pandas as pd
from flask import Flask, request, jsonify

xgb_model_file = 'xgb_cls.bin'
with open(xgb_model_file, 'rb') as f_in:
    model = pickle.load(f_in)

preprocessor_file = 'preprocessor.bin'
with open(preprocessor_file, 'rb') as f_in:
    preprocessor = pickle.load(f_in)


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
