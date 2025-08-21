import os
import json
import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
event_json_file = os.path.join(BASE_DIR, 'event.json')


with open(event_json_file, 'rt', encoding='utf-8') as f_in:
    student = json.load(f_in)


def test_predict():
    actual_result = predict.predict(student)
    expected_result = 5.0

    assert actual_result == expected_result
