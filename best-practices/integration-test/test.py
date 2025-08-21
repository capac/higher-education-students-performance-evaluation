import os
import json
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
event_json_file = os.path.join(BASE_DIR, 'event.json')


with open(event_json_file, 'rt', encoding='utf-8') as f_in:
    student = json.load(f_in)


LOCALHOST_URL = "http://localhost:9696/predict"
response = requests.post(LOCALHOST_URL, json=student)
print(f'Response text: {response.json()}')
print(f'Status code: {response.status_code}')
