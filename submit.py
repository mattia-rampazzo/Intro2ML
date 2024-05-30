from http.client import responses
import requests
import json


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        
preds = {}


res = {
    "images": preds,
    "groupname": "your_group_name"
}

submit(res)