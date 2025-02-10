import requests

url = "http://localhost:30000/update_weights"
data = {"model_path": "/root/vla-cot/checkpoint"}

response = requests.post(url, json=data)
response_json = response.json()
print(response.json())