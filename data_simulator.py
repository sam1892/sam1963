import random
import time
import datetime
import requests

url = "http://127.0.0.1:5000/ingest-data"

def generate_data():
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "temperature": round(random.uniform(70, 90), 2),
        "vibration": round(random.uniform(0.001, 0.005), 4),
        "wind_speed": round(random.uniform(10, 20), 2),
        "failure": random.choice([0, 1])
    }

while True:
    data = generate_data()
    try:
        response = requests.post(url, json=data)
        print(f"Data sent: {data}")
    except Exception as e:
        print(f"Error sending data: {e}")
    time.sleep(5)
