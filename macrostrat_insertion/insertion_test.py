import requests
import json
import os
import time

def make_request():
    request_url = "http://127.0.0.1:9543/record_run"
    data_dir = "extracted_example_relationships"
    for file_name in os.listdir(data_dir):
        if "json" not in file_name:
            continue
        
        # Get the request data
        file_name = "user_test.json"
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r") as reader:
            request_data = json.load(reader)
        
        # Make the request
        response = requests.post(url = request_url, json = request_data)
        if response.status_code != 200:
            print("Failed to process file", file_name, "due to error", response.json())
        
        time.sleep(0.2)
        break

if __name__ == "__main__":
    make_request()