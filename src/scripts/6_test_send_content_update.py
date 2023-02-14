
import requests
import json

from pathlib import Path

def send_content_update( content_update_json, api_key ):
    headers = {"Content-Type": "application/json; charset=utf-8",
               "api-key": api_key}
    response = requests.post("http://127.0.0.1:5000/fresh_content",
                             json=content_update_json,
                             headers=headers)
    return response

directory = "../data"
filename = "data_for_frontend.json"
content_full_path = Path(directory, filename)

with open( content_full_path, "r" ) as content_update_file:
    content_update_json = json.load( content_update_file )

valid_api_keys = []
api_keys_path = Path(directory, "valid_api_keys.txt")
with open( api_keys_path, "r" ) as api_keys_file:
    for key_string in api_keys_file:
        valid_api_keys.append( key_string )
api_key = valid_api_keys[0]

response = send_content_update( content_update_json, api_key )
print(response)

print("done")