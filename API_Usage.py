# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 01:48:46 2023

@author: Alex
"""

import requests

RCSBCODE = "7Z0X"
seq = "API_test"

api_url = f"http://127.0.0.1:5000/ProteinDesign/{RCSBCODE}/{seq}"
print(api_url)
post_data = {"Source": "API_test",
            "hbonds": 6,
            "BindingEnergy": -32.35931305264907,
            "contact surface area": 31.93952531149623
            }
            
response = requests.post(api_url, json=post_data)
print(response)
print(response.json())
print(response.status_code)


api_url = f"http://127.0.0.1:5000/ProteinDesign"
print(api_url)
response = requests.get(api_url)

print(type(response.json()))
#print(response.status_code)


