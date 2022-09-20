# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:21:00 2022

@author: user
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Review':'good'})

print(r.json()) 