# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 11:07:18 2022

@author: user
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Review':'ambiance is good'})

print(r.json()) 