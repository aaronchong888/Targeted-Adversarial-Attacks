# import necessary packages
import sys
import os
import argparse
import json
import numpy as np
import cv2

jsonPath = os.path.join(os.path.dirname(__file__), 'original_index.json')
with open(jsonPath) as json_file:
    mapping_dict = json.load(json_file)

new_dict = {}
for key in mapping_dict:
    print(key)
    str_list = key.lower().split(', ')
    print(str_list)
    for item in str_list:
        new_dict[item] = mapping_dict[key]
print(new_dict)

with open('imagenet_index.json', 'w') as fp:
    json.dump(new_dict, fp)
