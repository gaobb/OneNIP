#!/usr/bin/env python3
# coding:utf-8

# **********************************************************
# * Author        : danylgao (Bin-Bin Gao)
# * Email         : danylgao@tencent.com
# * Create time   : 2022-11-18 11:43
# * Last modified : 2022-11-18 11:43
# * Filename      : gen_train_val_json.py
# * Description   : 
# **********************************************************

import os
import csv
import json


def data_mapping(row):
    old_keys = ["object",  "label", "image", "mask"]
    map_keys = {"object": "clsname",  "label": "label", "image": "filename", "mask": "maskname"}

    obj = {map_keys[key]: row[key] for key in old_keys if key in row}
    if obj["label"] == "anomaly":
        obj["label"] = 1
        obj["label_name"] = "defective"
    
    if obj["label"] == "normal":
        obj["label"] = 0
        obj["label_name"] = "good"
    return obj


def write_csv_to_json(filename, split):
    json_file = f'./{split}.json'
    with open(filename, 'r', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                obj = data_mapping(row)
                with open(json_file, "a") as f:
                    f.write(json.dumps(obj))
                    f.write('\n')
                    
filename = "../data/VisA/split_csv/1cls.csv"
write_csv_to_json(filename, 'train')
write_csv_to_json(filename, 'test')