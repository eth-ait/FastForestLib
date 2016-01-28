import json

def read_forest(forest_json_file):
    with open(forest_json_file, 'r') as fp:
        obj = json.load(fp)
        return obj['forest']

