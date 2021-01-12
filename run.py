import sys
import os
import json

sys.path.insert(0, 'src')

from etl import get_data
from clean import clean_data

def main(targets):
    if 'data' in targets:
        with open('config/etl-params.json') as fh:
            data_cfg = json.load(fh)
        get_data(**data_cfg)

    if 'clean' in targets:
        with open('config/clean-params.json') as fh:
            clean_cfg = json.load(fh)
        clean_data(**clean_cfg)

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)