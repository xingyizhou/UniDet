import json
import sys
sys.path.insert(0, 'projects/UniDet/')
from unidet.data.datasets.det_categories import categories

if __name__ == '__main__':
    json.dump(categories, open('datasets/metadata/det_categories.json', 'w'))

