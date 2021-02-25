import sys
import json
import csv

COCO_ANN_PATH = 'datasets/coco/annotations/instances_val2017.json'
OID_ANN_PATH = 'datasets/oid/annotations/openimages_challenge_2019_val_v2_expanded.json'
OBJECTS365_ANN_PATH = 'datasets/objects365/annotations/objects365_val.json'
# OBJECTS365_ANN_PATH = 'datasets/objects365/annotations/objects365v2_val_0422.json'
COL = {'coco': 4, 'objects365': 3, 'oid': 1}

def csvread(file): 
    with open(file, 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        data = []
        for row in csv_f:
            data.append(row)
    return data


def get_unified_label_map(unified_label, cats):
    '''
    Inputs:

    Return:
        unified_label_map: dict of dict
            (dataset (string), cat_id (int)) --> unified_id (int)
    '''
    unified_label_map = {}
    for dataset in cats:
        unified_label_map[dataset] = {}
        col = COL[dataset]
        table_names = [x[col].lower().strip() for x in unified_label[1:]]
        cat_ids = sorted([x['id'] for x in cats[dataset]])
        id2contid = {x: i for i, x in enumerate(cat_ids)}
        for cat_info in cats[dataset]:
            if dataset != 'oid':
                cat_name = cat_info['name']
            else:
                cat_name = cat_info['freebase_id']
            cat_id = id2contid[cat_info['id']]
            if cat_name.lower().strip() in table_names:
                unified_id = table_names.index(cat_name.lower().strip())
                unified_label_map[dataset][cat_id] = unified_id
            else:
                print('ERROR!', cat_name, 'not find!')
        print(dataset, 'OK')
    return unified_label_map

if __name__ == '__main__':
    unified_label_path = sys.argv[1]
    unified_label = csvread(unified_label_path)
    cats = {}
    print('Loading')
    cats['coco'] = json.load(open(COCO_ANN_PATH, 'r'))['categories']
    cats['oid'] = json.load(open(OID_ANN_PATH, 'r'))['categories']
    cats['objects365'] = json.load(open(OBJECTS365_ANN_PATH, 'r'))['categories']
    
    unified_label_map = get_unified_label_map(unified_label, cats)
    unified_label_map_list = {d: [unified_label_map[d][i] \
        for i in range(len(cats[d]))] for d in cats}
    dataset_inds = {d: sorted(unified_label_map[d].values()) \
        for d in cats}
    dataset_mask = {d: [1 if i in dataset_inds[d] else 0 \
        for i in range(len(unified_label) - 1)] for d in cats}
    categories = [{'id': i, 'name': x[0]} for i, x in \
        enumerate(unified_label[1:])]

    out = {'categories': categories, 'label_map_dict': unified_label_map, 
        'label_map': unified_label_map_list,
        'raw_data': unified_label, 'dataset_inds': dataset_inds,
        'dataset_mask': dataset_mask}
    
    json.dump(out, open(
        '{}.json'.format(unified_label_path[:-4]), 'w'))
    for x in categories:
        print('  {' + "'id': {}, 'name': '{}'".format(x['id'], x['name']) + '},')

