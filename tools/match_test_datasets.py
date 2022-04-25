import argparse
import json
import numpy as np
import sys

ROOT_PATH = 'datasets/'
TEST_ANN_PATH = {
    'voc': ROOT_PATH + 'voc/annotations/pascal_test2007.json',
    'viper': ROOT_PATH + 'viper/val/viper_instances_val.json',
    'cityscapes': ROOT_PATH + 'cityscapes/cityscapes_fine_instance_seg_val_coco_format.json',
    'scannet': ROOT_PATH + 'scannet/scannet_instances_1.json',
    'wilddash': ROOT_PATH + 'wilddash/wd_public_02/wilddash_public.json',
    'crowdhuman': ROOT_PATH + 'crowdhuman/annotations/val.json',
    'kitti': ROOT_PATH + 'kitti/train.json',
    # 'coco': ROOT_PATH + 'coco/annotations/instances_val2017.json',
    # 'mapillary': ROOT_PATH + 'mapillary/annotations/validation.json',
    # 'objects365': ROOT_PATH + 'objects365/annotations/objects365_val.json',
    # 'oid': ROOT_PATH + 'oid/annotations/openimages_challenge_2019_val_v2_expanded.json',
}
# META_DATA_PATH = ROOT_PATH + 'metadata/det_categories.json'
GLOVE_PATH = ROOT_PATH + 'glove.42B.300d.txt'

TRAIN_ANN_PATH = {
    'coco': ROOT_PATH + 'coco/annotations/instances_val2017.json',
    'objects365': ROOT_PATH + 'objects365/annotations/objects365_val.json',
    'oid': ROOT_PATH + 'oid/annotations/oid_challenge_2019_val_expanded.json',
    'mapillary': ROOT_PATH + 'mapillary/annotations/validation.json',
}
cats = json.load(open(TRAIN_ANN_PATH['coco'], 'r'))['categories']
coconame2id = {x['name'].strip(): x['id'] for x in cats}
coconame2id[''] = 9999

cats = json.load(open(TRAIN_ANN_PATH['objects365'], 'r'))['categories']
o365name2id = {x['name'].strip(): x['id'] for x in cats}
o365name2id[''] = 9999

cats = json.load(open(TRAIN_ANN_PATH['oid'], 'r'))['categories']
oidname2id = {x['name'].strip(): x['id'] for x in cats}
oidname2id[''] = 9999

cats = json.load(open(TRAIN_ANN_PATH['mapillary'], 'r'))['categories']
mapname2id = {x['name'].strip(): x['id'] for x in cats}
mapname2id[''] = 9999

print('Loading glove dict')
glove_file = open(GLOVE_PATH, 'rb')
glove_dict = {}
for line in glove_file:
    x = line.decode('utf-8').split(' ')
    key = x[0]
    glove_dict[key] = np.array([float(y) for y in x[1:]], dtype=np.float32)

INF = 1
def glove_dist(u, v):
    if u not in glove_dict or v not in glove_dict:
        return INF
    u = glove_dict[u]
    v = glove_dict[v]
    return 1 - (u * v).sum() / ((u * u).sum()**0.5 * (v * v).sum() ** 0.5)

def huristic_dist(u, v):
    if u == v:
        return 0
    if u.endswith(v) or v.endswith(u):
        return 0.01
    if u.startswith(v) or v.startswith(u):
        return 0.02
    return glove_dist(u, v)

def get_matches(cur_name, train_names):
    matches = []
    for m, detect_name_list in enumerate(train_names):
        dist = 1
        for detect_name in detect_name_list:
            dist = min(dist, huristic_dist(cur_name, detect_name))
        matches.append((dist, m, detect_name_list))
    matches = sorted(matches)
    tie =  matches[0][0]
    K = 0
    while K < len(matches) and matches[K][0] == tie:
        K = K + 1
    return matches[:K]

def match_test_datasets(label_space_path, save=True):
    print('Loading', label_space_path)
    data = json.load(open(label_space_path, 'r'))
    categories = data['categories']
    raw_data = data['raw_data'][1:]

    train_categories = sorted(categories, key=lambda x: x['id'])
    predid2name = []
    predid2name_verbose = []
    sort_values = []
    for c, (cat, raw_name) in enumerate(zip(train_categories, raw_data)):
        predid2name.append(cat['name'])
        x = cat['name'].split('_')
        name_verbose = ''
        x = raw_name
        if x[4] != '':
            name_verbose = name_verbose + 'COCO_{},'.format(x[4])
            _ = coconame2id[x[4]]
        if x[3] != '':
            name_verbose = name_verbose + 'O365_{},'.format(x[3])
            _ = o365name2id[x[3]]
        if x[2] != '':
            name_verbose = name_verbose + 'OID_{},'.format(x[2])
            if x[2].endswith('2') or x[2].endswith('1'):
                x[2] = x[2][:-1]
            _ = oidname2id[x[2]]
        if x[5] != '':
            name_verbose = name_verbose + 'Mapillary_{},'.format(x[5])
            _ = mapname2id[x[5]]

        sort_value = (coconame2id[x[4]], o365name2id[x[3]], \
                        oidname2id[x[2]], mapname2id[x[5]])

        predid2name_verbose.append(name_verbose)
        sort_values.append(sort_value)
    print('len(categories[d])', len(categories))
    print('len(predid2name)', len(predid2name))
    # print(predid2name_verbose)

    train_names = []
    for x in predid2name:
        if '--' in x:
            x = x[x.find('--')+2:]
            x = x.replace('--', '_')
        if '/' in x:
            x = x.replace('/', '_')
        x = x.split('_')
        x = [u.lower().replace(' ', '').strip() for u in x if u != '']
        train_names.append(x)
    out = {}
    for d, v in TEST_ANN_PATH.items():
        print(d)
        out[d] = []
        test_categories = json.load(open(v, 'r'))['categories']
        test_categories = sorted(test_categories, key=lambda x: x['id'])
        for j, cat in enumerate(test_categories):
            cur_name = cat['name'].replace(' ', '').lower().strip()
            if '--' in cur_name:
                cur_name = cur_name[cur_name.rfind('--') + 2:]
            matches = get_matches(cur_name, train_names)

            matches_sort = sorted([(sort_values[x[1]], i) for i, x in enumerate(matches)])
            matches = [matches[x[1]] for x in matches_sort]

            matches = [matches[0]]
              
            out[d].append([x[1] for x in matches])
            print(cur_name, end='| ')
            for x in matches:
                print('[{} {}]'.format(x[0], predid2name_verbose[x[1]]), end=', ')
            print()
        print('====')
    if SAVE:
        out_path = label_space_path[:-5] + '_labelmap_test.json'
        print('Writing to', out_path)
        json.dump(out, open(out_path, 'w'))
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--label_space_path',
        help='path to the label space .json file')
    parser.add_argument(
        '--not_save', action='store_true')
    args = parser.parse_args()
    match_test_datasets(args.label_space_path, save=not args.not_save)