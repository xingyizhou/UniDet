import json
import sys
import copy

def _update_dict(initial_dict, update):
    for key, value_list in update.items():
        if key in initial_dict:
            initial_dict[key].update(value_list)
        else:
            initial_dict[key] = set(value_list)
def _build_plain_hierarchy(hierarchy, is_root=False):
    all_children = set([])
    all_keyed_parent = {}
    all_keyed_child = {}
    if 'Subcategory' in hierarchy:
        for node in hierarchy['Subcategory']:
            keyed_parent, keyed_child, children = _build_plain_hierarchy(node)
            _update_dict(all_keyed_parent, keyed_parent)
            _update_dict(all_keyed_child, keyed_child)
            all_children.update(children)

    if not is_root:
        all_keyed_parent[freebase2id[hierarchy['LabelName']]] = copy.deepcopy(all_children)
        all_children.add(freebase2id[hierarchy['LabelName']])
        for child, _ in all_keyed_child.items():
            all_keyed_child[child].add(freebase2id[hierarchy['LabelName']])
        all_keyed_child[freebase2id[hierarchy['LabelName']]] = set([])

    return all_keyed_parent, all_keyed_child, all_children


ann_path = sys.argv[1]
oid_hierarchy_path = sys.argv[2]
out_path = oid_hierarchy_path[:-5] + '-list.json'

dataset = json.load(open(ann_path))
oid_hierarchy = json.load(open(oid_hierarchy_path, 'r'))
cat_info = dataset['categories']
freebase2id = {x['freebase_id']: int(x['id']) for x in cat_info}
id2freebase = {x['id']: x['freebase_id'] for x in cat_info}
id2name = {x['id']: x['name'] for x in cat_info}

childs, fas, _ = _build_plain_hierarchy(oid_hierarchy, is_root=True)
childs = {int(x): sorted([z for z in y]) for x, y in childs.items()}
fas = {int(x): sorted([z for z in y]) for x, y in fas.items()}
parents_and_childs = {}

for x in fas:
    print(id2name[x], 'fas:', [id2name[y] for y in fas[x]], 
        '. chs:', [id2name[y] for y in childs[x]])
    parents_and_childs[int(x)] = sorted([y for y in fas[x]] + [y for y in childs[x]])

out = {'hierarchy': oid_hierarchy, 'categories': cat_info, 'childs': childs, 
    'parents': fas, 'parents_and_childs': parents_and_childs}

json.dump(out, open(out_path, 'w'))