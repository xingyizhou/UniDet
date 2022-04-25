# Modified from https://github.com/bethgelab/openimages2coco/blob/master/convert_annotations.py
# The original file is under MIT license by Bethge Lab
# Modified by Xingyi Zhou
# The modification includes:
#   - `file_name` now includes subfolders
#   - We include the `iscrowd` field that is required by the COCO api
#   - We removed fields like `IsOccluded` that are never used
#   - We read image size by loading the image instead of from a pre-processed file

import os
import csv
import json
import argparse
import warnings
# import imagesize # imagesize is buggy on my machine, it sometimes swaps hight and width
from PIL import Image

import numpy as np
import skimage.io as io

from tqdm import tqdm
from collections import defaultdict

def csvread(file):
    if file:       
        with open(file, 'r', encoding='utf-8') as f:
            csv_f = csv.reader(f)
            data = []
            for row in csv_f:
                data.append(row)
    else:
        data = None
        
    return data

def _url_to_license(licenses, mode='http'):
    # create dict with license urls as 
    # mode is either http or https
    
    # create dict
    licenses_by_url = {}

    for license in licenses:
        # Get URL
        if mode == 'https':
            url = 'https:' + license['url'][5:]
        else:
            url = license['url']
        # Add to dict
        licenses_by_url[url] = license
        
    return licenses_by_url

def _list_to_dict(list_data):
    
    dict_data = []
    columns = list_data.pop(0)
    for i in range(len(list_data)):
        dict_data.append({columns[j]: list_data[i][j] for j in range(len(columns))})
                         
    return dict_data

def convert_category_annotations(orginal_category_info):
    
    categories = []
    num_categories = len(orginal_category_info)
    for i in range(num_categories):
        cat = {}
        cat['id'] = i + 1
        cat['name'] = orginal_category_info[i][1]
        cat['freebase_id'] = orginal_category_info[i][0]
        
        categories.append(cat)
    
    return categories

def convert_image_annotations(original_image_metadata,
                              original_image_annotations,
                              original_image_sizes,
                              image_dir,
                              categories,
                              licenses,
                              origin_info=False):
    
    original_image_metadata_dict = _list_to_dict(original_image_metadata)
    original_image_annotations_dict = _list_to_dict(original_image_annotations)
    
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    if original_image_sizes:
        # import pdb; pdb.set_trace()
        image_size_dict = {x[0]:  [int(x[1]), int(x[2])] for x in original_image_sizes[1:]}
    else:
        image_size_dict = {}
    
    # Get dict with license urls
    licenses_by_url_http = _url_to_license(licenses, mode='http')
    licenses_by_url_https = _url_to_license(licenses, mode='https')
    
    # convert original image annotations to dicts
    pos_img_lvl_anns = defaultdict(list)
    neg_img_lvl_anns = defaultdict(list)
    for ann in original_image_annotations_dict[1:]:
        cat_of_ann = cats_by_freebase_id[ann['LabelName']]['id']
        if int(ann['Confidence']) == 1:
            pos_img_lvl_anns[ann['ImageID']].append(cat_of_ann)
        elif int(ann['Confidence']) == 0:
            neg_img_lvl_anns[ann['ImageID']].append(cat_of_ann)
    
    #Create list
    images = []

    # loop through entries skipping title line
    num_images = len(original_image_metadata_dict)
    for i in tqdm(range(num_images), mininterval=0.5):
        # Select image ID as key
        key = original_image_metadata_dict[i]['ImageID']
        
        # Copy information
        img = {}
        img['id'] = key
        img['file_name'] = key[0] + '/' + key + '.jpg'
        img['neg_category_ids'] = neg_img_lvl_anns.get(key, [])
        img['pos_category_ids'] = pos_img_lvl_anns.get(key, [])
        if origin_info:
            img['original_url'] = original_image_metadata_dict[i]['OriginalURL']
            license_url = original_image_metadata_dict[i]['License']
            # Look up license id
            try:
                img['license'] = licenses_by_url_https[license_url]['id']
            except:
                img['license'] = licenses_by_url_http[license_url]['id']

        # Extract height and width
        image_size = image_size_dict.get(key, None)
        if image_size is not None:
            img['width'], img['height'] = image_size
        else:
            filename = os.path.join(image_dir, img['file_name'])
            try:
                # img['width'], img['height'] = imagesize.get(filename)
                image = Image.open(open(filename, 'rb'))
                img['width'], img['height'] = image.shape[1], image.shape[0]
            except:
                print('No image!', filename)
            
        # Add to list of images
        images.append(img)
        
    return images


def convert_instance_annotations(
    original_annotations, images, categories, start_index=0, 
    is_train=True):
    
    original_annotations_dict = _list_to_dict(original_annotations)
    
    imgs = {img['id']: img for img in images}
    cats = {cat['id']: cat for cat in categories}
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    annotations = []
    
    # annotated_attributes = [
    #     attr for attr in ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'] if attr in original_annotations[0]]

    num_instances = len(original_annotations_dict)
    for i in tqdm(range(num_instances), mininterval=0.5):
        # set individual instance id
        # use start_index to separate indices between dataset splits
        key = i + start_index
        csv_line = i
        ann = {}
        ann['id'] = key
        image_id = original_annotations_dict[csv_line]['ImageID']
        ann['image_id'] = image_id
        # ann['freebase_id'] = original_annotations_dict[csv_line]['LabelName']
        freebase_id = original_annotations_dict[csv_line]['LabelName']
        ann['category_id'] = cats_by_freebase_id[freebase_id]['id']
        
        xmin = float(original_annotations_dict[csv_line]['XMin']) * imgs[image_id]['width']
        ymin = float(original_annotations_dict[csv_line]['YMin']) * imgs[image_id]['height']
        xmax = float(original_annotations_dict[csv_line]['XMax']) * imgs[image_id]['width']
        ymax = float(original_annotations_dict[csv_line]['YMax']) * imgs[image_id]['height']
        dx = xmax - xmin
        dy = ymax - ymin
        ann['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
        ann['area'] = round(dx * dy, 2)
        isgroupof = int(original_annotations_dict[csv_line]['IsGroupOf'])
        if isgroupof > 0 or not is_train:
            ann['iscrowd'] = isgroupof
        # for attribute in annotated_attributes:
        #     ann[attribute.lower()] = int(original_annotations_dict[csv_line][attribute])

        annotations.append(ann)
        
    return annotations


def filter_images(images, annotations):
    image_ids = set([ann['image_id'] for ann in annotations])
    filtered_images = [img for img in images if img['id'] in image_ids]
    return filtered_images


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', '--path', dest='path',
                        help='path to openimages data', 
                        type=str)
    parser.add_argument('--version',
                        default='challenge_2019',
                        choices=['v4', 'v5', 'v6', 'challenge_2019'],
                        type=str,
                        help='Open Images Version')
    parser.add_argument('--subsets',
                        type=str,
                        nargs='+',
                        default=['sample'],
                        choices=['train', 'val', 'test', 'sample'],
                        help='subsets to convert')
    parser.add_argument('--task',
                        type=str,
                        default='bbox',
                        choices=['bbox', 'panoptic'],
                        help='type of annotations')
    parser.add_argument('--expand_label', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
base_dir = args.path
if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]



for subset in args.subsets:
    # Convert annotations
    print('converting {} data'.format(subset))
    if args.expand_label:
        assert subset == 'val'

    # Select correct source files for each subset        
    if subset == 'train' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        image_size_sourcefile = 'train_sizes-00000-of-00001.csv'
    elif subset == 'val' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        if args.expand_label:
            annotation_sourcefile = 'challenge-2019-validation-detection-bbox_expanded.csv'
            image_label_sourcefile = 'challenge-2019-validation-detection-human-imagelabels_expanded.csv'
        else:
            annotation_sourcefile = 'challenge-2019-validation-detection-bbox.csv'
            image_label_sourcefile = 'challenge-2019-validation-detection-human-imagelabels.csv'
        image_size_sourcefile = 'validation_sizes-00000-of-00001.csv'
    elif subset == 'sample':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'challenge-2019-validation-detection-bbox_sample00.csv'
        image_label_sourcefile = 'challenge-2019-validation-detection-human-imagelabels.csv'
        image_size_sourcefile = 'validation_sizes-00000-of-00001.csv'
    else:
        assert 0

    # Load original annotations
    print('loading original annotations ...', end='\r')
    original_category_info = csvread(os.path.join(base_dir, 'annotations', category_sourcefile))
    original_image_metadata = csvread(os.path.join(base_dir, 'annotations', image_sourcefile))
    original_image_annotations = csvread(os.path.join(base_dir, 'annotations', image_label_sourcefile))
    original_image_sizes = csvread(os.path.join(base_dir, 'annotations', image_size_sourcefile))
    original_annotations = csvread(os.path.join(base_dir, 'annotations', annotation_sourcefile))

    print('loading original annotations ... Done')
    oi = {}

    # Add basic dataset info
    print('adding basic dataset info')
    oi['info'] = {'contributos': 'Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik',
                  'description': 'Open Images Dataset {}'.format(args.version),
                  'url': 'https://storage.googleapis.com/openimages/web/index.html',
                  'version': '{}'.format(args.version),
                  'year': 2020}

    # Add license information
    print('adding basic license info')
    oi['licenses'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'},
                      {'id': 2, 'name': 'Attribution-NonCommercial License', 'url': 'http://creativecommons.org/licenses/by-nc/2.0/'},
                      {'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/'},
                      {'id': 4, 'name': 'Attribution License', 'url': 'http://creativecommons.org/licenses/by/2.0/'},
                      {'id': 5, 'name': 'Attribution-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-sa/2.0/'},
                      {'id': 6, 'name': 'Attribution-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nd/2.0/'},
                      {'id': 7, 'name': 'No known copyright restrictions', 'url': 'http://flickr.com/commons/usage/'},
                      {'id': 8, 'name': 'United States Government Work', 'url': 'http://www.usa.gov/copyright.shtml'}]


    # Convert category information
    print('converting category info')
    oi['categories'] = convert_category_annotations(original_category_info)

    # Convert image mnetadata
    print('converting image info ...')
    image_dir = os.path.join(base_dir, subset)
    oi['images'] = convert_image_annotations(
        original_image_metadata, original_image_annotations, original_image_sizes, image_dir, oi['categories'], oi['licenses'])

    # Convert instance annotations
    print('converting annotations ...')
    # Convert annotations
    oi['annotations'] = convert_instance_annotations(
        original_annotations, oi['images'], oi['categories'], start_index=0,
        is_train=subset==('train'))
    
    print('Filtering annotations ...')
    if subset in ['train', 'sample']:
        oi['images'] = filter_images(oi['images'], oi['annotations'])
    
    print('Writing to disk ...')
    if args.expand_label:
        print('Saving to expand label!!')
        filename = os.path.join(
            base_dir, 'annotations/', 'oid_{}_{}_expanded.json'.format(args.version, subset))
    else:
        filename = os.path.join(
            base_dir, 'annotations/', 'oid_{}_{}.json'.format(args.version, subset))
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))
    print('Done')
