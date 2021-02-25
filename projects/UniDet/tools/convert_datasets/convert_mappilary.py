# git+git://github.com/waspinator/pycococreator.git@0.2.0
import argparse
import glob
import json
import shutil
from multiprocessing import Pool, Value, Lock
from os import path, mkdir

import numpy as np
import tqdm
from PIL import Image
from pycococreatortools import pycococreatortools as pct

parser = argparse.ArgumentParser(description="Convert Vistas to seamseg format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory of Vistas")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")

_SPLITS = ["training", "validation"]
_IMAGES_DIR, _IMAGES_EXT = "images", "jpg"
_LABELS_DIR, _LABELS_EXT = "instances", "png"


def main(args):
    print("Loading Vistas from", args.root_dir)

    # Process meta-data
    categories, version = _load_metadata(args.root_dir)
    cat_id_mvd_to_iss, cat_id_iss_to_mvd, num_stuff, num_thing = _cat_id_maps(categories)

    # Prepare directories
    lst_dir = path.join(args.out_dir, "lst")
    _ensure_dir(lst_dir)
    coco_dir = path.join(args.out_dir, "coco")
    _ensure_dir(coco_dir)

    # Run conversion
    images = []
    for split in _SPLITS:
        print("Converting", split, "...")

        # Find all image ids in the split
        img_ids = []
        for name in glob.glob(path.join(args.root_dir, split, _IMAGES_DIR, "*." + _IMAGES_EXT)):
            _, name = path.split(name)
            img_ids.append(name[:-(1 + len(_IMAGES_EXT))])

        # Write the list file
        with open(path.join(lst_dir, split + ".txt"), "w") as fid:
            fid.writelines(img_id + "\n" for img_id in img_ids)

        # Convert to COCO detection format
        coco_out = {
            "info": {"version": str(version)},
            "images": [],
            "categories": [],
            "annotations": []
        }
        for cat_id, cat_meta in enumerate(categories):
            if cat_meta["instances"]:
                coco_out["categories"].append({
                    "id": cat_id_mvd_to_iss[cat_id],
                    "name": cat_meta["name"]
                })

        # Process images in parallel
        worker = _Worker(categories, cat_id_mvd_to_iss, path.join(args.root_dir, split), args.out_dir)
        with Pool(initializer=_init_counter, initargs=(_Counter(0),)) as pool:
            total = len(img_ids)
            for img_meta, coco_img, coco_ann in tqdm.tqdm(pool.imap(worker, img_ids, 8), total=total):
                images.append(img_meta)

                # COCO annotation
                coco_out["images"].append(coco_img)
                coco_out["annotations"] += coco_ann

        # Write COCO detection format annotation
        try:
            with open(path.join(coco_dir, split + ".json"), "w") as fid:
                json.dump(coco_out, fid)
        except:
            pass
        # import pdb; pdb.set_trace()


def _cat_id_maps(categories):
    cat_id_mvd_to_iss = dict()
    cat_id_iss_to_mvd = dict()

    num_thing, num_stuff = 0, 0
    # Find stuff
    for cat_id, cat_meta in enumerate(categories):
        if not cat_meta["evaluate"]:
            continue

        if not cat_meta["instances"]:
            cat_id_mvd_to_iss[cat_id] = num_stuff
            cat_id_iss_to_mvd[num_stuff] = cat_id
            num_stuff += 1

    for cat_id, cat_meta in enumerate(categories):
        if not cat_meta["evaluate"]:
            continue

        if cat_meta["instances"]:
            cat_id_mvd_to_iss[cat_id] = num_thing + num_stuff
            cat_id_iss_to_mvd[num_thing + num_stuff] = cat_id
            num_thing += 1

    return cat_id_mvd_to_iss, cat_id_iss_to_mvd, num_stuff, num_thing


def _load_metadata(root_dir):
    with open(path.join(root_dir, "config.json")) as fid:
        metadata = json.load(fid)
    categories = metadata["labels"]
    version = metadata["version"]

    return categories, version


def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


class _Worker:
    def __init__(self, categories, cat_id_mvd_to_iss, root_dir, out_dir):
        self.categories = categories
        self.cat_id_mvd_to_iss = cat_id_mvd_to_iss
        self.root_dir = root_dir
        self.out_dir = out_dir

    def __call__(self, img_id):
        coco_ann = []

        # Load the annotation
        with Image.open(path.join(self.root_dir, _LABELS_DIR, img_id + "." + _LABELS_EXT)) as lbl_img:
            lbl = np.array(lbl_img, dtype=np.uint16)
            lbl_size = lbl_img.size

        mvd_ids = np.unique(lbl)

        # Compress the labels and compute cat
        lbl_out = np.zeros(lbl.shape, np.int32)
        cat = [255]
        iscrowd = [0]
        for mvd_id in mvd_ids:
            mvd_class_id = int(mvd_id // 255)
            category = self.categories[mvd_class_id]

            # If it's a void class just skip it
            if not category["evaluate"]:
                continue

            # Extract all necessary information
            iss_class_id = self.cat_id_mvd_to_iss[mvd_class_id]
            iss_instance_id = len(cat)
            iscrowd_i = 1 if "group" in category["name"] else 0
            mask_i = lbl == mvd_id

            # Save ISS format annotation
            cat.append(iss_class_id)
            iscrowd.append(iscrowd_i)
            lbl_out[mask_i] = iss_instance_id

            # Compute COCO detection format annotation
            if category["instances"]:
                category_info = {"id": iss_class_id, "is_crowd": iscrowd_i == 1}
                coco_ann_i = pct.create_annotation_info(
                    counter.increment(), img_id, category_info, mask_i, lbl_size, tolerance=2)
                if coco_ann_i is not None:
                    coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_id, img_id + "." + _IMAGES_EXT, lbl_size)

        # Write output
        out_msk_dir = path.join(self.out_dir, "msk")
        out_img_dir = path.join(self.out_dir, "img")
        _ensure_dir(out_msk_dir)
        _ensure_dir(out_img_dir)

        Image.fromarray(lbl_out).save(path.join(out_msk_dir, img_id + ".png"))
        shutil.copy(path.join(self.root_dir, _IMAGES_DIR, img_id + "." + _IMAGES_EXT),
                    path.join(out_img_dir, img_id + "." + _IMAGES_EXT))

        img_meta = {
            "id": img_id,
            "cat": cat,
            "size": (lbl_size[1], lbl_size[0]),
            "iscrowd": iscrowd
        }

        return img_meta, coco_img, coco_ann


def _init_counter(c):
    global counter
    counter = c


class _Counter:
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val


if __name__ == "__main__":
    main(parser.parse_args())