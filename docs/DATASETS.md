## Datasets

Our project involves 11 datasets. 
4 datasets (COCO, Objects365, OpenImages, Mapillary) are used in training/ evaluation and the rest are only used in testing.
For most experiments in the paper, we only need the 3 large datasets: COCO, Objects365, and OpenImages.
All datasets should be placed under `$UNIDET_ROOT/datasets/`. It's OK to only setup part of them. Our pre-processed annotation files can be directly downloaded [here](https://drive.google.com/drive/folders/1rW-oesL2L-9QbD_HRhpifJm15z5AOel-).

~~~
$UNIDET_ROOT/
    datasets/
        coco/
        objects365/
        oid/
        mapillary/
        voc/
        viper/
        cityscapes/
        scannet/
        wilddash/
        crowdhuman/
        kitti/
~~~


#### COCO

We follow the standard setup from detectron2 for COCO.
Download the data from the [official website](https://cocodataset.org/#download) and place them as below:

~~~
coco/
    annotations/
        instances_train2017.json
        instances_val2017.json
    train2017/
    val2017/
~~~

#### Objects365

Objects365 can be set up the same way as COCO:

Download the data from the [official website](https://www.objects365.org/download.html), and orgnize the data as below:

~~~
objects365/
    annotations/
        objects365_train.json
        objects365_val.json
    train/
    val/
~~~

#### OpenImages

We use the [challenge2019 version](https://storage.googleapis.com/openimages/web/challenge2019.html) of OpenImages. The easiest way to download the data is by using the [script](https://github.com/ozendelait/rvc_devkit/blob/master/objdet/download_oid_boxable.sh) provided in the RVC challenge. The total dataset size is around 527GB. Please make sure you have sufficient storage before downloading it.
In addition, download the label hierarchy file for evaluation

~~~
wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label500-hierarchy.json -P datasets/oid/annotations/
~~~


After downloading and extracting data, place the data as below:

~~~
oid/
    annotations/
        challenge-2019-train-detection-bbox.csv
        challenge-2019-validation-detection-bbox.csv
        challenge-2019-train-detection-human-imagelabels.csv
        challenge-2019-validation-detection-human-imagelabels.csv
        challenge-2019-classes-description-500.csv
        challenge-2019-label500-hierarchy.json
    images/
        0/
        1/
        2/
        ...
~~~

Then convert the annotation to COCO format:

~~~
python tools/convert_datasets/convert_oid.py -p datasets/oid/ --subsets train
python tools/convert_datasets/convert_oid.py -p datasets/oid/ --subsets val --expand_label
~~~

This will produce `oid_challenge_2019_train_bbox.json` and `oid_challenge_2019_val_expanded.json` under `oid/annotations/`.

The suffix `_expanded` means expanding the original labels with its label hierarchy.
This is used in the [officiel evaluation metric](https://storage.googleapis.com/openimages/web/evaluation.html#object_detection_eval).

Next, preprocess and convert the original label hierarchy.

~~~
python tools/convert_datasets/get_oid_hierarchy.py datasets/oid/annotations/oid_challenge_2019_val_expanded.json datasets/oid/annotations/challenge-2019-label500-hierarchy.json
~~~

This creates `challenge-2019-classes-description-500-list.json` under `oid/annotations/`. The file will be used by the hierarchical-aware loss and the OpenImage evaluation script.

For your convenience, we have packed up all converted annotation files [here]().


#### Mapillary

Download the dataset from the [official website](https://www.mapillary.com/dataset/vistas).

Unzip and place the data as the following:

~~~
mapillary/
    training/images/
    validation/images/
    annotations/
        training.json
        validation.json
~~~

#### Pascal VOC

We can use the built-in VOC dataset from detectron2.

Download the data from the [official website](http://host.robots.ox.ac.uk/pascal/VOC/index.html) and place it as:

~~~
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
~~~

For unified evaluation, we converted the annotation to COCO format for convenience. Our converted annotation can be found [here](https://drive.google.com/drive/folders/1C-8dNdnj8TbUEBKrjWXIsu62O_uIPgqN?usp=sharing).

#### Cityscapes

We can use the builtin Cityscapes dataset from detectron2.

Download the data from the [official website](https://www.cityscapes-dataset.com/downloads/) and place it as:

~~~
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
  leftImg8bit/
    train/
    val/
    test/
~~~

The dataset processing script requires installing the API

~~~
pip install git+https://github.com/mcordts/cityscapesScripts.git
~~~

For unified evaluation, we converted the annotation to COCO format for convenience. Our converted annotation can be found [here](https://drive.google.com/drive/folders/1C-8dNdnj8TbUEBKrjWXIsu62O_uIPgqN?usp=sharing).

#### CrowdHuman

Download the data from the [official website](https://www.crowdhuman.org/download.html) and place it as:

~~~
crowdhuman/
    CrowdHuman_train/
        Images/
    CrowdHuman_val/
        Images/
    annotation_train.odgt
    annotation_val.odgt
~~~

Convert them to COCO format:

~~~
python tools/convert_datasets/convert_crowdhuman.py
~~~

This creates `corwdhuman/annotations/train.json` and `corwdhuman/annotations/val.json`.

#### VIPER, ScanNet, WildDash, KITTI

We used the data preparing scripts from the [RVC challenge devkit](https://github.com/ozendelait/rvc_devkit/tree/master/segmentation). Please follow the instructions there. Our preprocessed annotations can be found [here](https://drive.google.com/drive/folders/1C-8dNdnj8TbUEBKrjWXIsu62O_uIPgqN?usp=sharing). Please note that you still need to download the images from the official websites and place them properly.