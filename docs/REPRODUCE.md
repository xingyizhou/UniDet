# REPRODUCE

This document provides instructions to reproduce the results in our paper. Before getting started, make sure you have finished installation and dataset setup. All models can be directly downloaded [here](https://drive.google.com/drive/folders/1rW-oesL2L-9QbD_HRhpifJm15z5AOel-).


## Partitioned detector

A partitioned detector is a detector with split-classifier for each dataset. During testing, it needs to know the source dataset for each image to determine which output head to look at. We provide configs and models for different datasets and schedules:

| Model                      | COCO | Objects365 | OpenImages | Mapillary | links   |
|----------------------------|------|------------|------------|-----------|---------|
| Partitioned_COI_R50_2x     | 41.8 | 20.6       | 62.7       | -         |[config](../configs/Partitioned_COI_R50_2x.yaml)/ [weights](https://drive.google.com/file/d/1lUvgRQGcdheN7BHa9zoJoGCOLc5_ziej)|
| Partitioned_COI_R50_6x     | 44.6 | 23.6       | 64.8       | -         |[config](../configs/Partitioned_COI_R50_6x.yaml)/ [weights](https://drive.google.com/file/d/1o2yP9ghS5UUyfEh6WVuMdjnGKpjsqDuE)|
| Partitioned_COI_R50_8x     | 45.5 | 24.6       | 65.7       | -         |[config](../configs/Partitioned_COI_R50_8x.yaml)/ [weights](https://drive.google.com/file/d/1D6kh1P2VnYOE3IkW03Vp_TEJeXPWXorE)|
| Partitioned_COIM_R50_6x+2x | 45.1 | 24.0       | 65.1       | 14.9      |[config](../configs/Partitioned_COIM_R50_6x+2x.yaml)/ [weights](https://drive.google.com/file/d/186NjyTDiWaFmkLlj_3kl-mojcJTzqER8)|
| Partitioned_COI_RS101_2x   | 48.5 | 27.7       | 67.7       | -         |[config](../configs/Partitioned_COI_RS101_2x.yaml)/ [weights](https://drive.google.com/file/d/110JSpmfNU__7T3IMSJwv0QSfLLo_AqtZ)|

To evaluate a partitioned detector on the validation set of its training datasets (e.g, Partitioned_COI_R50_2x), run 

~~~
python train_net.py --config-file configs/Partitioned_COI_R50_2x.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS models/Partitioned_COI_R50_2x.pth
~~~

With 8 GPUs, the evaluation will take 2 mins for COCO, >30mins for OpenImages, and >15mins for Objects365.
The results should meet the numbers above.

To train these models, run

~~~
python train_net.py --config-file configs/Partitioned_COI_R50_2x.yaml --num-gpus 8
~~~

Training a `2x` Res50 model takes ~24 hours on 8 RTX-2080-Tis. Both our `Partitioned_COI_R50_8x` model and `Partitioned_COIM_R50_6x+2x` are finetuned from the `Partitioned_COI_R50_6x` model. To finetune a model, run

~~~
python train_net.py --config-file configs/Partitioned_COI_R50_8x.yaml --num-gpus 8 MODEL.WEIGHTS output/UniDet/Partitioned_COI_R50_2x/model_final.pth SOLVER.RESET_ITER True SOLVER.MAX_ITER 180000 SOLVER.STEPS "(120000,160000)"
~~~

## Learning a unified label space

Once we have a partitioned detector, we run it on the validation sets of the training datasets to collect predictions. We will learn the class relations based on these predictions.

~~~
python train_net.py --config-file Partitioned_COI_R50_6x.yaml --num-gpus 8 --eval-only MULTI_DATASET.UNIFIED_EVAL True
~~~

This produces `output/UniDet/Partitioned_COI_R50_6x/inference_*/unified_instances_results.json`.
Next, we load the predictions and learn the unified label space using this [ipython notebook](../tools/UniDet_learn_labelspace_mAP.ipynb). 

The notebook prints spread-sheet-friendly text. We can copy-paste this to spreadsheet software (e.g., Google sheet) to visualize the label space.
Then download the spread-sheet as `.csv` file and put it under `dataset/label_spaces`, and run the following script to convert the `.csv` to a json file:

~~~
python tools/create_unified_label_json.py datasets/label_space/learned_mAP.csv
~~~

This produces` datasets/label_space/learned_mAP.json`.
We have already put all converted files used in this project in this folder.

## Unified detector

With a unified label space, we can turn a partitioned detector into a unified detector by merging its last classification layer. A unified detector does not need to know the image source during testing. To merge the `Partitioned_COI_R50_2x` model into a unified detector using our learned label space `datasets/label_spaces/learned_mAP.json`, run

~~~
python train_net.py --config-file configs/Partitioned_COI_R50_2x.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS models/Partitioned_COI_R50_2x.pth MULTI_DATASET.UNIFY_LABEL_TEST True MULTI_DATASET.UNIFIED_LABEL_FILE datasets/label_spaces/learned_mAP.json
~~~

The expected results should be 41.0mAP for COCO, 20.3 mAP for O365, and 62.6 mAP@0.5 for OpenImages.


To retrain a unified detector end-to-end using the learned label space, run

~~~
python train_net.py --config-file configs/Unified_COI_R50_2x.yaml --num-gpus 8
~~~

The trained model is expected to give 42.0mAP for COCO, 20.9 mAP for O365, and 62.8 mAP@0.5 for OpenImages.

We provide a zoo of retrained unified detectors below:

| Model                              | COCO | Objects365 | OpenImages | Mapillary | weights |
|------------------------------------|------|------------|------------|-----------|---------|
| Unified_human_COI_R50_2x           | 41.5 | 20.6       | 62.6       | -         |[config](../configs/Unified_human_COI_R50_2x.yaml)/ [weights](https://drive.google.com/file/d/1z5qw03bHK8XX_4Nnjvnk7IgfiSw84Cjw)|
| Unified_learned_COI_R50_2x         | 42.0 | 20.9       | 62.8       | -         |[config](../configs/Unified_learned_COI_R50_2x.yaml)/ [weights](https://drive.google.com/file/d/1RT0KCM1rr5Y8c_gsqa_mQ4parnlOog7P)|
| Unified_learned_COI_R50_6x         | 44.6 | 23.3       | 64.5       | -         |[config](../configs/Unified_learned_COI_R50_6x.yaml)/ [weights](https://drive.google.com/file/d/1DIzXuk5BZqjZy29YVMFhxCo9ZYpzMZO2)|
| Unified_learned_COI_R50_8x         | 45.4 | 24.4       | 66.0       | -         |[config](../configs/Unified_learned_COI_R50_8x.yaml)/ [weights](https://drive.google.com/file/d/1bFpYy7FzmTQj9JKbCgixuAwrb5kkepZ3)|
| Unified_learned_COIM_R50_6x+2x     | 44.9 | 23.9       | 65.7       | 14.8      |[config](../configs/Unified_learned_COIM_R50_6x+2x.yaml)/ [weights](https://drive.google.com/file/d/1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq)|
| Unified_learned_COI_RS200_6x       | 52.6 | 31.1       | 67.6       | -         |[config](../configs/Unified_learned_COI_RS200_6x.yaml)/ [weights](https://drive.google.com/file/d/1MP8B44_FbSUOY-u7_hkXLF28Zrk-JDRr)|
| Unified_learned_COIM_RS200_6x+2x   | 52.2 | 31.3       | 67.8       | 20.3      |[config](../configs/Unified_learned_COIM_RS200_6x+2x.yaml)/ [weights](https://drive.google.com/file/d/1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI)|

Similarly, the `Unified_learned_COI_R50_8x` and `Unified_learned_COI_R50_8x` models are finetuned from the `Unified_learned_COI_R50_6x` model in our implementation.

`Unified_learned_COIM_RS200_6x+2x` is the model with the same setting as our RVC challenge submission.
We are not able to release our original model for the RVC challenge as the model used a no-longer-available version (v2) of Objects365 dataset (with a different label space).
The model above gives a similar performance on the all validations sets.

## Zero-shot cross dataset evaluation

To evaluate our unified model on new test datasets, run

~~~
python train_net.py --config-file configs/Unified_OCIM_R50_6x+2x.yaml --num-gpus 8 --eval-only MATCH_NOVEL_CLASSES_FILE 'datasets/label_spaces/learned_mAP_labelmap_test.json' UNIFIED_EVAL True UNIFIED_NOVEL_CLASSES_EVAL True DATASETS.TEST "('voc_cocoformat_test','viper_val', 'scannet_val','wilddash_public','kitti_train','crowdhuman_val', 'cityscapes_cocoformat_val',)"
~~~

The results should be

|  VOC  | VIPER |  CityScapes  | ScanNet | WildDash | CrowdHuman | KITTI | mean |
|-------|-------|--------------|---------|----------|------------|-------|------|
| 82.9  | 21.3  | 52.6         | 29.8    | 34.7     | 70.7       | 39.9  | 47.3 |

The novel class matching file `datasets/label_spaces/learned_mAP+M_labelmap_test.json` is obtained from matching the test dataset labels to the unified label space by a word-embedding-based matching:

~~~
python tools/match_test_datasets.py datasets/label_spaces/learned_mAP+M.json
~~~

The scripts require the GloVe embedding dict, which is downloaded from [here](https://github.com/stanfordnlp/GloVe) and should be placed at `datasets/glove.42B.300d.txt`. We have already provided our matched file under `datasets/label_spaces/learned_mAP+M_labelmap_test.json` if you don't want to run the matching yourself.