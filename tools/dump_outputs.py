import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json
import numpy as np

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.data import transforms as T
from detectron2.utils.logger import log_every_n_seconds

from unidet.config import add_unidet_config
from unidet.checkpoint import CustomCheckpointer
from unidet.data.multi_dataset_dataloader import build_multi_dataset_train_loader
from unidet.evaluateion.oideval import OIDEvaluator
from unidet.evaluateion.multi_dataset_evaluator import get_unified_evaluator

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        if cfg.MULTI_DATASET.ENABLED:
            # TODO: refactor
            try:
                model.set_eval_dataset(dataset_name)
            except:
                try:
                    model.module.set_eval_dataset(dataset_name)
                except:
                    print('set eval dataset failed.')
        data_loader = build_detection_test_loader(cfg, dataset_name)
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} images".format(len(data_loader)))
        total = min(len(data_loader), cfg.DUMP_NUM_IMG)
        start_time = time.perf_counter()
        model.eval()

        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx > total:
                    break
                _ = model(inputs)
                total_seconds_per_img = (time.perf_counter() - start_time) / (idx + 1)
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. ETA={}".format(
                        idx + 1, total, str(eta)
                    ),
                    n=5,
                )

        if cfg.DUMP_CLS_SCORE:
            class_scores = model.roi_heads.class_scores
            class_scores = [[y.tolist() for y in x] for x in class_scores]
            json.dump(class_scores, open('{}/class_scores_{}.json'.format(
                cfg.OUTPUT_DIR, dataset_name), 'w'))
            model.roi_heads.class_scores = []
            if cfg.DUMP_BBOX:
                boxes = model.roi_heads.dump_boxes
                boxes = [[y.tolist() for y in x] for x in boxes]
                json.dump(boxes, open('{}/boxes_{}.json'.format(
                    cfg.OUTPUT_DIR, dataset_name), 'w'))
                model.roi_heads.dump_boxes = []

    return


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
