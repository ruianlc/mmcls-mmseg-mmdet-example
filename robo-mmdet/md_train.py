import time
import numpy as np
import os

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import mmcv
from mmcv import Config

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


# # Regist model so that we can access the class through str in configs
# @DATASETS.register_module()
# class MyDataset(CustomDataset):

#     CLASSES = ('person', 'bicycle', 'car', 'motorcycle')

#     def load_annotations(self, ann_file):
#         ann_list = mmcv.list_from_file(ann_file)

#         data_infos = []
#         for i, ann_line in enumerate(ann_list):
#             if ann_line != '#':
#                 continue

#             img_shape = ann_list[i + 2].split(' ')
#             width = int(img_shape[0])
#             height = int(img_shape[1])
#             bbox_number = int(ann_list[i + 3])

#             anns = ann_line.split(' ')
#             bboxes = []
#             labels = []
#             for anns in ann_list[i + 4:i + 4 + bbox_number]:
#                 bboxes.append([float(ann) for ann in anns[:4]])
#                 labels.append(int(anns[4]))

#             data_infos.append(
#                 dict(
#                     filename=ann_list[i + 1],
#                     width=width,
#                     height=height,
#                     ann=dict(
#                         bboxes=np.array(bboxes).astype(np.float32),
#                         labels=np.array(labels).astype(np.int64))
#                 ))

#         return data_infos

#     def get_ann_info(self, idx):
#         return self.data_infos[idx]['ann']


def data_train(backbone_file):
    cfg = Config.fromfile(backbone_file)

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # Modify number of classes as per the model head.
    cfg.model.roi_head.bbox_head.num_classes = 20
    # Comment/Uncomment this to training from scratch/fine-tune according to the 
    # model checkpoint path. 
    cfg.load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    # # Let's have a look at the final config used for finetuning
    # print(f'Config:\n{cfg.pretty_text}')


def main():
    # define constant
    config_file = 'configs/faster_rcnn_r50_fpn_1x_coco.py'

    # segmentor train
    data_train(config_file)

if __name__ == '__main__':
    main()