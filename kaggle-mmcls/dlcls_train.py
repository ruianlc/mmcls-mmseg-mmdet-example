import time
import numpy as np
import os

from mmcls.datasets import DATASETS, BaseDataset, build_dataset
from mmcls.apis import train_model
from mmcls.models import build_classifier
import mmcv
from mmcv import Config


# Regist model so that we can access the class through str in configs
@DATASETS.register_module()
class MyCifar10Dataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            # The ann_file is the annotation files we generate above.
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.img_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


def data_train(backbone_file):
    cfg = Config.fromfile(backbone_file)

    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # Build the classifier
    model = build_classifier(cfg.model)
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Begin finetuning
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        meta=dict())

    # # Let's have a look at the final config used for finetuning
    # print(f'Config:\n{cfg.pretty_text}')


def main():
    # define constant
    config_file = 'configs/myconfig_vit.py'

    # segmentor train
    data_train(config_file)

if __name__ == '__main__':
    main()