from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch

from util.base_util import plt


def main():
    config_file = r'configs/faster_rcnn_r50_fpn_1x_voc.py'
    
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    #checkpoint_file = r'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    checkpoint_file = r'data/output/epoch_2.pth'
    
    device = 'cpu'
    img = r'data/demo.jpg'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=device)

    result = inference_detector(model, img)
    # show the results
    show_result_pyplot(model, img, result)
    plt.show()

if __name__ == '__main__':
    main()