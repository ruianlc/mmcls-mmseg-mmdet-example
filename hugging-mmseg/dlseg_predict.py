from util.base_util import np, plt, os, get_project_rootpath, mk_dir

from PIL import Image
import torch
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def apply_mask(image, mask, palette, alpha=0.5):
    img_ann = image.copy() # 类别标注
    img_seg = image.copy() # 去除背景保留晾制烟叶区域
    for c in range(3):
        for l in range(1, len(palette)):
            img_ann[:, :, c] = np.where((mask==l), image[:, :, c] * (1 - alpha) + alpha * palette[l][c], img_ann[:, :, c]) # 对应类别原始图片与标注颜色各占一半权重
            
        img_seg[:, :, c] = np.where(mask==0, 0, image[:, :, c]) # 0 - 背景
    return img_ann, img_seg

def data_predict(model, img_path, palette):
    img = Image.open(img_path)
    img_masked = np.array(img).copy()

    result = inference_segmentor(model, img_path)[0]
    # # show the results
    # show_result_pyplot(model, img, result, palette=get_palette('cityscapes'), fig_size=(15, 10))
    img_ann, img_seg = apply_mask(img_masked, result, palette=palette)

    return img_ann, img_seg
    

def main():
    # define constant
    config_file = 'configs\\myconfig.py'
    checkpoint_file = 'data\\output\\iter_500.pth'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    img_root = 'data/testing1'
    save_ann_root = r'data/output/ann'
    mk_dir(save_ann_root)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=DEVICE)

    for ff in os.listdir(img_root):
        img_path = os.path.join(img_root, ff)
        img_ann, img_seg = data_predict(model, img_path, palette=PALETTE)

        img_ann_save = Image.fromarray(img_ann.astype(np.uint8))
        img_ann_save.save(save_ann_root+'/'+ff)

if __name__ == '__main__':
    main()
