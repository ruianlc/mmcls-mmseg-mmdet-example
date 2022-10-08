from os import listdir
from util.base_util import os, random, mk_dir

import matplotlib.patches as patches
from sklearn.utils import shuffle
from tqdm import tqdm_notebook



"""
将原始数据集整理成训练集、测试集、验证集以及相应标签文件格式
1、定义输出目录格式：
* meta
  * train.txt
  * val.txt
  * test.txt
* train
* val
* test
2、读取原始数据集
* 从train中划分出train和val
* 从test中读取test

"""
def main():
    data_root = r'F:\\开源分享\\数据集\\KaggleDataSet\\CIFAR-10-images'
    save_dir = r'E:\\代码仓库\\VSCodeProjects\\kaggle-mmcls\\data\\input\\cifar-10'

    meta_dir = os.path.join(save_dir,'meta')
    train_dir = os.path.join(save_dir,'train')
    val_dir = os.path.join(save_dir,'val')
    test_dir = os.path.join(save_dir,'test')
    mk_dir(meta_dir)
    mk_dir(train_dir)
    mk_dir(val_dir)
    mk_dir(test_dir)

    for ff in os.listdir(data_root):
        if ff=='train': # 划分train和val
            subdir = os.path.join(data_root, ff)
            for ll in os.listdir(subdir): # 单个类别
                ssubdir = os.path.join(subdir, ll)
                imgs_list = os.listdir(ssubdir)
                train_idxs = random.sample(range(0, len(imgs_list)), round(len(imgs_list) * 0.8))
                val_idxs = list(set(range(0, len(imgs_list))).difference(set(train_idxs)))

                train_imgs_list = [imgs_list[idx] for idx in train_idxs]
                val_imgs_list = [imgs_list[idx] for idx in val_idxs]


                aa=0
        elif ff=='test': #读取test
            aa=0
        else:
            return

    # # quick look at the label stats
    # print(data['label'].value_counts())

    # # random sampling
    # shuffled_data = shuffle(data)
    # fig, ax = plt.subplots(2,5, figsize=(20,8))
    # fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)
    # # Negatives
    # for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    #     path = os.path.join(train_path, idx)
    #     ax[0,i].imshow(cv_imread(path + '.tif'))
    #     # Create a Rectangle patch
    #     box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')
    #     ax[0,i].add_patch(box)
    # ax[0,0].set_ylabel('Negative samples', size='large')
    # # Positives
    # for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    #     path = os.path.join(train_path, idx)
    #     ax[1,i].imshow(cv_imread(path + '.tif'))
    #     # Create a Rectangle patch
    #     box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')
    #     ax[1,i].add_patch(box)
    # ax[1,0].set_ylabel('Tumor tissue samples', size='large')
    # plt.show()
    
    AA = 0


if __name__ == '__main__':
    main()
