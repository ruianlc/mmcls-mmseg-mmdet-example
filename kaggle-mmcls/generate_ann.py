import shutil
import os
import os.path as osp
from itertools import chain
import mmcv


# Generate mapping from class_name to label
def find_folders(root_dir):
    folders = [
        d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# Generate annotations
def gen_annotations(root_dir):
    annotations = dict()
    folder_to_idx = find_folders(root_dir)

    for cls_dir, label in folder_to_idx.items():
        cls_to_label = [
            '{} {}'.format(cls_dir + '/' + filename, label) 
            for filename in mmcv.scandir(osp.join(root_dir, cls_dir), suffix='.jpg')
        ]
        annotations[cls_dir] = cls_to_label
    return annotations


def main():
    data_root = './data/input/cifar-10/'
    train_dir = osp.join(data_root, 'train/')
    val_dir = osp.join(data_root, 'val/')
    test_dir = osp.join(data_root, 'test/')

    # Split train/val set
    mmcv.mkdir_or_exist(val_dir)
    class_dirs = [
        d for d in os.listdir(train_dir) if osp.isdir(osp.join(train_dir, d))
    ]
    for cls_dir in class_dirs:
        train_imgs = [filename for filename in mmcv.scandir(osp.join(train_dir, cls_dir), suffix='.jpg')]
        # Select first 4/5 as train set and the last 1/5 as val set
        train_length = int(len(train_imgs)*4/5)
        val_imgs = train_imgs[train_length:]
        # Move the val set into a new dir
        src_dir = osp.join(train_dir, cls_dir)
        tar_dir = osp.join(val_dir, cls_dir)
        mmcv.mkdir_or_exist(tar_dir)
        for val_img in val_imgs:
            shutil.move(osp.join(src_dir, val_img), osp.join(tar_dir, val_img))
        
    # Save train annotations
    with open(osp.join(data_root, 'train.txt'), 'w') as f:
        annotations = gen_annotations(train_dir)
        contents = chain(*annotations.values())
        f.writelines('\n'.join(contents))

    # Save val annotations
    with open(osp.join(data_root, 'val.txt'), 'w') as f:
        annotations = gen_annotations(val_dir)
        contents = chain(*annotations.values())
        f.writelines('\n'.join(contents))
        
    # Save test annotations
    with open(osp.join(data_root, 'test.txt'), 'w') as f:
        annotations = gen_annotations(test_dir)
        contents = chain(*annotations.values())
        f.writelines('\n'.join(contents))

    # Generate classes
    folder_to_idx = find_folders(train_dir)
    classes = list(folder_to_idx.keys())
    with open(osp.join(data_root, 'classes.txt'), 'w') as f:
        f.writelines('\n'.join(classes))


if __name__ == '__main__':
    main()