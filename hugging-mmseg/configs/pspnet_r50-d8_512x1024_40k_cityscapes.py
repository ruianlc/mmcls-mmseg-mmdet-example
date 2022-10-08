## % model
# Since we use ony one GPU, BN is used instead of SyncBN
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    #pretrained='open-mmlab://resnet50_v1c', # 在线模型下载
    #pretrained='pretrain_model/resnet50_v1c_trick-2cccc1ad.pth', # 本地模型加载
    # 主干网络 (backbone): 通常是卷积网络的堆叠，来做特征提取，例如 ResNet, HRNet
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # 解码头 (decoder head): 用于语义分割图的解码的组件（得到分割结果）
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,)),
    # 辅助头(auxiliary head)：It is a deep supervision trick to improve the accuracy
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

## % dataset
dataset_type = 'MyADE20KDataset'
data_root = 'data/ADEChallengeData2016'
img_dir = 'images'
ann_dir = 'annotations'
work_dir = 'data/output'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    ## 数据加载
    dict(type='LoadImageFromFile'), # 图像
    dict(type='LoadAnnotations'),   # 标注
    ## 预处理
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug', # 测试时数据增强
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir+'\\training',
        ann_dir=ann_dir+'\\training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir + '\\validation',
        ann_dir=ann_dir + '\\validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir + '\\validation',
        ann_dir=ann_dir + '\\validation',
        pipeline=test_pipeline),
)

## % default runtime
# 日志配置文件：log_config 包裹了许多日志钩 (logger hooks) 而且能去设置间隔 (intervals)
log_config = dict(
    interval=500, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'

# 使用预训练的模型权重来获取更好的性能，导入之前预训练模型参数，重新训练
load_from = ''#'checkpoint\\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

## % schedules
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(
    type='AdamW',
    lr=6e-04,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        ),
    ),
)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000, meta=dict(CLASSES=None, PALETTE=None))
# 评估配置文件：每执行多少次迭代评估模型表现
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

## % others
gpu_ids = range(1)
seed = 42
device = 'cpu'
