# optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-07, by_epoch=True)

optimizer = dict(type='Adam', lr=0.000001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=20, warmup=None) # interval = 1
work_dir = './gdrive/MyDrive/snxx4'

total_epochs = 100
checkpoint_config = dict(interval=10)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/content/gdrive/MyDrive/sn1/epoch_2.pth'
load_from = '/content/gdrive/MyDrive/snxx3/epoch_100.pth'
# load_from = '/content/pre.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

seed = 0
gpu_ids = range(0, 1)


model = dict(
    type='TextSnake',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN_UNet', in_channels=[256, 512, 1024, 2048], out_channels=32),
    bbox_head=dict(
        type='TextSnakeHead',
        in_channels=32,
        loss=dict(type='TextSnakeLoss'),
        postprocessor=dict(
            type='TextSnakePostprocessor', text_repr_type='poly')),
    train_cfg=None,
    test_cfg=None)
dataset_type = 'IcdarDataset'
data_root = 'datac'
train_file=data_root+'/instances_training.json'
# train_file=data_root+'/instances_test.json'
test_file=data_root+'/instances_test.json'

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=train_file,
                img_prefix=data_root+'/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadTextAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='RandomCropPolyInstances',
                instance_key='gt_masks',
                crop_ratio=0.65,
                min_side_ratio=0.3),
            dict(
                type='RandomRotatePolyInstances',
                rotate_ratio=0.5,
                max_angle=20,
                pad_with_fixed_color=False),
            dict(
                type='ScaleAspectJitter',
                img_scale=[(3000, 736)],
                ratio_range=(0.7, 1.3),
                aspect_ratio_range=(0.9, 1.1),
                multiscale_mode='value',
                long_size_bound=800,
                short_size_bound=480,
                resize_type='long_short_bound',
                keep_ratio=False),
            dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='TextSnakeTargets'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='CustomFormatBundle',
                keys=[
                    'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
                    'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
                ],
                visualize=dict(flag=False, boundary_key='gt_text_mask')),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
                    'gt_radius_map', 'gt_sin_map', 'gt_cos_map'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=test_file,
                img_prefix=data_root+'/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 736),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1333, 736), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='IcdarDataset',
                ann_file=test_file,
                img_prefix=data_root+'/imgs',
                pipeline=None)
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 736),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1333, 736), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=10, metric='hmean-iou')
