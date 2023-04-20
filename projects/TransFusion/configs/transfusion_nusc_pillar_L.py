custom_imports = dict(imports=["projects.TransFusion.transfusion"], allow_failed_imports=False)
default_scope = 'mmdet3d'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
metainfo = dict(classes=class_names)
voxel_size = [0.2, 0.2, 8]
out_size_factor = 4
evaluation = dict(interval=1)
dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
input_modality = dict(
    use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False
)
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="ObjectSample",
        db_sampler=dict(
            data_root=None,
            info_path=data_root + "nuscenes_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5,
                ),
            ),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2,
            ),
            points_loader=dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ),
        ),
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5],
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "/nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            box_type_3d="LiDAR",
        ),
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)
test_dataloader = val_dataloader
model = dict(
    type="TransFusionDetector",
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 60000))),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type="BN1d", eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(type="PointPillarsScatter", in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="TransFusionHead",
        num_proposals=200,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
        ),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        loss_heatmap=dict(type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="mmdet.FocalLossCost", gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[512, 512, 1],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range,
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=[512, 512, 1],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )
    ),
)
lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01), 
    clip_grad=dict(max_norm=0.1, norm_type=2)
)  # for 8gpu * 2sample_per_gpu
# lr_config = dict(policy="cyclic", target_ratio=(10, 0.0001), cyclic_times=1, step_ratio_up=0.4)
# momentum_config = dict(
#     policy="cyclic", target_ratio=(0.8947368421052632, 1), cyclic_times=1, step_ratio_up=0.4
# )
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()
backend_args = None
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = "INFO"
work_dir = None
load_from = None
resume_from = False
workflow = [("train", 1)]
gpu_ids = range(0, 8)
