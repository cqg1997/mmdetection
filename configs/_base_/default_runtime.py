checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'./work_dirs/RCAretinanet_r50_fpn_1x_coco/epoch_12.pth'
resume_from = None#'./work_dirs/RCAretinanet_r50_fpn_1x_coco/epoch_10.pth'
workflow = [('train', 1)]
