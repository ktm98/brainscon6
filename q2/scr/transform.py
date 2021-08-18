import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def get_transforms(cfg, data):
    transform = cfg['augmentation']['transform']

    transform = cfg['augmentation']['transform'][data]
    return A.Compose([
        # A.Resize(CFG.size, CFG.size),
        # A.PadIfNeeded(min_height=CFG.size, min_width=CFG.size, p=1.0),
        *(getattr(A, aug)(**arg) for aug, arg in transform.items()),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # *dropout,
        ToTensorV2(),
    ])

    # elif data == 'valid':
    #     transform = cfg['augmentation']['transform'][data]
    #     return A.Compose([
    #         # A.PadIfNeeded(min_height=CFG.size, min_width=CFG.size, p=1.0),
    #         *(getattr(A, aug)(**arg) for aug, arg in transform.items()),
    #         # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         ToTensorV2(),
    #     ])

def get_segmentation_transforms(cfg, data):
    transform = cfg['augmentation']['transform']

    transform = cfg['augmentation']['transform'][data]
    ret = A.Compose([
        # A.Resize(CFG.size, CFG.size),
        # A.PadIfNeeded(min_height=CFG.size, min_width=CFG.size, p=1.0),
        *(getattr(A, aug)(**arg) for aug, arg in transform.items() if aug != 'Normalize'),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # *dropout,
        # ToTensorV2(),
    ])

    return ret


def segmentation_preprocess(cfg, data):
    # mean = cfg['augmentation']['transform'][data]['Normalize']['mean']
    # std = cfg['augmentation']['transform'][data]['Normalize']['std']
    if 'encoder_name' in cfg.keys():
        encoder_name = cfg['encoder_name']
    else:
        encoder_name = 'resnet34'
    preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    return A.Compose([
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor)
    ])

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')