# from q2.scr.preprocess import map_location
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp

from .utils import fix_model_state_dict
from .vision_transformer import vit_small_patch16_256


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False, backbone=None, target_size=1):
        super().__init__()
        self.cfg = cfg

        if self.cfg['model_name'] == 'vit_small_patch16_256':
            self.model = vit_small_patch16_256(patch_size=16, num_classes=1)
        else:

            self.model = timm.create_model(self.cfg['model_name'], pretrained=False)
            
        if 'efficientnet' in self.cfg['model_name']:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            if self.cfg['model_name'] == 'tf_efficientnet_b0':
                self.model.load_state_dict(torch.load('./exp/ssl001/tf_efficientnet_b0_simsiam_best_collapse_level.pth', map_location='cpu'))

        elif 'resnet' in self.cfg['model_name']:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif 'nfnet' in self.cfg['model_name']:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        elif ('vit' in self.cfg['model_name']):
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()

            if 'small' in self.cfg['model_name']:
                
                if self.cfg['model_name'] == 'vit_small_patch16_224':
                    state_dict = torch.load('../output_dino_vit_small_patch16_224/checkpoint.pth', map_location='cpu')['student']
                elif self.cfg['model_name'] == 'vit_small_patch16_256':
                    state_dict = torch.load('../output_dino_vit_small_patch16_256_/checkpoint.pth', map_location='cpu')['student']
                else:
                    state_dict = torch.load('../output_dino_vit_small_wall/checkpoint.pth', map_location='cpu')['student']
                state_dict = fix_model_state_dict(state_dict)

                self.model.load_state_dict(
                    state_dict,
                )
        
        if backbone is not None:
            self.model.load_state_dict(backbone.state_dict())

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.n_features, target_size)
            )
        # self.n_features = self.model.head.fc.in_features
        # self.model.head.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, x):
        x = self.model(x)
        output = self.fc(x)
        return output


class SegmentationModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(smp, cfg['base_model'])(
            encoder_name=cfg['encoder_name'],
            encoder_weights='imagemet' if pretrained else None,
            classes=cfg['n_classes'],
            activation=None,
            aux_params=None if cfg['classification'] else cfg['aux_params']
        )
        if cfg['encoder_name'] == 'timm-efficientnet-b0':
            self.model.encoder.load_state_dict(torch.load('./exp/ssl000/tf_efficientnet_b0_simsiam_best_collapse_level.pth', map_location='cpu'))

    def forward(self, x):
        return self.model(x)