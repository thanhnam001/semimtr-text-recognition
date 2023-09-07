import logging
import torch.nn as nn

from semimtr.modules.attention import PositionAttention, Attention
from semimtr.modules.model import Model
from semimtr.modules.backbone import ResTranformer
from semimtr.modules.convvit import ConvViT
from semimtr.modules.resnet import resnet45
from semimtr.utils.utils import if_none


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = if_none(config.model_vision_loss_weight, 1.0)
        # self.out_channels = if_none(config.model_vision_d_model, 512)
        self.out_channels = if_none(config.model_vision_d_model, 768) # For MCMAE

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        elif config.model_vision_backbone == 'mcmae':
            self.backbone = ConvViT(config)
        else:
            self.backbone = resnet45()

        if config.model_vision_attention == 'position':
            # mode = if_none(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                in_channels=768, # For MCMAE
                max_length=config.dataset_max_length + 1,  # additional stop token
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                in_channels=768, # For MCMAE
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )
        else:
            raise NotImplementedError(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint,
                      submodule = config.model_vision_checkpoint_submodule,
                      exclude = config.model_vision_exclude
                      )

    def forward(self, images, *args, **kwargs):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, L, E), (N, L, H, W)
        logits = self.cls(attn_vecs)  # (N, L, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision',
                'backbone_feature': features}
