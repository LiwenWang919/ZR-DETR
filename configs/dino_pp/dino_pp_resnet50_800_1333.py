from torch import nn
from torchvision.ops import FrozenBatchNorm2d

from models.backbones.resnet import ResNetBackbone
from models.bricks.dino_transformer import (
    DINOTransformer,
    DINOTransformerDecoder,
    DINOTransformerDecoderLayer,
    DINOTransformerEncoder,
    DINOTransformerEncoderLayer,
)
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.set_criterion import SetCriterion
from models.detectors.dino import DINO
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper

# mostly changed parameters
embed_dim = 256
num_classes = 91
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

# instantiate model components
position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone = ResNetBackbone(
    arch="resnet50", norm_layer=FrozenBatchNorm2d, return_indices=(1, 2, 3), freeze_indices=(0,)
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = DINOTransformer(
    encoder=DINOTransformerEncoder(
        encoder_layer=DINOTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    decoder=DINOTransformerDecoder(
        decoder_layer=DINOTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
)

matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25, focal_gamma=2.0)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({
    k + f"_{i}": v
    for i in range(transformer_dec_layers - 1)
    for k, v in weight_dict.items()
})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})

criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, alpha=0.25, gamma=2.0)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

# combine above components to instantiate the model
model = DINO(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    min_size=800,
    max_size=1333,
)
