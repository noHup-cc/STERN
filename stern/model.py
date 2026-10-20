"""The STERN network: a Swin-L (ImageNet-22k) backbone fine-tuned as a Siamese network.

Training used a projector MLP (1536 -> 4096 -> 4096 -> 256) on the pooled backbone output; the slice
features used in the paper are the 1536-d pooled backbone output (the projector input), ``embed``.

Released weights (``stern_weights.safetensors``) hold ``backbone.*`` in the key layout of timm 0.6
(the training environment) without the constant buffers ``relative_position_index`` / ``attn_mask``,
and ``projector.*``. ``load_backbone_state`` converts them for newer timm (>= 0.9), whose Swin moves each
patch-merging layer to the start of the next stage.
"""
import re

import torch
import torch.nn as nn

FEATURE_DIM = 1536
PROJECTION_DIM = 256
MLP_DIM = 4096
_BUFFERS = ('relative_position_index', 'attn_mask')


def _timm_major_minor():
    import timm
    return tuple(int(x) for x in re.findall(r'\d+', timm.__version__)[:2])


def build_backbone():
    """Swin-L, patch 4, window 7, 224 x 224, without classifier (pooled 1536-d output)."""
    import timm
    name = ('swin_large_patch4_window7_224_in22k' if _timm_major_minor() < (0, 9)
            else 'swin_large_patch4_window7_224')
    return timm.create_model(name, pretrained=False, num_classes=0)


def build_projector():
    """The projector of the Siamese training (MoCo v3 style MLP with batch norm)."""
    layers, dims = [], [FEATURE_DIM, MLP_DIM, MLP_DIM, PROJECTION_DIM]
    for i in range(3):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i < 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
        else:
            layers.append(nn.BatchNorm1d(dims[i + 1], affine=False))
    return nn.Sequential(*layers)


def convert_backbone_state(state, model_keys):
    """Map backbone weights in the timm 0.6 layout to the layout of ``model_keys``."""
    state = {k: v for k, v in state.items() if not any(b in k for b in _BUFFERS)}
    if any(k.startswith('layers.0.downsample.') for k in model_keys):
        return state                                            # timm < 0.9: same layout
    out = {}
    for k, v in state.items():
        out[re.sub(r'^layers\.(\d+)\.downsample', lambda m: 'layers.%d.downsample' % (int(m.group(1)) + 1), k)] = v
    return out


def released_backbone_state(backbone):
    """The backbone's weights in the released (timm 0.6) layout, from any timm version."""
    state = {k: v for k, v in backbone.state_dict().items() if not any(b in k for b in _BUFFERS)}
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    if any(k.startswith('layers.0.downsample.') for k in state):
        return state
    out = {}
    for k, v in state.items():
        out[re.sub(r'^layers\.(\d+)\.downsample', lambda m: 'layers.%d.downsample' % (int(m.group(1)) - 1), k)] = v
    return out


def released_state(model):
    """``backbone.*`` / ``projector.*`` tensors of a STERN model in the released layout."""
    out = {'backbone.' + k: v for k, v in released_backbone_state(model.backbone).items()}
    out.update(('projector.' + k, v) for k, v in model.projector.state_dict().items())
    return out


def load_backbone_state(backbone, state):
    """Load released backbone weights; every parameter must be matched (constant buffers excepted)."""
    converted = convert_backbone_state(state, backbone.state_dict().keys())
    result = backbone.load_state_dict(converted, strict=False)
    missing = [k for k in result.missing_keys if not any(b in k for b in _BUFFERS)]
    if missing or result.unexpected_keys:
        raise RuntimeError('weights do not match the backbone: missing %s, unexpected %s'
                           % (missing[:5], list(result.unexpected_keys)[:5]))


class STERN(nn.Module):
    """``embed(x)``: (B, 3, 224, 224) normalised images -> (B, 1536) slice features (the paper's features);
    ``project(x)``: -> (B, 256) projector output used by the training loss."""

    def __init__(self, backbone=None, projector=None):
        super(STERN, self).__init__()
        self.backbone = backbone if backbone is not None else build_backbone()
        self.projector = projector if projector is not None else build_projector()

    def embed(self, x):
        return self.backbone(x)

    def project(self, x):
        return self.projector(self.backbone(x))

    def forward(self, x):
        return self.embed(x)


def split_state(state):
    backbone = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
    projector = {k[len('projector.'):]: v for k, v in state.items() if k.startswith('projector.')}
    if not backbone or len(backbone) + len(projector) != len(state):
        raise ValueError('expected only backbone.* and projector.* weights')
    return backbone, projector


def read_safetensors_numpy(path):
    """safetensors -> torch tensors through NumPy (torch < 1.10 has no torch.frombuffer)."""
    from safetensors.numpy import load_file
    return {k: torch.from_numpy(v.copy()) for k, v in load_file(str(path)).items()}


def read_weights(path):
    """Tensors of a released weights file (.safetensors, or a plain tensor dict saved with torch.save)."""
    path = str(path)
    if path.endswith('.safetensors'):
        if hasattr(torch, 'frombuffer'):
            from safetensors.torch import load_file
            return load_file(path)
        return read_safetensors_numpy(path)
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:                                            # torch < 1.13
        return torch.load(path, map_location='cpu')


def load_stern(weights, device='cpu'):
    """The STERN network with released weights, in evaluation mode, on ``device``."""
    backbone_state, projector_state = split_state(read_weights(weights))
    model = STERN()
    load_backbone_state(model.backbone, backbone_state)
    if projector_state:
        model.projector.load_state_dict(projector_state, strict=True)
    return model.eval().to(device)
