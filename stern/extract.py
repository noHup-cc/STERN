"""Slice features of volumes with the STERN network."""
import contextlib
import warnings

import numpy as np
import torch

from .model import load_stern
from .preprocess import check_paper_libraries, get_recipe, model_input

__all__ = ['Extractor']


@contextlib.contextmanager
def _tf32(enabled):
    """TF32 for matmul / cuDNN as in the paper's extraction (only meaningful on Ampere+ GPUs)."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class Extractor(object):
    """Features of every slice of a volume.

    weights: released weights file; recipe: 'mouse', 'heart' or 'zebrafish' (how slices become 8-bit
    images, see ``stern.preprocess``); jpeg: JPEG round trip as in the paper (default True);
    batch_size: slices per forward pass (the paper used 1; larger batches are faster and change features
    only at the level of floating-point rounding); tf32: None = the paper's setting for the recipe."""

    def __init__(self, weights, recipe='mouse', jpeg=True, device=None, batch_size=1, tf32=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.recipe = get_recipe(recipe, jpeg=jpeg)
        self.batch_size = int(batch_size)
        self.tf32 = self.recipe.tf32 if tf32 is None else tf32
        self.model = load_stern(weights, self.device)
        for note in check_paper_libraries(self.recipe):
            warnings.warn(note + '; features may differ slightly from the published ones', stacklevel=2)

    def _forward(self, tensors):
        out = []
        with torch.no_grad(), _tf32(self.tf32):
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self.device)
                out.append(self.model.embed(batch).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def embed_slices(self, slices):
        """Iterable of 2D arrays -> (n, 1536) float32."""
        return self._forward([model_input(self.recipe.pil_image(s)) for s in slices])

    def embed_volume(self, volume):
        """(Z, Y, X) array -> (Z, 1536); row k is slice z = k + 1."""
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError('expected a (Z, Y, X) volume, got shape %s' % (volume.shape,))
        return self.embed_slices(volume)

    def embed_files(self, paths):
        """Already-written slice images (e.g. the paper's JPEG files) -> (n, 1536), read as the paper's
        data loader read them (no further conversion)."""
        from PIL import Image
        tensors = []
        for p in paths:
            with Image.open(str(p)) as image:
                tensors.append(model_input(image.convert('RGB')))
        return self._forward(tensors)
