"""From one 2D slice to the model input, as in the paper.

The paper's pipeline wrote every slice to a JPEG file and the model read those files:

    mouse embryos (KLB), mouse hearts (TIFF)   cv2.imwrite(<t>z<k>.jpg, slice)         (OpenCV, quality 95)
    zebrafish (OME-TIFF)                       1-99 % percentile scaling to 8 bit, then
                                               imageio / Pillow JPEG, quality 95
    model input                                PIL open -> RGB -> resize 224 x 224 (bilinear) -> tensor
                                               -> ImageNet mean / std

``Recipe`` reproduces this in memory (the JPEG is encoded and decoded without writing files). With
``jpeg=False`` the 8-bit slice goes to the model directly; the features then differ slightly from the
paper's (the network was trained on JPEG slices, so the JPEG round trip is the default).

Bit-identical features need the paper's library versions: the JPEG bytes depend on the encoder
(opencv-contrib-python 4.6.0.66 for mouse / heart, Pillow 12.0 via imageio for zebrafish) and decoding /
resizing on Pillow (5.4.1 in the paper environment). Other versions give slightly different features.
"""
import io

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
JPEG_QUALITY = 95
PAPER_OPENCV = '4.6.0'


def to_uint8_saturate(image):
    """OpenCV's conversion before JPEG encoding of a non-8-bit slice: round, then clip to 0..255."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(image)
    return np.clip(image, 0, 255).astype(np.uint8)


def to_uint8_percentile(image, low=1.0, high=99.0):
    """The zebrafish scaling of the paper's conversion code: percentiles of the slice ->
    0..1 -> x 255, truncated to uint8."""
    x = np.asarray(image).astype(np.float32)
    lo, hi = np.percentile(x, (low, high))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def encode_jpeg_opencv(image8, quality=JPEG_QUALITY):
    """JPEG bytes as cv2.imwrite writes them (default quality 95)."""
    import cv2
    params = [] if quality == JPEG_QUALITY else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode('.jpg', image8, params)
    if not ok:
        raise RuntimeError('OpenCV JPEG encoding failed')
    return buf.tobytes()


def encode_jpeg_pillow(image8, quality=JPEG_QUALITY):
    """JPEG bytes as imageio.v3.imwrite(path, image, quality=95) writes them (Pillow plug-in)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image8).save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


class Recipe(object):
    """How a raw slice becomes the 8-bit image the network sees.

    scale: 'saturate' (mouse, heart) or 'percentile' (zebrafish); encoder: 'opencv' or 'pillow';
    jpeg: False skips the JPEG round trip; tf32: TF32 setting of the paper's extraction on GPU."""

    def __init__(self, name, scale, encoder, tf32, jpeg=True, quality=JPEG_QUALITY):
        if scale not in ('saturate', 'percentile') or encoder not in ('opencv', 'pillow'):
            raise ValueError('unknown scale / encoder: %s / %s' % (scale, encoder))
        self.name, self.scale, self.encoder, self.tf32 = name, scale, encoder, tf32
        self.jpeg, self.quality = jpeg, quality

    def __repr__(self):
        return 'Recipe(%r, scale=%r, encoder=%r, jpeg=%r, tf32=%r)' % (self.name, self.scale, self.encoder,
                                                                      self.jpeg, self.tf32)

    def to_uint8(self, image):
        return to_uint8_percentile(image) if self.scale == 'percentile' else to_uint8_saturate(image)

    def jpeg_bytes(self, image):
        """The JPEG file the paper's pipeline would have written for this slice."""
        image8 = self.to_uint8(image)
        if self.encoder == 'opencv':
            return encode_jpeg_opencv(image8, self.quality)
        return encode_jpeg_pillow(image8, self.quality)

    def pil_image(self, image):
        """The slice as the network's data loader opened it (PIL, RGB)."""
        from PIL import Image
        if self.jpeg:
            return Image.open(io.BytesIO(self.jpeg_bytes(image))).convert('RGB')
        return Image.fromarray(self.to_uint8(image)).convert('RGB')

    def without_jpeg(self):
        return Recipe(self.name, self.scale, self.encoder, self.tf32, jpeg=False, quality=self.quality)


RECIPES = {
    'mouse': Recipe('mouse', 'saturate', 'opencv', tf32=False),
    'heart': Recipe('heart', 'saturate', 'opencv', tf32=False),
    'zebrafish': Recipe('zebrafish', 'percentile', 'pillow', tf32=True),
}


def get_recipe(name, jpeg=True):
    try:
        recipe = RECIPES[name]
    except KeyError:
        raise ValueError('unknown recipe %r (expected one of %s)' % (name, ', '.join(RECIPES)))
    return recipe if jpeg else recipe.without_jpeg()


def model_input(pil_image):
    """PIL image -> normalised (3, 224, 224) float tensor (the paper's ``image_to_array``)."""
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transform(pil_image)


def check_paper_libraries(recipe):
    """Warnings for library versions that differ from the paper's (features will differ slightly)."""
    import PIL
    notes = []
    if recipe.jpeg and recipe.encoder == 'opencv':
        try:
            import cv2
            if not cv2.__version__.startswith(PAPER_OPENCV):
                notes.append('OpenCV %s (paper: %s): JPEG bytes may differ' % (cv2.__version__, PAPER_OPENCV))
        except ImportError:
            notes.append('OpenCV not installed (needed for the mouse / heart JPEG round trip)')
    if not PIL.__version__.startswith('5.4.'):
        notes.append('Pillow %s (paper: 5.4.1): JPEG decoding / resizing may differ slightly' % PIL.__version__)
    return notes
