"""Tests of the network and the preprocessing (need torch, timm; OpenCV / imageio where noted)."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timm')
from stern import preprocess as P  # noqa: E402
from stern.model import (STERN, build_backbone, load_stern, released_backbone_state, released_state,  # noqa: E402
                         split_state)

DATA = Path(__file__).resolve().parent / 'data'


# ----------------------------------------------------------------------------- preprocessing
def test_percentile_scaling_equals_the_original():
    def original(img16, qlow, qhigh):                            # the paper's conversion code, verbatim
        x = img16.astype(np.float32)
        lo, hi = np.percentile(x, (qlow, qhigh))
        if hi <= lo:
            return np.zeros_like(img16, dtype=np.uint8)
        x = (x - lo) / (hi - lo)
        x = np.clip(x, 0, 1)
        return (x * 255).astype(np.uint8)
    img = np.random.default_rng(0).integers(0, 4000, (50, 60)).astype(np.uint16)
    assert np.array_equal(P.to_uint8_percentile(img), original(img, 1.0, 99.0))
    assert not P.to_uint8_percentile(np.full((4, 4), 7, np.uint16)).any()


def test_opencv_recipe_writes_the_same_jpeg_as_cv2_imwrite():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(1)
    for img in (rng.integers(0, 256, (40, 50)).astype(np.uint8), rng.integers(0, 700, (40, 50)).astype(np.uint16)):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'a.jpg')
            assert cv2.imwrite(path, img)
            assert Path(path).read_bytes() == P.get_recipe('mouse').jpeg_bytes(img)


def test_pillow_recipe_writes_the_same_jpeg_as_imageio():
    iio = pytest.importorskip('imageio.v3')
    img = np.random.default_rng(2).integers(0, 5000, (40, 50)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'a.jpg')
        iio.imwrite(path, P.to_uint8_percentile(img), quality=95)
        assert Path(path).read_bytes() == P.get_recipe('zebrafish').jpeg_bytes(img)


def test_model_input_matches_reading_the_jpeg_file():
    pytest.importorskip('cv2')
    from PIL import Image
    img = np.random.default_rng(3).integers(0, 256, (300, 280)).astype(np.uint8)
    recipe = P.get_recipe('heart')
    a = P.model_input(recipe.pil_image(img))
    b = P.model_input(Image.open(io.BytesIO(recipe.jpeg_bytes(img))).convert('RGB'))
    assert a.shape == (3, 224, 224) and torch.equal(a, b)
    c = P.model_input(P.get_recipe('heart', jpeg=False).pil_image(img))
    assert not torch.equal(a, c)


# ----------------------------------------------------------------------------- network and weights
@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return STERN().eval()


def test_released_layout_matches_the_published_weight_shapes(model):
    shapes = json.loads((DATA / 'released_weight_shapes.json').read_text())
    ours = {k: list(v.shape) for k, v in released_state(model).items()}
    assert ours == shapes


def test_weights_round_trip_and_feature_shape(model, tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / 'w.safetensors'
    save_file({k: v.contiguous() for k, v in released_state(model).items()}, str(path))
    loaded = load_stern(path)
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        a, b = model.embed(x), loaded.embed(x)
        p = loaded.project(x)
    assert a.shape == (2, 1536) and p.shape == (2, 256) and torch.equal(a, b)
    backbone, projector = split_state(released_state(model))
    assert set(released_backbone_state(build_backbone())) == set(backbone) and len(projector) == 16


def test_safetensors_through_numpy_equals_torch_loader(model, tmp_path):
    """The NumPy path used with torch < 1.10 reads the same tensors."""
    from safetensors.torch import save_file
    from stern.model import read_safetensors_numpy
    state = {k: v.contiguous() for k, v in list(released_state(model).items())[:20]}
    save_file(state, str(tmp_path / 's.safetensors'))
    loaded = read_safetensors_numpy(tmp_path / 's.safetensors')
    assert set(loaded) == set(state) and all(torch.equal(loaded[k], state[k]) for k in state)


def test_export_of_a_checkpoint_with_classes_that_are_not_installed(tmp_path):
    """The training checkpoint pickles its configuration (a yacs CfgNode); export works without yacs."""
    import importlib
    import sys
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    export = importlib.import_module('export_weights')
    (tmp_path / 'fake_config_module.py').write_text('class Config(dict):\n    pass\n')
    sys.path.insert(0, str(tmp_path))
    config = importlib.import_module('fake_config_module').Config(EPOCHS=50)
    state = {'swin_22k.layers.0.blocks.0.attn.relative_position_index': torch.zeros(2, dtype=torch.long),
             'swin_22k.norm.weight': torch.ones(3), 'swin_22k.head.0.weight': torch.ones(2, 3)}
    torch.save({'model': state, 'config': config, 'epoch': 49}, str(tmp_path / 'checkpoint.pth'))
    sys.path.remove(str(tmp_path))
    del sys.modules['fake_config_module']
    out = tmp_path / 'w.safetensors'
    export.main([str(tmp_path / 'checkpoint.pth'), str(out)])
    record = json.loads(out.with_suffix('.json').read_text())
    assert record['classes_not_imported'] == ['fake_config_module.Config'] and record['epoch'] == 49
    assert record['checkpoint_keys'] == ['config', 'epoch', 'model'] and record['tensors'] == 2
    from stern.model import read_weights
    assert sorted(read_weights(out)) == ['backbone.norm.weight', 'projector.0.weight']


def test_wrong_weights_are_refused(model, tmp_path):
    from safetensors.torch import save_file
    state = released_state(model)
    state.pop(sorted(k for k in state if k.startswith('backbone.layers.2'))[0])
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp_path / 'bad.safetensors'))
    with pytest.raises(RuntimeError):
        load_stern(tmp_path / 'bad.safetensors')
