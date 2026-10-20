#!/usr/bin/env python3
"""Training checkpoint -> released weights (``stern_weights.safetensors``) and a JSON record.

    python scripts/export_weights.py checkpoint.pth stern_weights.safetensors [--sha256 EXPECTED]

The checkpoint of the Siamese training holds ``state['model']`` with keys ``swin_22k.*`` (the timm 0.6
Swin-L whose ``head`` was replaced by the projector MLP) and, possibly, optimiser / scheduler state.
Only the network is exported: ``swin_22k.head.*`` -> ``projector.*``, the rest -> ``backbone.*``, without the
constant buffers relative_position_index / attn_mask (rebuilt by the model). The record lists the
checkpoint's top-level keys and the epoch it stores, the parameter count and both SHA-256 values.
"""
import argparse
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

BUFFERS = ('relative_position_index', 'attn_mask')


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


class _Placeholder(dict):
    """Stands for a class the checkpoint references but this environment lacks (e.g. yacs.config.CfgNode
    of the training configuration); only the network tensors are exported."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


def checkpoint_pickle_module(missing):
    """A pickle module for torch.load that records and replaces classes that cannot be imported."""
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super(Unpickler, self).find_class(module, name)
            except (ImportError, AttributeError):
                missing.add('%s.%s' % (module, name))
                return type(name, (_Placeholder,), {'__module__': module})
    return types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, loads=pickle.loads, __name__='pickle')


def released_state(model_state):
    out = {}
    for key, value in model_state.items():
        if not key.startswith('swin_22k.'):
            raise ValueError('unexpected key %s (expected swin_22k.*)' % key)
        key = key[len('swin_22k.'):]
        if any(b in key for b in BUFFERS):
            continue
        name = 'projector.' + key[len('head.'):] if key.startswith('head.') else 'backbone.' + key
        out[name] = value.detach().cpu().contiguous()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('out')
    p.add_argument('--sha256', help='expected SHA-256 of the checkpoint')
    args = p.parse_args(argv)
    import torch
    from safetensors.torch import save_file
    source_sha = sha256(args.checkpoint)
    if args.sha256 and source_sha != args.sha256:
        raise SystemExit('checkpoint SHA-256 %s differs from the expected %s' % (source_sha, args.sha256))
    missing = set()
    state = torch.load(args.checkpoint, map_location='cpu', pickle_module=checkpoint_pickle_module(missing))
    tensors = released_state(state['model'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={'format': 'stern-1', 'source_sha256': source_sha})
    record = {
        'checkpoint': Path(args.checkpoint).name, 'checkpoint_sha256': source_sha,
        'checkpoint_keys': sorted(state) if isinstance(state, dict) else None,
        'epoch': state.get('epoch') if isinstance(state, dict) else None,
        'classes_not_imported': sorted(missing),
        'tensors': len(tensors), 'parameters': int(sum(v.numel() for v in tensors.values())),
        'weights': out.name, 'weights_sha256': sha256(out), 'weights_bytes': out.stat().st_size,
    }
    for key in ('epoch',):
        value = record[key]
        if hasattr(value, 'item'):
            record[key] = value.item()
    out.with_suffix('.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps(record, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
