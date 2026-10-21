import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import model


# AUGMENTATION STRENGTHS -- MIRROR THE DEFAULTS IN dataloader.py, KEEP IN SYNC
GAIN_FACTOR   = 0.05   # GainScaling(factor=)
OFFSET_FACTOR = 0.05   # BandOffset(factor=)
NOISE_SIGMA   = 0.03   # GaussianNoise(sigma=)
BLUR_KSIZE    = 9      # GaussianBlur(kernel_size=)
BLUR_SIGMA    = (0.1, 1.0)


def effective_sigma(k, sigma):
    '''
    Sigma actually delivered once the kernel is truncated to k taps and renormalized.
    torchvision does not warn when k is too small for the requested sigma, it just
    hands back a weaker blur, so compare this against what you asked for.
    '''
    r = k // 2
    i = torch.arange(-r, r + 1, dtype=torch.float64)
    w = torch.exp(-0.5 * (i / sigma) ** 2)
    w = w / w.sum()
    return float((w * i ** 2).sum().sqrt())


def load_chips(cache_dir, split, n, crop):
    '''
    n chips spread evenly over the split, memory mapped so this stays cheap on the
    24 GB train cache. Returns raw 0-255 floats [n,10,256,256] and the centre crop
    offset used as the unaugmented reference.
    '''
    path = f"{cache_dir}/{split}_rgb.npy"
    arr  = np.load(path, mmap_mode='r')
    assert crop <= arr.shape[-1], f"crop {crop} larger than chip {arr.shape[-1]}"

    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
    raw = torch.from_numpy(np.array(arr[idx])).float()
    return raw, (arr.shape[-1] - crop) // 2


def build_encoder(hp, device, checkpoint=None):
    '''
    Encoder in eval mode. Pass a .pth.tar to measure a trained model, otherwise the
    numbers describe how the architecture propagates each perturbation, not how a
    trained representation responds to it.
    '''
    net = model.SentinelBarlowTwins(
        model_id=hp['id'],
        bands=hp['bands'],
        init_chan=hp['init_chan'],
        cnn_layers=hp['cnn_layers'],
        proj_dim=hp['proj_dim']
    )

    if checkpoint is not None:
        chkpt = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict(chkpt['model_state_dict'])
        print(f"Loaded {checkpoint} (epoch {chkpt['epoch']}, v_err {chkpt['v_err']:.5f})")

    return net.to(device).eval()


@torch.no_grad()
def embed(net, x, device, chunk=32):
    '''
    Pooled representation [B,32*init_chan], the vector the probe eventually sees.
    Float32 on purpose: bf16 autocast adds error of the same size as the weaker
    augmentations we are trying to measure.
    '''
    out = []
    for i in range(0, x.shape[0], chunk):
        out.append(net(x[i:i + chunk].to(device)).float())
    return torch.cat(out)


def perturbation(r0, r1):
    '''
    How far an augmentation moves the pooled representation.
    '''
    cos = torch.nn.functional.cosine_similarity(r0, r1, dim=1).mean().item()
    rl2 = ((r1 - r0).norm(dim=1) / r0.norm(dim=1)).mean().item()
    return cos, 100 * rl2


def build_augmentations(raw, off, crop, seed=0):
    '''
    (group, name, normalized tensor) for each augmentation, against a centre-crop
    reference. Gain is applied to raw 0-255 and normalized after, the way the
    transforms in dataloader.py do it; offset and noise act on normalized values.
    '''
    norm = lambda t: (t - dataloader.MEAN) / dataloader.STDD
    ref  = raw[:, :, off:off + crop, off:off + crop]
    x    = norm(ref)
    B, C = raw.shape[:2]

    g    = torch.Generator().manual_seed(seed)
    sym  = lambda f, *s: 1 + f * (torch.rand(*s, generator=g) * 2 - 1)   # gain in [1-f,1+f]
    blur = lambda k, s: v2.GaussianBlur(k, (s, s))(tv_tensors.Image(x)).as_subclass(torch.Tensor)

    augs = [
        # GEOMETRIC -- ImageViewTransform.geometric
        ('geometric', 'rot90',                        torch.rot90(x, 1, (2, 3)).contiguous()),
        ('geometric', 'horizontal flip',              x.flip(-1)),
        ('geometric', 'vertical flip',                x.flip(-2)),
        ('geometric', f'corner {crop} crop',          norm(raw[:, :, :crop, :crop])),

        # BLUR -- RandomApply([GaussianBlur]) IN .spectral
        ('blur', f'sigma {BLUR_SIGMA[0]}, k={BLUR_KSIZE}', blur(BLUR_KSIZE, BLUR_SIGMA[0])),
        ('blur', f'sigma {BLUR_SIGMA[1]}, k={BLUR_KSIZE}', blur(BLUR_KSIZE, BLUR_SIGMA[1])),

        # PHOTOMETRIC -- ONE SCALAR FOR ALL BANDS, AS WRITTEN
        ('photometric', f'GainScaling +/-{GAIN_FACTOR:.0%}, one scalar',
            norm(ref * sym(GAIN_FACTOR, B, 1, 1, 1))),
        ('photometric', f'BandOffset +/-{OFFSET_FACTOR}, one scalar',
            x + OFFSET_FACTOR * (torch.randint(0, 2, (B, 1, 1, 1), generator=g) * 2 - 1)),
        ('photometric', f'GaussianNoise sigma={NOISE_SIGMA}',
            x + torch.randn(x.shape, generator=g) * NOISE_SIGMA),

        # PHOTOMETRIC -- INDEPENDENT PER BAND, (B,C,1,1) INSTEAD OF (B,1,1,1)
        ('per-band', f'gain +/-{GAIN_FACTOR:.0%}, per band',  norm(ref * sym(GAIN_FACTOR, B, C, 1, 1))),
        ('per-band',  'gain +/-10%, per band',                norm(ref * sym(0.10, B, C, 1, 1))),
        ('per-band',  'gain +/-20%, per band',                norm(ref * sym(0.20, B, C, 1, 1))),
    ]
    return x, augs


def check_augmentations(net, raw, off, crop, device, seed=0):
    '''
    Rank the augmentations by how much they move the pooled representation. An
    augmentation that barely moves it cannot teach much invariance, however much
    CPU it costs to produce.
    '''
    x, augs = build_augmentations(raw, off, crop, seed)
    r0 = embed(net, x, device)

    rows = [(group, name) + perturbation(r0, embed(net, xa, device)) for group, name, xa in augs]
    rows.sort(key=lambda r: -r[3])

    print(f"\nEMBEDDING PERTURBATION ({raw.shape[0]} chips, {crop}x{crop} crop), ranked:\n")
    for group, name, cos, rl2 in rows:
        print(f"  {name:<40} cos {cos:.5f} | rel-L2 {rl2:6.2f}%   [{group}]")
    return rows


def check_blur(net, raw, off, crop, device, ksizes=(5, 7, 9, 13), sigmas=(0.5, 1.0, 1.5, 2.0)):
    '''
    Kernel size against sigma. Cost is ~k^2 on CPU (torchvision convolves a dense
    kxk kernel) but ~2k on GPU (GPUImageViewTransform convolves two 1-D kernels),
    so the kernel you can afford depends on where the augmentation runs.
    '''
    norm = lambda t: (t - dataloader.MEAN) / dataloader.STDD
    x  = norm(raw[:, :, off:off + crop, off:off + crop])
    r0 = embed(net, x, device)

    print(f"\nBLUR, EFFECTIVE SIGMA DELIVERED / REL-L2 MOVED:\n")
    print('  requested |' + ''.join(f'{"k=" + str(k):>18}' for k in ksizes))
    for s in sigmas:
        cells = []
        for k in ksizes:
            xb = v2.GaussianBlur(k, (s, s))(tv_tensors.Image(x)).as_subclass(torch.Tensor)
            _, rl2 = perturbation(r0, embed(net, xb, device))
            cells.append(f'{effective_sigma(k, s):8.2f} {rl2:6.2f}%')
        print(f'  {s:9.2f} |' + ''.join(f'{c:>18}' for c in cells))
    print('\n  k is not a free knob: it has to satisfy k >= 2*ceil(3*sigma)+1 or the')
    print('  requested sigma is silently clipped to the value in the left of each cell.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rank the training augmentations by how much they move the pooled representation.')
    parser.add_argument('--data-dir', type=str, required=True, help='Cache dir holding <split>_rgb.npy')
    parser.add_argument('--split', type=str, default='train', help='train/val/test')
    parser.add_argument('--n-chips', type=int, default=64, help='Chips sampled evenly over the split')
    parser.add_argument('--crop', type=int, default=224, help='Crop size, matches the transform under test')
    parser.add_argument('--device', type=int, default=0, help='GPU index')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional .pth.tar to measure a trained encoder')
    parser.add_argument('-p', '--params', type=str, default=None, help='JSON-lines hyperparameter file')
    parser.add_argument('--id', type=int, default=0, help='Model key id within --params')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), f"Data dir not found at {args.data_dir}"
    assert os.path.isfile(f"{args.data_dir}/{args.split}_rgb.npy"), \
        f"No {args.split}_rgb.npy in {args.data_dir}, build it with dataloader.py first"
    if args.checkpoint is not None:
        assert os.path.isfile(args.checkpoint), f"Checkpoint not found at {args.checkpoint}"
    assert args.device < torch.cuda.device_count(), "Out of range GPU id"

    return args


def load_hyperparameters(params, model_id):
    '''
    Architecture under test. Defaults to the smallest entry in the grid so the
    script runs without a hyperparameter file.
    '''
    if params is None:
        return {'id': model_id, 'bands': 10, 'init_chan': 32, 'cnn_layers': 3, 'proj_dim': 1024}

    with open(params) as fp:
        hps = {row['id']: row for row in (json.loads(l) for l in fp if l.strip())}
    assert model_id in hps, f"model id '{model_id}' not in {params}"
    return hps[model_id]


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)

    hp     = load_hyperparameters(args.params, args.id)
    device = torch.device(f'cuda:{args.device}')

    net = build_encoder(hp, device, args.checkpoint)
    raw, off = load_chips(args.data_dir, args.split, args.n_chips, args.crop)

    stride = 2 ** 5  # five stride-2 downsamples in ConvEncoder
    print(f"\nid {hp['id']} | init_chan {hp['init_chan']} | cnn_layers {hp['cnn_layers']} | "
          f"proj_dim {hp['proj_dim']} | representation_dim {hp['init_chan'] * 32}")
    print(f"encoder stride {stride} -> {args.crop}x{args.crop} becomes a "
          f"{args.crop // stride}x{args.crop // stride} grid, then mean(dim=(2,3)) collapses it, "
          f"so input detail below {stride} px is largely filtered before the probe sees it")

    check_augmentations(net, raw, off, args.crop, device, args.seed)
    check_blur(net, raw, off, args.crop, device)

    if args.checkpoint is None:
        print("\nNOTE: untrained encoder. These numbers show how the architecture propagates each")
        print("perturbation, not how a trained representation responds. Rerun with --checkpoint")
        print("against a saved model to measure that.")
