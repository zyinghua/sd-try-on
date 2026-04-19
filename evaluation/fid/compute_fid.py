# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import sys
from pathlib import Path
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import Dataset
from PIL import Image
import zipfile
import random

import dnnlib
from torch_utils import distributed as dist

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Simple PyTorch dataset for loading images from a folder or zip file."""
    
    def __init__(self, path, max_size=None, random_seed=0):
        self.path = path
        self.max_size = max_size
        
        # Collect image files
        self.image_files = []
        
        if path.endswith('.zip'):
            # Load from zip file
            with zipfile.ZipFile(path, 'r') as zip_ref:
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                for name in zip_ref.namelist():
                    if any(name.lower().endswith(ext) for ext in image_extensions):
                        self.image_files.append(name)
        else:
            # Load from directory (Recursive walk to find subfolders like Part1, Part2)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        self.image_files.append(os.path.join(root, file))
        
        # Sort for deterministic ordering
        self.image_files = sorted(self.image_files)
        
        # Use random seed to select subset deterministically
        if max_size is not None and len(self.image_files) > max_size:
            rng = random.Random(random_seed)
            self.image_files = rng.sample(self.image_files, max_size)
            self.image_files = sorted(self.image_files)  # Re-sort after sampling
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        
        # Load image
        if self.path.endswith('.zip'):
            with zipfile.ZipFile(self.path, 'r') as zip_ref:
                with zip_ref.open(file_path) as f:
                    img = Image.open(f).convert('RGB')
        else:
            img = Image.open(file_path).convert('RGB')
        
        # Convert to tensor: [C, H, W] format, values in [0, 1] range
        img_array = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        return img_tensor, 0  # Return dummy label

#----------------------------------------------------------------------------

def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=None, prefetch_factor=2, device=None,
):
    if device is None:
        device = get_device()
    
    # On Mac, use num_workers=0 to avoid multiprocessing issues
    if num_workers is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            num_workers = 0  # MPS doesn't support multiprocessing well
        elif not torch.cuda.is_available():
            num_workers = 0  # CPU mode often has issues with multiprocessing
        else:
            num_workers = 3  # CUDA can use multiprocessing
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    model_device = torch.device('cpu') if device.type == 'mps' else device
    if device.type == 'mps':
        dist.print0('Note: Using CPU for Inception model (MPS has limited support for some operations)')
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(model_device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    loader_kwargs = {'batch_sampler': rank_batches, 'num_workers': num_workers}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    data_loader = torch.utils.data.DataLoader(dataset_obj, **loader_kwargs)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')

    use_float64 = device.type != 'mps'
    accum_dtype = torch.float64 if use_float64 else torch.float32
    compute_dtype = torch.float32 if device.type == 'mps' else torch.float64
    
    accum_device = torch.device('cpu') if device.type == 'mps' else device
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=accum_device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=accum_device)
    
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        
        # Run inference
        features = detector_net(images.to(model_device), **detector_kwargs).to(compute_dtype)
        features_accum = features.to(torch.float64).to(accum_device)
        mu += features_accum.sum(0)
        sigma += features_accum.T @ features_accum

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate Frechet Inception Distance (FID)."""

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    device = get_device()
    if dist.get_rank() == 0:
        print(f'Using device: {device}')

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        # --- FIX: Handle local file paths ---
        if dnnlib.util.is_url(ref_path):
            with dnnlib.util.open_url(ref_path) as f:
                ref = dict(np.load(f))
        else:
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference file not found: {ref_path}")
            with open(ref_path, 'rb') as f:
                ref = dict(np.load(f))
        # ------------------------------------

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, device=device)
    dist.print0('Calculating FID...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'FID: {fid:.4f}')
    torch.distributed.barrier()

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    device = get_device()
    if dist.get_rank() == 0:
        print(f'Using device: {device}')

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch, device=device)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()