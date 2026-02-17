import math
from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Default ImageNet normalization (matches training)
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocess_tile(tile: np.ndarray, mean=DEFAULT_MEAN, std=DEFAULT_STD, device='cuda'):
    # tile: HxWx3 uint8 or float [0..255]
    t = tile.astype(np.float32) / 255.0
    t = (t - mean) / std
    t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
    return t

def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str = 'cuda',
    tile_size: int = 512,
    stride: Optional[int] = None,
    batch_size: int = 4,
    num_classes: Optional[int] = None,
    mean: np.ndarray = DEFAULT_MEAN,
    std: np.ndarray = DEFAULT_STD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run sliding-window inference on a large image and return (pred_mask, prob_map).
    - image: HxWx3 (uint8)
    - pred_mask: HxW uint8 class ids
    - prob_map: C x H x W float32 (per-class probabilities)
    Notes:
      - Uses reflect padding.
      - Processes tiles in batches for GPU efficiency.
    """
    assert image.ndim == 3 and image.shape[2] == 3, "image must be HxWx3 numpy array"
    device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
    tile_size = int(tile_size)
    stride = tile_size if stride is None else int(stride)
    H, W = image.shape[:2]

    # compute number of tiles and padded size so tiles cover image with given stride
    n_y = max(1, math.ceil((H - tile_size) / stride) + 1) if H > tile_size else 1
    n_x = max(1, math.ceil((W - tile_size) / stride) + 1) if W > tile_size else 1
    H_pad = stride * (n_y - 1) + tile_size
    W_pad = stride * (n_x - 1) + tile_size
    pad_h = H_pad - H
    pad_w = W_pad - W

    # pad image (bottom/right)
    if pad_h > 0 or pad_w > 0:
        image_p = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
    else:
        image_p = image

    # prepare accumulators
    # infer num_classes on first forward if not provided
    prob_map = None
    count_map = np.zeros((H_pad, W_pad), dtype=np.float32)

    coords = []
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            coords.append((y, x))

    model = model.to(device)
    model.eval()
    tiles_batch = []
    coords_batch = []

    with torch.no_grad():
        for (y, x) in coords:
            tile = image_p[y:y + tile_size, x:x + tile_size]
            tiles_batch.append(_preprocess_tile(tile, mean=mean, std=std, device=device))
            coords_batch.append((y, x))

            if len(tiles_batch) == batch_size:
                inp = torch.cat(tiles_batch, dim=0)  # [B,3,H,W]
                out = model(inp)  # [B,C,H,W] logits
                probs = F.softmax(out, dim=1).cpu().numpy()  # [B,C,H,W]

                if prob_map is None:
                    C = probs.shape[1] if num_classes is None else num_classes
                    prob_map = np.zeros((C, H_pad, W_pad), dtype=np.float32)

                for i, (yy, xx) in enumerate(coords_batch):
                    prob_map[:, yy:yy + tile_size, xx:xx + tile_size] += probs[i]
                    count_map[yy:yy + tile_size, xx:xx + tile_size] += 1.0

                tiles_batch = []
                coords_batch = []

        # process remaining
        if tiles_batch:
            inp = torch.cat(tiles_batch, dim=0)
            out = model(inp)
            probs = F.softmax(out, dim=1).cpu().numpy()
            if prob_map is None:
                C = probs.shape[1] if num_classes is None else num_classes
                prob_map = np.zeros((C, H_pad, W_pad), dtype=np.float32)
            for i, (yy, xx) in enumerate(coords_batch):
                prob_map[:, yy:yy + tile_size, xx:xx + tile_size] += probs[i]
                count_map[yy:yy + tile_size, xx:xx + tile_size] += 1.0

    # normalize
    count_map = np.maximum(count_map, 1e-6)[None, :, :]
    prob_map = prob_map / count_map

    # crop to original size and compute argmax
    prob_map = prob_map[:, :H, :W]
    pred = np.argmax(prob_map, axis=0).astype(np.uint8)

    return pred, prob_map