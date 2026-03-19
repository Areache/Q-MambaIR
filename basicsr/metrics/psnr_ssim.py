import cv2
import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    import warnings
    warnings.warn('LPIPS package is not installed. Please install it with: pip install lpips')


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, net='alex', use_gpu=True, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Ref: https://github.com/richzhang/PerceptualSimilarity

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the LPIPS calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        net (str): Network to use for LPIPS. Options: 'alex', 'vgg', 'squeeze'.
            Default: 'alex'.
        use_gpu (bool): Whether to use GPU. Default: True.

    Returns:
        float: LPIPS result. Lower means more similar.
    """
    if not LPIPS_AVAILABLE:
        raise ImportError('LPIPS package is not installed. Please install it with: pip install lpips')

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    # Reorder to HWC
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to torch tensor
    # LPIPS expects: Nx3xHxW, RGB, range [-1, 1]
    # Input is: HWC, BGR, range [0, 255]
    
    # Convert BGR to RGB
    if img.shape[2] == 3:
        img = img[:, :, ::-1]  # BGR to RGB
        img2 = img2[:, :, ::-1]  # BGR to RGB
    
    # Convert to CHW and normalize to [-1, 1]
    img = img.astype(np.float32).transpose(2, 0, 1)  # HWC -> CHW
    img2 = img2.astype(np.float32).transpose(2, 0, 1)  # HWC -> CHW
    
    img = img / 127.5 - 1.0  # [0, 255] -> [-1, 1]
    img2 = img2 / 127.5 - 1.0  # [0, 255] -> [-1, 1]
    
    # Add batch dimension: CHW -> 1xCxHxW
    img = torch.from_numpy(img).unsqueeze(0)
    img2 = torch.from_numpy(img2).unsqueeze(0)
    
    # Move to GPU if available
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    img = img.to(device)
    img2 = img2.to(device)
    
    # Initialize LPIPS model (singleton pattern to avoid reloading)
    # Use net as key to cache different network models
    cache_key = f'{net}_{device}'
    if not hasattr(calculate_lpips, 'loss_fn_cache'):
        calculate_lpips.loss_fn_cache = {}
    
    if cache_key not in calculate_lpips.loss_fn_cache:
        try:
            calculate_lpips.loss_fn_cache[cache_key] = lpips.LPIPS(net=net, verbose=False)
            if use_gpu and torch.cuda.is_available():
                calculate_lpips.loss_fn_cache[cache_key] = calculate_lpips.loss_fn_cache[cache_key].to(device)
            calculate_lpips.loss_fn_cache[cache_key].eval()
        except (OSError, IOError, RuntimeError) as e:
            error_msg = (
                f'\n{"="*60}\n'
                f'Failed to load LPIPS model (net={net}).\n'
                f'Error: {str(e)}\n\n'
                f'This is likely due to network issues downloading pretrained weights.\n'
                f'\nSolutions:\n'
                f'1. Download model manually and place in ~/.cache/torch/hub/checkpoints/\n'
                f'   For AlexNet: wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth\n'
                f'2. Set skip_on_error=True in LPIPS config to skip calculation\n'
                f'3. Ensure network connectivity\n'
                f'{"="*60}\n'
            )
            raise RuntimeError(error_msg) from e
    
    loss_fn = calculate_lpips.loss_fn_cache[cache_key]
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = loss_fn.forward(img, img2)
    
    return lpips_value.item()


# Initialize LPIPS model cache
calculate_lpips.loss_fn_cache = {}
