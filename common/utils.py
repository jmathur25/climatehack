import pyproj
from scipy import fftpack
import numpy as np
import torch
import common.loss_utils as loss_utils
import cv2


def check_times(tstart, tend):
    """
    Given two times in as numpy datetimes, find out if they are 3 hours apart.
    """
    return int((tend - tstart) / np.timedelta64(1, "m")) == 175


def warp_flow(img, flow):
    """
    Given an image and flow vectors, apply the flow to create a new image.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return res


def get_msssim(x, y, average=True, inner_64=True):
    """
    A generic function to get the MS-SSIM for images. It will work with numpy arrays or
    PyTorch tensors. MAKE SURE THE DATA IS NORMALIZED TO BE BETWEEN 0 AND 1024.
    It can handle various shaped inputs.
    (h x w): just get the MS-SSIM of two images
    (t x h x w): get the MS-SSIM for a sequence of images
    (b x t x h x w): get the MS-SSIM for batches of sequences of images

    In the case where a batch/temporal dimension is provded, setting `average=True` will
    simply average the results. Setting `average=False` means you want the MS-SSIM per
    batch/timestep. In the case of (b x t x h x w), you'll get back something of shape
    (b x t) which will be the MS-SSIM per (b,t).

    `inner_64` means to get the MS-SSIM for the inner 64x64.

    Note: this is different that using a loss function for backprop. Use `loss_utils`
    for that. This function is mainly for analysis.
    """
    was_numpy = False
    if isinstance(x, np.ndarray):
        was_numpy = True
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
    expected_shape = None
    # now work with various shapes to make them of size 5
    if len(x.shape) == 2:
        # (h x w)
        # these become (1 x 1 x h x w)
        x = x.unsqueeze(dim=0).unsqueeze(dim=0)
        y = y.unsqueeze(dim=0).unsqueeze(dim=0)
        expected_shape = 1
    elif len(x.shape) == 3:
        # (t x h x w)
        # these become (t x 1 x h x w)
        t, h, w = x.shape
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
        expected_shape = t
    elif len(x.shape) == 4:
        # (b x t x h x w)
        # these become (b*t x 1 x h x w)
        b, t, h, w = x.shape
        x = x.reshape(b * t, h, w)
        y = y.reshape(b * t, h, w)
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
        expected_shape = (b, t)
    else:
        raise ValueError("Can only accept 2, 3, and 4-dimensional shapes")

    if inner_64:
        x = x[:, :, 32:96, 32:96]
        y = y[:, :, 32:96, 32:96]
    criterion = loss_utils.MS_SSIM(
        data_range=1023.0, size_average=average, win_size=3, channel=1
    )
    scores = criterion(y, x)
    if average is False:
        scores = scores.reshape(expected_shape)
    if was_numpy:
        scores = scores.numpy()
    return scores


class Transformers:
    """
    Class to store transformation from one Grid to another.
    Its good to make this only once, but need the
    option of updating them, due to out of data grids.
    """

    # OSGB is also called "OSGB 1936 / British National Grid -- United
    # Kingdom Ordnance Survey".  OSGB is used in many UK electricity
    # system maps, and is used by the UK Met Office UKV model.  OSGB is a
    # Transverse Mercator projection, using 'easting' and 'northing'
    # coordinates which are in meters.  See https://epsg.io/27700
    OSGB = 27700

    # WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
    # latitude and longitude.
    WGS84 = 4326
    WGS84_CRS = f"EPSG:{WGS84}"

    def __init__(self):
        """Init"""
        self._osgb_to_lat_lon = None
        self._lat_lon_to_osgb = None
        self.make_transformers()

    def make_transformers(self):
        """
        Make transformers
         Nice to only make these once, as it makes calling the functions below quicker
        """
        self._osgb_to_lat_lon = pyproj.Transformer.from_crs(
            crs_from=Transformers.OSGB, crs_to=Transformers.WGS84
        )
        self._lat_lon_to_osgb = pyproj.Transformer.from_crs(
            crs_from=Transformers.WGS84, crs_to=Transformers.OSGB
        )

    @property
    def osgb_to_lat_lon(self):
        """OSGB to lat-lon property"""
        return self._osgb_to_lat_lon

    @property
    def lat_lon_to_osgb(self):
        """lat-lon to OSGB property"""
        return self._lat_lon_to_osgb


transformers = Transformers()


def osgb_to_lat_lon(x, y):
    """
    Change OSGB coordinates to lat, lon
    Args:
        x: osgb east-west
        y: osgb north-south
    Return: 2-tuple of latitude (north-south), longitude (east-west).
    """
    return transformers.osgb_to_lat_lon.transform(x, y)


def get_idct_filter():
    """
    Returns a callable `idct_filter` which undoes the DCT-as-feature maps operation.

    dct_maps: (batch_size, 64, 16, 16)

    64 because we did an 8x8 DCT, so 64 coefficients
    16x16 because the original image is of size 128, and we split
    it into 8x8 blocks, so we have 16x16 of these blocks

    """
    e = fftpack.dct(np.eye(8), norm="ortho")
    # for any 2D matrix `i`, you can get the 2D DCT with
    # p = fftpack.dctn(i, norm='ortho')
    # this is equal to:
    # p == e.T @ i @ e

    # we use these to undo the DCT
    ei = np.linalg.inv(e)
    eti = ei.T
    # you can recover the original image by doing:
    # ih = eti @ p @ ei
    # np.allclose(ih, i)

    eti_t = torch.FloatTensor(eti)
    ei_t = torch.FloatTensor(ei)

    def idct_filter(dct_maps):
        nonlocal eti_t, ei_t
        dct_maps = dct_maps.permute(0, 2, 3, 1)
        b, blocks_h, blocks_w, f = dct_maps.shape
        # turn flattened 64 -> 8x8
        dct_maps = dct_maps.reshape(b, blocks_h, blocks_w, 8, 8)

        eti_t = eti_t.to(dct_maps.device)
        ei_t = ei_t.to(dct_maps.device)

        # this undoes DCT on the 8x8 frequency coefficients
        pixel_maps = eti_t @ dct_maps @ ei_t

        # now, we reorganize on the width to make it contiguous
        # dct_maps will not have all the width pixels represented contiguously
        pixel_maps = pixel_maps.permute(0, 1, 3, 2, 4)
        pixel_maps = pixel_maps.reshape(b, blocks_h, 8, blocks_w * 8)
        # notice the height is also stored contiguously, we just need to reshape to get it
        pixel_img = pixel_maps.reshape(b, blocks_h * 8 * blocks_w * 8).reshape(
            -1, blocks_h * 8, blocks_w * 8
        )
        return pixel_img

    return idct_filter


def _create_dct_filter(k1, k2, n):
    """
    Basically straight from wikipedia: https://en.wikipedia.org/wiki/Discrete_cosine_transform.
    Normalization is applied to keep the transform orthogonal. This should be identical
    to what scipy.fftpack is doing, it just gets the weights for a specific (k1, k2) for
    convolution.
    """
    f = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            f[i, j] = np.cos(np.pi / n * (i + 1 / 2) * k1) * np.cos(
                np.pi / n * (j + 1 / 2) * k2
            )
            # apply normalization
            if k1 == 0 and k2 == 0:
                f[i, j] *= 1.0 / n
            elif k1 == 0 or k2 == 0:
                f[i, j] *= np.sqrt(2) / n
            else:
                f[i, j] *= 2 / n
    return f


def create_conv_dct_filter():
    """
    Create conv filters that work on a single channel image img. Make sure that:
    img.shape == (1, h, w) such that h and w are divisble by 8

    This convolution layer outputs 64 feature maps of size h//8, w//8. Accessing
    index (i,j,k) will get the ith 8x8 DCT coefficient (flattened) for the (j,w)
    block. You can think of a block as an 8x8 portion of an image that is non-overlapping
    with other blocks. So (i,0,0) gets the ith DCT coefficient in the top-left block.
    (i,2,4) gets the ith DCT coefficient on img[2*8:2*8+8,4*8+4*8+8].

    The point of this is to have 64 feature maps that represent the image in the frequency domain.
    Each feature map contains the same DCT coefficient, just on a different 8x8 block in the image.
    """
    n = 8
    conv = torch.nn.Conv2d(1, n * n, kernel_size=(n, n), stride=n)
    for i in range(n * n):
        idxi = i // n
        idxj = i - n * idxi
        f = _create_dct_filter(idxi, idxj, n)
        conv.weight.data[i, 0] = torch.FloatTensor(f)
    # no bias
    conv.bias.data = torch.zeros_like(conv.bias.data)
    # turn off gradient calculation
    conv.weight.requires_grad = False
    conv.bias.requires_grad = False
    return conv


if __name__ == "__main__":
    # run a test to ensure we are doing DCT correctly
    bs = 3
    x = np.random.randn(bs, 128, 128)
    conv_dct = create_conv_dct_filter()
    conv_idct = get_idct_filter()

    x = torch.FloatTensor(x)
    # add an input channel
    x = torch.unsqueeze(x, dim=1)

    dct_maps = conv_dct(x)
    blocks_h = x.shape[2] // 8
    blocks_w = x.shape[3] // 8
    for b in range(bs):
        for i in range(blocks_h):
            for j in range(blocks_w):
                # check every block for the correct DCT coefficients
                block = x[b, 0, i * 8 : i * 8 + 8, j * 8 : j * 8 + 8].numpy()
                conv_output = dct_maps[b, :, i, j].reshape(8, 8)
                correct_output = fftpack.dctn(block, norm="ortho")
                assert np.allclose(conv_output.numpy(), correct_output, atol=1e-6)

    xhat = conv_idct(dct_maps)

    xhat = np.squeeze(xhat.numpy())
    x = np.squeeze(x.numpy())

    assert np.allclose(x, xhat, atol=1e-6)

    print("Success!")
