## ----------------- PYTHON VERSIONS OF THE MATLAB BUILT-IN FUNCTIONS USED BY THESE SCRIPTS ----------------------------------------------

# MATLAB has these functions built in, Python doesn't, so they are defined here once and shared by every script.
# Each one is written to behave the same as the MATLAB function of the same name.

# Pixel positions follow MATLAB's convention (1-based: the top left pixel is x = 1, y = 1),
# so measured vesicle centres match the MATLAB scripts. Subtract 1 to plot them onto an image in Python.

import numpy as np
from matplotlib.patches import Circle
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import local_maxima, reconstruction


def imread(filename):
    # Same as MATLAB imread for .png files: returns the image without its transparency (alpha) channel
    image = Image.open(filename)
    if image.mode == 'P':  # palette images are expanded to colour
        image = image.convert('RGBA')
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] in (2, 4):  # MATLAB returns alpha as a separate output, so drop it
        img = img[:, :, :-1]
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def im2gray(img):
    # Same as MATLAB im2gray: colour images are converted with the rgb2gray weights, greyscale images are unchanged
    if img.ndim == 2:
        return img
    grey = img[:, :, :3].astype(np.float64) @ np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    if np.issubdtype(img.dtype, np.integer):
        grey = np.floor(grey + 0.5).astype(img.dtype)  # MATLAB rounds back to the original data type
    return grey


def im2single(img):
    # Same as MATLAB im2single: integer images are scaled to 0-1, e.g. uint8 is divided by 255 and uint16 by 65535
    if np.issubdtype(img.dtype, np.integer):
        return (img.astype(np.float64) / np.iinfo(img.dtype).max).astype(np.float32)
    return img.astype(np.float32)


def matlab_round(x):
    # Same as MATLAB round: halves are rounded away from zero (numpy's round sends halves to the nearest even number)
    x = np.asarray(x)
    return np.trunc(x + np.copysign(0.5, x).astype(x.dtype))


def trimmean(x, percent):
    # Same as MATLAB trimmean: mean after removing percent/2 % of the values from each end
    # eg. trimmean(4 values, 50) removes the highest & lowest and averages the 2 middle values
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    k = n * percent / 200
    k0 = int(matlab_round(k - np.spacing(k)))
    if n == 0:
        return np.nan
    return np.mean(x[k0:n - k0])


def bwperim(mask):
    # Same as MATLAB bwperim: the outline pixels of each object (pixels with a 4-connected neighbour outside the object)
    return mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(2, 1), border_value=0)


def imfindcircles(A, radiusRange, sensitivity):
    # Same as MATLAB imfindcircles(A, radiusRange, 'Sensitivity', sensitivity) with its other default options:
    # bright circles, phase coding method and automatic edge threshold.
    # Returns centers (one [x, y] row per circle, in MATLAB 1-based pixel positions) and radii,
    # sorted from the strongest circle to the weakest.
    accumMatrix = _chaccum(A, radiusRange)
    if not np.any(accumMatrix):
        return np.zeros((0, 2)), np.zeros(0)

    accumThresh = 1 - sensitivity
    centers, metric = _chcenters(accumMatrix, accumThresh)

    # keep circles with a strong enough accumulator value
    keep = metric >= accumThresh
    centers = centers[keep]
    if len(centers) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    radii = _chradiiphcode(centers, accumMatrix, radiusRange)
    return centers, radii


def _chaccum(A, radiusRange):
    # Circular Hough transform: every edge pixel votes for the possible centres of circles it could belong to
    if A.ndim == 3:
        A = im2gray(A)
    A = im2single(A)

    # Sobel gradient, image edges padded by replicating the border pixels
    # (convolution, as in MATLAB's imfindcircles, so the gradient points from bright to dark and bright circles vote for their centre)
    hy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    Gx = ndi.convolve(A, hy.T, mode='nearest')
    Gy = ndi.convolve(A, hy, mode='nearest')
    gradientImg = np.hypot(Gx, Gy)

    # edge pixels = gradient above the Otsu threshold of the gradient image
    Gmax = gradientImg.max()
    edgeThresh = _graythresh(gradientImg / Gmax)
    t = Gmax * np.float32(edgeThresh)
    Ey, Ex = np.nonzero((gradientImg > t).T)[::-1]  # transposed so the edge pixels are in MATLAB's (column by column) order
    Gx_E = (Gx[Ey, Ex] / gradientImg[Ey, Ex])[:, None]
    Gy_E = (Gy[Ey, Ex] / gradientImg[Ey, Ex])[:, None]
    Ex = (Ex + 1).astype(np.float32)[:, None]  # MATLAB 1-based positions
    Ey = (Ey + 1).astype(np.float32)[:, None]

    # radii to vote for, every 0.5 pixel, each coded as a phase so the radius can be read back from the centre's vote
    radii = np.arange(radiusRange[0], radiusRange[1] + 0.25, 0.5)
    lnR = np.log(radii)
    phi = ((lnR - lnR[0]) / (lnR[-1] - lnR[0]) * 2 * np.pi) - np.pi
    w0 = np.exp(1j * phi) / (2 * np.pi * radii)
    RR = radii.astype(np.float32)

    M, N = A.shape
    accumReal = np.zeros(M * N)
    accumImag = np.zeros(M * N)
    chunk = max(1, 5_000_000 // len(RR))  # edge pixels processed at once, to limit memory
    for i in range(0, len(Ex), chunk):
        xc = matlab_round(Ex[i:i + chunk] - RR * Gx_E[i:i + chunk])
        yc = matlab_round(Ey[i:i + chunk] - RR * Gy_E[i:i + chunk])
        w = np.broadcast_to(w0, xc.shape)

        # only count votes inside the image (MATLAB's own code uses yc < M here, so the last row never gets votes)
        inside = (xc >= 1) & (xc <= N) & (yc >= 1) & (yc < M)
        idx = (yc[inside].astype(np.int64) - 1) * N + (xc[inside].astype(np.int64) - 1)
        accumReal += np.bincount(idx, weights=w.real[inside], minlength=M * N)
        accumImag += np.bincount(idx, weights=w.imag[inside], minlength=M * N)

    return (accumReal + 1j * accumImag).reshape(M, N)


def _graythresh(img):
    # Same as MATLAB graythresh: Otsu threshold (0-1) calculated from a 256 bin histogram
    counts = np.bincount(matlab_round(img.ravel().astype(np.float64) * 255).astype(np.int64), minlength=256)
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257))
    mu_t = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    maxval = np.nanmax(np.where(np.isfinite(sigma_b_squared), sigma_b_squared, np.nan))
    idx = np.mean(np.nonzero(sigma_b_squared == maxval)[0] + 1)
    return (idx - 1) / 255


def _chcenters(accumMatrix, suppThreshold):
    # Finds circle centres as the peaks of the accumulator, returns centres & peak heights strongest first
    accumMatrix = np.abs(accumMatrix)
    if np.all(accumMatrix == accumMatrix.flat[0]):
        return np.zeros((0, 2)), np.zeros(0)

    Hd = ndi.median_filter(accumMatrix, size=5, mode='constant', cval=0)  # same as medfilt2 (zero padded)
    suppThreshold = max(suppThreshold - np.spacing(suppThreshold), 0)
    Hd = reconstruction(Hd - suppThreshold, Hd, method='dilation')  # same as imhmax: removes peaks lower than suppThreshold
    bw = local_maxima(Hd, connectivity=2, allow_borders=True)  # same as imregionalmax

    # weighted centroid of each peak, numbered in MATLAB's (column by column) order
    labels = label(bw.T, connectivity=2).T
    n = labels.max()
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    rows, cols = np.indices(accumMatrix.shape)
    total = ndi.sum(accumMatrix, labels, np.arange(1, n + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        cx = ndi.sum(accumMatrix * (cols + 1), labels, np.arange(1, n + 1)) / total
        cy = ndi.sum(accumMatrix * (rows + 1), labels, np.arange(1, n + 1)) / total
    centers = np.column_stack([cx, cy])
    centers = centers[~np.isnan(centers).any(axis=1)]
    if len(centers) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    metric = Hd[matlab_round(centers[:, 1]).astype(int) - 1, matlab_round(centers[:, 0]).astype(int) - 1]
    order = np.argsort(-metric, kind='stable')
    return centers[order], metric[order]


def _chradiiphcode(centers, accumMatrix, radiusRange):
    # Reads each circle's radius back from the phase of the vote at its centre
    cenPhase = np.angle(accumMatrix[matlab_round(centers[:, 1]).astype(int) - 1, matlab_round(centers[:, 0]).astype(int) - 1])
    lnR = np.log(radiusRange)
    return np.exp(((cenPhase + np.pi) / (2 * np.pi) * (lnR[-1] - lnR[0])) + lnR[0])


def imshow(ax, img):
    # Same as MATLAB imshow(img): integer images are shown on their full data range (eg. 0-65535 for 16-bit), not contrast stretched
    if np.issubdtype(img.dtype, np.integer):
        img_max = np.iinfo(img.dtype).max
        if img.ndim == 3:
            ax.imshow(img.astype(np.float64) / img_max)
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=img_max)
    else:
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=1)
    ax.axis('off')


def imshow_scaled(ax, img):
    # Same as MATLAB imshow(img, []): greyscale images are contrast stretched from min to max, colour images are shown as they are
    if img.ndim == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
    ax.axis('off')


def viscircles(ax, centers, radii, color, linestyle='-', linewidth=2):
    # Similar to MATLAB viscircles: draws circles from MATLAB 1-based centres, radii can be one value for every circle
    radii = np.broadcast_to(radii, (len(centers),))
    for (x, y), r in zip(centers, radii):
        ax.add_patch(Circle((x - 1, y - 1), r, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth))
