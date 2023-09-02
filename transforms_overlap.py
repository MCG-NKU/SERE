# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import division

import numpy as np
import torch
import math
import random
from PIL import Image
import warnings

from torchvision.transforms import functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class ComposeOverLap(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mapping=None):
        for t in self.transforms:
            if "RandomResizedCropOverLap" in t.__class__.__name__:
                img, mapping = t(img)
            elif "FlipOverLap" in t.__class__.__name__:
                img, mapping = t(img, mapping)
            elif "ComposeOverLap" in t.__class__.__name__:
                img, mapping = t(img, mapping)
            else:
                img = t(img)
        mapping = np.array(mapping)
        return img, mapping

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlipOverLap(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mapping):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            mapping.append(1)
            return F.hflip(img), mapping
        mapping.append(0)
        return img, mapping

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomResizedCropOverLap(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=Image.BILINEAR,
    ):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        mapping = [j, i, w, h]
        return (
            F.resized_crop(img, i, j, h, w, self.size, self.interpolation),
            mapping,
        )

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def get_grid(rectq, rectk, size):
    grid = float(size - 1)
    overlap = [
        max(rectq[0], rectk[0]),
        max(rectq[1], rectk[1]),
        min(rectq[0] + rectq[2], rectk[0] + rectk[2]),
        min(rectq[1] + rectq[3], rectk[1] + rectk[3]),
    ]
    if overlap[0] < overlap[2] and overlap[1] < overlap[3]:
        q_overlap = torch.FloatTensor(
            [
                (overlap[0] - rectq[0]) / rectq[2],
                (overlap[1] - rectq[1]) / rectq[3],
                (overlap[2] - overlap[0]) / rectq[2],
                (overlap[3] - overlap[1]) / rectq[3],
            ]
        )
        k_overlap = torch.FloatTensor(
            [
                (overlap[0] - rectk[0]) / rectk[2],
                (overlap[1] - rectk[1]) / rectk[3],
                (overlap[2] - overlap[0]) / rectk[2],
                (overlap[3] - overlap[1]) / rectk[3],
            ]
        )

        q_grid = torch.zeros(size=(size, size, 2), dtype=torch.float32)
        k_grid = torch.zeros(size=(size, size, 2), dtype=torch.float32)
        q_grid[:, :, 0] = torch.FloatTensor(
            [q_overlap[0] + i * q_overlap[2] / grid for i in range(size)]
        ).view(1, size)
        q_grid[:, :, 1] = torch.FloatTensor(
            [q_overlap[1] + i * q_overlap[3] / grid for i in range(size)]
        ).view(size, 1)
        k_grid[:, :, 0] = torch.FloatTensor(
            [k_overlap[0] + i * k_overlap[2] / grid for i in range(size)]
        ).view(1, size)
        k_grid[:, :, 1] = torch.FloatTensor(
            [k_overlap[1] + i * k_overlap[3] / grid for i in range(size)]
        ).view(size, 1)

        # flip
        if rectq[4] > 0:
            q_grid[:, :, 0] = 1 - q_grid[:, :, 0]

        if rectk[4] > 0:
            k_grid[:, :, 0] = 1 - k_grid[:, :, 0]

        k_grid = 2 * k_grid - 1
        q_grid = 2 * q_grid - 1

    else:
        # fill zero
        q_grid = torch.full(fill_value=-2, size=(size, size, 2), dtype=torch.float32)
        k_grid = torch.full(fill_value=-2, size=(size, size, 2), dtype=torch.float32)

    return q_grid, k_grid
