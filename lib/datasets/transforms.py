import random

import torchvision.transforms.functional as F
from torchvision.transforms import *


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = []
        for t in transforms:
            if t is not None:
                self.transforms.append(t)

    def __call__(self, img, boxes):
        for t in self.transforms:
            if isinstance(t, RandomHorizontalFlip):
                img, boxes = t(img, boxes)
            else:
                img = t(img)
        return img, boxes


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a probability of 0.5.
        Return image with boxes and flipped boxes.
    """

    def __call__(self, img, boxes):

        flip = random.random() < 0.5
        if flip:
            width = img.width
            flipped_boxes = boxes.clone()
            flipped_boxes[:, 2] = width - boxes[:, 0] - 1
            flipped_boxes[:, 0] = width - boxes[:, 2] - 1
            img = F.hflip(img)
            return img, flipped_boxes
        else:
            return img, boxes
