import matplotlib.pyplot as plt
import numpy as np


def read_split_image(img):
    width, height = img.size
    box1 = (0, 0, height, height)  # (left, upper, right, lower) - tuple
    box2 = (height, 0, height * 2, height)
    box3 = (height * 2, 0, width, height)
    tra_ch = img.crop(box1)  # target
    sou_seal = img.crop(box2)  # source
    dst_seal = img.crop(box3)
    return tra_ch, sou_seal, dst_seal


def plot_tensor(tensor):
    img = np.transpose(tensor.data, (1, 2, 0))
    plt.imshow(img)
