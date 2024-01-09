import glob

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_msssim import ms_ssim, ssim

"""
Returns structural similarity score between two images
"""


def calculate_pairwise_sim(img1, img2):
    return ssim(img1, img2, data_range=255, size_average=False)


def ssim_for_all_images():
    pattern = "../images/all/blood_*/*"
    imgs = []
    img_paths = []
    for img in glob.glob(pattern):
        print(img)
        imgs.append(cv2.imread(img)/ 255)
        img_paths.append(img.split("/")[-1])

    l1 = np.array(imgs, dtype=float)
    ssim_values = np.zeros((len(l1), len(l1)))

    for i, img1 in tqdm(enumerate(l1)):
        for j, img2 in enumerate(l1):
            if j < i:
                continue
            elif i == j:
                ssim_values[i][j] = 1.0
                ssim_values[j][i] = 1.0
            else:
                sim = ssim(
                torch.from_numpy(img1.reshape(1, 3, 336, 336)),
                torch.from_numpy(img2.reshape(1, 3, 336, 336)))
                ssim_values[i][j] = sim
                ssim_values[j][i] = sim
        pd.DataFrame(ssim_values, columns=img_paths).to_csv("../images/sim_blood2")


if __name__ == '__main__':
    ssim_for_all_images()
