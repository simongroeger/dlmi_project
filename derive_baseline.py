from typing import Any

import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

def main():
    src_dir = "./pytorch-image-models/images"
    split = "val_split"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    classcount = np.zeros(len(cls_folder_list))

    ss = []
    s_all = []
    vs = []
    v_all = []
    for cls_i, cls_folder in enumerate(cls_folder_list):
        print(cls_folder)
        img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
        s = []
        v = []
        for i in trange(min(100000, len(img_list))):
            img_name = img_list[i]
            if "_rot_" in img_name or "_shear_" in img_name:
                continue
            img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name)) / 255.
            img = img[30:-30, 30:-30, :]
            hsv_image = matplotlib.colors.rgb_to_hsv(img)

            mean_s = get_masked_means(hsv_image, 1)
            mean_v = get_masked_means(hsv_image, 2)

            s.append(mean_s)
            s_all.append(mean_s)
            v.append(mean_v)
            v_all.append(mean_v)
            classcount[cls_i] += 1
        ss.append(s)
        vs.append(v)
    plot_posterior(classcount, s_all, cls_folder_list, ss)
    plot_posterior(classcount, v_all, cls_folder_list, vs)



"""
Calculate and return mean h for all num_patches patches of the image
"""


def mean_h_for_patches(img, num_patches):
    patch_means = []
    patch_size = img.shape[0] // num_patches
    for i in range(num_patches):
        if i == num_patches - 1:
            average = np.mean(img[i * patch_size:, i * patch_size:, 0])
        else:
            average = np.mean(img[i * patch_size:(i + 1) * patch_size, i * patch_size:(i + 1) * patch_size, 0])
        patch_means.append(average)
    return patch_means

def plot_posterior(classcount, all, cls_folder_list, values):
    classprior = classcount / classcount.sum()

    fig, ax = plt.subplots(3, 1)

    all_counts, bins = np.histogram(all, 100, density=True)
    print(bins)

    for cls_i, cls_folder in enumerate(cls_folder_list):
        likelihood, _ = np.histogram(values[cls_i], bins, density=True)
        ax[0].stairs(likelihood, bins, fill=False, label="p(mean hue | " + cls_folder + ")")

        likelihood_prior = likelihood * classprior[cls_i]
        ax[1].stairs(likelihood_prior, bins, fill=False, label="p(mean hue , " + cls_folder + ")")

        posterior = likelihood_prior / all_counts * 100
        ax[2].stairs(posterior, bins, fill=False, label="p(" + cls_folder + " | mean hue)")

    ax[0].set_title("Likelihood")
    ax[1].set_title("Likelihood x Prior")
    ax[2].set_title("Posterior")

    for i in range(3):
        ax[i].set_xlabel("mean hue")
        ax[i].set_ylabel("probability density")
        ax[i].legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()

def get_masked_means(img, hsv_index):
    lower1 = np.array([0, 100, 20]) / 255.
    upper1 = np.array([10, 255, 255]) / 255.
    lower2 = np.array([160, 100, 20]) / 255.
    upper2 = np.array([179, 255, 255]) / 255.

    mask1 = cv2.inRange(img, lower1, upper1)
    mask2 = cv2.inRange(img, lower2, upper2)
    mask = mask1 + mask2
    result = cv2.bitwise_and(img, img, mask=mask)
    if not result.any():
        return 1
    result[result == 0] = np.nan
    return np.nanmean(result[:, :, hsv_index])

main()

# blood if average_h > 0.6015 and average_h < 0.6057 or average_h > 0.6088
