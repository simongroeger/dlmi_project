import math
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

# Upper and lower bounds for red pixels hues
lower1 = np.array([0, 100, 20]) / 255.
upper1 = np.array([10, 255, 255]) / 255.
lower2 = np.array([160, 100, 20]) / 255.
upper2 = np.array([179, 255, 255]) / 255.

debug = False

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

"""
Plot likelihood, prior and posterior with decision boundaries for blood class
"""
def plot_posterior(class_count, all_values, cls_folder_list, values, value_description: str):
    boundaries = []
    # Replace with prior from original dataset
    class_prior = class_count / class_count.sum()

    fig, ax = plt.subplots(3, 1)

    all_counts, bins = np.histogram(all_values, 100, density=True)

    for cls_i, cls_folder in enumerate(cls_folder_list):
        likelihood, _ = np.histogram(values[cls_i], bins, density=True)
        ax[0].stairs(likelihood, bins, fill=False, label="p(mean " + value_description + " | " + cls_folder + ")")

        likelihood_prior = likelihood * class_prior[cls_i]
        ax[1].stairs(likelihood_prior, bins, fill=False, label="p(mean " + value_description + " , " + cls_folder + ")")

        posterior = likelihood_prior / all_counts * 100
        posterior = [0 if math.isnan(x) else x for x in posterior]
        if cls_folder == "blood":
            boundaries += [bins[min(np.nonzero(posterior)[0])], bins[max(np.nonzero(posterior)[-1])+1]]
            print(value_description, "lower boundary", boundaries[0])
            print(value_description, "upper boundary", boundaries[1])
            ax[2].axvline(boundaries[0], color="red", label="decision boundary")
            ax[2].axvline(boundaries[1], color="red")
        ax[2].stairs(posterior, bins, fill=False, label="p(" + cls_folder + " | mean " + value_description + ")")

    ax[0].set_title("Likelihood")
    ax[1].set_title("Likelihood x Prior")
    ax[2].set_title("Posterior")

    for i in range(3):
        ax[i].set_xlabel("mean " + value_description)
        ax[i].set_ylabel("probability density")
        ax[i].legend()

    plt.subplots_adjust(hspace=1)
    plt.show()

"""
Mask given image by removing all non-red pixels
"""
def get_masked_means(img, hsv_index):
    mask1 = cv2.inRange(img, lower1, upper1)
    mask2 = cv2.inRange(img, lower2, upper2)
    mask = mask1 + mask2
    result = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow("result", result)
        cv2.waitKey(0)
    if not result.any():
        return 1
    result[result == 0] = np.nan
    return np.nanmean(result[:, :, hsv_index])

if __name__ == "__main__":
    src_dir = "../pytorch-image-models/images"
    split = "train_split"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    class_count = np.zeros(len(cls_folder_list))

    s_values = []
    s_all = []
    v_values = []
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
            hsv_image = matplotlib.colors.rgb_to_hsv(img)
            if debug:
                cv2.imshow("img", img)

            mean_s = get_masked_means(hsv_image, 1)
            mean_v = get_masked_means(hsv_image, 2)

            s.append(mean_s)
            s_all.append(mean_s)
            v.append(mean_v)
            v_all.append(mean_v)
            class_count[cls_i] += 1
        s_values.append(s)
        v_values.append(v)
    plot_posterior(class_count, s_all, cls_folder_list, s_values, "saturation")
    plot_posterior(class_count, v_all, cls_folder_list, v_values, "value")