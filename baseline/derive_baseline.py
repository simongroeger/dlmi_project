import math
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from baseline.helpers import BaselineHelper
from tqdm import trange

"""
Calculate and return mean h for all num_patches patches of the image
"""


class Baseline:
    debug = False
    hsv_names = np.array(["hue", "saturation", "value"])

    def mean_h_for_patches(self, img, num_patches):
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

    def plot_posterior(self, class_count, cls_folder_list, values, value_description: str, comp, index):
        all_values = [item for sub_list in values for item in sub_list]
        # Replace with prior from original dataset
        class_prior = class_count / class_count.sum()

        fig, ax = plt.subplots(3, 1)

        all_counts, bins = np.histogram(all_values, 100, density=True)

        posteriors = {}
        for cls_i, cls_folder in enumerate(cls_folder_list):
            likelihood, _ = np.histogram(values[cls_i], bins, density=True)
            ax[0].stairs(likelihood, bins, fill=False, label="p(mean " + value_description + " | " + cls_folder + ")")

            likelihood_prior = likelihood * class_prior[cls_i]
            ax[1].stairs(likelihood_prior, bins, fill=False,
                         label="p(mean " + value_description + " , " + cls_folder + ")")

            posterior = likelihood_prior / all_counts * 100
            posterior = [0 if math.isnan(x) else x for x in posterior]
            if cls_folder == "blood":
                boundaries = [bins[min(np.nonzero(posterior)[0])], bins[max(np.nonzero(posterior)[-1]) + 1]]
                print(value_description, "lower boundary", boundaries[0])
                print(value_description, "upper boundary", boundaries[1])
                ax[2].axvline(boundaries[0], color="red", label="no FN decision boundary")
                ax[2].axvline(boundaries[1], color="red")
            ax[2].stairs(posterior, bins, fill=False, label="p(" + cls_folder + " | mean " + value_description + ")")
            posteriors[cls_folder] = posterior

        indices = [i for i, (val1, val2) in enumerate(zip(posteriors["blood"], posteriors["other"])) if
                   comp(val1, val2)]
        bayesian_boundaries = [bins[min(indices, default=0)], bins[max(indices, default=0) + 1]]
        ax[2].axvline(bayesian_boundaries[0], color="mediumseagreen", label="bayesian decision boundary")
        ax[2].axvline(bayesian_boundaries[1], color="seagreen")
        print(value_description, "lower bayesian boundary", bayesian_boundaries[0])
        print(value_description, "upper bayesian boundary", bayesian_boundaries[1])

        ax[0].set_title("Likelihood")
        ax[1].set_title("Likelihood x Prior")
        ax[2].set_title("Posterior")

        for i in range(3):
            ax[i].set_xlabel("mean " + value_description)
            ax[i].set_ylabel("probability density")
            ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(hspace=2)
        plt.savefig(value_description + '.png')
        plt.show()



    """
    Mask given image by removing all non-red pixels
    """

    @staticmethod
    def get_masked_means(img, mask, extract_methods):
        result = img
        if mask is not None:
            mask1 = cv2.inRange(img, mask[0], mask[1]) # lower red values
            mask2 = cv2.inRange(img, mask[2], mask[3]) # higher red values
            mask = mask1 + mask2
            result = cv2.bitwise_and(img, img, mask=mask)
            if Baseline.debug:
                cv2.imshow("result", result)
                cv2.waitKey(0)
            if not result.any():
                return [1, 1, 0]
            result[result == 0] = np.nan
        return [extract_methods[j](result[:, :, j]) for j in range(3)]

    """
    @param m: mask, comp: comparator, cfl: class folder list, cc: class count, extr_methods: extraction methods
    """
    def derive_baseline(self, m, comp, cfl, extr_methods):
        values = [[[],[]], [[],[]], [[],[]]]  # first index is h, s, v and second cls_i
        class_count = np.zeros(len(cls_folder_list))

        for cls_i, cls_folder in enumerate(cfl):
            print(cls_folder)
            img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
            for i in trange(min(100000, len(img_list))):
                img_name = img_list[i]
                if "_rot_" in img_name or "_shear_" in img_name:
                    continue
                img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name)) / 255.
                hsv_image = matplotlib.colors.rgb_to_hsv(img)
                if self.debug:
                    cv2.imshow("img", img)

                means = self.get_masked_means(hsv_image, mask, extr_methods)
                if self.debug:
                    print(means)

                for k in range(len(self.hsv_names)):
                    values[k][cls_i].append(means[k])

                class_count[cls_i] += 1
        for i, name in enumerate(self.hsv_names):
            self.plot_posterior(class_count, cfl, values[i], self.hsv_names[i], comp, i)


if __name__ == "__main__":
    src_dir = "../pytorch-image-models/images"
    split = "test_split"
    mask = BaselineHelper.First_try
    # mask = BaselineHelper.Second_try
    # mask = BaselineHelper.Third_try
    comparator = BaselineHelper.greater
    cls_folder_list = os.listdir(os.path.join(src_dir, split))
    extract_methods = [np.nanmean, np.nanmean, np.nanmean]

    baseline = Baseline()
    baseline.derive_baseline(mask, comparator, cls_folder_list, extract_methods)