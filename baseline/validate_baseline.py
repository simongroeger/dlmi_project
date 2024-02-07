import numpy as np
import cv2
import os
import matplotlib
from tqdm import trange

from baseline.derive_baseline import Baseline
from baseline.helpers import BaselineHelper


# def infere_h(average_h):
#     return (0, "blood") if (average_h > 0.6015 and average_h < 0.6057) or average_h > 0.6088 else (1, "other")

# -- First Try --
def g_infere_masked_h(average_h):
    return (0, "blood") if average_h > 0.6350 and average_h < 0.6568 else (1, "other")
def g_infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.5967 and average_s < 0.8153 else (1, "other")
def g_infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.2518 and average_v < 0.6818 else (1, "other")
def g_infer_combined(averag_hsvs: list):
    return (0, "blood") if (g_infere_masked_h(averag_hsvs[0])[0] == 0 or
                            g_infere_masked_v(averag_hsvs[0])[0] == 0 or
                            g_infere_masked_s(averag_hsvs[1])[0] == 0) else (1, "other")
# using bayesian inspired borders
def gb_infere_masked_h(average_h):
    return (0, "blood") if average_h > 0.6457 and average_h < 0.6494 else (1, "other")
def gb_infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.6162 and average_s < 0.6745 else (1, "other")
def gb_infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.4370 and average_v < 0.5863 else (1, "other")
def gb_infer_combined(averag_hsvs: list):
    return (0, "blood") if (gb_infere_masked_h(averag_hsvs[0])[0] == 0 or
                            gb_infere_masked_v(averag_hsvs[0])[0] == 0 or
                            gb_infere_masked_s(averag_hsvs[1])[0] == 0) else (1, "other")

# -- Second Try --
def g_infere_masked_h2(average_h):
    return (0, "blood") if average_h > 0.6347 and average_h < 0.6572 else (1, "other") #lb - u
def g_infere_masked_s2(average_s):
    return (0, "blood") if average_s > 0.5739 and average_s < 0.8089 else (1, "other")
def g_infere_masked_v2(average_v):
    return (0, "blood") if average_v > 0.2390 and average_v < 0.6700 else (1, "other") # l - ub
def g_infer_combined2(averag_hsvs: list):
    return (0, "blood") if (g_infere_masked_h2(averag_hsvs[0])[0] == 0 or
                            g_infere_masked_v2(averag_hsvs[2])[0] == 0 or
                            g_infere_masked_s2(averag_hsvs[1])[0] == 0) else (1, "other")
# using bayesian inspired borders
def gb_infere_masked_h2(average_h):
    return (0, "blood") if average_h > 0.6459 and average_h < 0.6492 else (1, "other") #lb - u
def gb_infere_masked_s2(average_s):
    return (0, "blood") if average_s > 0.5739 and average_s < 0.6131 else (1, "other")
def gb_infere_masked_v2(average_v):
    return (0, "blood") if average_v > 0.4254 and average_v < 0.5768 else (1, "other") # l - ub
def gb_infer_combined2(averag_hsvs: list):
    return (0, "blood") if (gb_infere_masked_h2(averag_hsvs[0])[0] == 0 or
                            gb_infere_masked_v2(averag_hsvs[2])[0] == 0 or
                            gb_infere_masked_s2(averag_hsvs[1])[0] == 0) else (1, "other")
# -- Third Try --
def g_infere_masked_h3(average_h):
    return (0, "blood") if average_h > 0.6342 and average_h < 0.6576 else (1, "other") #lb - u
def g_infere_masked_s3(average_s):
    return (0, "blood") if average_s > 0.6538 and average_s < 0.8306 else (1, "other")
def g_infere_masked_v3(average_v):
    return (0, "blood") if average_v > 0.2812 and average_v < 0.6451 else (1, "other") # l - ub
def g_infer_combined3(averag_hsvs: list):
    return (0, "blood") if (g_infere_masked_h3(averag_hsvs[0])[0] == 0 or
                            g_infere_masked_v3(averag_hsvs[2])[0] == 0 or
                            g_infere_masked_s3(averag_hsvs[1])[0] == 0) else (1, "other")

# using bayesian inspired borders
def gb_infere_masked_h3(average_h):
    return (0, "blood") if average_h > 0.6459 and average_h < 0.6496 else (1, "other") #lb - u
def gb_infere_masked_s3(average_s):
    return (0, "blood") if average_s > 0.6943 and average_s < 0.7312 else (1, "other")
def gb_infere_masked_v3(average_v):
    return (0, "blood") if average_v > 0.4245 and average_v < 0.5734 else (1, "other") # l - ub
def gb_infer_combined3(averag_hsvs: list):
    return (0, "blood") if (gb_infere_masked_h3(averag_hsvs[0])[0] == 0 or
                            gb_infere_masked_v3(averag_hsvs[2])[0] == 0 or
                            gb_infere_masked_s3(averag_hsvs[1])[0] == 0) else (1, "other")


"""
Calculate metrics
"""
def validate_for_hsv_index(hsv_indices, pred_functions, split, mask, extract_methods):
    src_dir = "../pytorch-image-models/images"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    class_count = np.zeros((len(hsv_indices), len(cls_folder_list)))

    res = np.zeros((len(hsv_indices), 2, 2))

    for cls_i, cls_folder in enumerate(cls_folder_list):
        print(cls_folder)
        img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
        for i in trange(min(100000, len(img_list))):
            img_name = img_list[i]
            if "_rot_" in img_name or "_shear_" in img_name:
                continue
            img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name)) / 255.0
            hsv_image = matplotlib.colors.rgb_to_hsv(img)
            for j, hsv_index_list in enumerate(hsv_indices):
                if len(hsv_index_list) > 1:
                    mean = []
                    for hsv_index in hsv_index_list:
                        mean.append(Baseline.get_masked_means(hsv_image, mask, extract_methods)[hsv_index])
                else:
                    mean = Baseline.get_masked_means(hsv_image, mask, extract_methods)[hsv_index_list[0]]

                pred_i, pred_cls = pred_functions[j](mean)
                res[j, pred_i, cls_i] += 1
                class_count[j, cls_i] += 1
                print(pred_cls)

    for i, index in enumerate(hsv_indices):
        print()
        print(res[i])
        r = res[i]

        print()
        print(r)

        accuracy = (r[0, 0] + r[1, 1]) / np.sum(r)
        precision = r[0, 0] / (r[0, 0] + r[0, 1]) # error before [0,1] and [1,0] swapped
        recall = r[0, 0] / (r[0, 0] + r[1, 0])
        f1score = 2 * (precision * recall) / (precision + recall)

        print("Performance for", pred_functions[i].__name__,
              "\n accuracy", accuracy,
              "\n precision", precision,
              "\n recall", recall,
              "\n f1-score", f1score)


if __name__ == "__main__":
    m = BaselineHelper.First_try
    clfs_all = [g_infere_masked_s]
    validate_for_hsv_index(
        [[1]],
        clfs_all,
        "train_split",
        m,
        [np.nanmean, np.nanmean, np.nanmean]
    )

