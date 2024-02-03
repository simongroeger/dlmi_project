import numpy as np
import cv2
import os
import matplotlib
from tqdm import trange

from baseline.derive_baseline import get_masked_means


def infere_h(average_h):
    return (0, "blood") if (average_h > 0.6015 and average_h < 0.6057) or average_h > 0.6088 else (1, "other")


# No FN
def infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.5968 and average_s < 0.8154 else (1, "other")
def infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.2519 and average_v < 0.6819 else (1, "other")
def infere_masked_h(average_h):
    return (0, "blood") if average_h > 0.6351 and average_h < 0.6569 else (1, "other")
def infer_combined(averag_hvss: list):
    return (0, "blood") if (infere_masked_h(averag_hvss[0])[0] == 0 and
                            infere_masked_s(averag_hvss[1])[0] == 0 and
                            infere_masked_v(averag_hvss[2])[0] == 0) else (1, "other")

# Allowing FN (first occurence where p(b) >=  p(n) - last occurence where P(b) >= P(n))
def geq_infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.5191 and average_s < 0.7134 else (1, "other")
def geq_infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.4370 and average_v < 0.5803 else (1, "other")
def geq_infere_masked_h(average_h):
    return (0, "blood") if average_h > 0.6458 and average_h < 0.6498 else (1, "other")
def geq_infer_combined(averag_hsvs: list):
    return (0, "blood") if (geq_infere_masked_h(averag_hsvs[0])[0] == 0 and
                            geq_infere_masked_s(averag_hsvs[1])[0] == 0 and
                            geq_infere_masked_v(averag_hsvs[2])[0] == 0) else (1, "other")

# Allowing FN (first occurence where p(b) >  p(n) - last occurence where P(b) >= P(n))
def g_infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.6162 and average_s < 0.6697 else (1, "other")
def g_infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.4370 and average_v < 0.5803 else (1, "other")
def g_infere_masked_h(average_h):
    return (0, "blood") if average_h > 0.6458 and average_h < 0.6491 else (1, "other")
def g_infer_combined(averag_hsvs: list):
    return (0, "blood") if (g_infere_masked_h(averag_hsvs[0])[0] == 0 and
                            g_infere_masked_s(averag_hsvs[1])[0] == 0 and
                            g_infere_masked_v(averag_hsvs[2])[0] == 0) else (1, "other")
def validate_for_hsv_index(hsv_indices, pred_functions, split):
    src_dir = "../pytorch-image-models/images"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    class_count = np.zeros((len(hsv_indices),len(cls_folder_list)))

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
            for i, hsv_index_list in enumerate(hsv_indices):
                if len(hsv_index_list) > 1:
                    mean = []
                    for hsv_index in hsv_indices:
                        mean.append(get_masked_means(hsv_image, hsv_index))
                else:
                    mean = get_masked_means(hsv_image, hsv_index_list[0])

                pred_i, pred_cls = pred_functions[i](mean)
                res[i, pred_i, cls_i] += 1
                class_count[i, cls_i] += 1

    for i in range(res.shape[0]):
        print()
        print(res[i])
        r = res[i]
        r = r / np.sum(class_count[i])

        print()
        print(r)

        accuracy = (r[0, 0] + r[1, 1]) / np.sum(r)
        precision = r[0, 0] / (r[0, 0] + r[1, 0])
        recall = r[0, 0] / (r[0, 0] + r[0, 1])
        f1score = 2 * (precision * recall) / (precision + recall)

        print("Performance for", pred_functions[i].__name__,
              "\n accuracy", accuracy,
              "\n precision", precision,
              "\n recall", recall,
              "\n f1-score", f1score)


if __name__ == "__main__":
    validate_for_hsv_index(
        [[0], [1], [2], [0, 1, 2]],
        [g_infere_masked_h, g_infere_masked_s, g_infere_masked_v, geq_infer_combined],
        "val_split"
    )
