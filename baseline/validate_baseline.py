import numpy as np
import cv2
import os
import matplotlib
from tqdm import trange

from baseline.derive_baseline import get_masked_means


def infere_h(average_h):
    return (0, "blood") if (average_h > 0.6015 and average_h < 0.6057) or average_h > 0.6088 else (1, "other")


def infere_masked_s(average_s):
    return (0, "blood") if average_s > 0.5968 and average_s < 0.8154 else (1, "other")


def infere_masked_v(average_v):
    return (0, "blood") if average_v > 0.2519 and average_v < 0.6819 else (1, "other")


def infer_combined(averag_vs: list):
    return (0, "blood") if infere_masked_s(averag_vs[0])[0] == 0 and infere_masked_v(averag_vs[1])[0] == 0 else (
    1, "other")


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
        [[1], [2], [1, 2]],
        [infere_masked_s, infere_masked_v, infer_combined],
        "val_split"
    )
