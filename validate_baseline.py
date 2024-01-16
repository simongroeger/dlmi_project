import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.numeric import allclose
from tqdm import trange


def infere(average_h):
    return (0, "blood") if (average_h > 0.6015 and average_h < 0.6057) or average_h > 0.6088 else (1, "other")

def main():

    src_dir = "/home/simon/dlmi_project/pytorch-image-models/images"
    split = "test_split"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    classcount = np.zeros(len(cls_folder_list))

    res = np.zeros((2,2))
    
    for cls_i, cls_folder in enumerate(cls_folder_list): 
        print(cls_folder)
        img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
        for i in trange(min(100000, len(img_list))):
            img_name = img_list[i]
            if "_rot_" in img_name or "_shear_" in img_name:
                continue
            img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name))/255.0
            hsv_image = matplotlib.colors.rgb_to_hsv(img)
            average_h = np.mean(hsv_image[:, :, 0])

            pred_i, pred_cls = infere(average_h)
            res[pred_i, cls_i] += 1
            classcount[cls_i] += 1

    print()
    print(res)

    res = res / np.sum(classcount)

    print()
    print(res)

    accuracy = (res[0,0] + res[1,1]) / np.sum(res)
    precision = res[0, 0] / (res[0, 0] + res[1, 0])
    recall = res[0, 0] / (res[0, 0] + res[0, 1])
    f1score = 2 * (precision*recall)/(precision+recall)

    print(accuracy, precision, recall, f1score)




main()

