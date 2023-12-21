import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

def main():

    src_dir = "/home/simon/dlmi_project/pytorch-image-models/images"
    split = "raw_split_1"


    cls_folder_list = os.listdir(os.path.join(src_dir, split))
    hs = []


    classcount = np.zeros(len(cls_folder_list))
    for i, cls_folder in enumerate(cls_folder_list): 
        classcount[i] = len(os.listdir(os.path.join(src_dir, split, cls_folder)))
    classprior = classcount / classcount.sum()

    fig, ax = plt.subplots(2, 1)

    for cls_i, cls_folder in enumerate(cls_folder_list): 
        print(cls_folder)
        img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
        h = []
        for i in trange(min(100, len(img_list))):
            img_name = img_list[i]
            img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name))/255.0
            hsv_image = matplotlib.colors.rgb_to_hsv(img)
            average_h = np.mean(hsv_image[:, :, 0])
            h.append(average_h)

        counts, bins = np.histogram(h, 25, density=True)
        ax[0].stairs(counts, bins, fill=False, label=cls_folder)


        counts *= classprior[cls_i]
        ax[1].stairs(counts, bins, fill=False, label=cls_folder)

        print(cls_folder, bins)

    ax[1].axvline(x=0.59438768, label="decision_boundary")

    plt.legend()
    plt.show()




main()

# other if average_h > 0.59438768