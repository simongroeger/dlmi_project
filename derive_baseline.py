import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from numpy.core.numeric import allclose
from tqdm import trange

def main():

    src_dir = "/home/simon/dlmi_project/pytorch-image-models/images"
    split = "val_split"

    cls_folder_list = os.listdir(os.path.join(src_dir, split))

    classcount = np.zeros(len(cls_folder_list))
    
    hs = []
    h_all = []
    for cls_i, cls_folder in enumerate(cls_folder_list): 
        print(cls_folder)
        img_list = os.listdir(os.path.join(src_dir, split, cls_folder))
        h = []
        for i in trange(min(100000, len(img_list))):
            img_name = img_list[i]
            if "_rot_" in img_name or "_shear_" in img_name:
                continue
            img = cv2.imread(os.path.join(src_dir, split, cls_folder, img_name))/255.0
            hsv_image = matplotlib.colors.rgb_to_hsv(img)
            average_h = np.mean(hsv_image[:, :, 0])
            h.append(average_h)
            h_all.append(average_h)
            classcount[cls_i] += 1
        hs.append(h)

    classprior = classcount / classcount.sum()

    fig, ax = plt.subplots(3, 1)

    all_counts, bins = np.histogram(h_all, 100, density=True)
    print(bins)

    for cls_i, cls_folder in enumerate(cls_folder_list): 
        likelihood, _ = np.histogram(hs[cls_i], bins, density=True)
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




main()

# blood if average_h > 0.6015 and average_h < 0.6057 or average_h > 0.6088