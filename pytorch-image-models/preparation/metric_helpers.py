import glob
import os
import shutil
from itertools import chain

import cv2
import numpy as np
import pandas as pd
import torch
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
        imgs.append(cv2.imread(img) / 255)
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


"""
Create clusters based on saved similarity scores
"""


def get_clusters(num_clusters, df, specific_num_cluster=None, threshold=0.0009):
    if num_clusters is not None:
        cluster_sizes = [len(df) // num_clusters] * num_clusters
        c = [[] for _ in range(num_clusters)]
    elif specific_num_cluster is not None:
        cluster_sizes = specific_num_cluster
        c = [[] for _ in range(len(specific_num_cluster))]
    else:
        raise Exception("PLEASE DEFINE CLUSTER SIZE")

    files = df.columns
    current_cluster = 0
    for name, column in tqdm(df.items()):
        if df.sum().sum() == 0:
            break

        cluster = column[1 - column < threshold]
        if len(cluster) > 0:
            c[current_cluster] += (list(map(lambda x: files[x], cluster.index)))

        for i in cluster.index:
            df[name] = 0
            df.iloc[i] = 0
        if len(c[current_cluster]) >= cluster_sizes[current_cluster]:
            current_cluster += 1

    print(len(c), "clusters created with a total of", len(list(chain(*c))), "elements")
    [print("cluster", i, len(cluster)) for i, cluster in enumerate(c)]
    return c


def save_clusters(c, split_names):
    # Check if the directory already exists
    if not os.path.exists("../images/metric_split"):
        # Create the directory
        os.makedirs("../images/metric_split")
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

    print(c)
    for i, cluster in tqdm(enumerate(c)):
        dst_dir = "../images/metric_split/c" + str(i)
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

        image_names = []
        for name in cluster:
            src_file = os.path.join("../images", name)
            dst_file = os.path.join(dst_dir, name.split('\\')[-1])
            image_names += [name.split('\\')[-1]]
            shutil.copy(src_file, dst_file)

        current_df = pd.read_csv("../../csvs/splits_by_video/" + split_names[i])
        current_df = current_df.drop(current_df[current_df.label == "Blood"].index)
        data = pd.DataFrame({
            'label': ["Blood"] * len(image_names),
            'filename': image_names})
        current_df = pd.concat(
            [current_df, data], ignore_index=True
        )
        current_df.to_csv("../../csvs/splits_by_metric/" + split_names[i])


if __name__ == '__main__':
    # ssim_for_all_images()
    df = pd.read_csv("../../csvs/splits_by_metric/ssim_blood.csv", index_col=0)
    clusters = get_clusters(None, df, [398, 30, 30], 0.0002)
    save_clusters(clusters, ["train_split.csv", "val_split.csv", "test_split.csv"])
