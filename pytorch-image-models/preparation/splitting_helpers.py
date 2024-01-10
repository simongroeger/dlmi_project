import glob

import pandas as pd

# Gets file names of csv, merges them together and add movie information
def create_full_set(splits_to_merge: list):
    splits = []
    for split_name in splits_to_merge:
        split = pd.read_csv("../images/" + split_name)
        splits.append(split)

    result = pd.concat(splits)
    result[['video', 'image']] = result['filename'].str.split('_', n=1, expand=True)

    return result

def create_and_save_full_set():
    splits_to_merge = ["split_0.csv", "split_1.csv"]

    data = create_full_set(splits_to_merge)
    data.to_csv("../images/full_dataset.csv")

    return data

"""
Return missing files list of (file_name, folder_name)
"""
def get_missing_files(df) -> (str, str):
    uniques = df["filename"].unique()
    pattern = "../images/all/*/*"
    files = []
    for img in glob.glob(pattern):
        if img.split('\\')[-1] not in uniques:
            files.append((img.split('\\')[-1], img.split('\\')[-2]))

    print(files)
    return files
def add_missing_files_to_df(df, missing_files):
    print(len(df))
    for file_name, folder_name in missing_files:
        data = [len(df.index), file_name, class_dict[folder_name], file_name.split("_")[0], file_name.split("_")[1]]
        df.loc[len(df.index)] = data

    print(len(df))
    df.to_csv("../images/full_dataset_mod.csv")


def print_nested_dict(dictionary: dict, title: str):
    for d in dictionary:
        print("\n\n", title, ":", d)
        print("{:<20} {:<15}".format('Label', 'Count'))
        for k, item in dictionary[d].items():
            print("{:<20} {:<15}".format(k, item))

def save_video_dict():
    classes = dataset["label"].unique()
    video_dict = {}

    for video, group_df in dataset.groupby("video", as_index=False):
        current_dict = dict.fromkeys(classes, 0)
        for label in classes:
            if label in group_df["label"].unique():
                count = group_df["label"].value_counts()[label]
                current_dict[label] += count
        video_dict[video] = current_dict

    pd.DataFrame.from_dict(video_dict).to_csv("../../csvs/video_dict.csv")
    return video_dict

class_dict = {
    "blood_hematin": "Blood",
    "ampulla_of_vater": "Ampulla",
    "polyp": "Polyp"
}

if __name__ == '__main__' :
    # When running for the first time create dataset
    # dataset = create_full_set()
    dataset = pd.read_csv("../images/full_dataset_mod.csv")
    # Add missing files if necessary
    # missing_files = get_missing_files(dataset)
    # add_missing_files_to_df(dataset, missing_files)