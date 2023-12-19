'''
prepares classification dataset maas_custom
enforces class balance by setting a maximum of images per class so the rare classes are not overwhelmed
creates train val splits
'''

import os
import shutil
import numpy as np
import math
import argparse
import random
import csv




# for sorting list
def takeSecond(elem):
    return elem[1]


def copy_image_folder(source, dest, class_folder, thresh):

    print('class: ' + class_folder)

    # make folder at dest if necessary
    try: 
        os.mkdir(os.path.join(dest, class_folder))
        print('created directory at destination')
    except:
        pass

    img_list = os.listdir(os.path.join(source, class_folder))
    img_count = len(img_list)

    # thresh is maximum number of images per class to limit class imbalance
    if img_count >= thresh:
        # only copy images with higher resolution since they are assumed to be most distinctive
        img_list_sizes = []
        counter = 0
        for img in img_list:
            if '.csv' in img or '.txt' in img: continue
            img_src_path = os.path.join(source, class_folder, img)
            cv_img = cv2.imread(img_src_path)
            img_area = cv_img.shape[0] * cv_img.shape[1]
            img_list_sizes.append([img, img_area])

            counter += 1
            if counter >= 10 * thresh:
                break
        
        # natural sorting to get same order as in folder
        img_list_sizes.sort(key=takeSecond)

        counter = 0
        for img in img_list_sizes:
            img_src_path = os.path.join(source, class_folder, img[0])
            img_dst_path = os.path.join(dest, class_folder, img[0])
            shutil.copyfile(img_src_path, img_dst_path)
            counter += 1

            if counter > thresh:
                break
        
        print('copied ' + str(counter) + ' images')            
    else:
        #copy everything
        counter = 0
        for img in img_list:
            img_src_path = os.path.join(source, class_folder, img)
            img_dst_path = os.path.join(dest, class_folder, img)
            shutil.copyfile(img_src_path, img_dst_path)
            counter += 1
        print('copied ' + str(counter) + ' images')


#copy all data to custom dataset
def copy_images_all(source, dest, thresh):

    print('Copying images to ' + dest + ' from: '+ source)

    # check existance of source folder
    if not os.path.isdir(source):
        raise Exception('Error. Source folder not found: ' + source)
    
    # make destination folder if necessary
    try:
        os.mkdir(dest)
        print('created destination folder')
    except:
        pass
    
    # get list of classes the classifier should use

    class_list = os.listdir(source)
    for class_folder in class_list:

        # copy images from each folder
        #if class_folder in maas_custom_cls:

        copy_image_folder(source, dest, class_folder, thresh)

    print('Completed copying images\n')




# data does not have predefined splits, create them
def create_splits(src_dataset, src_trainset):
    
    print('Splitting images into train and val')

    cls_folder_list = os.listdir(os.path.join(src_dataset))

    # create new dirs at destination if necessary
    for split in ['split_0', 'split_1']:
        try:
            os.mkdir( os.path.join(src_trainset, split) )
            print('created directory')
        except:
            pass

        for cls_folder in cls_folder_list:
            try:
                os.mkdir(os.path.join(src_trainset, split, cls_folder))
                print('created directory')
            except:
                pass


    img_list = []
    for cls_folder in cls_folder_list: 
        img_list.append(os.listdir(os.path.join(src_dataset, cls_folder)))


    for split in ['split_0', 'split_1']:
        print(split)
        # read split.csv data
        with open(src_trainset + "/" + split + ".csv", newline='') as csvfile:
            r = csv.reader(csvfile, delimiter=',')
            first = True
            for row in r:
                if first: 
                    first = False
                    continue    
                img_name = row[0]
                found = False
                for cls_i, cls_folder in enumerate(cls_folder_list): 
                    if row[0] in img_list[cls_i]:
                        #print(img_name, "is in", cls_folder)
                        if found: 
                            print("Warning: file duplicated", img_name)
                        found = True

                        img_path_src = os.path.join(src_dataset, cls_folder, img_name)
                        if not os.path.isfile(img_path_src):
                            print('Warning: not a file or does not exist: ' + img_path_src )
                            print('if file is duplicated: no problem')
                            continue

                        dest_filepath = os.path.join(src_trainset, split, cls_folder, img_name)
                        shutil.move(img_path_src, dest_filepath)
                if not found:
                    print("Warning: file from csv not found", img_name)


    print('Finished splitting images into train and val')



if __name__ == '__main__' :

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', type=int, default=100000, help='maximum number of images per class')
    parser.add_argument('--images-src', type=str, default='/home/simon/dlmi_project/pytorch-image-models/images', \
        help='path to directory where images are in all')
    parser.add_argument('--datadir', type=str, default="all")

    args = parser.parse_args()
    print(args)

    # balance dataset
    copy_images_all(args.images_src + "/" + args.datadir, args.images_src+"/tmp", args.thresh)

    create_splits(args.images_src+"/tmp", args.images_src)

