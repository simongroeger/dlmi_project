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

import cv2
from numpy.lib import index_tricks


def shear_image(image, o_shear_x, o_shear_y):
    shear_x = abs(o_shear_x)
    shear_y = abs(o_shear_y)
    M = np.float32([[1, shear_x, 0],
                    [shear_y, 1  , 0],
                    [0, 0  , 1]])
    rows_prev, cols_prev, dim = image.shape
    rows = rows_prev + shear_y*cols_prev
    cols = cols_prev + shear_x*rows_prev
    
    img = image
    if o_shear_x< 0: img =  cv2.flip(img, 0)      
    if o_shear_y< 0: img =  cv2.flip(img, 1)      
    img = cv2.warpPerspective(img,M,(int(cols),int(rows)), borderMode=cv2.BORDER_REPLICATE)
    if o_shear_x< 0: img =  cv2.flip(img, 0)      
    if o_shear_y< 0: img =  cv2.flip(img, 1) 
    return img



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result



def transform(path, degrees, shears):
    img = cv2.imread(path)
    count = 0
    for degree in degrees:
        cv2.imwrite(path[:-4] + "_rot_" + str(degree) + ".png", rotate_image(img, degree))
        count += 1
    
    for shear_x in shears:
        for shear_y in shears:
            if shear_x != 0 or shear_y != 0:
                cv2.imwrite(path[:-4] + "_shear_"+("%.2f" % round(shear_x, 2))+ "_" + ("%.2f" % round(shear_y, 2)) + ".png", shear_image(img, shear_x, shear_y))
                count += 1

    return count



def fillup_splitted_folder(dir, class_folder, thresh):
    img_list = os.listdir(os.path.join(dir, class_folder))
    img_count = len(img_list)
    amount =  thresh - img_count
    print(class_folder, "got", img_count, "needs", amount)

    i = 0
    for img in img_list:
        additional = transform(os.path.join(dir, class_folder, img), degrees=[-2, 2], shears=[-0.05, 0, 0.05])
        i+=additional
        if i > amount:
            return
    for img in img_list:
        additional = transform(os.path.join(dir, class_folder, img), degrees=[-10, -5, 5, 10], shears=[-0.1, 0, 0.1])
        i += additional
        if i > amount:
            return
       


def fillup_splitted(dir):
    class_list = os.listdir(dir)
    print("fillup", len(class_list))

    img_count = []
    for class_folder in class_list:
        img_count.append(len(os.listdir(os.path.join(dir, class_folder))))

    min_class_i = img_count.index(min(img_count))

    fillup_splitted_folder(dir, class_list[min_class_i], max(img_count))

    print("finished filling up")





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
def copy_images_all(source, dest, thresh=1000000):

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
def create_splits(src_dataset, src_trainset, thresh, src_csv):
    
    print('Splitting images into train and val')

    cls_folder_list = os.listdir(os.path.join(src_dataset))
    target_class_list = ['blood', 'other']
    splits = ['train_split', 'val_split', 'test_split']

    # create new dirs at destination if necessary
    for split in splits:
        try:
            os.mkdir( os.path.join(src_trainset, split) )
            print('created directory')
        except:
            pass

        for cls_folder in target_class_list:
            try:
                os.mkdir(os.path.join(src_trainset, split, cls_folder))
                print('created directory')
            except:
                pass

    class_count = np.zeros((len(splits), len(cls_folder_list)))

    img_list = []
    for cls_folder in cls_folder_list: 
        img_list.append(os.listdir(os.path.join(src_dataset, cls_folder)))

    # split according to csv
    for split_i, split in enumerate(splits):
        print(split)
        # read split.csv data
        with open(src_csv + "/" + split + ".csv", newline='') as csvfile:
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

                        if class_count[split_i, cls_i] > thresh and split == train_split:
                            os.remove(img_path_src)
                            continue
                        else:
                            class_count[split_i, cls_i] += 1

                        target_class = 'blood' if 'blood' in cls_folder else 'other'
                        dest_filepath = os.path.join(src_trainset, split, target_class, img_name)
                        shutil.move(img_path_src, dest_filepath)
                if not found:
                    print("Warning: file from csv not found", img_name)


    print('Finished splitting images into train, val and test')



if __name__ == '__main__' :

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', type=int, default=100000, help='maximum number of images per class')
    parser.add_argument('--images-src', type=str, default='pytorch-image-models/images',
        help='path to directory where images are in all')
    parser.add_argument('--datadir', type=str, default="all")

    args = parser.parse_args()
    print(args)

    train_split = "train_split"

    # balance dataset
    copy_images_all(args.images_src + "/" + args.datadir, args.images_src+"/tmp")

    create_splits(args.images_src+"/tmp", args.images_src, args.thresh, "csvs/splits_by_video")

    fillup_splitted(args.images_src+"/"+train_split)


