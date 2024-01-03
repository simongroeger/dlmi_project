'''
prepares classification dataset maas_custom
enforces class balance by setting a maximum of images per class so the rare classes are not overwhelmed
creates train val splits
'''

import os
import shutil
import cv2
import numpy as np
import math
import argparse
import random



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
       


def fillup_splitted(dir, thresh):
    class_list = os.listdir(dir)
    print(len(class_list))
    for class_folder in class_list:
        fillup_splitted_folder(dir, class_folder, thresh)
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


def fillup_other_with_random(tmppath):
    print('Starting Fillup with random')

    amout_cone_data = len(os.listdir(os.path.join(tmppath, "cone")))
    amout_other_data = len(os.listdir(os.path.join(tmppath, "other")))

    print('Generating '  + str(amout_cone_data - amout_other_data) +  ' random images')

    for i in range(amout_cone_data - amout_other_data):
        img_width = 36
        img_height = 27
        mat = np.zeros((img_width, img_height))

        circlesize = 1

        for j in range(random.randint(20, 150)):
            y = random.randint(2, img_width-3)
            x = random.randint(2, img_height-3)
            intensity = random.randint(50,255)

            cv2.circle(mat, (x, y), int(circlesize/2), intensity, -1)
        
        cv2.imwrite(os.path.join(tmppath, "other", "random_" + str(i) + ".png"), mat)

    print('Completed filling up')

    
def create_random_unknown(tmppath):
    print('Starting random unknown')

    try:
        os.mkdir(tmppath + "/unknown")
        print('created destination folder')
    except:
        pass

    amout_blue_data = len(os.listdir(os.path.join(tmppath, "blue")))

    print('Generating '  + str(amout_blue_data) +  ' random images')

    for i in range(amout_blue_data):
        img_width = 36
        img_height = 27
        mat = np.zeros((img_width, img_height))

        circlesize = 1

        for j in range(random.randint(20, 150)):
            y = random.randint(2, img_width-3)
            x = random.randint(2, img_height-3)
            intensity = random.randint(50,255)

            cv2.circle(mat, (x, y), int(circlesize/2), intensity, -1)
        
        cv2.imwrite(os.path.join(tmppath, "unknown", "random_" + str(i) + ".png"), mat)

    print('Completed filling up')


def clear_empty_cls(folder):

    print('Beginning to remove empty class folders.')

    counter = 0
    cls_folder_list = os.listdir(folder)
    for cls_folder in cls_folder_list:

        number_images = len(os.listdir(os.path.join(folder, cls_folder)))

        if number_images == 0:
            print(os.path.join(folder, cls_folder))
            counter += 1
            os.rmdir(os.path.join(folder, cls_folder))

    print('Finished removing ' + str(counter) + ' folders\n')


# data does not have predefined splits, create them
def create_splits(src_dataset, src_trainset, ratio_train):
    
    print('Splitting images into train and val')

    cls_folder_list = os.listdir(os.path.join(src_dataset))

    # create new dirs at destination if necessary
    for split in ['train', 'validation']:
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


    # start splitting files
    for cls_folder in cls_folder_list: 

        split = 'validation'
        img_list = os.listdir(os.path.join(src_dataset, cls_folder))

        print("split", len(img_list) * ratio_train)

        # move files with step given by ratio_train into val folder (i.e. every 5th image is val)
        for i in range(0, len(img_list), int(1/(1-ratio_train)) ):

            img_name = img_list[i]

            img_path_src = os.path.join(src_dataset, cls_folder, img_name)

            if not os.path.isfile(img_path_src):
                print('val: Warning, not a file or does not exist: ' + img_path_src )
                continue

            dest_filepath = os.path.join(src_trainset, split, cls_folder, img_name)

            shutil.move(img_path_src, dest_filepath)

        # move other files into train folder
        split = 'train'
        file_list = os.listdir(os.path.join(src_dataset, cls_folder))
        for img_name in file_list:

            img_path_src = os.path.join(src_dataset, cls_folder, img_name)

            if not os.path.isfile(img_path_src):
                print('train: Warning, not a file or does not exist: ' + img_path_src )
                continue

            dest_filepath = os.path.join(src_trainset, split, cls_folder, img_name)

            shutil.move(img_path_src, dest_filepath)

    print('Finished splitting images into train and val')



if __name__ == '__main__' :

    # user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', type=int, default=100000, help='maximum number of images per class')
    parser.add_argument('--images-src', type=str, default='.\images',
        help='path to directory where images are in all')
    parser.add_argument('--datadir', type=str, default="all")

    args = parser.parse_args()
    print(args)

    # some image folders may be empty
    clear_empty_cls(args.images_src + "\\" + args.datadir)

    # balance dataset
    copy_images_all(args.images_src + "\\" + args.datadir, args.images_src+ "\\tmp", args.thresh)

    #fillup_other_with_random(args.images_src+"\\tmp")
    # create_random_unknown(args.images_src+"\\tmp")

    # create data splits for all maas_custom folders
    create_splits(args.images_src+ "\\tmp", args.images_src, 0.8)

    #fillup_splitted(args.images_src+"\\train", args.thresh * 0.8)
    #fillup_splitted(args.images_src+"\\validation", args.thresh * 0.2)


