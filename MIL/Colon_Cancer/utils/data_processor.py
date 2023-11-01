import glob
import random

import imageio
import numpy as np


def generate_batch(path):
    bags = []
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.bmp')
        num_ins = len(img_path)

        label = int(each_path.split('/')[-2])

        if label == 1:
            curr_label = np.ones(num_ins,dtype=np.uint8)
        else:
            curr_label = np.zeros(num_ins, dtype=np.uint8)
        for each_img in img_path:
            img_data = np.asarray(imageio.imread(each_img), dtype=np.float32)
            #img_data -= 255
            img_data[:, :, 0] -= 123.68
            img_data[:, :, 1] -= 116.779
            img_data[:, :, 2] -= 103.939
            img_data /= 255
            # sci.imshow(img_data)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))

    return bags


def Get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val