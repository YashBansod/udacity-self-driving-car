#!/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
import csv
# noinspection PyPackageRequirements
from tqdm import tqdm
# noinspection PyPackageRequirements
import numpy as np
import matplotlib.image as mpimg
from glob import glob


# ******************************************    Func Declaration Start      ****************************************** #
def csv_data_reader(csv_file_path_list):

    data_csv_lines = []

    for csv_file_path in csv_file_path_list:
        dir_path = csv_file_path[:csv_file_path.find("driving_log.csv")]
        temp_lines = []
        with open(csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            for _line in reader:
                _line[0] = dir_path + _line[0][_line[0].find("IMG/"):]
                _line[1] = dir_path + _line[1][_line[1].find("IMG/"):]
                _line[2] = dir_path + _line[2][_line[2].find("IMG/"):]
                temp_lines.append(_line)
        data_csv_lines.extend(temp_lines)

    return data_csv_lines

# ******************************************    Func Declaration End        ****************************************** #


# ******************************************    Func Declaration Start      ****************************************** #
def dataset_reader(data_csv_lines):
    num_data_samples = len(data_csv_lines)
    features_array = np.empty(shape=(num_data_samples * 4, 160, 320, 3), dtype=np.uint8)
    labels_array = np.empty(shape=num_data_samples * 4, dtype=np.float32)

    prog_bar = tqdm(desc="Reading dataset", total=num_data_samples * 4, unit=" img")
    for index in range(num_data_samples):
        line = data_csv_lines[index]
        steering_correction = 0.12

        center_img_path = line[0]
        center_measurement = float(line[3])

        left_img_path = line[1]
        left_measurement = float(line[3]) + steering_correction

        right_img_path = line[2]
        right_measurement = float(line[3]) - steering_correction

        center_image = mpimg.imread(center_img_path)
        left_image = mpimg.imread(left_img_path)
        right_image = mpimg.imread(right_img_path)

        features_array[index] = center_image
        features_array[num_data_samples + index] = np.fliplr(center_image)
        features_array[num_data_samples * 2 + index] = left_image
        features_array[num_data_samples * 3 + index] = right_image

        labels_array[index] = center_measurement
        labels_array[num_data_samples + index] = -center_measurement
        labels_array[num_data_samples * 2 + index] = left_measurement
        labels_array[num_data_samples * 3 + index] = right_measurement

        prog_bar.update(n=4)
    prog_bar.close()

    rand_indices = np.arange(num_data_samples * 4)
    np.random.shuffle(rand_indices)
    features_array = features_array[rand_indices]
    labels_array = labels_array[rand_indices]

    return features_array, labels_array

# ******************************************    Func Declaration Start      ****************************************** #


# ******************************************        Main Program Start      ****************************************** #
def main():
    csv_lines = csv_data_reader(csv_file_path_list=glob("./SimRec_?/*Rec*/driving_log.csv"))
    features, labels = dataset_reader(data_csv_lines=csv_lines)

    np.save('c_l_r_features.npy', features)
    np.save('c_l_r_labels.npy', labels)

# ******************************************        Main Program End        ****************************************** #


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProcess interrupted by user. Bye!')

"""
Author: Yash Bansod
Project: CarND-Behavioral-Cloning
"""
