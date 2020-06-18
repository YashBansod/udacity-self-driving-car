#!/usr/bin/env python

# ******************************************    Libraries to be imported    ****************************************** #
# noinspection PyPackageRequirements
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, LeakyReLU, MaxPooling2D, Cropping2D, Dropout


# ******************************************        Main Program Start      ****************************************** #
def main():
    features = np.load('c_l_r_features.npy')
    labels = np.load('c_l_r_labels.npy')

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(16, kernel_size=3))
    model.add(LeakyReLU())
    model.add(Conv2D(24, kernel_size=3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    model.add(Conv2D(48, kernel_size=3))
    model.add(LeakyReLU())
    model.add(Conv2D(64, kernel_size=3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    history_object = model.fit(x=features, y=labels, batch_size=64, shuffle=True, validation_split=0.1, epochs=3)

    print(history_object.history.keys())

    plt.figure(figsize=(10, 10))
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('training_loss_viz.png')
    plt.show()

    model.save('model.h5')

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
