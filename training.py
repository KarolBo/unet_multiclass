from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.python.lib import io

from generators import MyGenerator
from helpers import validate_generator
from models import build_unet
from metrics import iou

import numpy as np
import matplotlib.pyplot as plt


def train(model, train_data, test_data, lr, epochs):
    opt = Adam(lr)
    metrics = ["acc", iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
    callbacks = [
        ModelCheckpoint("unet_checkpoint_model.h5",
                        monitor='val_loss', verbose=1, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    ]
    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              callbacks=callbacks)


def predict(model, img_path, input_size):
    img = load_img(img_path, grayscale=True)
    input_arr = img_to_array(img.resize(input_size))
    input_arr /= 255
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)

    return prediction


if __name__ == '__main__':
    # Macroparameters
    CLASSES = 3
    BATCH_SIZE = 1
    EPOCHS = 1
    IMG_SIZE = (256, 256)
    learning_rate = 1e-3
    dropout = 9 * [0.25]
    weight_zeros = 1
    weight_ones = 1
    aug_dict = {
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'zoom_range': 0.001,
        'horizontal_flip': True,
        'rotation_range': 30
    }

    train_generator = MyGenerator(BATCH_SIZE, 'data', 'images',
                                  ('lesions_malignant', 'microcalc_benign'),
                                  aug_dict, IMG_SIZE)
    # validate_generator(train_generator)
    # model = build_unet(IMG_SIZE, CLASSES)
    # train(model, train_generator, train_generator, learning_rate, EPOCHS)
    model = load_model('unet_checkpoint_model.h5', custom_objects={'iou': iou})
    prediction = predict(model, 'test.png', IMG_SIZE)
    print(prediction.shape, np.min(prediction), np.max(prediction))
    plt.imshow(prediction[0, :, :, :])
    plt.show()
