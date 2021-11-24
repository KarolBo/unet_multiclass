from glob import glob
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_unet(input_shape, num_of_classes):
    num_filters = [96, 128, 160, 192]
    inputs = Input((input_shape[0], input_shape[1], 1))  # grayscale

    skip_x = []
    x = inputs

    # Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)
        x = Dropout(0.25)(x)

    # Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()

    # Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = Dropout(0.33)(x)
        x = conv_block(x, f)

    # Output
    x = Dropout(0.33)(x)
    x = Conv2D(num_of_classes, (1, 1), padding="same")(x)
    x = Activation(activation="sigmoid")(x)

    return Model(inputs, x)
