from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense,
    DepthwiseConv2D, Flatten, GlobalAveragePooling2D,
    Input, MaxPooling2D, SeparableConv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def build_mini_xception(input_shape=(64, 64, 1), num_classes=7, l2_reg=0.01):
    """
    Mini-XCEPTION architecture for FER-2013 emotion classification.
    Reference: Arriaga et al. (2017) - Real-time Convolutional Neural Networks
    for Emotion and Gender Classification
    """
    inputs = Input(input_shape)

    # base
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(l2_reg), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(16, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = x + residual

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(32, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = x + residual

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = x + residual

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = x + residual

    x = Conv2D(num_classes, (3, 3), padding="same", kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation("softmax")(x)

    return Model(inputs, outputs, name="mini_XCEPTION")
