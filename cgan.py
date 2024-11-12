import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class cGAN:
    def __init__(self, img_shape, num_classes, noise_dim):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        noise = layers.Input(shape=(self.noise_dim,))
        label = layers.Input(shape=(1,))

        label_embedding = layers.Embedding(self.num_classes, self.noise_dim)(label)
        input = layers.multiply([noise, label_embedding])

        x = layers.Dense(1024)(input)  # Thay đổi kích thước thành 1024
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((8, 8, 16))(x)  # Chuyển thành kích thước 8x8x16

        # Tiếp tục thêm các lớp để hoàn thành bộ Generator
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

        return tf.keras.Model([noise, label], x)

    def build_discriminator(self):
        img = layers.Input(shape=self.img_shape)
        label = layers.Input(shape=(1,))

        label_embedding = layers.Embedding(self.num_classes, np.prod(self.img_shape))(label)
        label_embedding = layers.Reshape(self.img_shape)(label_embedding)

        x = layers.concatenate([img, label_embedding])
        x = layers.Conv2D(64, kernel_size=3, strides=2)(x)
        x = layers.LeakyReLU()(x)

        return tf.keras.Model([img, label], x)
