import tensorflow as tf
from cnn import create_cnn  # Import hàm tạo mô hình CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Đường dẫn tới các dữ liệu
train_data_dir = "archive/train"  # Dữ liệu gốc
generated_images_dir = "generated_images"  # Dữ liệu ảnh sinh ra từ GAN

# Tạo ImageDataGenerator để xử lý ảnh
datagen_train = ImageDataGenerator(rescale=1./255)

# Kết hợp dữ liệu gốc và ảnh giả tạo từ GAN
train_dataset_original = datagen_train.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),  # Kích thước ảnh đầu vào
    color_mode="grayscale",  # Ảnh là grayscale
    class_mode="categorical",  # Nhãn là one-hot encoding
    batch_size=32
)

# Tải dữ liệu từ thư mục generated_images đã được chia thành các nhãn
train_dataset_generated = datagen_train.flow_from_directory(
    generated_images_dir,
    target_size=(64, 64),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32
)

# Hàm chuyển dữ liệu từ generator thành tf.data.Dataset
def generator_to_dataset(generator):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, 7), dtype=tf.float32))
    )
    return dataset

# Chuyển các generator thành tf.data.Dataset
dataset_original = generator_to_dataset(train_dataset_original)
dataset_generated = generator_to_dataset(train_dataset_generated)

# Kết hợp cả hai dataset
combined_dataset = dataset_original.concatenate(dataset_generated)

# Tạo mô hình CNN
input_shape = (64, 64, 1)  # Kích thước ảnh đầu vào
num_classes = 7  # 7 loại cảm xúc

model = create_cnn(input_shape, num_classes)

# Huấn luyện mô hình
model.fit(
    combined_dataset,
    epochs=10,
    steps_per_epoch=200
)

# Lưu mô hình sau khi huấn luyện
model.save('cnn_emotion_recognition_model.h5')
