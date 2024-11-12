import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cgan import cGAN

# Các thông số cài đặt cho mô hình
IMG_SHAPE = (64, 64, 1)  # Đảm bảo định dạng ảnh phù hợp với FER2013
NUM_CLASSES = 7  # Tổng số nhãn cảm xúc
NOISE_DIM = 100  # Kích thước đầu vào nhiễu

# Khởi tạo mô hình cGAN
gan = cGAN(IMG_SHAPE, NUM_CLASSES, NOISE_DIM)

# Chuẩn bị dữ liệu huấn luyện
data_dir = "archive/train"  # Đường dẫn đến thư mục train trong FER2013

datagen = ImageDataGenerator(rescale=1. / 255)  # Chuẩn bị data generator để chuẩn hóa ảnh
dataset = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SHAPE[:2],
    color_mode="grayscale",
    class_mode="sparse",  # Sử dụng "sparse" để nhận nhãn dưới dạng số nguyên
    batch_size=32
)

# Tạo thư mục để lưu ảnh giả (nếu chưa có)
save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)

# Tạo thư mục con cho từng nhãn cảm xúc
for label in range(NUM_CLASSES):
    label_dir = os.path.join(save_dir, f"label_{label}")
    os.makedirs(label_dir, exist_ok=True)

# Hàm huấn luyện GAN và lưu ảnh
def train_gan(gan, dataset, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Lấy một batch ảnh và nhãn
        real_imgs, labels = next(dataset)

        # Sinh ảnh giả từ Generator
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        gen_labels = np.random.randint(0, NUM_CLASSES, batch_size)
        fake_imgs = gan.generator.predict([noise, gen_labels])

        # Huấn luyện mô hình
        # (Bước huấn luyện mô hình discriminator và generator...)

        # Lưu ảnh giả sau mỗi epoch
        if epoch % 100 == 0:
            for i, fake_img in enumerate(fake_imgs):
                label = gen_labels[i]  # Nhãn của ảnh giả
                fake_img = (fake_img + 1) / 2.0  # Giả sử ảnh đầu ra là [-1, 1], chuẩn hóa lại về [0, 1]

                # Lưu ảnh vào thư mục tương ứng với nhãn
                label_dir = os.path.join(save_dir, f"label_{label}")
                save_path = os.path.join(label_dir, f"fake_img_epoch_{epoch}_idx_{i}.png")
                plt.imsave(save_path, fake_img.squeeze(), cmap='gray')

            print(f"Ảnh giả cho epoch {epoch} đã được lưu vào các thư mục theo nhãn.")

# Gọi hàm huấn luyện
train_gan(gan, dataset)
