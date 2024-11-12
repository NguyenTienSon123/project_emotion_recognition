# test_cnn.py
from cnn import create_cnn  # Import lại mô hình CNN
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# Các tham số cấu hình
input_shape = (64, 64, 1)
num_classes = 7

# Tải mô hình đã huấn luyện
model = load_model('cnn_emotion_recognition_model.h5')

# Chuẩn bị dữ liệu kiểm tra
test_data_dir = "archive/train"

datagen_test = ImageDataGenerator(rescale=1./255)

dataset_test = datagen_test.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False  # Đảm bảo không xáo trộn dữ liệu khi dự đoán
)

# Dự đoán với dữ liệu kiểm tra
predictions = model.predict(dataset_test, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Nhãn thực tế từ dữ liệu kiểm tra
true_classes = dataset_test.classes

# Đánh giá mô hình
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# Thống kê các chỉ số khác
report = classification_report(true_classes, predicted_classes, target_names=dataset_test.class_indices.keys())
print("Báo cáo phân loại:\n", report)
