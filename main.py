from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
from keras.models import load_model

class Ui_Khuon_Mat_Cam_Xuc(object):
    def setupUi(self, Khuon_Mat_Cam_Xuc):
        Khuon_Mat_Cam_Xuc.setObjectName("Khuon_Mat_Cam_Xuc")
        Khuon_Mat_Cam_Xuc.resize(677, 484)
        self.centralwidget = QtWidgets.QWidget(parent=Khuon_Mat_Cam_Xuc)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 20, 561, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.bt_UL = QtWidgets.QPushButton(parent=self.centralwidget)
        self.bt_UL.setGeometry(QtCore.QRect(130, 100, 93, 28))
        self.bt_UL.setObjectName("bt_UL")
        self.bt_WC = QtWidgets.QPushButton(parent=self.centralwidget)
        self.bt_WC.setGeometry(QtCore.QRect(480, 100, 131, 28))
        self.bt_WC.setObjectName("bt_WC")
        self.anh_goc = QtWidgets.QListView(parent=self.centralwidget)
        self.anh_goc.setGeometry(QtCore.QRect(30, 150, 361, 271))
        self.anh_goc.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.anh_goc.setObjectName("anh_goc")
        self.anh_mat = QtWidgets.QListView(parent=self.centralwidget)
        self.anh_mat.setGeometry(QtCore.QRect(470, 180, 151, 111))
        self.anh_mat.setObjectName("anh_mat")
        self.cam_xuc = QtWidgets.QListView(parent=self.centralwidget)
        self.cam_xuc.setGeometry(QtCore.QRect(440, 350, 201, 31))
        self.cam_xuc.setObjectName("cam_xuc")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 160, 171, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 330, 151, 16))
        self.label_3.setObjectName("label_3")
        Khuon_Mat_Cam_Xuc.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=Khuon_Mat_Cam_Xuc)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 677, 26))
        self.menubar.setObjectName("menubar")
        Khuon_Mat_Cam_Xuc.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=Khuon_Mat_Cam_Xuc)
        self.statusbar.setObjectName("statusbar")
        Khuon_Mat_Cam_Xuc.setStatusBar(self.statusbar)

        self.retranslateUi(Khuon_Mat_Cam_Xuc)
        QtCore.QMetaObject.connectSlotsByName(Khuon_Mat_Cam_Xuc)

        self.bt_UL.clicked.connect(self.load_image)  # Kết nối nút bt_UL với hàm load_image

        # Tải mô hình cảm xúc đã huấn luyện
        self.emotion_model = load_model('cnn_emotion_recognition_model.h5')

    def retranslateUi(self, Khuon_Mat_Cam_Xuc):
        _translate = QtCore.QCoreApplication.translate
        Khuon_Mat_Cam_Xuc.setWindowTitle(_translate("Khuon_Mat_Cam_Xuc", "Nhận diện khuôn mặt và cảm xúc"))
        self.label.setText(_translate("Khuon_Mat_Cam_Xuc", "Nhận diện cảm xúc từ khuôn mặt"))
        self.bt_UL.setText(_translate("Khuon_Mat_Cam_Xuc", "Tải ảnh lên"))
        self.bt_WC.setText(_translate("Khuon_Mat_Cam_Xuc", "Sử dụng webcam"))
        self.label_2.setText(_translate("Khuon_Mat_Cam_Xuc", "Khuôn mặt nhận diện được"))
        self.label_3.setText(_translate("Khuon_Mat_Cam_Xuc", "Cảm xúc nhận diện được"))

    def display_image(self, image_path, view):
        pixmap = QPixmap(image_path)
        # Scale the pixmap to fit the view's size, keeping the aspect ratio
        pixmap = pixmap.scaled(view.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                               QtCore.Qt.TransformationMode.SmoothTransformation)

        view.setModel(QtGui.QStandardItemModel())  # Reset display model
        item = QtGui.QStandardItem()
        item.setData(pixmap, QtCore.Qt.ItemDataRole.DecorationRole)
        view.model().appendRow(item)

    def load_image(self):
        try:
            # Chọn file ảnh từ hệ thống
            file_name, _ = QFileDialog.getOpenFileName(None, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg)")
            if not file_name:
                return  # Nếu không chọn ảnh thì thoát

            # Đọc ảnh bằng OpenCV
            image = cv2.imread(file_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Phát hiện khuôn mặt
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                raise ValueError("Không phát hiện thấy khuôn mặt trong ảnh!")

            # Chọn khuôn mặt đầu tiên phát hiện được
            (x, y, w, h) = faces[0]
            face = image[y:y + h, x:x + w]

            # Hiển thị ảnh gốc lên giao diện
            self.display_image(file_name, self.anh_goc)

            # Cắt khuôn mặt và hiển thị nó
            face_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_image_qt = QImage(face_image.data, face_image.shape[1], face_image.shape[0], face_image.strides[0],
                                   QImage.Format.Format_RGB888)

            pixmap = QPixmap.fromImage(face_image_qt)
            pixmap = pixmap.scaled(self.anh_mat.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                   QtCore.Qt.TransformationMode.SmoothTransformation)

            self.anh_mat.setModel(QtGui.QStandardItemModel())  # Reset display model
            item = QtGui.QStandardItem()
            item.setData(pixmap, QtCore.Qt.ItemDataRole.DecorationRole)  # Thêm hình ảnh vào item
            self.anh_mat.model().appendRow(item)

            # Dự đoán cảm xúc
            face_resized = cv2.resize(face, (64, 64))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            face_gray = np.expand_dims(face_gray, axis=-1)  # Mở rộng để thành (64, 64, 1)
            face_gray = np.expand_dims(face_gray, axis=0)  # Mở rộng để thành (1, 64, 64, 1)
            face_gray = face_gray / 255.0  # Chuẩn hóa

            emotion_prediction = self.emotion_model.predict(face_gray)
            emotion_class = np.argmax(emotion_prediction)

            # Cập nhật label cảm xúc
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            emotion = emotions[emotion_class]
            self.cam_xuc.setModel(QtGui.QStandardItemModel())  # Reset display model
            self.cam_xuc.model().appendRow(QtGui.QStandardItem(emotion))

        except Exception as e:
            # Hiển thị thông báo lỗi
            QMessageBox.critical(None, "Lỗi", str(e))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Khuon_Mat_Cam_Xuc = QtWidgets.QMainWindow()
    ui = Ui_Khuon_Mat_Cam_Xuc()
    ui.setupUi(Khuon_Mat_Cam_Xuc)
    Khuon_Mat_Cam_Xuc.show()
    sys.exit(app.exec())
