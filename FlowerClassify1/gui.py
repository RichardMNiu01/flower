import sys
import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout
)

from PyQt5.QtGui import (
    QPixmap,
    QImage
)

from PyQt5.QtCore import (
    QTimer
)


from torchvision.models import resnet18


# =====================
# 模型加载
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classes = []

with open("classes.txt","r") as f:
    classes = [
        x.strip()
        for x in f.readlines()
    ]


model = resnet18()

model.fc = torch.nn.Linear(
    512,
    len(classes)
)


model.load_state_dict(
    torch.load(
        "checkpoints/best-ckpt2.pt",
        map_location=device
    )
)


model.to(device)

model.eval()



# =====================
# 图片预处理
# =====================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])



# =====================
# 预测函数
# =====================

def predict(img):

    img = Image.fromarray(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )


    img = transform(img)

    img = img.unsqueeze(0)


    img = img.to(device)


    with torch.no_grad():

        output = model(img)


        prob = torch.softmax(
            output,
            dim=1
        )


        confidence, index = torch.max(
            prob,
            1
        )


    return (
        classes[index.item()],
        confidence.item()
    )



# =====================
# GUI
# =====================

class FlowerGUI(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Flower Recognition System"
        )

        self.resize(
            900,
            700
        )


        self.image_label = QLabel()

        self.result_label = QLabel(
            "Result:"
        )


        self.btn_image = QPushButton(
            "选择图片"
        )


        self.btn_camera = QPushButton(
            "打开摄像头"
        )


        self.btn_stop = QPushButton(
            "关闭摄像头"
        )


        layout = QVBoxLayout()


        layout.addWidget(
            self.image_label
        )


        layout.addWidget(
            self.result_label
        )


        button_layout = QHBoxLayout()


        button_layout.addWidget(
            self.btn_image
        )


        button_layout.addWidget(
            self.btn_camera
        )


        button_layout.addWidget(
            self.btn_stop
        )


        layout.addLayout(
            button_layout
        )


        self.setLayout(layout)



        # 摄像头
        self.camera = None


        self.timer = QTimer()


        self.timer.timeout.connect(
            self.camera_detect
        )


        # 按钮绑定

        self.btn_image.clicked.connect(
            self.open_image
        )


        self.btn_camera.clicked.connect(
            self.open_camera
        )


        self.btn_stop.clicked.connect(
            self.stop_camera
        )



    # =====================
    # 图片检测
    # =====================

    def open_image(self):

        path,_ = QFileDialog.getOpenFileName(

            self,

            "选择图片",

            "",

            "Images (*.jpg *.png)"

        )


        if path:

            img=cv2.imread(path)


            label,score=predict(img)


            self.result_label.setText(

                f"{label}  {score:.2%}"

            )


            self.show_image(img)



    # =====================
    # 摄像头
    # =====================

    def open_camera(self):

        self.camera=cv2.VideoCapture(0)


        self.timer.start(
            30
        )



    def camera_detect(self):

        ret,frame=self.camera.read()


        if ret:

            label,score=predict(frame)


            cv2.putText(

                frame,

                f"{label}:{score:.2%}",

                (30,50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                (0,255,0),

                2

            )


            self.show_image(frame)



    def stop_camera(self):

        if self.camera:

            self.timer.stop()

            self.camera.release()

            self.camera=None



    # =====================
    # 显示图片
    # =====================

    def show_image(self,img):

        img=cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )


        h,w,c=img.shape


        qimg=QImage(

            img.data,

            w,

            h,

            w*c,

            QImage.Format_RGB888

        )


        pixmap=QPixmap.fromImage(qimg)


        self.image_label.setPixmap(

            pixmap.scaled(

                self.image_label.size(),

                aspectRatioMode=1

            )

        )



# =====================
# 启动
# =====================

if __name__=="__main__":

    app=QApplication(sys.argv)


    window=FlowerGUI()


    window.show()


    sys.exit(
        app.exec_()
    )