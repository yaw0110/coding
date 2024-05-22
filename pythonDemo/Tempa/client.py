import sys
import json
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer

class DataGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_data)
        self.file = None

    def start(self):
        if not self.file:
            self.file = open(self.filename, 'w')
        self.timer.start(1000)  # 每秒发送一次数据

    def stop(self):
        if self.file:
            self.timer.stop()
            self.file.close()
            self.file = None
            print("File closed")

    def generate_data(self):
        if self.file and not self.file.closed:
            data = {
                'temperature': round(random.uniform(20.0, 30.0), 2),
                'humidity': round(random.uniform(50.0, 80.0), 2),
                'location': 'Beijing',
                'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            json.dump(data, self.file)
            self.file.write('\n')  # 写入换行符，以便每行数据单独占一行
            self.file.flush()  # 立即将数据写入文件
            print(data)  # 发送到终端上
        else:
            print("File not open")


    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.file_name = filename
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_data)
        self.file_open = True

    def start(self):
        self.timer.start(1000)  # 每秒发送一次数据

    def stop(self):
        self.timer.stop()
        if self.file_open:
            self.file.close()
            self.file_open = False
            print("File closed")

    def restart(self):
        if not self.file_open:
            self.file = open(self.file_name, 'w')
            self.file_open = True
            print("File reopened")

    def generate_data(self):
        if self.file_open:
            data = {
                'temperature': round(random.uniform(20.0, 30.0), 2),
                'humidity': round(random.uniform(50.0, 80.0), 2),
                'location': 'Beijing',
                'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            json.dump(data, self.file)
            self.file.write('\n')  # 写入换行符，以便每行数据单独占一行
            self.file.flush()  # 立即将数据写入文件
            print(data)  # 发送到终端上
        else:
            print("File not open")


    def __init__(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_data)
        self.file = open('data.json', 'w')

    def start(self):
        self.timer.start(1000)  # 每秒发送一次数据

    def stop(self):
        self.timer.stop()
        self.file.close()

    def generate_data(self):
        data = {
            'temperature': random.uniform(20.0, 30.0),
            'humidity': random.uniform(50.0, 80.0),
            'location': 'Beijing',
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        json.dump(data, self.file)
        print(data)  # 发送到终端上

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.button = QPushButton('Start')
        self.button.clicked.connect(self.on_button_click)

        layout = QVBoxLayout()
        layout.addWidget(self.button)

        self.setLayout(layout)

        self.generator = DataGenerator()

    def on_button_click(self):
        if self.generator.timer.isActive():
            self.generator.stop()
            self.button.setText('Start')
        else:
            self.generator.start()
            self.button.setText('Stop')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
