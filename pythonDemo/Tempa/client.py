import sys
import random
import socket
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QTextEdit
from PyQt5.QtCore import QTimer  # 导入 QTimer
import time
import os

class DataCollectorClient(QMainWindow):
    sensor_id = f'sensor_{random.randint(1, 100)}'

    def __init__(self, server_address):
        super().__init__()
        self.server_address = server_address
        self.log_file_path = 'pythonDemo/Tempa/logs/' + self.sensor_id + str(time.time())+ '.log'
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.heartbeat_timer = QTimer()  # 创建一个 QTimer 用于发送心跳
        self.initUI()
 # 检查文件是否存在
        if not os.path.exists(self.log_file_path):
            # 文件不存在，创建一个空文件并立即关闭
            with open(self.log_file_path, 'a'):
                pass
            print(f"文件'{self.log_file_path}'已创建。")
        else:
            print(f"文件'{self.log_file_path}'已存在。")
        self.log_file = open(self.log_file_path, 'a')  # Open log file in append mode
        self.connect_to_server()

    def initUI(self):
        self.setWindowTitle('Data Collector Client')
        self.setGeometry(100, 100, 400, 300)

        # Layout and widgets
        layout = QVBoxLayout()

        self.status_label = QLabel('Disconnected', self)
        layout.addWidget(self.status_label)

        self.data_log = QTextEdit(self)
        self.data_log.setReadOnly(True)
        layout.addWidget(self.data_log)

        btn_layout = QVBoxLayout()

        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_to_server)
        btn_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton('Disconnect')
        self.disconnect_button.clicked.connect(self.disconnect_from_server)
        btn_layout.addWidget(self.disconnect_button)

        self.send_data_button = QPushButton('Send Data')
        self.send_data_button.clicked.connect(self.send_data)
        btn_layout.addWidget(self.send_data_button)

        layout.addLayout(btn_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)  # 将定时器的信号连接到发送心跳消息的槽函数

    def connect_to_server(self):
        try:
            self.client_socket.connect(self.server_address)
            self.status_label.setText('Connected')
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.send_data_button.setEnabled(True)
            # 启动心跳定时器，每30秒发送一次心跳
            self.heartbeat_timer.start(30000)
        except socket.error as e:
            self.status_label.setText(f'Connection failed: {e}')


    def disconnect_from_server(self):
        self.heartbeat_timer.stop()  # 停止心跳定时器
        self.client_socket.close()
        self.status_label.setText('Disconnected')
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.send_data_button.setEnabled(False)
        socket.close()

    def send_data(self):
        # Generate random data
        temperature = random.uniform(20, 30)
        humidity = random.uniform(30, 50)
        data = {
            'type': 'data_report',
            'payload': {
                'sensor_id': self.sensor_id,
                'temperature': temperature,
                'humidity': humidity
            }
        }
        data_json = json.dumps(data)

        # Write data to log file
        self.log_file.write(data_json + '\n')
        self.log_file.flush()  # Ensure data is written to file immediately

        # Send data over the network
        try:
            self.client_socket.sendall(data_json.encode('utf-8'))
            self.data_log.append(f'Sent: {data_json}')
        except socket.error as e:
            self.data_log.append(f'Send failed: {e}')


    def send_heartbeat(self):
        heartbeat_msg = {
            'type': 'heartbeat',
            'payload': {
                'sensor_id': self.sensor_id
            }
        }
        try:
            self.client_socket.sendall(json.dumps(heartbeat_msg).encode('utf-8'))
            self.data_log.append('Sent heartbeat')  # 记录发送的心跳
        except socket.error as e:
            self.data_log.append(f'Heartbeat send failed: {e}')

    def closeEvent(self, event):
        # Close the log file when the client is closing
        self.log_file.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    server_address = ('127.0.0.1', 12345)  # Replace with your server address and port
    client = DataCollectorClient(server_address)
    client.show()
    sys.exit(app.exec_())