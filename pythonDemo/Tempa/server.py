import sys
import socket
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTextEdit, QPushButton
from threading import Thread
import time
import os

class DataCollectorServer(QMainWindow):
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.log_file_path = 'pythonDemo/Tempa/logs/server' + str(time.time())+ '.log'
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen()
        self.clients = {}  # 用于跟踪客户端连接
        # self.heartbeat_queue = Queue()  # 用于发送心跳消息的队列
        # self.lock = Lock()  # 用于线程间同步的锁
        self.initUI()
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'a'):
                pass
            print(f"文件'{self.log_file_path}'已创建。")
        else:
            print(f"文件'{self.log_file_path}'已存在。")
        self.log_file = open(self.log_file_path, 'a')

    # 修改后的方法，用于启动心跳发送线程
    def start_heartbeat_thread(self):
        heartbeat_thread = Thread(target=self.send_heartbeat)
        heartbeat_thread.daemon = True  # 设置为守护线程，使得在应用程序关闭时能够自动退出
        heartbeat_thread.start()

    # 修改后的方法，用于发送心跳消息
    def send_heartbeat(self):
        while True:
            with self.lock:
                for address, client_socket in list(self.clients.items()):
                    try:
                        client_socket.sendall(json.dumps({"type": "heartbeat"}).encode("utf-8"))
                    except socket.error as e:
                        print(f"发送心跳失败: {e}")
                        client_socket.close()
                        del self.clients[address]
            time.sleep(30)  # 心跳间隔，可以根据需要调整


    def initUI(self):
        self.setWindowTitle('Data Collector Server')
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel('Server is running and waiting for connections...', self)
        layout.addWidget(self.status_label)

        # Text edit to display received data
        self.data_display = QTextEdit(self)
        self.data_display.setReadOnly(True)
        layout.addWidget(self.data_display)

        # Start/Stop server button
        self.server_control_button = QPushButton('Stop Server', self)
        self.server_control_button.clicked.connect(self.stop_server)
        layout.addWidget(self.server_control_button)

        # Central widget
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Accept connections in a separate thread
        self.accept_thread = Thread(target=self.accept_connections)
        self.accept_thread.start()

    def accept_connections(self):
        while True:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"Connected by {client_address}")
                self.clients[client_address] = client_socket
                client_thread = Thread(target=self.handle_client, args=(client_socket, client_address))
                client_thread.start()
            except socket.error as e:
                self.data_display.append(f'Error: {e}')

    def handle_client(self, client_socket, client_address):
        while True:
            try:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                if self.is_heartbeat(data):  # 如果收到心跳消息
                    self.data_display.append(f'Heartbeat received from {client_address}.')  # 记录收到心跳
                else:
                    self.log_file.write(data + '\n')
                    self.log_file.flush()
                    self.data_display.append(f'Received from {client_address}: {data}')
                    message = json.loads(data)
                    self.process_message(message)
            except socket.error as e:
                self.data_display.append(f'Error: {e}')
                break
        self.remove_client(client_address)

    def is_heartbeat(self, data):
        try:
            message = json.loads(data)
            return message.get('type') == 'heartbeat'  # 检查消息类型是否为心跳
        except json.JSONDecodeError:
            return False  # 若解析失败，则不是心跳消息

    def process_message(self, message):
        # Example processing: simply log the received data
        if message.get('type') == 'data_report':
            sensor_id = message.get('payload', {}).get('sensor_id', 'Unknown')
            temperature = message.get('payload', {}).get('temperature', 'N/A')
            humidity = message.get('payload', {}).get('humidity', 'N/A')
            self.data_display.append(f'Sensor {sensor_id}: Temp={temperature}, Humidity={humidity}')

    def remove_client(self, client_address):
        if client_address in self.clients:
            self.clients[client_address].close()
            self.clients.pop(client_address)
            self.data_display.append(f'Connection {client_address} closed.')

    def stop_server(self):
        self.server_socket.close()
        self.server_control_button.setText('Start Server')
        self.log_file.close()
        self.status_label.setText('Server is stopped.')
        self.data_display.append('Server stopped and all connections closed.')

    def closeEvent(self, event):
        self.log_file.close()
        self.stop_server()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    port = 12345  # Replace with your server port
    server = DataCollectorServer(port)
    server.show()
    sys.exit(app.exec_())