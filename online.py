import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import torch
from scipy.signal import butter, lfilter, freqz
import copy
from socketdemo import handle_client, send_npc_info, receive_npc_info, NPCInfo
import socket
import threading
from dataclasses import dataclass
import json
def butter_bandpass_filter(data, sampling_rate, lowcut, highcut, order=5):
    nyquist_freq = sampling_rate / 2.
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def send_npc_info(client_socket, npc_info):
    # npc_info = NPCInfo("机器人", 100, 10, 1.5)  # 示例数据
    while True:
        # user_input = input("输入P发送信息")
        # if user_input.lower() == 'p':
        #     json_data = json.dumps(npc_info.__dict__)   # 转换为JSON格式
        #     json_data += '\n'   # 添加换行符作为分隔符
            # client_socket.sendall(json_data.encode())   # 发送JSON数据
        update_event.wait() # 等待事件更新
        update_event.clear() # 重置事件
        print(f"Attack updated to {npc_info.Attack}")
        json_data = json.dumps(npc_info.__dict__)   # 转换为JSON格式
        json_data += '\n'   # 添加换行符作为分隔符
        client_socket.sendall(json_data.encode())   # 发送JSON数据
        print('msg sent')

def handle_client(client_socket, npc_info):
    send_thread = threading.Thread(target=send_npc_info, args=(client_socket, npc_info))
    receive_thread = threading.Thread(target=receive_npc_info, args=(client_socket,))

    send_thread.start()
    receive_thread.start()

# 从客户端接收JSON数据并解码为NPCInfo实例
def receive_npc_info(client_socket):
    while True:
        received_data = client_socket.recv(1024).decode()   # 接收数据并解码为字符串
        if not received_data:
            break
        # 解码JSON数据为NPCInfo实例
        npc_data = json.loads(received_data)
        npc_info = NPCInfo(**npc_data)
        print("收到Unity信息:", npc_info)


def update_jump_value():
    time.sleep(1.5)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data0 = board.get_board_data()# get all data and remove it from internal buffer
    data0 = data0[1:5] 
    ''' 
    执行该语句会延时，数据流还在传递
    '''
    x = butter_bandpass_filter(data0, 200, 1, 48, 6).astype('float32') 
    x = torch.from_numpy(x[:, -300:].reshape(1, 1, 4,300))
    '''print(x.type())
    print(model.conv_spatial.weight.cpu().type())'''
    out = model(x).squeeze()
    out = out.data.numpy()
    out = np.argmax(out, axis=0)
    count = np.bincount(out)
    y = np.argmax(count)
    print(y)
    while True:

        time.sleep(0.48) # 根据系统延时确定, 做到每0.5s响应一次
        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_board_data()  # get all data and remove it from internal buffer
        #print(data.shape)
        data = np.concatenate((data0, data[1:5]), -1)
        data0 = copy.deepcopy(data)
        
        x = butter_bandpass_filter(data0, 200, 1, 48, 6).astype('float32') 
        x = torch.from_numpy(x[:, -300:].reshape(1, 1, 4,300))
        out = model(x).squeeze()
        out = out.data.numpy()
        out = np.argmax(out, axis=0)
        count = np.bincount(out)
        y = np.argmax(count)
        # print(y)
        if(y==0):
            share_info.Attack = -1
        else:
            share_info.Attack = int(y)
        print(share_info.Attack)
        update_event.set()
    



if __name__ == "__main__":
    subject = 'niuxu'
    model = torch.load('Result/%s/optimal/model.pth' % subject)

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = "COM5"
    #params.mac_address = args.mac_address
    board = BoardShim(BoardIds.GANGLION_BOARD, params)
    board.prepare_session()

    ''' 开启流'''
    board.start_stream ()

    # 创建TCP socket并绑定IP地址和端口号
    # 定义服务器的IP地址和端口号
    host, port = "127.0.0.1", 25001

    share_info = NPCInfo("shareVar", 100, 1, 1.5)
    update_event = threading.Event()

    # 创建TCP socket并绑定IP地址和端口号
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"正在监听 {host}:{port}")

    update_thread = threading.Thread(target=update_jump_value)
    update_thread.start()

    while True:
        # 等待客户端连接
        client_socket, _ = server_socket.accept()
        print(f"成功连接到客户端 {_}")

        # 启动一个线程来处理客户端连接
        client_thread = threading.Thread(target=handle_client, args=(client_socket,share_info))
        client_thread.start()


        


    ''' 关闭流 '''
    board.stop_stream()
    board.release_session()    
    