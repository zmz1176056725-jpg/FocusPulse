import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import time
import re
import csv
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier

# 数据类，用于存储传感器数据
class SensorData:
    def __init__(self):
        self.raw_data = []  # 原始传感器数据
        self.bpm = 0        # 心率
        self.ibi = 0        # 心跳间隔
        self.data_length = 200  # 显示的数据长度
        self.signal_quality = False  # 信号质量：True表示有有效信号（手指已放置），False表示无效信号
        self.min_amplitude = 30  # 有效信号的最小振幅阈值（降低过滤强度）
        self.min_std_dev = 10    # 有效信号的最小标准差阈值（降低过滤强度）
        self.min_bpm = 40        # 有效心率的最小值（扩大范围）
        self.max_bpm = 120       # 有效心率的最大值（扩大范围）
    
    def add_raw_data(self, value):
        self.raw_data.append(value)
        if len(self.raw_data) > self.data_length:
            self.raw_data.pop(0)
        # 添加新数据后更新信号质量
        self.update_signal_quality()
    
    def update_bpm(self, value):
        # 只有当信号质量良好且心率在有效范围内时才更新心率
        if self.signal_quality and (self.min_bpm <= value <= self.max_bpm):
            self.bpm = value
        else:
            self.bpm = 0
    
    def update_ibi(self, value):
        # 只有当信号质量良好时才更新心跳间隔
        if self.signal_quality:
            self.ibi = value
        else:
            self.ibi = 0
    
    def update_signal_quality(self):
        """更新信号质量状态"""
        if len(self.raw_data) < 50:  # 需要足够的数据来判断
            self.signal_quality = False
            return
        
        # 计算信号的最大值和最小值
        max_val = max(self.raw_data)
        min_val = min(self.raw_data)
        amplitude = max_val - min_val
        
        # 计算信号的标准差
        mean_val = sum(self.raw_data) / len(self.raw_data)
        variance = sum((x - mean_val) ** 2 for x in self.raw_data) / len(self.raw_data)
        std_dev = variance ** 0.5
        
        # 计算基于当前数据的估计心率范围（仅作为参考，不再严格限制）
        # 通过计算信号频率来估计心率
        # 这是一个简化的估算，用于辅助判断信号质量
        if amplitude > self.min_amplitude:
            # 寻找峰值数量来估算心率
            peaks = 0
            for i in range(1, len(self.raw_data) - 1):
                if self.raw_data[i] > self.raw_data[i-1] and self.raw_data[i] > self.raw_data[i+1]:
                    peaks += 1
            
            # 根据峰值数量估算心率（假设数据长度为200，采样频率为50Hz）
            if peaks > 0:
                estimated_bpm = (peaks / len(self.raw_data)) * 50 * 60
                # 放宽心率范围限制，不再直接拒绝信号
                if not (self.min_bpm * 0.5 <= estimated_bpm <= self.max_bpm * 1.5):
                    # 仅在极端情况下降低信号质量判断要求
                    # 但仍继续进行其他信号质量判断
                    pass
        
        # 判断信号质量
        if amplitude > self.min_amplitude and std_dev > self.min_std_dev:
            self.signal_quality = True
        else:
            self.signal_quality = False

# 串口通信类
class SerialCommunication:
    def __init__(self, sensor_data, baudrate=115200):
        self.sensor_data = sensor_data
        self.baudrate = baudrate
        self.serial_port = None
        self.is_connected = False
        self.thread = None
        self.stop_event = threading.Event()
    
    def list_ports(self):
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port):
        """连接到指定的串口"""
        try:
            self.serial_port = serial.Serial(port, self.baudrate, timeout=1)
            self.is_connected = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.read_data)
            self.thread.daemon = True
            self.thread.start()
            return True
        except serial.SerialException as e:
            print(f"串口连接错误: {e}")
            return False
    
    def disconnect(self):
        """断开串口连接"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.is_connected = False
    
    def read_data(self):
        """读取串口数据"""
        while not self.stop_event.is_set() and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    self.parse_data(line)
            except (serial.SerialException, UnicodeDecodeError) as e:
                continue
            time.sleep(0.001)
    
    def parse_data(self, line):
        """解析接收到的数据"""
        # 使用正则表达式匹配数据格式
        match = re.match(r'([SBQ])(\d+)', line)
        if match:
            data_type = match.group(1)
            value = int(match.group(2))
            
            if data_type == 'S':  # 原始传感器数据
                self.sensor_data.add_raw_data(value)
            elif data_type == 'B':  # 心率
                self.sensor_data.update_bpm(value)
            elif data_type == 'Q':  # 心跳间隔
                self.sensor_data.update_ibi(value)

# GUI应用类
class PulseSensorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pulse Sensor Monitor with Attention Detection")
        self.root.geometry("900x700")
        
        # 创建传感器数据实例
        self.sensor_data = SensorData()
        
        # 创建串口通信实例
        self.serial_comm = SerialCommunication(self.sensor_data)
        
        # 模式状态
        self.mode = "monitor"  # monitor, learning, inference
        self.is_learning = False
        self.is_inferencing = False
        
        # 学习数据
        self.learning_data = []
        self.current_sequence = []
        self.sequence_length = 50  # 每个学习序列的数据点数量
        self.model = None  # 用于存储KNN模型
        
        # 创建GUI组件
        self.create_widgets()
        
        # 初始化图表
        self.init_plot()
        
        # 开始动画
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)
    
    def create_widgets(self):
        """创建GUI组件"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 串口选择
        ttk.Label(control_frame, text="串口:").pack(side=tk.LEFT, padx=5)
        self.port_combo = ttk.Combobox(control_frame, values=self.serial_comm.list_ports())
        self.port_combo.pack(side=tk.LEFT, padx=5)
        
        # 刷新串口按钮
        refresh_button = ttk.Button(control_frame, text="刷新", command=self.refresh_ports)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # 连接/断开按钮
        self.connect_button = ttk.Button(control_frame, text="连接", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="未连接")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # 模式选择面板
        mode_frame = ttk.Frame(self.root, padding="10", relief=tk.RAISED, borderwidth=1)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="模式选择:", font= ("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # 监控模式按钮
        self.monitor_button = ttk.Button(mode_frame, text="监控模式", command=lambda: self.set_mode("monitor"))
        self.monitor_button.pack(side=tk.LEFT, padx=5)
        
        # 学习模式按钮
        self.learning_button = ttk.Button(mode_frame, text="学习模式", command=lambda: self.set_mode("learning"))
        self.learning_button.pack(side=tk.LEFT, padx=5)
        
        # 推理模式按钮
        self.inference_button = ttk.Button(mode_frame, text="推理模式", command=lambda: self.set_mode("inference"))
        self.inference_button.pack(side=tk.LEFT, padx=5)
        
        # 当前模式显示
        self.current_mode_label = ttk.Label(mode_frame, text="当前模式: 监控模式", font= ("Arial", 12, "bold"))
        self.current_mode_label.pack(side=tk.LEFT, padx=20)
        
        # 中间数据显示区域
        data_frame = ttk.Frame(self.root, padding="10")
        data_frame.pack(fill=tk.X)
        
        # 心率显示
        ttk.Label(data_frame, text="心率:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.bpm_label = ttk.Label(data_frame, text="0", font=("Arial", 24, "bold"))
        self.bpm_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(data_frame, text="BPM", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        
        # 心跳间隔显示
        ttk.Label(data_frame, text="心跳间隔:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.ibi_label = ttk.Label(data_frame, text="0", font=("Arial", 16))
        self.ibi_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(data_frame, text="ms", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        
        # 信号质量显示
        ttk.Label(data_frame, text="信号质量:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.signal_quality_label = ttk.Label(data_frame, text="无信号", font=("Arial", 16, "bold"), foreground="red")
        self.signal_quality_label.pack(side=tk.LEFT, padx=10)
        
        # 学习模式控制面板
        self.learning_frame = ttk.Frame(self.root, padding="10", relief=tk.RAISED, borderwidth=1)
        self.learning_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.learning_frame, text="学习模式控制:", font= ("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # 开始/停止学习按钮
        self.learn_start_button = ttk.Button(self.learning_frame, text="开始学习", command=self.start_learning)
        self.learn_start_button.pack(side=tk.LEFT, padx=5)
        
        self.learn_stop_button = ttk.Button(self.learning_frame, text="停止学习", command=self.stop_learning, state=tk.DISABLED)
        self.learn_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 学习状态显示
        self.learning_status_label = ttk.Label(self.learning_frame, text="学习状态: 未学习", foreground="blue")
        self.learning_status_label.pack(side=tk.LEFT, padx=20)
        
        # 标注按钮（是/否）
        ttk.Label(self.learning_frame, text="注意力状态: ", font= ("Arial", 12)).pack(side=tk.LEFT, padx=10)
        
        self.label_yes_button = ttk.Button(self.learning_frame, text="集中", command=lambda: self.label_data(True), state=tk.DISABLED)
        self.label_yes_button.pack(side=tk.LEFT, padx=5)
        
        self.label_no_button = ttk.Button(self.learning_frame, text="不集中", command=lambda: self.label_data(False), state=tk.DISABLED)
        self.label_no_button.pack(side=tk.LEFT, padx=5)
        
        # 保存和加载按钮
        self.save_button = ttk.Button(self.learning_frame, text="保存学习数据", command=self.save_learning_data)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = ttk.Button(self.learning_frame, text="加载学习数据", command=self.load_learning_data)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # 模型保存和加载按钮
        self.save_model_button = ttk.Button(self.learning_frame, text="保存模型", command=self.save_model)
        self.save_model_button.pack(side=tk.LEFT, padx=5)
        
        self.load_model_button = ttk.Button(self.learning_frame, text="加载模型", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)
        
        # 推理模式控制面板
        self.inference_frame = ttk.Frame(self.root, padding="10", relief=tk.RAISED, borderwidth=1)
        self.inference_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.inference_frame, text="推理模式控制:", font= ("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        
        # 开始/停止推理按钮
        self.inference_start_button = ttk.Button(self.inference_frame, text="开始推理", command=self.start_inference)
        self.inference_start_button.pack(side=tk.LEFT, padx=5)
        
        self.inference_stop_button = ttk.Button(self.inference_frame, text="停止推理", command=self.stop_inference, state=tk.DISABLED)
        self.inference_stop_button.pack(side=tk.LEFT, padx=5)
        
        # 推理状态显示
        self.inference_status_label = ttk.Label(self.inference_frame, text="推理状态: 未推理", foreground="blue")
        self.inference_status_label.pack(side=tk.LEFT, padx=20)
        
        # 注意力状态显示
        ttk.Label(self.inference_frame, text="注意力状态: ", font= ("Arial", 12)).pack(side=tk.LEFT, padx=10)
        self.attention_status_label = ttk.Label(self.inference_frame, text="-", font= ("Arial", 16, "bold"), foreground="black")
        self.attention_status_label.pack(side=tk.LEFT, padx=10)
        
        # 初始化时隐藏学习和推理面板
        self.learning_frame.pack_forget()
        self.inference_frame.pack_forget()
        
        # 底部图表区域
        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def refresh_ports(self):
        """刷新串口列表"""
        self.port_combo['values'] = self.serial_comm.list_ports()
    
    def toggle_connection(self):
        """切换串口连接状态"""
        if self.serial_comm.is_connected:
            self.serial_comm.disconnect()
            self.connect_button.config(text="连接")
            self.status_label.config(text="未连接")
        else:
            selected_port = self.port_combo.get()
            if selected_port:
                if self.serial_comm.connect(selected_port):
                    self.connect_button.config(text="断开")
                    self.status_label.config(text="已连接")
    
    def set_mode(self, mode):
        """切换操作模式"""
        # 停止当前可能正在进行的学习或推理
        if self.is_learning:
            self.stop_learning()
        if self.is_inferencing:
            self.stop_inference()
        
        # 设置新模式
        self.mode = mode
        
        # 更新模式显示
        mode_names = {
            "monitor": "监控模式",
            "learning": "学习模式",
            "inference": "推理模式"
        }
        self.current_mode_label.config(text=f"当前模式: {mode_names[mode]}")
        
        # 显示或隐藏相应的控制面板
        if mode == "learning":
            self.learning_frame.pack(fill=tk.X, pady=5)
            self.inference_frame.pack_forget()
        elif mode == "inference":
            self.learning_frame.pack_forget()
            self.inference_frame.pack(fill=tk.X, pady=5)
        else:  # monitor
            self.learning_frame.pack_forget()
            self.inference_frame.pack_forget()
    
    def start_learning(self):
        """开始学习模式"""
        if not self.serial_comm.is_connected:
            self.learning_status_label.config(text="学习状态: 请先连接串口", foreground="red")
            return
        
        self.is_learning = True
        self.current_sequence = []
        self.learn_start_button.config(state=tk.DISABLED)
        self.learn_stop_button.config(state=tk.NORMAL)
        self.label_yes_button.config(state=tk.DISABLED)
        self.label_no_button.config(state=tk.DISABLED)
        self.learning_status_label.config(text="学习状态: 正在采集数据...", foreground="green")
    
    def stop_learning(self):
        """停止学习模式"""
        self.is_learning = False
        self.learn_start_button.config(state=tk.NORMAL)
        self.learn_stop_button.config(state=tk.DISABLED)
        self.label_yes_button.config(state=tk.DISABLED)
        self.label_no_button.config(state=tk.DISABLED)
        self.learning_status_label.config(text="学习状态: 学习已停止", foreground="blue")
    
    def label_data(self, is_attentive):
        """手动标注数据"""
        if not self.current_sequence or len(self.current_sequence) < self.sequence_length:
            return
        
        # 提取特征
        features = self.extract_features(self.current_sequence)
        
        # 保存标注数据
        self.learning_data.append({
            "features": features,
            "label": 1 if is_attentive else 0
        })
        
        # 清空当前序列，准备下一次采集
        self.current_sequence = []
        
        # 更新学习状态
        self.learning_status_label.config(
            text=f"学习状态: 已标注 {len(self.learning_data)} 个样本", 
            foreground="green"
        )
        
        # 禁用标注按钮，等待新的序列采集
        self.label_yes_button.config(state=tk.DISABLED)
        self.label_no_button.config(state=tk.DISABLED)
    
    def extract_features(self, sequence):
        """从心率序列中提取特征，增强对序列变化趋势的学习"""
        if not sequence:
            return []
        
        # 计算基本统计特征
        mean_bpm = sum(sequence) / len(sequence)
        max_bpm = max(sequence)
        min_bpm = min(sequence)
        
        # 计算心率变异性（HRV）相关特征
        if len(sequence) > 1:
            # 相邻心率差值的平均值
            avg_diff = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence))) / (len(sequence) - 1)
            # 相邻心率差值的标准差
            if len(sequence) > 2:
                diffs = [abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence))]
                diff_mean = sum(diffs) / len(diffs)
                diff_std = (sum((d - diff_mean)**2 for d in diffs) / len(diffs))**0.5
            else:
                diff_std = 0
        else:
            avg_diff = 0
            diff_std = 0
        
        # 增加序列变化趋势特征
        trend_features = []
        if len(sequence) > 1:
            # 心率上升和下降的次数
            up_count = 0
            down_count = 0
            for i in range(1, len(sequence)):
                if sequence[i] > sequence[i-1]:
                    up_count += 1
                elif sequence[i] < sequence[i-1]:
                    down_count += 1
            
            # 计算心率变化的斜率（线性回归近似）
            x = list(range(len(sequence)))
            y = sequence
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi*yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi**2 for xi in x)
            
            if n * sum_x2 - sum_x**2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            else:
                slope = 0
            
            trend_features = [up_count, down_count, slope]
        
        # 合并所有特征
        return [mean_bpm, max_bpm, min_bpm, avg_diff, diff_std] + trend_features
    
    def save_learning_data(self):
        """保存学习数据到CSV文件"""
        if not self.learning_data:
            self.learning_status_label.config(text="学习状态: 没有可保存的数据", foreground="red")
            return
        
        try:
            # 定义CSV文件名
            filename = "attention_learning_data.csv"
            
            # 写入CSV文件
            with open(filename, 'w', newline='') as csvfile:
                # 定义CSV表头，包含新增的趋势特征
                fieldnames = ['mean_bpm', 'max_bpm', 'min_bpm', 'avg_diff', 'diff_std', 
                            'up_count', 'down_count', 'slope', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                for data in self.learning_data:
                    # 确保特征数量足够
                    features = data['features']
                    feature_dict = {
                        'mean_bpm': features[0] if len(features) > 0 else 0,
                        'max_bpm': features[1] if len(features) > 1 else 0,
                        'min_bpm': features[2] if len(features) > 2 else 0,
                        'avg_diff': features[3] if len(features) > 3 else 0,
                        'diff_std': features[4] if len(features) > 4 else 0,
                        'up_count': features[5] if len(features) > 5 else 0,
                        'down_count': features[6] if len(features) > 6 else 0,
                        'slope': features[7] if len(features) > 7 else 0,
                        'label': data['label']
                    }
                    writer.writerow(feature_dict)
            
            self.learning_status_label.config(text=f"学习状态: 数据已保存到 {filename}", foreground="green")
        except Exception as e:
            self.learning_status_label.config(text=f"学习状态: 保存失败 - {str(e)}", foreground="red")
            return
    
    def save_model(self):
        """将学习好的模型保存到本地"""
        if not self.learning_data:
            self.learning_status_label.config(text="学习状态: 没有足够的数据来训练模型", foreground="red")
            return
        
        try:
            # 准备训练数据
            X = [data['features'] for data in self.learning_data]
            y = [data['label'] for data in self.learning_data]
            
            # 训练KNN模型
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.model.fit(X, y)
            
            # 保存模型到文件
            model_filename = "attention_model.joblib"
            joblib.dump(self.model, model_filename)
            
            self.learning_status_label.config(text=f"学习状态: 模型已保存到 {model_filename}", foreground="green")
        except Exception as e:
            self.learning_status_label.config(text=f"学习状态: 模型保存失败 - {str(e)}", foreground="red")
    
    def load_model(self):
        """从本地加载模型"""
        try:
            model_filename = "attention_model.joblib"
            if not os.path.exists(model_filename):
                self.learning_status_label.config(text="学习状态: 找不到模型文件", foreground="red")
                return
            
            # 加载模型
            self.model = joblib.load(model_filename)
            self.learning_status_label.config(text="学习状态: 模型已加载", foreground="green")
        except Exception as e:
            self.learning_status_label.config(text=f"学习状态: 模型加载失败 - {str(e)}", foreground="red")
    
    def load_learning_data(self):
        """从CSV文件加载学习数据"""
        try:
            # 定义CSV文件名
            filename = "attention_learning_data.csv"
            
            if not os.path.exists(filename):
                self.learning_status_label.config(text="学习状态: 找不到数据文件", foreground="red")
                return
            
            # 读取CSV文件
            self.learning_data = []
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # 解析特征和标签
                    # 支持旧版和新版CSV格式
                    features = [
                        float(row['mean_bpm']),
                        float(row['max_bpm']),
                        float(row['min_bpm']),
                        float(row['avg_diff']),
                        float(row['diff_std'])
                    ]
                    
                    # 添加新增的趋势特征（如果存在）
                    if 'up_count' in row:
                        features.append(float(row['up_count']))
                    else:
                        features.append(0)
                    
                    if 'down_count' in row:
                        features.append(float(row['down_count']))
                    else:
                        features.append(0)
                    
                    if 'slope' in row:
                        features.append(float(row['slope']))
                    else:
                        features.append(0)
                    
                    label = int(row['label'])
                    
                    # 添加到学习数据
                    self.learning_data.append({
                        "features": features,
                        "label": label
                    })
            
            self.learning_status_label.config(
                text=f"学习状态: 已加载 {len(self.learning_data)} 个样本", 
                foreground="green"
            )
        except Exception as e:
            self.learning_status_label.config(text=f"学习状态: 加载失败 - {str(e)}", foreground="red")
    
    def start_inference(self):
        """开始推理模式"""
        if not self.serial_comm.is_connected:
            self.inference_status_label.config(text="推理状态: 请先连接串口", foreground="red")
            return
        
        if not self.learning_data:
            self.inference_status_label.config(text="推理状态: 没有可用的学习数据", foreground="red")
            return
        
        self.is_inferencing = True
        self.inference_sequence = []
        self.inference_start_button.config(state=tk.DISABLED)
        self.inference_stop_button.config(state=tk.NORMAL)
        self.inference_status_label.config(text="推理状态: 正在推理...", foreground="green")
    
    def stop_inference(self):
        """停止推理模式"""
        self.is_inferencing = False
        self.inference_start_button.config(state=tk.NORMAL)
        self.inference_stop_button.config(state=tk.DISABLED)
        self.inference_status_label.config(text="推理状态: 推理已停止", foreground="blue")
        self.attention_status_label.config(text="-", foreground="black")
    
    def knn_inference(self, features, k=3):
        """使用KNN算法进行推理"""
        if not self.learning_data:
            return None
        
        # 计算与所有训练样本的欧氏距离
        distances = []
        for data in self.learning_data:
            train_features = data['features']
            label = data['label']
            
            # 计算欧氏距离
            distance = sum((f1 - f2)**2 for f1, f2 in zip(features, train_features))**0.5
            distances.append((distance, label))
        
        # 按距离排序
        distances.sort(key=lambda x: x[0])
        
        # 选择K个最近邻
        k_nearest = distances[:k]
        
        # 统计标签
        label_count = {0: 0, 1: 0}
        for _, label in k_nearest:
            label_count[label] += 1
        
        # 返回最常见的标签
        return 1 if label_count[1] > label_count[0] else 0
    
    def init_plot(self):
        """初始化图表"""
        self.ax.set_title('Pulse Sensor Waveform')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_ylim(0, 1023)  # ADC范围
        self.ax.grid(True)
        
        # 创建数据线
        self.line, = self.ax.plot([], [], 'b-', lw=2)
        
        # 创建心跳标记线
        self.beat_line = self.ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        self.beat_line.set_visible(False)
    
    def update_plot(self, frame):
        """更新图表"""
        # 更新数据线
        x_data = range(len(self.sensor_data.raw_data))
        self.line.set_data(x_data, self.sensor_data.raw_data)
        
        # 更新坐标轴范围
        self.ax.set_xlim(0, len(self.sensor_data.raw_data))
        
        # 更新心率和心跳间隔显示
        self.bpm_label.config(text=str(self.sensor_data.bpm))
        self.ibi_label.config(text=str(self.sensor_data.ibi))
        
        # 更新信号质量显示
        if self.sensor_data.signal_quality:
            self.signal_quality_label.config(text="有信号", foreground="green")
        else:
            self.signal_quality_label.config(text="无信号", foreground="red")
        
        # 学习模式下的数据采集
        if self.mode == "learning" and self.is_learning:
            # 只收集有效的心率数据（非零）
            if self.sensor_data.bpm > 0:
                self.current_sequence.append(self.sensor_data.bpm)
                
                # 如果序列长度超过设定值，只保留最新的sequence_length个数据
                if len(self.current_sequence) > self.sequence_length:
                    self.current_sequence = self.current_sequence[-self.sequence_length:]
                
                # 更新学习状态显示
                self.learning_status_label.config(
                    text=f"学习状态: 已采集 {len(self.current_sequence)}/{self.sequence_length} 个数据点", 
                    foreground="green"
                )
                
                # 当采集到足够的数据点时，启用标注按钮
                if len(self.current_sequence) >= self.sequence_length:
                    self.label_yes_button.config(state=tk.NORMAL)
                    self.label_no_button.config(state=tk.NORMAL)
                    self.learning_status_label.config(
                        text="学习状态: 数据采集完成，请标注注意力状态", 
                        foreground="orange"
                    )
        
        # 推理模式下的数据处理
        if self.mode == "inference" and self.is_inferencing and self.sensor_data.signal_quality:
            # 只收集有效的心率数据（非零）
            if self.sensor_data.bpm > 0:
                self.inference_sequence.append(self.sensor_data.bpm)
                
                # 如果序列长度超过设定值，只保留最新的sequence_length个数据
                if len(self.inference_sequence) > self.sequence_length:
                    self.inference_sequence = self.inference_sequence[-self.sequence_length:]
                
                # 当采集到足够的数据点时，进行推理
                if len(self.inference_sequence) >= self.sequence_length:
                    # 提取特征
                    features = self.extract_features(self.inference_sequence)
                    
                    # 使用KNN算法进行推理
                    if self.model:
                        # 如果有训练好的模型，使用模型进行推理
                        result = self.model.predict([features])[0]
                    else:
                        # 否则使用传统的KNN函数
                        result = self.knn_inference(features, k=3)
                    
                    # 更新注意力状态显示
                    if result is not None:
                        if result == 1:
                            self.attention_status_label.config(text="集中", foreground="green")
                        else:
                            self.attention_status_label.config(text="不集中", foreground="red")
        
        return self.line, self.beat_line

# 主函数
if __name__ == "__main__":
    root = tk.Tk()
    app = PulseSensorGUI(root)
    root.mainloop()