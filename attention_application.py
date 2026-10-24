import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
import os
from datetime import datetime
import webbrowser
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
import threading
import time
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# 数据类，用于存储传感器数据
class SensorData:
    def __init__(self):
        self.raw_data = []  # 原始传感器数据
        self.bpm = 0        # 心率
        self.ibi = 0        # 心跳间隔
        self.data_length = 200  # 显示的数据长度
        self.signal_quality = False  # 信号质量：True表示有有效信号（手指已放置），False表示无效信号
        self.min_amplitude = 30  # 有效信号的最小振幅阈值（初始值）
        self.min_std_dev = 10    # 有效信号的最小标准差阈值（初始值）
        self.min_bpm = 40        # 有效心率的最小值
        self.max_bpm = 120       # 有效心率的最大值
        self.threshold_window = 50  # 计算动态阈值的窗口大小
        self.amplitude_history = []  # 存储历史振幅
        self.std_dev_history = []    # 存储历史标准差
        self.history_size = 20      # 历史数据点数量
    
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
        """更新信号质量状态，使用动态阈值"""
        if len(self.raw_data) < self.threshold_window:  # 需要足够的数据来判断
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
        
        # 更新历史记录
        self.amplitude_history.append(amplitude)
        self.std_dev_history.append(std_dev)
        
        # 限制历史记录的大小
        if len(self.amplitude_history) > self.history_size:
            self.amplitude_history.pop(0)
        if len(self.std_dev_history) > self.history_size:
            self.std_dev_history.pop(0)
        
        # 计算动态阈值（使用历史数据的平均值乘以一个系数）
        if len(self.amplitude_history) >= 5:  # 至少需要5个历史数据点
            dynamic_min_amplitude = max(20, sum(self.amplitude_history) / len(self.amplitude_history) * 0.8)
            dynamic_min_std_dev = max(5, sum(self.std_dev_history) / len(self.std_dev_history) * 0.8)
        else:
            # 如果历史数据不足，使用初始阈值
            dynamic_min_amplitude = self.min_amplitude
            dynamic_min_std_dev = self.min_std_dev
        
        # 判断信号质量
        if amplitude > dynamic_min_amplitude and std_dev > dynamic_min_std_dev:
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

class AttentionApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("注意力集中分析应用程序")
        self.root.geometry("900x700")
        
        # 创建传感器数据实例
        self.sensor_data = SensorData()
        
        # 创建串口通信实例
        self.serial_comm = SerialCommunication(self.sensor_data)
        
        # 模型和推理相关
        self.model = None
        self.inference_sequence = []
        self.sequence_length = 50  # 推理所需的心率序列长度
        self.inference_data = []  # 存储推理数据用于生成报告
        
        # GUI组件
        self.create_widgets()
        
        # 初始化图表
        self.init_plot()
        
        # 开始动画
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)
        
        # 加载模型
        self.load_model()
        
        # 启动主循环
        self.root.mainloop()
    
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
        
        # 中间数据显示区域
        data_frame = ttk.Frame(self.root, padding="10")
        data_frame.pack(fill=tk.X)
        
        # 心率显示
        ttk.Label(data_frame, text="心率:", font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        self.bpm_label = ttk.Label(data_frame, text="0", font=('Arial', 24, 'bold'))
        self.bpm_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(data_frame, text="BPM", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        
        # 心跳间隔显示
        ttk.Label(data_frame, text="心跳间隔:", font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        self.ibi_label = ttk.Label(data_frame, text="0", font=('Arial', 16))
        self.ibi_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(data_frame, text="ms", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        
        # 信号质量显示
        ttk.Label(data_frame, text="信号质量:", font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        self.signal_quality_label = ttk.Label(data_frame, text="无信号", font=('Arial', 16, 'bold'), foreground="red")
        self.signal_quality_label.pack(side=tk.LEFT, padx=10)
        
        # 注意力状态显示
        ttk.Label(data_frame, text="注意力状态:", font=('Arial', 12)).pack(side=tk.LEFT, padx=10)
        self.attention_status_label = ttk.Label(data_frame, text="-", font=('Arial', 16, 'bold'), foreground="black")
        self.attention_status_label.pack(side=tk.LEFT, padx=10)
        
        # 报告生成按钮
        report_button = ttk.Button(data_frame, text="生成报告", command=self.generate_report)
        report_button.pack(side=tk.RIGHT, padx=10)
        
        # 创建Notebook组件，用于切换主界面和报告界面
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 主界面标签页
        main_tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_tab_frame, text="实时监控")
        
        # 创建主界面图表
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_tab_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 报告界面标签页
        self.report_tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_tab_frame, text="报告")
        
        # 初始化报告界面内容
        self.init_report_tab()
    
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
    
    def load_model(self):
        """加载已保存的注意力模型"""
        try:
            self.model = joblib.load("attention_model.joblib")
            print("模型加载成功！")
        except FileNotFoundError:
            print("错误：未找到模型文件 attention_model.joblib")
            print("请先运行 attention_analyzer.py 进行模型训练")
    
    def extract_features(self, sequence):
        """从心率序列提取特征"""
        if len(sequence) < 10:
            return None
        
        sequence_array = np.array(sequence)
        
        # 基础统计特征
        mean_bpm = np.mean(sequence_array)
        max_bpm = np.max(sequence_array)
        min_bpm = np.min(sequence_array)
        
        # 差分特征
        diffs = np.diff(sequence_array)
        avg_diff = np.mean(np.abs(diffs))
        diff_std = np.std(diffs)
        
        # 趋势特征
        up_count = np.sum(diffs > 0)
        down_count = np.sum(diffs < 0)
        
        # 斜率特征
        if len(sequence_array) >= 2:
            x = np.arange(len(sequence_array))
            slope = np.polyfit(x, sequence_array, 1)[0]
        else:
            slope = 0
        
        # 返回与训练时一致的8个特征
        return [mean_bpm, max_bpm, min_bpm, avg_diff, diff_std, up_count, down_count, slope]
    
    def analyze_sequence(self, sequence):
        """分析单个心率序列的注意力状态"""
        if self.model is None:
            return None
        
        features = self.extract_features(sequence)
        if features is None:
            return None
        
        return self.model.predict([features])[0]
    
    def init_report_tab(self):
        """初始化报告标签页的UI组件"""
        # 创建报告内容框架
        self.report_content_frame = ttk.Frame(self.report_tab_frame)
        self.report_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建报告标题
        self.report_title = ttk.Label(self.report_content_frame, text="注意力集中情况报告", 
                                      font=('Arial', 16, 'bold'))
        self.report_title.pack(side=tk.TOP, pady=5)
        
        # 创建报告生成时间标签
        self.report_time = ttk.Label(self.report_content_frame, text="")
        self.report_time.pack(side=tk.TOP, pady=2)
        
        # 创建报告摘要框架
        self.summary_frame = ttk.LabelFrame(self.report_content_frame, text="报告摘要", padding=10)
        self.summary_frame.pack(fill=tk.X, pady=10)
        
        # 创建统计数据框架
        self.stats_frame = ttk.Frame(self.summary_frame)
        self.stats_frame.pack(fill=tk.X, pady=5)
        
        # 创建三个统计信息框
        self.total_data_frame = ttk.LabelFrame(self.stats_frame, text="数据概览", padding=10)
        self.total_data_frame.pack(side=tk.LEFT, expand=True, padx=5)
        
        self.attention_counts_frame = ttk.LabelFrame(self.stats_frame, text="注意力统计", padding=10)
        self.attention_counts_frame.pack(side=tk.LEFT, expand=True, padx=5)
        
        self.attention_ratio_frame = ttk.LabelFrame(self.stats_frame, text="注意力比例", padding=10)
        self.attention_ratio_frame.pack(side=tk.LEFT, expand=True, padx=5)
        
        # 初始化统计信息标签
        self.total_data_label = ttk.Label(self.total_data_frame, text="")
        self.total_data_label.pack(fill=tk.X)
        
        self.attention_counts_label = ttk.Label(self.attention_counts_frame, text="")
        self.attention_counts_label.pack(fill=tk.X)
        
        self.attention_ratio_label = ttk.Label(self.attention_ratio_frame, text="")
        self.attention_ratio_label.pack(fill=tk.X)
        
        # 创建注意力评估标签
        self.evaluation_label = ttk.Label(self.summary_frame, text="")
        self.evaluation_label.pack(fill=tk.X, pady=10)
        
        # 创建图表区域
        self.chart_frame = ttk.LabelFrame(self.report_content_frame, text="心率波形与注意力状态", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 初始化报告图表
        self.report_fig, self.report_ax = plt.subplots(figsize=(10, 5), dpi=100)
        self.report_canvas = FigureCanvasTkAgg(self.report_fig, master=self.chart_frame)
        self.report_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建图例框架
        self.legend_frame = ttk.Frame(self.report_content_frame)
        self.legend_frame.pack(fill=tk.X, pady=5)
        
        # 创建注意力状态图例
        self.focused_legend = ttk.Label(self.legend_frame, text="注意力集中", 
                                        background='#4CAF50', foreground='white', 
                                        padding=(5, 2), font=('Arial', 10, 'bold'))
        self.focused_legend.pack(side=tk.LEFT, padx=5)
        
        self.not_focused_legend = ttk.Label(self.legend_frame, text="注意力不集中", 
                                           background='#f44336', foreground='white', 
                                           padding=(5, 2), font=('Arial', 10, 'bold'))
        self.not_focused_legend.pack(side=tk.LEFT, padx=5)
        
        # 创建详细分析标签
        self.detail_label = ttk.Label(self.report_content_frame, 
                                     text="报告基于心率序列的动态变化趋势进行注意力状态评估。\n" +
                                     "注意力集中状态（绿色）表示心率变化呈现与集中状态相关的模式，\n" +
                                     "注意力不集中状态（红色）表示心率变化呈现与分散状态相关的模式。",
                                     justify=tk.LEFT)
        self.detail_label.pack(fill=tk.X, pady=10)
    
    def init_plot(self):
        """初始化图表"""
        self.ax.set_title('Pulse Sensor Waveform')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_ylim(0, 1023)  # ADC范围
        self.ax.grid(True)
        
        # 创建数据线
        self.line, = self.ax.plot([], [], 'b-', lw=2)
    
    def update_plot(self, frame):
        """更新图表和实时推理"""
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
        
        # 实时推理
        if self.serial_comm.is_connected and self.sensor_data.bpm > 0:
            # 收集心率数据
            self.inference_sequence.append(self.sensor_data.bpm)
            self.inference_data.append({"time": datetime.now(), "bpm": self.sensor_data.bpm})
            
            # 只保留最新的sequence_length个数据用于推理
            if len(self.inference_sequence) > self.sequence_length:
                self.inference_sequence = self.inference_sequence[-self.sequence_length:]
            
            # 当有足够的数据时进行推理
            if len(self.inference_sequence) >= self.sequence_length and self.model:
                result = self.analyze_sequence(self.inference_sequence)
                if result is not None:
                    if result == 1:
                        self.attention_status_label.config(text="集中", foreground="green")
                    else:
                        self.attention_status_label.config(text="不集中", foreground="red")
        
        return self.line,  # 返回更新的对象
    
    def generate_report(self):
        """生成注意力集中情况报告"""
        if not self.inference_data:
            print("错误：没有可用的推理数据")
            return
        
        # 提取BPM数据用于分析
        bpm_data = [data["bpm"] for data in self.inference_data if data["bpm"] > 0]
        if not bpm_data:
            print("错误：没有可用的有效心率数据")
            return
        
        # 分析注意力状态
        attention_results = []
        
        # 滑动窗口分析
        window_size = self.sequence_length
        for i in range(len(bpm_data) - window_size + 1):
            window = bpm_data[i:i+window_size]
            result = self.analyze_sequence(window)
            if result is not None:
                attention_results.append(result)
            else:
                attention_results.append(0)
        
        # 填充前面的结果
        padding = [0] * (window_size - 1)
        attention_results = padding + attention_results
        
        # 计算注意力集中的比例
        if attention_results:
            focused_count = sum(attention_results)
            focused_ratio = focused_count / len(attention_results)
        else:
            focused_count = 0
            focused_ratio = 0
        
        # 更新报告标签页中的内容
        
        # 更新报告标题和时间
        self.report_title.config(text="注意力集中情况报告")
        self.report_time.config(text=f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 更新统计数据
        self.total_data_label.config(text=f"总数据点: {len(bpm_data)}\n"
                                          f"分析窗口: {window_size} 个数据点")
        
        self.attention_counts_label.config(text=f"注意力集中次数: {focused_count}\n"
                                                 f"注意力不集中次数: {len(attention_results) - focused_count}")
        
        avg_bpm = np.mean(bpm_data)
        self.attention_ratio_label.config(text=f"注意力集中比例: {focused_ratio:.1%}\n"
                                               f"平均心率: {avg_bpm:.1f} BPM")
        
        # 更新注意力评估
        evaluation_text = self.get_attention_evaluation(focused_ratio)
        self.evaluation_label.config(text=evaluation_text)
        
        # 绘制图表
        self.report_ax.clear()
        
        # 绘制心率波形
        x_data = range(len(bpm_data))
        self.report_ax.plot(x_data, bpm_data, 'b-', label='心率 (BPM)', linewidth=1.5)
        
        # 标记注意力状态
        if attention_results:
            # 确保注意力结果与心率数据长度匹配
            if len(attention_results) < len(bpm_data):
                attention_results = [0] * (len(bpm_data) - len(attention_results)) + attention_results
            
            # 绘制注意力状态背景
            for i in range(len(bpm_data) - 1):
                if attention_results[i] == 1:
                    self.report_ax.axvspan(i, i+1, color='#4CAF50', alpha=0.3)
                else:
                    self.report_ax.axvspan(i, i+1, color='#f44336', alpha=0.3)
        
        # 设置图表属性
        self.report_ax.set_title('心率波形与注意力状态分析')
        self.report_ax.set_xlabel('时间 (样本点)')
        self.report_ax.set_ylabel('心率 (BPM)')
        self.report_ax.grid(True, alpha=0.3)
        self.report_ax.legend()
        
        # 调整布局
        self.report_ax.figure.tight_layout()
        
        # 刷新画布
        self.report_canvas.draw()
        
        # 切换到报告标签页
        self.notebook.select(self.report_tab_frame)
        
        print("报告已生成并显示在应用内")
        return True
    
    def generate_heart_rate_chart(self, bpm_data, attention_results, chart_path="heart_rate_chart.png"):
        """生成心率波形图"""
        plt.figure(figsize=(12, 6))
        
        # 绘制心率波形
        plt.plot(bpm_data, label='心率 (BPM)', color='blue', linewidth=1.5)
        
        # 标记注意力状态
        if attention_results:
            # 确保注意力结果与心率数据长度匹配
            if len(attention_results) < len(bpm_data):
                attention_results = [0] * (len(bpm_data) - len(attention_results)) + attention_results
            
            # 绘制注意力状态背景
            for i in range(len(bpm_data) - 1):
                if attention_results[i] == 1:
                    plt.axvspan(i, i+1, color='#4CAF50', alpha=0.3)
                else:
                    plt.axvspan(i, i+1, color='#f44336', alpha=0.3)
        
        plt.title('心率波形与注意力状态分析')
        plt.xlabel('时间 (样本点)')
        plt.ylabel('心率 (BPM)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_attention_evaluation(self, focused_ratio):
        """根据注意力集中比例生成评估"""
        if focused_ratio >= 0.8:
            return "优秀：您大部分时间都保持着高度集中的注意力！"
        elif focused_ratio >= 0.6:
            return "良好：您的注意力集中情况总体不错，继续保持！"
        elif focused_ratio >= 0.4:
            return "一般：您的注意力容易分散，建议适当休息和调整！"
        else:
            return "较差：您的注意力集中时间较少，建议调整学习或工作方式！"

if __name__ == "__main__":
    app = AttentionApplication()