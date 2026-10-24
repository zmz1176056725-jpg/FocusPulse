import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
import serial.tools.list_ports
import time
import re
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import webbrowser

class AttentionAnalyzer:
    def __init__(self):
        self.model = None
        self.model_filename = "attention_model.joblib"
        self.sequence_length = 50  # 每个分析序列的数据点数量
        
    def load_model(self):
        """加载已保存的模型"""
        if not os.path.exists(self.model_filename):
            print(f"错误: 找不到模型文件 {self.model_filename}")
            return False
        
        try:
            self.model = joblib.load(self.model_filename)
            print(f"成功加载模型: {self.model_filename}")
            return True
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False
    
    def extract_features(self, sequence):
        """从心率序列中提取特征，与训练时使用的特征提取方法一致"""
        if not sequence or len(sequence) < 2:
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
    
    def analyze_sequence(self, sequence):
        """分析心率序列，返回注意力状态"""
        if not self.model:
            print("错误: 未加载模型")
            return None
        
        if len(sequence) < self.sequence_length:
            print(f"错误: 序列长度不足 {self.sequence_length} 个数据点")
            return None
        
        # 提取特征
        features = self.extract_features(sequence)
        if not features:
            print("错误: 特征提取失败")
            return None
        
        # 使用模型进行预测
        try:
            result = self.model.predict([features])[0]
            return bool(result)
        except Exception as e:
            print(f"预测失败: {str(e)}")
            return None
    
    def read_serial_data(self, port, baudrate=115200, duration=60):
        """从串口读取实时心率数据"""
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            print(f"已连接到串口 {port}")
        except serial.SerialException as e:
            print(f"无法连接到串口 {port}: {str(e)}")
            return None
        
        bpm_data = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        # 使用正则表达式匹配心率数据 (B前缀)
                        match = re.match(r'B(\d+)', line)
                        if match:
                            bpm = int(match.group(1))
                            if bpm > 0:  # 只保留有效心率
                                bpm_data.append(bpm)
                                print(f"心率: {bpm} BPM")
                except (serial.SerialException, UnicodeDecodeError) as e:
                    continue
                time.sleep(0.001)
        finally:
            ser.close()
            print(f"已从串口读取 {len(bpm_data)} 个心率数据点")
        
        return bpm_data
    
    def load_csv_data(self, filename):
        """从CSV文件加载历史心率数据"""
        try:
            df = pd.read_csv(filename)
            # 假设CSV文件中有一个'bpm'列
            if 'bpm' in df.columns:
                bpm_data = df['bpm'].dropna().tolist()
                print(f"已从CSV文件加载 {len(bpm_data)} 个心率数据点")
                return bpm_data
            else:
                print("错误: CSV文件中没有找到'bpm'列")
                return None
        except Exception as e:
            print(f"加载CSV文件失败: {str(e)}")
            return None
    
    def generate_report(self, bpm_data, report_filename="attention_report.pdf"):
        """生成注意力集中情况报告"""
        if not bpm_data:
            print("错误: 没有心率数据可用于生成报告")
            return False
        
        if not self.model:
            print("错误: 未加载模型")
            return False
        
        try:
            # 创建PDF报告
            with PdfPages(report_filename) as pdf:
                # 第1页: 报告概览
                fig = plt.figure(figsize=(10, 7))
                plt.title(f'注意力集中情况报告\n生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
                plt.axis('off')
                
                # 添加报告信息
                info_text = [
                    f'总数据点数量: {len(bpm_data)}',
                    f'分析时间范围: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    f'平均心率: {round(np.mean(bpm_data), 2)} BPM',
                    f'最高心率: {max(bpm_data)} BPM',
                    f'最低心率: {min(bpm_data)} BPM',
                    f'心率标准差: {round(np.std(bpm_data), 2)} BPM'
                ]
                
                plt.text(0.1, 0.8, '报告概览', fontsize=14, fontweight='bold')
                for i, line in enumerate(info_text):
                    plt.text(0.1, 0.7 - i*0.05, line, fontsize=12)
                
                pdf.savefig(fig)
                plt.close()
                
                # 第2页: 心率波形图
                fig = plt.figure(figsize=(10, 7))
                plt.plot(bpm_data, 'b-', linewidth=1)
                plt.title('心率波形图', fontsize=14)
                plt.xlabel('时间 (数据点)', fontsize=12)
                plt.ylabel('心率 (BPM)', fontsize=12)
                plt.grid(True)
                pdf.savefig(fig)
                plt.close()
                
                # 第3页: 注意力状态分析
                if len(bpm_data) >= self.sequence_length:
                    # 将数据分割成多个序列
                    num_sequences = len(bpm_data) // self.sequence_length
                    attention_results = []
                    
                    for i in range(num_sequences):
                        start_idx = i * self.sequence_length
                        end_idx = start_idx + self.sequence_length
                        sequence = bpm_data[start_idx:end_idx]
                        
                        attention = self.analyze_sequence(sequence)
                        if attention is not None:
                            attention_results.append(attention)
                    
                    if attention_results:
                        # 计算注意力集中的比例
                        attentive_ratio = sum(attention_results) / len(attention_results)
                        
                        fig = plt.figure(figsize=(10, 7))
                        
                        # 子图1: 注意力状态时间序列
                        ax1 = fig.add_subplot(2, 1, 1)
                        ax1.plot(attention_results, 'go-', markersize=5)
                        ax1.set_title('注意力状态时间序列', fontsize=12)
                        ax1.set_xlabel('时间窗口', fontsize=10)
                        ax1.set_ylabel('注意力状态', fontsize=10)
                        ax1.set_yticks([0, 1])
                        ax1.set_yticklabels(['不集中', '集中'])
                        ax1.grid(True)
                        
                        # 子图2: 注意力状态分布
                        ax2 = fig.add_subplot(2, 1, 2)
                        labels = ['集中', '不集中']
                        sizes = [sum(attention_results), len(attention_results) - sum(attention_results)]
                        colors = ['#4CAF50', '#F44336']
                        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax2.set_title('注意力状态分布', fontsize=12)
                        ax2.axis('equal')  # 确保饼图是圆形
                        
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()
                        
                        # 添加注意力集中比例到概览
                        print(f"注意力集中比例: {round(attentive_ratio * 100, 1)}%")
                
            print(f"报告已生成: {report_filename}")
            
            # 尝试打开报告
            try:
                webbrowser.open(report_filename)
                print("正在打开报告...")
            except Exception as e:
                print(f"无法自动打开报告: {str(e)}")
                
            return True
        except Exception as e:
            print(f"生成报告失败: {str(e)}")
            return False
    
    def generate_html_report(self, bpm_data, report_filename="attention_report.html"):
        """生成HTML格式的注意力集中情况报告"""
        if not bpm_data:
            print("错误: 没有心率数据可用于生成报告")
            return False
        
        if not self.model:
            print("错误: 未加载模型")
            return False
        
        try:
            # 生成心率波形图
            plt.figure(figsize=(10, 5))
            plt.plot(bpm_data, 'b-', linewidth=1)
            plt.title('心率波形图', fontsize=14)
            plt.xlabel('时间 (数据点)', fontsize=12)
            plt.ylabel('心率 (BPM)', fontsize=12)
            plt.grid(True)
            
            # 保存心率图为PNG
            heart_rate_img = "heart_rate_chart.png"
            plt.savefig(heart_rate_img, dpi=100, bbox_inches='tight')
            plt.close()
            
            # 计算统计信息
            avg_bpm = round(np.mean(bpm_data), 2)
            max_bpm = max(bpm_data)
            min_bpm = min(bpm_data)
            std_bpm = round(np.std(bpm_data), 2)
            
            # 注意力状态分析
            attention_results = []
            if len(bpm_data) >= self.sequence_length:
                num_sequences = len(bpm_data) // self.sequence_length
                
                for i in range(num_sequences):
                    start_idx = i * self.sequence_length
                    end_idx = start_idx + self.sequence_length
                    sequence = bpm_data[start_idx:end_idx]
                    
                    attention = self.analyze_sequence(sequence)
                    if attention is not None:
                        attention_results.append(attention)
            
            # 生成HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>注意力集中情况报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .stats table {{ border-collapse: collapse; width: 100%; }}
                    .stats td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                    .chart {{ margin-bottom: 30px; text-align: center; }}
                    .chart img {{ max-width: 100%; }}
                    .attention-status {{ margin-top: 20px; }}
                    .status-box {{ display: inline-block; padding: 10px 20px; margin: 10px; border-radius: 5px; font-weight: bold; }}
                    .attentive {{ background-color: #d4edda; color: #155724; }}
                    .inattentive {{ background-color: #f8d7da; color: #721c24; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>注意力集中情况报告</h1>
                    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <div class="stats">
                        <h2>心率统计信息</h2>
                        <table>
                            <tr><td>总数据点数量:</td><td>{len(bpm_data)}</td></tr>
                            <tr><td>平均心率:</td><td>{avg_bpm} BPM</td></tr>
                            <tr><td>最高心率:</td><td>{max_bpm} BPM</td></tr>
                            <tr><td>最低心率:</td><td>{min_bpm} BPM</td></tr>
                            <tr><td>心率标准差:</td><td>{std_bpm} BPM</td></tr>
                        </table>
                    </div>
                    
                    <div class="chart">
                        <h2>心率波形图</h2>
                        <img src="{heart_rate_img}" alt="心率波形图">
                    </div>
                    
                    <div class="attention-status">
                        <h2>注意力状态分析</h2>
                        """
            
            # 添加注意力分析结果
            if attention_results:
                attentive_count = sum(attention_results)
                inattentive_count = len(attention_results) - attentive_count
                attentive_ratio = round(attentive_count / len(attention_results) * 100, 1)
                
                html_content += f"""
                <p>分析的时间窗口数量: {len(attention_results)}</p>
                <div class="status-box attentive">集中: {attentive_count} ({attentive_ratio}%)</div>
                <div class="status-box inattentive">不集中: {inattentive_count} ({round(100 - attentive_ratio, 1)}%)</div>
                
                <h3>注意力状态时间序列</h3>
                <p>"●" 表示集中，"○" 表示不集中</p>
                <div style="font-family: monospace; white-space: pre; margin: 20px 0;">
                """
            
                # 生成注意力状态可视化
                status_str = ""
                for i, status in enumerate(attention_results):
                    if i > 0 and i % 50 == 0:
                        status_str += "\n"
                    status_str += "● " if status else "○ "
                
                html_content += status_str + "\n</div>"
            else:
                html_content += "<p>数据量不足，无法进行注意力状态分析</p>"
            
            # 结束HTML内容
            html_content += """
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 写入HTML文件
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML报告已生成: {report_filename}")
            
            # 尝试打开HTML报告
            try:
                webbrowser.open(report_filename)
                print("正在打开HTML报告...")
            except Exception as e:
                print(f"无法自动打开HTML报告: {str(e)}")
            
            return True
        except Exception as e:
            print(f"生成HTML报告失败: {str(e)}")
            return False

def main():
    """主函数"""
    print("=== 注意力分析器 ===")
    
    # 创建分析器实例
    analyzer = AttentionAnalyzer()
    
    # 加载模型
    if not analyzer.load_model():
        return
    
    # 选择数据来源
    print("\n请选择数据来源:")
    print("1. 实时串口数据")
    print("2. CSV文件历史数据")
    
    choice = input("请输入选项 (1/2): ")
    
    bpm_data = None
    
    if choice == "1":
        # 显示可用串口
        print("\n可用串口:")
        ports = [port.device for port in serial.tools.list_ports.comports()]
        for i, port in enumerate(ports):
            print(f"{i+1}. {port}")
        
        if not ports:
            print("没有找到可用串口")
            return
        
        # 选择串口
        port_choice = input("请选择串口 (输入序号): ")
        try:
            port_idx = int(port_choice) - 1
            selected_port = ports[port_idx]
        except (ValueError, IndexError):
            print("无效的选择")
            return
        
        # 输入采集时长
        try:
            duration = int(input("请输入采集时长 (秒): "))
        except ValueError:
            print("无效的时长")
            return
        
        # 读取串口数据
        print(f"\n开始从串口 {selected_port} 采集 {duration} 秒的数据...")
        bpm_data = analyzer.read_serial_data(selected_port, duration=duration)
        
    elif choice == "2":
        # 输入CSV文件名
        filename = input("请输入CSV文件名 (包含路径): ")
        bpm_data = analyzer.load_csv_data(filename)
    
    else:
        print("无效的选择")
        return
    
    if not bpm_data:
        print("无法获取心率数据")
        return
    
    # 选择报告格式
    print("\n请选择报告格式:")
    print("1. PDF格式")
    print("2. HTML格式")
    
    report_choice = input("请输入选项 (1/2): ")
    
    if report_choice == "1":
        # 生成PDF报告
        analyzer.generate_report(bpm_data)
    elif report_choice == "2":
        # 生成HTML报告
        analyzer.generate_html_report(bpm_data)
    else:
        print("无效的选择")
        return
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()
