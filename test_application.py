import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import webbrowser
import os

# 加载模型
try:
    model = joblib.load("attention_model.joblib")
    print("模型加载成功！")
except FileNotFoundError:
    print("错误：未找到模型文件 attention_model.joblib")
    exit(1)

# 从测试文件中加载模拟BPM数据
df = pd.read_csv("test_bpm_data.csv")
bpm_data = df["bpm"].tolist()

# 特征提取函数
def extract_features(sequence):
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
    
    return [mean_bpm, max_bpm, min_bpm, avg_diff, diff_std, up_count, down_count, slope]

# 分析序列函数
def analyze_sequence(sequence):
    """分析单个心率序列的注意力状态"""
    features = extract_features(sequence)
    if features is None:
        return None
    
    return model.predict([features])[0]

# 生成心率波形图函数
def generate_heart_rate_chart(bpm_data, attention_results, chart_path="heart_rate_chart.png"):
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
    print(f"图表已生成: {chart_path}")

# 获取注意力评估函数
def get_attention_evaluation(focused_ratio):
    """根据注意力集中比例生成评估"""
    if focused_ratio >= 0.8:
        return "优秀：您大部分时间都保持着高度集中的注意力！"
    elif focused_ratio >= 0.6:
        return "良好：您的注意力集中情况总体不错，继续保持！"
    elif focused_ratio >= 0.4:
        return "一般：您的注意力容易分散，建议适当休息和调整！"
    else:
        return "较差：您的注意力集中时间较少，建议调整学习或工作方式！"

# 生成报告函数
def generate_report(bpm_data):
    """生成注意力集中情况报告"""
    if not bpm_data:
        print("错误：没有可用的心率数据")
        return False
    
    # 分析注意力状态
    attention_results = []
    sequence_length = 50  # 与应用程序中的参数保持一致
    
    # 滑动窗口分析
    for i in range(len(bpm_data) - sequence_length + 1):
        window = bpm_data[i:i+sequence_length]
        result = analyze_sequence(window)
        if result is not None:
            attention_results.append(result)
        else:
            attention_results.append(0)
    
    # 填充前面的结果
    padding = [0] * (sequence_length - 1)
    attention_results = padding + attention_results
    
    # 计算注意力集中的比例
    if attention_results:
        focused_count = sum(attention_results)
        focused_ratio = focused_count / len(attention_results)
    else:
        focused_count = 0
        focused_ratio = 0
    
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>注意力集中情况报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .stats {{ display: flex; gap: 30px; margin: 20px 0; }}
            .stat-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; min-width: 150px; }}
            .chart-container {{ margin: 30px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .legend {{ margin: 20px 0; }}
            .legend-item {{ display: inline-block; margin-right: 20px; }}
            .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; vertical-align: middle; }}
            .focused {{ background-color: #4CAF50; }}
            .not-focused {{ background-color: #f44336; }}
        </style>
    </head>
    <body>
        <h1>注意力集中情况报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>报告摘要</h2>
            <div class="stats">
                <div class="stat-box">
                    <strong>总数据点:</strong> {len(bpm_data)}<br>
                    <strong>分析窗口:</strong> {sequence_length} 个数据点
                </div>
                <div class="stat-box">
                    <strong>注意力集中次数:</strong> {focused_count}<br>
                    <strong>注意力不集中次数:</strong> {len(attention_results) - focused_count}
                </div>
                <div class="stat-box">
                    <strong>注意力集中比例:</strong> {focused_ratio:.2%}<br>
                    <strong>平均心率:</strong> {np.mean(bpm_data):.1f} BPM
                </div>
            </div>
            
            <h3>注意力状态评估</h3>
            <p>{get_attention_evaluation(focused_ratio)}</p>
        </div>
        
        <div class="chart-container">
            <h2>心率波形与注意力状态</h2>
            <img src="heart_rate_chart.png" alt="心率波形图">
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color focused"></span> 注意力集中
                </div>
                <div class="legend-item">
                    <span class="legend-color not-focused"></span> 注意力不集中
                </div>
            </div>
        </div>
        
        <h2>详细分析</h2>
        <p>报告基于心率序列的动态变化趋势进行注意力状态评估。</p>
        <p>注意力集中状态（绿色）表示心率变化呈现与集中状态相关的模式，</p>
        <p>注意力不集中状态（红色）表示心率变化呈现与分散状态相关的模式。</p>
    </body>
    </html>
    """
    
    # 生成心率波形图
    generate_heart_rate_chart(bpm_data, attention_results)
    
    # 保存HTML报告
    try:
        report_path = "attention_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"报告已生成: {report_path}")
        
        # 自动打开报告
        webbrowser.open(report_path)
        return True
    except Exception as e:
        print(f"生成报告时出错：{e}")
        return False

# 运行测试
if __name__ == "__main__":
    print(f"测试数据加载成功，共 {len(bpm_data)} 个数据点")
    print(f"数据范围：{min(bpm_data)} - {max(bpm_data)} BPM")
    print(f"平均心率：{np.mean(bpm_data):.1f} BPM")
    print()
    
    # 生成报告
    if generate_report(bpm_data):
        print("测试完成！")
    else:
        print("测试失败！")
