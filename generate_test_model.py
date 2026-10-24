import csv
import joblib
from sklearn.neighbors import KNeighborsClassifier
import os

def main():
    """生成测试模型"""
    print("生成测试模型...")
    
    # 检查学习数据文件是否存在
    learning_data_file = "attention_learning_data.csv"
    if not os.path.exists(learning_data_file):
        print(f"错误: 找不到学习数据文件 {learning_data_file}")
        return
    
    # 加载学习数据
    print(f"加载学习数据文件: {learning_data_file}")
    learning_data = []
    
    try:
        with open(learning_data_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 解析特征
                features = [
                    float(row['mean_bpm']),
                    float(row['max_bpm']),
                    float(row['min_bpm']),
                    float(row['avg_diff']),
                    float(row['diff_std']),
                    float(row['up_count']) if 'up_count' in row else 0,
                    float(row['down_count']) if 'down_count' in row else 0,
                    float(row['slope']) if 'slope' in row else 0
                ]
                
                # 解析标签
                label = int(row['label'])
                
                learning_data.append({
                    'features': features,
                    'label': label
                })
    
        print(f"成功加载 {len(learning_data)} 个学习样本")
    
        if not learning_data:
            print("错误: 学习数据文件为空")
            return
        
        # 准备训练数据
        X = [data['features'] for data in learning_data]
        y = [data['label'] for data in learning_data]
        
        # 训练KNN模型
        print("训练KNN模型...")
        model = KNeighborsClassifier(n_neighbors=5)  # 使用5个邻居提高模型鲁棒性
        model.fit(X, y)
        
        # 保存模型到文件
        model_filename = "attention_model.joblib"
        joblib.dump(model, model_filename)
        
        print(f"模型已成功保存到: {model_filename}")
        print(f"模型训练完成，共使用 {len(learning_data)} 个样本")
    
    except Exception as e:
        print(f"生成模型时出错: {str(e)}")
        return

if __name__ == "__main__":
    main()
