import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_dataset_stats():
    # 1. Khởi tạo dữ liệu từ bảng thống kê
    data = {
        'Dataset': ['14LAP', '14LAP', '14LAP', '14RES', '14RES', '14RES', 
                    '15RES', '15RES', '15RES', '16RES', '16RES', '16RES'],
        'Split': ['Train', 'Dev', 'Test', 'Train', 'Dev', 'Test', 
                  'Train', 'Dev', 'Test', 'Train', 'Dev', 'Test'],
        '#S': [906, 219, 328, 1266, 310, 492, 605, 148, 322, 857, 210, 326],
        '#T': [1460, 346, 543, 2338, 577, 994, 1013, 249, 485, 1394, 339, 514]
    }
    
    df = pd.DataFrame(data)
    datasets = df['Dataset'].unique()
    splits = df['Split'].unique()
    
    # 2. Thiết lập cấu hình biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(datasets))
    width = 0.25
    
    metrics = ['#S', '#T']
    titles = ['Số lượng câu (#S) theo tập', 'Số lượng bộ ba (#T) theo tập']
    colors = ['#66b3ff', '#99ff99', '#ffcc99'] # Blue, Green, Orange
    
    # 3. Vẽ biểu đồ cho từng chỉ số
    for i, metric in enumerate(metrics):
        for j, split in enumerate(splits):
            subset = df[df['Split'] == split]
            axes[i].bar(x + j*width, subset[metric], width, label=split, color=colors[j])
        
        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_xticks(x + width)
        axes[i].set_xticklabels(datasets)
        axes[i].set_ylabel('Số lượng')
        axes[i].legend()
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Thêm số liệu trên đầu cột
        for p in axes[i].patches:
            axes[i].annotate(str(int(p.get_height())), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='center', xytext=(0, 5), 
                             textcoords='offset points', fontsize=9)

    plt.tight_layout()
    
    # Thay plt.show() bằng lệnh lưu file
    plt.savefig('dataset_distribution.png', dpi=300)
    print("Biểu đồ đã được lưu tại: dataset_distribution.png")

# Gọi hàm thực thi
visualize_dataset_stats()