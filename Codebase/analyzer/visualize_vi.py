"""
Data Visualization for ASTE Dataset - Vietnamese labels + correct paths
Trực quan hóa dữ liệu ASTE
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict, Counter
import seaborn as sns

# Set style
rcParams['figure.figsize'] = (16, 10)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

class DataVisualizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.datasets = ['14lap', '14res', '15res', '16res']
        self.splits = ['train', 'dev', 'test']
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        self.output_dir = '../images'
        
    def load_dataset(self, dataset_name, split):
        """Load a single dataset split"""
        path = os.path.join(
            self.data_dir, 
            dataset_name, 
            f'{split}_dep_triple_polarity_result.json'
        )
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def analyze_all_data(self):
        """Analyze all datasets"""
        self.sentiment_dist = defaultdict(lambda: defaultdict(int))
        self.sentiment_dist_by_split = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.dataset_stats = defaultdict(lambda: defaultdict(int))
        self.span_lengths = defaultdict(list)
        self.triplets_per_sentence = defaultdict(list)
        self.triplets_per_sentence_by_split = defaultdict(list)
        
        for dataset_name in self.datasets:
            for split in self.splits:
                data = self.load_dataset(dataset_name, split)
                if data is None:
                    continue
                
                for sample in data:
                    self.dataset_stats[dataset_name]['sentences'] += 1
                    
                    # Count entities by type
                    entities_by_type = defaultdict(list)
                    for entity in sample.get('entities', []):
                        ent_type = entity['type']
                        entities_by_type[ent_type].append(entity)
                    
                    # Count triplets and sentiments
                    sentiments = sample.get('sentiments', [])
                    self.dataset_stats[dataset_name]['sentiments'] += len(sentiments)
                    self.triplets_per_sentence[dataset_name].append(len(sentiments))
                    self.triplets_per_sentence_by_split[f"{dataset_name}_{split}"].append(len(sentiments))
                    
                    # Count sentiment types
                    for sentiment in sentiments:
                        sent_type = sentiment.get('type', 'NEUTRAL')
                        self.sentiment_dist[dataset_name][sent_type] += 1
                        self.sentiment_dist_by_split[dataset_name][split][sent_type] += 1
                    
                    # Track entity span lengths
                    for entity in sample.get('entities', []):
                        span_length = entity['end'] - entity['start']
                        self.span_lengths[dataset_name].append(span_length)
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution per dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phân bố nhãn cảm xúc theo tập dữ liệu', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        sentiment_types = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        sentiment_labels = {'POSITIVE': 'Tích cực', 'NEGATIVE': 'Tiêu cực', 'NEUTRAL': 'Trung lập'}
        colors_sentiment = ['#2ECC71', '#E74C3C', '#95A5A6']
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            
            sentiments = self.sentiment_dist[dataset_name]
            values = [sentiments.get(st, 0) for st in sentiment_types]
            
            label_display = [sentiment_labels[st] for st in sentiment_types]
            bars = ax.bar(label_display, values, color=colors_sentiment, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            total = sum(values)
            for i, (bar, val) in enumerate(zip(bars, values)):
                pct = (val / total * 100) if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2., val/2,
                       f'{pct:.1f}%',
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'sentiment_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {output_path}")
        plt.close()
    
    def plot_dataset_comparison(self):
        """Compare datasets by metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('So sánh tập dữ liệu: số câu vs số bộ ba', fontsize=14, fontweight='bold')
        
        datasets = list(self.datasets)
        sentences = [self.dataset_stats[d]['sentences'] for d in datasets]
        sentiments = [self.dataset_stats[d]['sentiments'] for d in datasets]
        
        ax1 = axes[0]
        bars1 = ax1.bar(datasets, sentences, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Tổng số câu trên mỗi tập dữ liệu', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Số câu', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2 = axes[1]
        bars2 = ax2.bar(datasets, sentiments, color='#E67E22', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_title('Tổng bộ ba trên mỗi tập dữ liệu', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Số bộ ba', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'dataset_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {output_path}")
        plt.close()
    
    def plot_span_length_distribution(self):
        """Plot distribution of entity span lengths"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phân bố độ dài khoảng Thực thể', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            lengths = self.span_lengths[dataset_name]
            
            if not lengths:
                ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
                continue
            
            ax.hist(lengths, bins=range(1, min(max(lengths)+2, 20)), 
                   color='#45B7D1', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Độ dài khoảng (tokens)', fontsize=10)
            ax.set_ylabel('Tần số', fontsize=10)
            ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'span_length_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {output_path}")
        plt.close()
    
    def plot_class_imbalance(self):
        """Highlight class imbalance problem"""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle('Vấn đề mất cân bằng lớp', fontsize=14, fontweight='bold')
        
        datasets = list(self.datasets)
        positive_pcts = []
        negative_pcts = []
        neutral_pcts = []
        
        for dataset in datasets:
            sentiments = self.sentiment_dist[dataset]
            total = sum(sentiments.values())
            if total == 0:
                positive_pcts.append(0)
                negative_pcts.append(0)
                neutral_pcts.append(0)
            else:
                positive_pcts.append(100 * sentiments.get('POSITIVE', 0) / total)
                negative_pcts.append(100 * sentiments.get('NEGATIVE', 0) / total)
                neutral_pcts.append(100 * sentiments.get('NEUTRAL', 0) / total)
        
        x = np.arange(len(datasets))
        width = 0.25
        
        ax.bar(x - width, positive_pcts, width, label='Tích cực', color='#2ECC71', edgecolor='black')
        ax.bar(x, negative_pcts, width, label='Tiêu cực', color='#E74C3C', edgecolor='black')
        ax.bar(x + width, neutral_pcts, width, label='Trung lập', color='#95A5A6', edgecolor='black')
        
        ax.set_ylabel('Phần trăm (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Tập dữ liệu', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # ax.text(0.5, 0.95, 'Nhãn Trung lập chiếm ưu thế trong tất cả các tập dữ liệu,',
        #         transform=ax.transAxes, fontsize=11, fontweight='bold',
        #         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        #         ha='center', va='top')
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'class_imbalance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {output_path}")
        plt.close()
    
    def plot_triplets_per_sentence(self):
        """Plot distribution of triplets per sentence"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phân bố số bộ ba trên câu', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            counts = self.triplets_per_sentence[dataset_name]
            
            if not counts:
                ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
                continue
            
            max_count = max(counts) + 1
            ax.hist(counts, bins=range(0, max_count+1), 
                   color='#4ECDC4', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Số bộ ba trên câu', fontsize=10)
            ax.set_ylabel('Tần số', fontsize=10)
            ax.set_title(f'{dataset_name} (Trung bình: {np.mean(counts):.1f})', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'triplets_per_sentence.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {output_path}")
        plt.close()
    
    def plot_sentiment_distribution_by_split(self):
        """Visualize sentiment distribution broken down by train/dev/test splits"""
        print('Tạo biểu đồ phân bố Cảm xúc theo Tập...')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phân bố nhãn cảm xúc theo tập train/dev/test', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        sentiment_order = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        sentiment_labels = {'POSITIVE': 'Tích cực', 'NEGATIVE': 'Tiêu cực', 'NEUTRAL': 'Trung lập'}
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            
            splits_data = []
            x_labels = []
            
            for split in self.splits:
                data = self.sentiment_dist_by_split[dataset_name][split]
                total = sum(data.values())
                if total > 0:
                    splits_data.append([data.get(s, 0) / total * 100 for s in sentiment_order])
                    x_labels.append(split.upper())
            
            if splits_data:
                x = np.arange(len(x_labels))
                width = 0.25
                
                for i, sentiment in enumerate(sentiment_order):
                    values = [splits_data[j][i] for j in range(len(splits_data))]
                    ax.bar(x + i*width, values, width, label=sentiment_labels[sentiment], 
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
                
                ax.set_xlabel('Tập dữ liệu', fontsize=11, fontweight='bold')
                ax.set_ylabel('Tỷ lệ %', fontsize=11, fontweight='bold')
                ax.set_title(f'{dataset_name}: Phân bố nhãn cảm xúc theo tập', fontsize=12, fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels(x_labels, fontsize=10)
                ax.legend(fontsize=10, loc='upper right')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([0, 100])
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'sentiment_distribution_by_split.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {path}")
        plt.close()
    
    def plot_dataset_stats_by_split(self):
        """Visualize dataset statistics broken down by splits"""
        print('Tạo biểu đồ Thống kê Tập dữ liệu theo Tập...')
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Số câu và số bộ ba theo tập train/dev/test', fontsize=14, fontweight='bold')
        
        # Prepare data for sentences and triplets per split
        datasets = []
        sentences_data = {split: [] for split in self.splits}
        triplets_data = {split: [] for split in self.splits}
        
        for dataset_name in self.datasets:
            datasets.append(dataset_name)
            for split in self.splits:
                # Count sentences in this split
                data = self.load_dataset(dataset_name, split)
                if data is not None:
                    sentences_data[split].append(len(data))
                    triplets_data[split].append(sum(self.triplets_per_sentence_by_split[f"{dataset_name}_{split}"]))
                else:
                    sentences_data[split].append(0)
                    triplets_data[split].append(0)
        
        # Plot sentences per split
        x = np.arange(len(datasets))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, split in enumerate(self.splits):
            axes[0].bar(x + i*width, sentences_data[split], width, label=split.upper(), 
                       color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        axes[0].set_xlabel('Tập dữ liệu', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Số câu', fontsize=11, fontweight='bold')
        axes[0].set_title('Số câu theo tập dữ liệu', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(datasets, fontsize=10)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot triplets per split
        for i, split in enumerate(self.splits):
            axes[1].bar(x + i*width, triplets_data[split], width, label=split.upper(), 
                       color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        axes[1].set_xlabel('Tập dữ liệu', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Số bộ ba', fontsize=11, fontweight='bold')
        axes[1].set_title('Số bộ ba theo tập dữ liệu', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(datasets, fontsize=10)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'dataset_stats_by_split.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✅ Lưu: {path}")
        plt.close()
    
    def plot_class_distribution_heatmap(self):
        """Visualize sentiment class distribution as heatmap"""
        print('Tạo bản đồ nhiệt phân bố lớp theo tập...')
        
        sentiment_order = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        data_matrix = []
        labels = []
        
        for dataset_name in self.datasets:
            for split in self.splits:
                data = self.sentiment_dist_by_split[dataset_name][split]
                total = sum(data.values())
                if total > 0:
                    row = [data.get(s, 0) / total * 100 for s in sentiment_order]
                    data_matrix.append(row)
                    labels.append(f"{dataset_name}\n({split.upper()})")
        
        if data_matrix:
            data_matrix = np.array(data_matrix)
            fig, ax = plt.subplots(figsize=(10, 12))
            
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            sentiment_labels = ['Tích cực', 'Tiêu cực', 'Trung lập']
            ax.set_xticks(np.arange(len(sentiment_labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(sentiment_labels, fontsize=11, fontweight='bold')
            ax.set_yticklabels(labels, fontsize=9)
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(sentiment_order)):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                                 ha="center", va="center", color="black", fontsize=9, fontweight='bold')
            
            ax.set_title('Phân bố lớp cảm xúc theo tập (Bản đồ nhiệt)', fontsize=13, fontweight='bold', pad=20)
            cbar = fig.colorbar(im, ax=ax, label='Tỷ lệ (%)')
            
            plt.tight_layout()
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, 'class_distribution_heatmap.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"✅ Lưu: {path}")
            plt.close()
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print('Tạo biểu đồ Phân bố Cảm xúc...')
        self.plot_sentiment_distribution()
        
        print('Tạo biểu đồ So sánh Tập dữ liệu...')
        self.plot_dataset_comparison()
        
        print('Tạo biểu đồ Độ dài Khoảng...')
        self.plot_span_length_distribution()
        
        print('Tạo biểu đồ Mất cân bằng Lớp...')
        self.plot_class_imbalance()
        
        print('Tạo biểu đồ Bộ ba trên Câu...')
        self.plot_triplets_per_sentence()
        
        print('Tạo biểu đồ Phân bố Cảm xúc chi tiết theo Tập...')
        self.plot_sentiment_distribution_by_split()
        
        print('Tạo biểu đồ Thống kê chi tiết theo Tập...')
        self.plot_dataset_stats_by_split()
        
        print('Tạo bản đồ nhiệt Phân bố Lớp theo Tập...')
        self.plot_class_distribution_heatmap()


if __name__ == '__main__':
    data_dir = '../data'
    visualizer = DataVisualizer(data_dir)
    
    print('\n' + '='*80)
    print('TRỰC QUAN HÓA DỮ LIỆU ASTE')
    print('='*80 + '\n')
    
    visualizer.analyze_all_data()
    visualizer.create_all_visualizations()
    
    print('\n' + '='*80)
    print('✅ HOÀN THÀNH - Tất cả biểu đồ đã được lưu vào thư mục images/')
    print('='*80 + '\n')
