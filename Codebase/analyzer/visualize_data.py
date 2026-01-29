"""
Data Visualization for ABSTE Dataset
Tạo các biểu đồ đẹp để trực quan hóa dữ liệu
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
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
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
        self.dataset_stats = defaultdict(lambda: defaultdict(int))
        self.span_lengths = defaultdict(list)
        
        for dataset_name in self.datasets:
            for split in self.splits:
                data = self.load_dataset(dataset_name, split)
                if data is None:
                    continue
                
                for sent in data:
                    # Sentiment distribution
                    sentiments = sent.get('sentiments', [])
                    for sentiment in sentiments:
                        stype = sentiment.get('type', 'unknown')
                        self.sentiment_dist[dataset_name][stype] += 1
                    
                    # Dataset stats
                    self.dataset_stats[dataset_name]['sentences'] += 1
                    self.dataset_stats[dataset_name]['sentiments'] += len(sentiments)
                    
                    # Entity span lengths
                    entities = sent.get('entities', [])
                    for entity in entities:
                        start = entity.get('start', 0)
                        end = entity.get('end', 0)
                        length = end - start
                        self.span_lengths[dataset_name].append(length)
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution per dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phân bố Cảm xúc theo Tập dữ liệu', fontsize=14, fontweight='bold')
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
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add percentage
            total = sum(values)
            for i, (bar, val) in enumerate(zip(bars, values)):
                pct = (val / total * 100) if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{pct:.1f}%',
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            ax.set_title(f'{dataset_name} - Phân bố Cảm xúc', fontsize=12, fontweight='bold')
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
        
        datasets = list(self.datasets)
        sentences = [self.dataset_stats[d]['sentences'] for d in datasets]
        sentiments = [self.dataset_stats[d]['sentiments'] for d in datasets]
        
        # Plot 1: Number of sentences
        ax1 = axes[0]
        bars1 = ax1.bar(datasets, sentences, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Total Sentences per Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Sentences', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Number of sentiments (triplets)
        ax2 = axes[1]
        bars2 = ax2.bar(datasets, sentiments, color='#E67E22', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_title('Total Triplets per Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Triplets', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: dataset_comparison.png")
        plt.close()
    
    def plot_span_length_distribution(self):
        """Plot entity span length distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            
            lengths = self.span_lengths[dataset_name]
            if not lengths:
                continue
            
            # Count distribution
            length_counter = Counter(lengths)
            max_len = min(10, max(length_counter.keys()))  # Show up to 10
            
            x_vals = list(range(1, max_len + 1))
            y_vals = [length_counter.get(i, 0) for i in x_vals]
            
            bars = ax.bar(x_vals, y_vals, color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
            
            # Add percentage for 1-token and 2+ tokens
            total = sum(y_vals)
            one_token = y_vals[0]
            multi_token = sum(y_vals[1:])
            pct_one = (one_token / total * 100) if total > 0 else 0
            pct_multi = (multi_token / total * 100) if total > 0 else 0
            
            ax.set_title(f'{dataset_name} - Entity Span Length\n(1-token: {pct_one:.1f}%, Multi: {pct_multi:.1f}%)',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Span Length (tokens)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_xticks(x_vals)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('span_length_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: span_length_distribution.png")
        plt.close()
    
    def plot_class_imbalance_issue(self):
        """Highlight class imbalance problem"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sentiment_types = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        colors_sentiment = ['#2ECC71', '#E74C3C', '#95A5A6']
        
        dataset_totals = defaultdict(lambda: defaultdict(int))
        
        for dataset_name in self.datasets:
            sentiments = self.sentiment_dist[dataset_name]
            for stype in sentiment_types:
                dataset_totals[dataset_name][stype] = sentiments.get(stype, 0)
        
        # Prepare data for grouped bar chart
        x = np.arange(len(self.datasets))
        width = 0.25
        
        pos_counts = [dataset_totals[d]['POSITIVE'] for d in self.datasets]
        neg_counts = [dataset_totals[d]['NEGATIVE'] for d in self.datasets]
        neu_counts = [dataset_totals[d]['NEUTRAL'] for d in self.datasets]
        
        bars1 = ax.bar(x - width, pos_counts, width, label='POSITIVE', color=colors_sentiment[0], alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, neg_counts, width, label='NEGATIVE', color=colors_sentiment[1], alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, neu_counts, width, label='NEUTRAL', color=colors_sentiment[2], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Imbalance Problem: POSITIVE >> NEGATIVE > NEUTRAL', 
                    fontsize=13, fontweight='bold', color='#E74C3C')
        ax.set_xticks(x)
        ax.set_xticklabels(self.datasets)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add annotation
        ax.text(0.5, 0.95, '⚠️ Strong bias towards POSITIVE - needs weighted loss or oversampling',
               transform=ax.transAxes, fontsize=11, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.8, edgecolor='#E74C3C', linewidth=2))
        
        plt.tight_layout()
        plt.savefig('class_imbalance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: class_imbalance.png")
        plt.close()
    
    def plot_triplets_per_sentence(self):
        """Analyze triplets per sentence distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, dataset_name in enumerate(self.datasets):
            ax = axes[idx]
            
            triplets_per_sent = defaultdict(int)
            
            for split in self.splits:
                data = self.load_dataset(dataset_name, split)
                if data is None:
                    continue
                
                for sent in data:
                    triplet_count = len(sent.get('sentiments', []))
                    triplets_per_sent[triplet_count] += 1
            
            # Plot distribution
            x_vals = sorted(triplets_per_sent.keys())
            y_vals = [triplets_per_sent[x] for x in x_vals]
            
            bars = ax.bar(x_vals, y_vals, color='#16A085', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{dataset_name} - Triplets per Sentence', fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Triplets', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('triplets_per_sentence.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: triplets_per_sentence.png")
        plt.close()
    
    def visualize_all(self):
        """Generate all visualizations"""
        print("\n📊 Analyzing data...")
        self.analyze_all_data()
        
        print("📈 Creating visualizations...\n")
        self.plot_sentiment_distribution()
        self.plot_dataset_comparison()
        self.plot_span_length_distribution()
        self.plot_class_imbalance_issue()
        self.plot_triplets_per_sentence()
        
        print("\n" + "="*60)
        print("✅ All visualizations created successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. sentiment_distribution.png")
        print("  2. dataset_comparison.png")
        print("  3. span_length_distribution.png")
        print("  4. class_imbalance.png")
        print("  5. triplets_per_sentence.png")
        print("\nOpen these files to see detailed analysis!")
        print("="*60 + "\n")


def main():
    data_dir = '/home/haiyan/DESS-main/DESS-main/Codebase/data'
    
    visualizer = DataVisualizer(data_dir)
    visualizer.visualize_all()


if __name__ == '__main__':
    main()
