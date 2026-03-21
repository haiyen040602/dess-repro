"""
Comprehensive analysis of cameraCOQE dataset with visualizations
"""

import json
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class CameraCOQEAnalyzer:
    def __init__(self, data_dir='data/cameraCOQE'):
        self.data_dir = data_dir
        self.splits = ['train', 'dev', 'test']
        self.data = {}
        self.stats = defaultdict(lambda: defaultdict(dict))
        
    def load_data(self):
        """Load all splits"""
        for split in self.splits:
            filepath = os.path.join(self.data_dir, f'{split}_dep_triple_polarity_result.json')
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data[split] = json.load(f)
        print(f"✓ Loaded {sum(len(self.data[s]) for s in self.splits)} samples total")
    
    def compute_stats(self):
        """Compute comprehensive statistics"""
        for split in self.splits:
            samples = self.data[split]
            
            # Basic counts
            self.stats[split]['total_samples'] = len(samples)
            
            # Initialize counters
            all_tokens = []
            all_sent_labels = Counter()
            all_entity_types = Counter()
            entity_lengths = []
            sample_lengths = []
            triplets_per_sample = []
            sentiments_per_sample = []
            entities_per_sample = []
            targets_per_sample = []  # Count targets per sample
            opinions_per_sample = []  # Count opinions per sample
            unique_sentences = set()
            
            for sample in samples:
                tokens = sample.get('tokens', [])
                entities = sample.get('entities', [])
                sentiments = sample.get('sentiments', [])
                
                # Get unique sentence
                sentence_text = ' '.join(tokens)
                unique_sentences.add(sentence_text)
                
                # Token stats
                sample_lengths.append(len(tokens))
                all_tokens.extend(tokens)
                
                # Entity stats
                entities_per_sample.append(len(entities))
                target_count = 0
                opinion_count = 0
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    all_entity_types[entity_type] += 1
                    
                    # Count by type
                    if entity_type == 'target':
                        target_count += 1
                    elif entity_type == 'opinion':
                        opinion_count += 1
                    
                    # Entity length
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    entity_lengths.append(end - start)
                
                targets_per_sample.append(target_count)
                opinions_per_sample.append(opinion_count)
                
                # Sentiment stats
                sentiments_per_sample.append(len(sentiments))
                for sent in sentiments:
                    sent_type = sent.get('type', 'unknown')
                    all_sent_labels[sent_type] += 1
                
                # Triplets (entity pairs with sentiment)
                triplets_per_sample.append(len(sentiments))
            
            # Store computed stats
            self.stats[split]['unique_sentences'] = len(unique_sentences)
            self.stats[split]['total_triplets'] = sum(triplets_per_sample)
            self.stats[split]['avg_triplets_per_sample'] = np.mean(triplets_per_sample) if triplets_per_sample else 0
            self.stats[split]['max_triplets_per_sample'] = max(triplets_per_sample) if triplets_per_sample else 0
            self.stats[split]['triplet_dist'] = self._get_triplet_distribution(triplets_per_sample)
            
            self.stats[split]['total_tokens'] = len(all_tokens)
            self.stats[split]['unique_tokens'] = len(set(all_tokens))
            self.stats[split]['avg_tokens_per_sample'] = np.mean(sample_lengths)
            self.stats[split]['max_tokens_per_sample'] = max(sample_lengths) if sample_lengths else 0
            
            self.stats[split]['total_entities'] = sum(all_entity_types.values())
            self.stats[split]['avg_entities_per_sample'] = np.mean(entities_per_sample)
            self.stats[split]['entity_types'] = dict(all_entity_types)
            
            # Separate target and opinion averages
            self.stats[split]['avg_targets_per_sample'] = np.mean(targets_per_sample) if targets_per_sample else 0
            self.stats[split]['avg_opinions_per_sample'] = np.mean(opinions_per_sample) if opinions_per_sample else 0
            
            self.stats[split]['avg_entity_length'] = np.mean(entity_lengths) if entity_lengths else 0
            self.stats[split]['entity_length_dist'] = self._get_length_distribution(entity_lengths)
            
            self.stats[split]['total_sentiments'] = sum(all_sent_labels.values())
            self.stats[split]['avg_sentiments_per_sample'] = np.mean(sentiments_per_sample)
            self.stats[split]['sentiment_labels'] = dict(all_sent_labels)
            
            self.stats[split]['sample_length_dist'] = self._get_length_distribution(sample_lengths)
            
    def _get_length_distribution(self, lengths):
        """Get length distribution buckets"""
        if not lengths:
            return {}
        dist = Counter()
        for length in lengths:
            if length == 1:
                dist['1'] += 1
            elif length == 2:
                dist['2'] += 1
            elif length <= 5:
                dist['3-5'] += 1
            elif length <= 7:
                dist['6-7'] += 1
            else:
                dist['8+'] += 1
        return dict(dist)
    
    def _get_triplet_distribution(self, triplets):
        """Get triplet count distribution"""
        dist = Counter(triplets)
        return dict(sorted(dist.items()))
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("CAMERACOQUE DATASET ANALYSIS")
        print("="*80)
        
        # Overall summary
        print("\n📊 DATASET OVERVIEW")
        print("-" * 80)
        total_samples = sum(self.stats[s]['total_samples'] for s in self.splits)
        total_unique_sentences = sum(self.stats[s]['unique_sentences'] for s in self.splits)
        total_entities = sum(self.stats[s]['total_entities'] for s in self.splits)
        total_triplets = sum(self.stats[s]['total_triplets'] for s in self.splits)
        total_sentiments = sum(self.stats[s]['total_sentiments'] for s in self.splits)
        
        print(f"Total Samples:           {total_samples:>6}")
        print(f"Total Unique Sentences:  {total_unique_sentences:>6}")
        print(f"Total Triplets:          {total_triplets:>6}")
        print(f"Total Entities:          {total_entities:>6}")
        print(f"Total Sentiments:        {total_sentiments:>6}")
        print(f"Avg Entities/Sample:     {total_entities/total_samples:>6.2f}")
        print(f"Avg Triplets/Sample:     {total_triplets/total_samples:>6.2f}")
        print(f"Avg Sentiments/Sample:   {total_sentiments/total_samples:>6.2f}")
        
        # Per-split statistics
        print("\n" + "-" * 80)
        print(f"{'Split':<10} {'Samples':<10} {'Sents':<10} {'Triplets':<10} {'Entities':<12} {'Avg Tokens':<12}")
        print("-" * 80)
        
        for split in self.splits:
            s = self.stats[split]
            print(f"{split:<10} {s['total_samples']:<10} {s['unique_sentences']:<10} "
                  f"{s['total_triplets']:<10} {s['total_entities']:<12} {s['avg_tokens_per_sample']:<12.1f}")
        
        # Entity types
        print("\n📋 LOẠI THỰC THỂ")
        print("-" * 80)
        all_entity_types = Counter()
        for split in self.splits:
            for etype, count in self.stats[split]['entity_types'].items():
                all_entity_types[etype] += count
        
        print("Tổng thể:")
        total_entities_overall = sum(all_entity_types.values())
        for etype, count in all_entity_types.most_common():
            pct = 100 * count / total_entities_overall
            print(f"  {etype:<15}: {count:>6} ({pct:>5.1f}%)")
        
        # Per-split entity types
        print("\nChi tiết theo tập:")
        print(f"{'Split':<10} {'Target':<12} {'Opinion':<12} {'Total':<10}")
        print("-" * 80)
        for split in self.splits:
            entity_types = self.stats[split]['entity_types']
            target = entity_types.get('target', 0)
            opinion = entity_types.get('opinion', 0)
            total = target + opinion
            print(f"{split:<10} {target:<12} {opinion:<12} {total:<10}")
        
        # Entity length distribution
        print("\n📏 ENTITY LENGTH DISTRIBUTION (Tokens)")
        print("-" * 80)
        print(f"{'Split':<10} {'1':<10} {'2':<10} {'3-5':<10} {'6-7':<10} {'8+':<10} {'Avg Length':<12}")
        print("-" * 80)
        
        for split in self.splits:
            dist = self.stats[split]['entity_length_dist']
            print(f"{split:<10} "
                  f"{dist.get('1', 0):<10} "
                  f"{dist.get('2', 0):<10} "
                  f"{dist.get('3-5', 0):<10} "
                  f"{dist.get('6-7', 0):<10} "
                  f"{dist.get('8+', 0):<10} "
                  f"{self.stats[split]['avg_entity_length']:<12.2f}")
        
        # Triplet distribution per sample
        print("\n📈 TRIPLETS PER SAMPLE DISTRIBUTION")
        print("-" * 80)
        for split in self.splits:
            print(f"\n{split.upper()}:")
            dist = self.stats[split]['triplet_dist']
            print(f"  Avg triplets/sample: {self.stats[split]['avg_triplets_per_sample']:.2f}")
            print(f"  Max triplets/sample: {self.stats[split]['max_triplets_per_sample']}")
            print(f"  Distribution:")
            for count in sorted(dist.keys()):
                freq = dist[count]
                pct = 100 * freq / self.stats[split]['total_samples']
                bar_len = int(pct / 2)
                bar = '█' * bar_len + '░' * (30 - bar_len)
                print(f"    {count} triplet(s): {freq:>4} samples ({pct:>5.1f}%) {bar}")
        
        # Target and Opinion per-sample averages
        print("\n👥 TARGET & OPINION PER-SAMPLE AVERAGES")
        print("-" * 80)
        print(f"{'Split':<10} {'Avg Targets':<15} {'Avg Opinions':<15}")
        print("-" * 80)
        for split in self.splits:
            avg_targets = self.stats[split]['avg_targets_per_sample']
            avg_opinions = self.stats[split]['avg_opinions_per_sample']
            print(f"{split:<10} {avg_targets:<15.2f} {avg_opinions:<15.2f}")
        
        # Overall averages
        overall_avg_targets = sum(self.stats[s]['avg_targets_per_sample'] * self.stats[s]['total_samples'] for s in self.splits) / total_samples
        overall_avg_opinions = sum(self.stats[s]['avg_opinions_per_sample'] * self.stats[s]['total_samples'] for s in self.splits) / total_samples
        print(f"{'Overall':<10} {overall_avg_targets:<15.2f} {overall_avg_opinions:<15.2f}")
        
        # Sentiment distribution
        print("\n💭 SENTIMENT DISTRIBUTION")
        print("-" * 80)
        
        all_sentiments = Counter()
        for split in self.splits:
            for stype, count in self.stats[split]['sentiment_labels'].items():
                all_sentiments[stype] += count
        
        total_sent = sum(all_sentiments.values())
        for stype in ['BETTER', 'EQUAL', 'WORSE', 'DIFFERENT']:
            count = all_sentiments.get(stype, 0)
            pct = 100 * count / total_sent if total_sent > 0 else 0
            bar_len = int(pct / 2)
            bar = '█' * bar_len + '░' * (50 - bar_len)
            print(f"  {stype:<12}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        # Per-split sentiment
        print("\n" + "-" * 80)
        print("Sentiment distribution per split:")
        print("-" * 80)
        
        for split in self.splits:
            print(f"\n{split.upper()}:")
            sentiments = self.stats[split]['sentiment_labels']
            total = sum(sentiments.values())
            for stype in ['BETTER', 'EQUAL', 'WORSE', 'DIFFERENT']:
                count = sentiments.get(stype, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"  {stype:<12}: {count:>5} ({pct:>5.1f}%)")
        
        # Token statistics
        print("\n📝 TOKEN STATISTICS")
        print("-" * 80)
        print(f"{'Split':<10} {'Total Tokens':<15} {'Unique':<10} {'Avg/Sample':<12} {'Max':<10}")
        print("-" * 80)
        
        for split in self.splits:
            s = self.stats[split]
            print(f"{split:<10} {s['total_tokens']:<15} {s['unique_tokens']:<10} "
                  f"{s['avg_tokens_per_sample']:<12.1f} {s['max_tokens_per_sample']:<10}")
        
        print("\n" + "="*80)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        
        # Vietnamese label mapping
        label_map = {
            'BETTER': 'BETTER',
            'EQUAL': 'EQUAL',
            'WORSE': 'WORSE',
            'DIFFERENT': 'DIFFERENT',
            'train': 'train',
            'dev': 'dev',
            'test': 'test'
        }
        
        # Create figure with subplots - 2 rows x 3 columns = 6 charts
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        
        # 1. Sentiment distribution (overall pie chart)
        ax1 = axes[0, 0]
        all_sentiments = Counter()
        for split in self.splits:
            for stype, count in self.stats[split]['sentiment_labels'].items():
                all_sentiments[stype] += count
        
        labels = ['BETTER', 'EQUAL', 'WORSE', 'DIFFERENT']
        labels_vi = [label_map[l] for l in labels]
        sizes = [all_sentiments.get(l, 0) for l in labels]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels_vi, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
        ax1.set_title('Phân bố cảm xúc tổng thể', fontsize=12, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 2. Sentiment distribution (bar chart per split)
        ax2 = axes[0, 1]
        splits_data = {split: self.stats[split]['sentiment_labels'] for split in self.splits}
        
        x = np.arange(len(labels))
        width = 0.25
        
        for i, split in enumerate(self.splits):
            sentiments = splits_data[split]
            total = sum(sentiments.values())
            values = [100 * sentiments.get(l, 0) / total if total > 0 else 0 for l in labels]
            ax2.bar(x + i*width, values, width, label=label_map[split], alpha=0.8)
        
        ax2.set_xlabel('Loại cảm xúc', fontweight='bold')
        ax2.set_ylabel('Phần trăm (%)', fontweight='bold')
        ax2.set_title('Phân bố cảm xúc theo tập dữ liệu', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(labels_vi)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Sample and entity counts
        ax3 = axes[0, 2]
        sample_counts = [self.stats[s]['total_samples'] for s in self.splits]
        entity_counts = [self.stats[s]['total_entities'] for s in self.splits]
        
        x = np.arange(len(self.splits))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, sample_counts, width, label='Câu', alpha=0.8, color='#3498db')
        bars2 = ax3.bar(x + width/2, entity_counts, width, label='Thực thể', alpha=0.8, color='#e74c3c')
        
        ax3.set_xlabel('Tập dữ liệu', fontweight='bold')
        ax3.set_ylabel('Số lượng', fontweight='bold')
        ax3.set_title('Số lượng câu và thực thể', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([label_map[s] for s in self.splits])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 4. Entity length distribution
        ax4 = axes[1, 0]
        length_labels = ['1', '2', '3-5', '6-7', '8+']
        
        for split in self.splits:
            dist = self.stats[split]['entity_length_dist']
            values = [dist.get(l, 0) for l in length_labels]
            ax4.plot(length_labels, values, marker='o', label=label_map[split], linewidth=2, markersize=8)
            
            # Add count labels on data points
            for i, (label, value) in enumerate(zip(length_labels, values)):
                ax4.text(i, value, str(value), ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Độ dài thực thể (từ)', fontweight='bold')
        ax4.set_ylabel('Số lượng', fontweight='bold')
        ax4.set_title('Phân bố độ dài thực thể', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Targets and opinions per sample (separated)
        ax5 = axes[1, 1]
        avg_targets = [self.stats[s]['avg_targets_per_sample'] for s in self.splits]
        avg_opinions = [self.stats[s]['avg_opinions_per_sample'] for s in self.splits]
        avg_sentiments = [self.stats[s]['avg_sentiments_per_sample'] for s in self.splits]
        
        x = np.arange(len(self.splits))
        width = 0.25
        
        bars1 = ax5.bar(x - width, avg_targets, width, label='Đối tượng', alpha=0.8, color='#3498db')
        bars2 = ax5.bar(x, avg_opinions, width, label='Ý kiến', alpha=0.8, color='#f39c12')
        bars3 = ax5.bar(x + width, avg_sentiments, width, label='Nhãn cảm xúc', alpha=0.8, color='#2ecc71')
        
        ax5.set_xlabel('Tập dữ liệu', fontweight='bold')
        ax5.set_ylabel('Số lượng trung bình', fontweight='bold')
        ax5.set_title('Trung bình đối tượng, ý kiến và nhãn cảm xúc mỗi câu', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([label_map[s] for s in self.splits])
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7)
        
        # 6. Tokens per sample
        ax6 = axes[1, 2]
        avg_tokens = [self.stats[s]['avg_tokens_per_sample'] for s in self.splits]
        max_tokens = [self.stats[s]['max_tokens_per_sample'] for s in self.splits]
        
        x = np.arange(len(self.splits))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, avg_tokens, width, label='TB từ', alpha=0.8, color='#3498db')
        bars2 = ax6.bar(x + width/2, max_tokens, width, label='Max từ', alpha=0.8, color='#e74c3c')
        
        ax6.set_xlabel('Tập dữ liệu', fontweight='bold')
        ax6.set_ylabel('Số từ', fontweight='bold')
        ax6.set_title('Số từ mỗi câu', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([label_map[s] for s in self.splits])
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Save figure
        output_path = 'cameracoque_analysis.png'
        plt.savefig(output_path, dpi=300)
        print(f"\n✓ Saved visualization: {output_path}")
        
        # Create additional detailed sentiment chart
        fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for idx, split in enumerate(self.splits):
            sentiments = self.stats[split]['sentiment_labels']
            labels_vi_list = [label_map[l] for l in labels]
            sizes = [sentiments.get(l, 0) for l in labels]
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            
            wedges, texts, autotexts = axes[idx].pie(sizes, labels=labels_vi_list, autopct='%1.1f%%',
                                                       colors=colors, startangle=90)
            axes[idx].set_title(f'{label_map[split]} (n={sum(sizes)})', fontsize=11, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        
        plt.tight_layout()
        output_path2 = 'cameracoque_sentiment_by_split.png'
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {output_path2}")
        
        plt.close('all')

def main():
    analyzer = CameraCOQEAnalyzer()
    analyzer.load_data()
    analyzer.compute_stats()
    analyzer.print_summary()
    analyzer.create_visualizations()
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
