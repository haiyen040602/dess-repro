"""
Data Analysis Script for Aspect-Based Sentiment Triplet Extraction Dataset
Phân tích dữ liệu cơ bản từ 4 tập: 14lap, 14res, 15res, 16res
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.datasets = ['14lap', '14res', '15res', '16res']
        self.splits = ['train', 'dev', 'test']
        
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
    
    def analyze_dataset_split(self, dataset_name, split):
        """Analyze a single dataset split"""
        data = self.load_dataset(dataset_name, split)
        if data is None:
            return None
        
        stats = {
            'sentences': 0,
            'entities': 0,
            'sentiments': 0,
            'tokens_total': 0,
            'entity_types': Counter(),
            'sentiment_types': Counter(),
            'avg_tokens': 0,
            'avg_entities_per_sent': 0,
            'avg_sentiments_per_sent': 0,
        }
        
        for sent in data:
            stats['sentences'] += 1
            
            # Count tokens
            tokens = sent.get('tokens', [])
            stats['tokens_total'] += len(tokens)
            
            # Count entities
            entities = sent.get('entities', [])
            stats['entities'] += len(entities)
            for entity in entities:
                stats['entity_types'][entity.get('type', 'unknown')] += 1
            
            # Count sentiments
            sentiments = sent.get('sentiments', [])
            stats['sentiments'] += len(sentiments)
            for sentiment in sentiments:
                stats['sentiment_types'][sentiment.get('type', 'unknown')] += 1
        
        # Calculate averages
        if stats['sentences'] > 0:
            stats['avg_tokens'] = stats['tokens_total'] / stats['sentences']
            stats['avg_entities_per_sent'] = stats['entities'] / stats['sentences']
            stats['avg_sentiments_per_sent'] = stats['sentiments'] / stats['sentences']
        
        return stats
    
    def print_analysis(self):
        """Print comprehensive analysis of all datasets"""
        print("\n" + "="*100)
        print("DATA ANALYSIS REPORT - ASPECT-BASED SENTIMENT TRIPLET EXTRACTION")
        print("="*100)
        
        for dataset_name in self.datasets:
            print(f"\n{'='*100}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*100}")
            
            dataset_totals = {
                'sentences': 0,
                'entities': 0,
                'sentiments': 0,
                'tokens': 0,
            }
            
            for split in self.splits:
                stats = self.analyze_dataset_split(dataset_name, split)
                if stats is None:
                    print(f"  {split}: FILE NOT FOUND")
                    continue
                
                print(f"\n  {split.upper()} SET:")
                print(f"    Sentences: {stats['sentences']}")
                print(f"    Total tokens: {stats['tokens_total']}")
                print(f"    Avg tokens/sentence: {stats['avg_tokens']:.2f}")
                print(f"    Total entities: {stats['entities']}")
                print(f"    Avg entities/sentence: {stats['avg_entities_per_sent']:.2f}")
                print(f"    Total sentiments: {stats['sentiments']}")
                print(f"    Avg sentiments/sentence: {stats['avg_sentiments_per_sent']:.2f}")
                
                # Entity types distribution
                if stats['entity_types']:
                    print(f"    Entity types distribution:")
                    for etype, count in stats['entity_types'].most_common():
                        pct = (count / stats['entities'] * 100) if stats['entities'] > 0 else 0
                        print(f"      - {etype}: {count} ({pct:.1f}%)")
                
                # Sentiment types distribution
                if stats['sentiment_types']:
                    print(f"    Sentiment types distribution:")
                    for stype, count in stats['sentiment_types'].most_common():
                        pct = (count / stats['sentiments'] * 100) if stats['sentiments'] > 0 else 0
                        print(f"      - {stype}: {count} ({pct:.1f}%)")
                
                # Update dataset totals
                dataset_totals['sentences'] += stats['sentences']
                dataset_totals['entities'] += stats['entities']
                dataset_totals['sentiments'] += stats['sentiments']
                dataset_totals['tokens'] += stats['tokens_total']
            
            # Dataset summary
            print(f"\n  DATASET TOTAL:")
            print(f"    All sentences: {dataset_totals['sentences']}")
            print(f"    All entities: {dataset_totals['entities']}")
            print(f"    All sentiments: {dataset_totals['sentiments']}")
            print(f"    All tokens: {dataset_totals['tokens']}")
        
        # Overall statistics
        self.print_overall_stats()
    
    def print_overall_stats(self):
        """Print overall statistics across all datasets"""
        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")
        
        all_sentences = 0
        all_entities = 0
        all_sentiments = 0
        
        for dataset_name in self.datasets:
            for split in self.splits:
                stats = self.analyze_dataset_split(dataset_name, split)
                if stats is not None:
                    all_sentences += stats['sentences']
                    all_entities += stats['entities']
                    all_sentiments += stats['sentiments']
        
        print(f"  Total sentences across all datasets: {all_sentences}")
        print(f"  Total entities across all datasets: {all_entities}")
        print(f"  Total sentiments across all datasets: {all_sentiments}")
        
        if all_sentences > 0:
            print(f"  Average entities per sentence: {all_entities / all_sentences:.2f}")
            print(f"  Average sentiments per sentence: {all_sentiments / all_sentences:.2f}")
            print(f"  Entity-to-Sentiment ratio: {all_entities / all_sentiments:.2f}" if all_sentiments > 0 else "  N/A")


def main():
    data_dir = '../data'
    
    print("\n🔍 Bắt đầu Phân tích Dữ liệu...")
    analyzer = DataAnalyzer(data_dir)
    analyzer.print_analysis()
    
    print(f"\n{'='*100}")
    print("📊 Phân tích Hoàn thành!")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
