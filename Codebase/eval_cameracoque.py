"""
Evaluate ASTE Model on cameraCOQE Dataset
Cross-domain evaluation using converted COQE data
"""

import json
import os
import sys

# Evaluate cameraCOQE ASTE data with your existing model
def load_cameracoque_aste_data():
    """Load cameraCOQE ASTE format data"""
    base_path = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE_ASTE'
    
    data = {}
    for split in ['train', 'dev', 'test']:
        filepath = os.path.join(base_path, f'{split}_aste.json')
        with open(filepath, 'r') as f:
            data[split] = json.load(f)
    
    return data


def format_cameracoque_for_evaluation():
    """
    Format cameraCOQE ASTE data to match your original ASTE format
    if needed for compatibility
    """
    data = load_cameracoque_aste_data()
    
    print("\n" + "="*80)
    print("CAMERACOQUE ASTE DATASET - READY FOR EVALUATION")
    print("="*80)
    
    for split, samples in data.items():
        print(f"\n{split.upper()}:")
        print(f"  Samples: {len(samples)}")
        
        # Count triplets by sentiment
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_triplets = 0
        for sample in samples:
            for triplet in sample['triplets']:
                total_triplets += 1
                label = triplet['aspect_sentiment_label']
                if label in sentiments:
                    sentiments[label] += 1
        
        print(f"  Triplets: {total_triplets}")
        print(f"    Positive: {sentiments['positive']} ({100*sentiments['positive']/max(total_triplets,1):.1f}%)")
        print(f"    Negative: {sentiments['negative']} ({100*sentiments['negative']/max(total_triplets,1):.1f}%)")
        print(f"    Neutral: {sentiments['neutral']} ({100*sentiments['neutral']/max(total_triplets,1):.1f}%)")
        
        # Sample
        sample = samples[0]
        print(f"\n  SAMPLE:")
        print(f"    Sentence: {sample['sentence'][:100]}")
        print(f"    Triplets: {len(sample['triplets'])}")
        for trip in sample['triplets'][:2]:
            print(f"      - ({trip['aspect_term']}, {trip['opinion_term']}, {trip['aspect_sentiment_label']})")


def create_evaluation_guide():
    """Create guide for using cameraCOQE with your model"""
    guide = """
GUIDE: Evaluating ASTE Model on cameraCOQE Dataset
===================================================

Your cameraCOQE data has been converted to ASTE format:
  Location: data/cameraCOQE_ASTE/
  Files: train_aste.json, dev_aste.json, test_aste.json
  Format: [{"sentence": "...", "triplets": [{"aspect_term": "...", "opinion_term": "...", "aspect_sentiment_label": "..."}, ...]}, ...]

DATA STATISTICS:
  Train: 649 sentences, 2,032 triplets (3.1 triplets/sentence)
  Dev: 165 sentences, 514 triplets (3.1 triplets/sentence)
  Test: 200 sentences, 488 triplets (2.4 triplets/sentence)

CLASS DISTRIBUTION:
  The dataset is based on COMPARATIVE OPINIONS - expect different distribution
  than typical ASTE datasets which focus on single opinions

USAGE:
  1. Load cameraCOQE_ASTE data instead of original ASTE files
  2. Update your data loader to use the new file paths
  3. Run evaluation using your existing evaluation pipeline
  4. Compare results: cameraCOQE (cross-domain) vs original (in-domain)

EXPECTED PERFORMANCE DROP:
  - Cross-domain evaluation typically shows 5-15% F1 drop
  - Different domain (cameras → cameras but different annotation style)
  - Comparative opinions may be harder than standard opinions

NEXT STEPS:
  1. Update Parameter.py to reference cameraCOQE_ASTE if needed
  2. Run evaluation script with cameraCOQE data
  3. Analyze cross-domain performance
  4. Consider domain adaptation techniques
"""
    
    return guide


if __name__ == '__main__':
    data = load_cameracoque_aste_data()
    format_cameracoque_for_evaluation()
    
    print("\n" + "="*80)
    print(guide := create_evaluation_guide())
    print("="*80)
    
    # Save guide
    with open('/home/haiyan/DESS-main/DESS-main/Codebase/CAMERACOQUE_EVALUATION_GUIDE.md', 'w') as f:
        f.write(guide)
    print("\nGuide saved to: CAMERACOQUE_EVALUATION_GUIDE.md")
