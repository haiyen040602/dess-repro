"""
Convert cameraCOQE to ASTE training format (CORRECT PARSING)
cameraCOQE Format:
  - Odd lines: sentence<TAB>label (0 or 1)
  - Even lines: single quintuple [subjects];[objects];[predicates];[sentiments];[comparators]

Quintuple format:
  [item1&&word1, item2&&word2, ...];[...];[...];[...];[...]
  Each item: order_in_sentence&&word_text
"""

import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()


def parse_quintuple_component(component_str):
    """Parse single component like [7&&all 8&&of 9&&the] -> [(7, 'all'), (8, 'of'), (9, 'the')]"""
    component_str = component_str.strip()
    if not component_str or component_str == '[]':
        return []
    
    # Remove brackets
    if component_str.startswith('['):
        component_str = component_str[1:]
    if component_str.endswith(']'):
        component_str = component_str[:-1]
    
    items = component_str.split()
    result = []
    for item in items:
        if '&&' in item:
            try:
                idx_str, word = item.split('&&', 1)
                idx = int(idx_str)
                result.append((idx, word))
            except:
                pass
    
    return result


def parse_quintuple_line(quintuple_str):
    """
    Parse quintuple line: [subjects];[objects];[predicates];[sentiments];[comparators]
    Returns: (subjects, objects, predicates, sentiments, comparators)
    """
    # Split by semicolon
    parts = quintuple_str.split(';')
    if len(parts) < 5:
        # Fill missing parts
        parts.extend(['[]'] * (5 - len(parts)))
    
    subjects = parse_quintuple_component(parts[0])
    objects = parse_quintuple_component(parts[1])
    predicates = parse_quintuple_component(parts[2])
    sentiments = parse_quintuple_component(parts[3])
    comparators = parse_quintuple_component(parts[4])
    
    return subjects, objects, predicates, sentiments, comparators


def determine_sentiment(sentiment_text, context=""):
    """VADER-based sentiment classification"""
    full_text = f"{context} {sentiment_text}" if context else sentiment_text
    scores = vader_analyzer.polarity_scores(full_text)
    compound = scores['compound']
    
    if compound >= 0.1:
        return 'positive'
    elif compound <= -0.1:
        return 'negative'
    else:
        sentiment_lower = sentiment_text.lower()
        positive_words = {'good', 'great', 'excellent', 'best', 'better', 'nice', 'high', 
                          'faster', 'higher', 'sharper', 'brighter', 'clearer', 'superior', 
                          'improved', 'stronger', 'longer', 'easier', 'smoother'}
        negative_words = {'bad', 'poor', 'worse', 'worst', 'slow', 'slower', 'lower', 
                          'blurry', 'dull', 'dark', 'cheap', 'weak', 'inferior', 
                          'harder', 'difficult', 'complicated', 'shorter'}
        
        if any(word in sentiment_lower for word in positive_words):
            return 'positive'
        elif any(word in sentiment_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'


def parse_cameracoque_file(filepath):
    """
    Parse cameraCOQE format file
    Returns list of samples with label=1 (comparative sentences)
    """
    samples = []
    total_lines = 0
    total_label_0 = 0
    total_label_1 = 0
    total_empty_quintuple = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read sentence line
        sent_line = lines[i].rstrip('\r\n')
        i += 1
        total_lines += 1
        
        if not sent_line.strip():
            continue
        
        # Parse sentence line: text<TAB>label
        tab_pos = sent_line.rfind('\t')
        if tab_pos == -1:
            continue
        
        text = sent_line[:tab_pos].strip()
        try:
            label = int(sent_line[tab_pos+1:].strip())
        except:
            continue
        
        if label == 0:
            total_label_0 += 1
        else:
            total_label_1 += 1
        
        # Read quintuple line
        if i >= len(lines):
            continue
        
        quintuple_line = lines[i].rstrip('\r\n')
        i += 1
        
        try:
            subjects, objects, predicates, sentiments, comparators = parse_quintuple_line(quintuple_line)
        except:
            continue
        
        # Only keep label=1 (comparative sentences)
        if label == 1:
            # Check if empty
            if not predicates or not sentiments:
                total_empty_quintuple += 1
            
            samples.append({
                'text': text,
                'label': label,
                'subjects': subjects,
                'objects': objects,
                'predicates': predicates,
                'sentiments': sentiments,
                'comparators': comparators
            })
    
    print(f"  Parsed: {total_lines} lines | label=0: {total_label_0}, label=1: {total_label_1} | empty: {total_empty_quintuple}")
    return samples


def create_triplets(sample):
    """Create ASTE triplets from sample"""
    triplets = []
    text = sample['text']
    tokens = text.split()
    
    subjects = sample['subjects']  # [(idx, word), ...]
    objects = sample['objects']
    predicates = sample['predicates']
    sentiments = sample['sentiments']
    comparators = sample['comparators']
    
    # Extract text from items
    def get_text(items):
        return ' '.join([word for idx, word in sorted(items)])
    
    # Map indices to text
    pred_texts = get_text(predicates) if predicates else None
    sent_texts = get_text(sentiments) if sentiments else None
    subj_texts = get_text(subjects) if subjects else None
    obj_texts = get_text(objects) if objects else None
    
    # Use predicate as aspect, sentiment as opinion
    if pred_texts and sent_texts:
        sentiment_label = determine_sentiment(sent_texts, context=pred_texts)
        triplets.append({
            'aspect_term': pred_texts,
            'opinion_term': sent_texts,
            'aspect_sentiment_label': sentiment_label
        })
    elif subj_texts and obj_texts:
        # Fallback: use subject/object if no predicate/sentiment
        sentiment_label = determine_sentiment(obj_texts, context=subj_texts)
        triplets.append({
            'aspect_term': subj_texts,
            'opinion_term': obj_texts,
            'aspect_sentiment_label': sentiment_label
        })
    
    return triplets


def convert_to_aste_format(samples):
    """Convert samples to ASTE format"""
    aste_samples = []
    total_triplets = 0
    
    for sample in samples:
        triplets = create_triplets(sample)
        total_triplets += len(triplets)
        
        aste_sample = {
            'sentence': sample['text'],
            'triplets': triplets,
            'source': 'cameraCOQE',
            'is_comparative': True
        }
        aste_samples.append(aste_sample)
    
    return aste_samples, total_triplets


def main():
    base_path = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE'
    aste_output_dir = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE_ASTE'
    os.makedirs(aste_output_dir, exist_ok=True)
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        input_file = os.path.join(base_path, f'{split}.txt')
        output_file = os.path.join(aste_output_dir, f'{split}_aste.json')
        
        print(f"\n{'='*60}")
        print(f"Converting {split.upper()}")
        print(f"{'='*60}")
        
        # Parse cameraCOQE
        samples = parse_cameracoque_file(input_file)
        print(f"✓ Loaded {len(samples)} comparative samples (label=1)")
        
        # Convert to ASTE
        aste_samples, total_triplets = convert_to_aste_format(samples)
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aste_samples, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Samples: {len(aste_samples)}")
        print(f"  Total triplets: {total_triplets}")


if __name__ == '__main__':
    main()
