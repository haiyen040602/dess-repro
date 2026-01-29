"""
COQE to ASTE Converter
Convert Comparative Opinion Extraction dataset to ASTE format
"""

import json
import os
from collections import defaultdict

def parse_coqe_anno_list(anno_str):
    """Parse COQE annotation list"""
    anno_str = anno_str.strip()
    if anno_str.startswith('[') and anno_str.endswith(']'):
        anno_str = anno_str[1:-1]
    
    parts = anno_str.split('];[')
    result = []
    for part in parts:
        part = part.strip().strip('[]')
        if not part:
            result.append([])
        else:
            items = part.split()
            result.append(items)
    
    return result


def extract_spans(annotation_items):
    """Extract (token_idx, text) from items like ['9&&battery', '10&&item']"""
    spans = []
    for item in annotation_items:
        if '&&' in item:
            try:
                idx_str, text = item.split('&&', 1)
                idx = int(idx_str)
                spans.append((idx, text))
            except:
                pass
    return spans


def tokenize_text(text):
    """Simple tokenization"""
    # Split on whitespace and punctuation
    import re
    tokens = re.findall(r'\b\w+\b|[.,!?;:\'-]', text)
    return tokens


def get_span_boundaries(tokens, target_indices):
    """
    Find character boundaries for token indices in original text
    """
    if not target_indices:
        return []
    
    # Get unique sorted indices
    indices = sorted(set([idx for idx, _ in target_indices]))
    
    # Map indices to actual token boundaries
    text_copy = ' '.join(tokens)
    word_boundaries = []
    pos = 0
    
    for token in tokens:
        start = text_copy.find(token, pos)
        end = start + len(token)
        word_boundaries.append((start, end))
        pos = end
    
    spans = []
    for idx in indices:
        if 0 <= idx < len(word_boundaries):
            start, end = word_boundaries[idx]
            spans.append((start, end))
    
    return spans


def convert_coqe_to_aste(coqe_text, subjects, objects, predicates, sentiments):
    """
    Convert COQE format to ASTE triplet format
    COQE: (subject, object, predicate, sentiment)
    ASTE: (aspect_term, opinion_term, sentiment_label)
    
    Mapping:
    - aspect = subject (if exists) else object
    - opinion = predicate  
    - sentiment = sentiment label (map from text to positive/negative/neutral)
    """
    
    tokens = coqe_text.split()
    triplets = []
    
    # Extract spans
    subject_spans = extract_spans(subjects)
    pred_spans = extract_spans(predicates)
    sent_spans = extract_spans(sentiments)
    
    # Create mapping from index to text
    pred_idx_map = {idx: text for idx, text in pred_spans}
    sent_idx_map = {idx: text for idx, text in sent_spans}
    
    # Generate triplets: predicate + sentiment pairs
    for pred_idx, pred_text in pred_spans:
        # Find sentiments
        for sent_idx, sent_text in sent_spans:
            # Determine sentiment label based on sentiment text
            sentiment_label = determine_sentiment(sent_text, pred_text)
            
            # Use predicate as aspect, sentiment text as opinion
            if 0 <= pred_idx < len(tokens):
                aspect = tokens[pred_idx]
            else:
                aspect = pred_text
            
            triplet = (aspect, sent_text, sentiment_label)
            triplets.append(triplet)
    
    return triplets


def determine_sentiment(sentiment_text, context=""):
    """
    Determine sentiment label from text
    For COQE, need to infer if comparative is positive/negative
    """
    # Simple heuristic based on common words
    sentiment_text_lower = sentiment_text.lower()
    
    positive_words = {'good', 'great', 'excellent', 'best', 'better', 'nice', 'high', 
                      'faster', 'higher', 'sharper', 'brighter', 'clearer', 'quality'}
    negative_words = {'bad', 'poor', 'worse', 'worst', 'slow', 'slower', 'lower', 
                      'blurry', 'dull', 'dark', 'quality', 'cheaper', 'weak'}
    
    if any(word in sentiment_text_lower for word in positive_words):
        return 'positive'
    elif any(word in sentiment_text_lower for word in negative_words):
        return 'negative'
    else:
        return 'neutral'


def parse_coqe_file(filepath):
    """Load COQE file"""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        i += 1
        
        if not line.strip():
            continue
        
        tab_pos = line.rfind('\t')
        if tab_pos == -1:
            continue
        
        text = line[:tab_pos]
        label_str = line[tab_pos+1:].strip()
        
        try:
            label = int(label_str)
        except:
            continue
        
        if i >= len(lines):
            continue
        
        anno_str = lines[i].rstrip('\r\n')
        i += 1
        
        try:
            subjects, objects, predicates, sentiments, comparators = parse_coqe_anno_list(anno_str)
            
            # Only keep samples with opinions
            if label == 1 and predicates and sentiments:
                samples.append({
                    'text': text.strip(),
                    'label': label,
                    'subjects': subjects,
                    'objects': objects,
                    'predicates': predicates,
                    'sentiments': sentiments,
                    'comparators': comparators
                })
        except:
            pass
    
    return samples


def create_aste_format(coqe_sample):
    """
    Convert single COQE sample to ASTE format
    Returns a dict with sentence and triplet list
    """
    text = coqe_sample['text']
    triplets_aste = []
    
    # Extract spans
    pred_spans = extract_spans(coqe_sample['predicates'])
    sent_spans = extract_spans(coqe_sample['sentiments'])
    
    # For each predicate-sentiment pair, create a triplet
    for pred_idx, pred_text in pred_spans:
        for sent_idx, sent_text in sent_spans:
            sentiment_label = determine_sentiment(sent_text)
            
            # Create triplet: (aspect_term, opinion_term, aspect_sentiment_label)
            triplet = {
                "aspect_term": pred_text,
                "opinion_term": sent_text,
                "aspect_sentiment_label": sentiment_label
            }
            triplets_aste.append(triplet)
    
    return {
        "sentence": text,
        "triplets": triplets_aste,
        "source": "cameraCOQE",
        "is_comparative": True  # Tất cả sample đã filter có label=1
    }


# Main conversion
def convert_all_coqe_datasets():
    """Convert all COQE datasets"""
    base_path = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE'
    splits = ['train', 'dev', 'test']
    
    output_dir = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE_ASTE'
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        input_file = os.path.join(base_path, f'{split}.txt')
        output_file = os.path.join(output_dir, f'{split}_aste.json')
        
        print(f"\nConverting {split}...")
        coqe_samples = parse_coqe_file(input_file)
        print(f"  Loaded {len(coqe_samples)} COQE samples with opinions")
        
        # Convert to ASTE format
        aste_samples = []
        for sample in coqe_samples:
            aste_sample = create_aste_format(sample)
            aste_samples.append(aste_sample)
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aste_samples, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(aste_samples)} ASTE samples to {output_file}")

        # Statistics
        total_triplets = sum(len(s['triplets']) for s in aste_samples)
        print(f"  Total triplets: {total_triplets}")


if __name__ == '__main__':
    convert_all_coqe_datasets()
