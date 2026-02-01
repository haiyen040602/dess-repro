"""
COQE to ASTE Converter
Convert Comparative Opinion Extraction dataset to ASTE format
"""

import json
import os
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

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
    Determine sentiment label from text using VADER sentiment analyzer
    For COQE, need to infer if comparative is positive/negative
    """
    # Combine sentiment text with context for better analysis
    full_text = f"{context} {sentiment_text}" if context else sentiment_text
    
    # Use VADER to get sentiment scores
    scores = vader_analyzer.polarity_scores(full_text)
    compound = scores['compound']
    
    # VADER compound score interpretation:
    # compound >= 0.05: positive
    # compound <= -0.05: negative  
    # -0.05 < compound < 0.05: neutral
    
    # For comparative opinions, use more sensitive thresholds
    if compound >= 0.1:  # More conservative threshold for positive
        return 'positive'
    elif compound <= -0.1:  # More conservative threshold for negative
        return 'negative'
    else:
        # Fallback to keyword-based for borderline cases
        sentiment_text_lower = sentiment_text.lower()
        positive_words = {'good', 'great', 'excellent', 'best', 'better', 'nice', 'high', 
                          'faster', 'higher', 'sharper', 'brighter', 'clearer', 'superior', 
                          'improved', 'stronger', 'longer', 'easier', 'smoother'}
        negative_words = {'bad', 'poor', 'worse', 'worst', 'slow', 'slower', 'lower', 
                          'blurry', 'dull', 'dark', 'cheaper', 'weak', 'inferior', 
                          'harder', 'difficult', 'complicated', 'shorter'}
        
        if any(word in sentiment_text_lower for word in positive_words):
            return 'positive'
        elif any(word in sentiment_text_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'


def parse_coqe_file(filepath):
    """Load COQE file - groups quintuples from same sentence"""
    samples = []
    total_parsed = 0
    total_label_1 = 0
    total_quintuples = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n').strip()
        i += 1
        
        if not line:
            continue
        
        # Check if this is a sentence line (contains text\tLABEL)
        tab_pos = line.rfind('\t')
        if tab_pos == -1:
            continue
        
        text = line[:tab_pos]
        label_str = line[tab_pos+1:].strip()
        
        try:
            label = int(label_str)
            total_parsed += 1
            if label == 1:
                total_label_1 += 1
        except:
            continue
        
        # Read ALL quintuples for this sentence (next lines starting with [[)
        quintuples_for_sentence = []
        
        while i < len(lines):
            quintuple_line = lines[i].rstrip('\r\n').strip()
            
            # Check if this is a quintuple line (starts with [[)
            if not quintuple_line.startswith('[['):
                break  # Next sentence found
            
            i += 1
            total_quintuples += 1
            
            try:
                subjects, objects, predicates, sentiments, comparators = parse_coqe_anno_list(quintuple_line)
                
                # Store each quintuple separately
                quintuples_for_sentence.append({
                    'text': text.strip(),
                    'label': label,
                    'subjects': subjects,
                    'objects': objects,
                    'predicates': predicates,
                    'sentiments': sentiments,
                    'comparators': comparators
                })
            except Exception as e:
                continue
        
        # Keep ALL samples with label=1 and valid quintuples
        if label == 1 and quintuples_for_sentence:
            samples.append(quintuples_for_sentence)
    
    print(f"  Debug: sentences={total_parsed}, label=1: {total_label_1}, quintuples: {total_quintuples}")
    print(f"  Debug: sentences with quintuple(s): {len(samples)}")
    return samples


def create_aste_format(coqe_samples_list):
    """
    Convert COQE samples to ASTE format
    coqe_samples_list: list of samples for SAME sentence but different quintuples
    Each quintuple becomes ONE triplet
    Returns a dict with sentence and triplet list
    """
    # All samples should be from same sentence
    if not coqe_samples_list:
        return None
    
    text = coqe_samples_list[0]['text']
    triplets_aste = []
    
    # Process each quintuple separately (each is one sample in the list)
    for coqe_sample in coqe_samples_list:
        # Extract spans from THIS quintuple
        pred_spans = extract_spans(coqe_sample['predicates'])
        sent_spans = extract_spans(coqe_sample['sentiments'])
        subj_spans = extract_spans(coqe_sample['subjects'])
        obj_spans = extract_spans(coqe_sample['objects'])
        
        # Fallback: if no pred/sent, try subject/object
        if not pred_spans and subj_spans:
            pred_spans = subj_spans
        if not sent_spans and obj_spans:
            sent_spans = obj_spans
        
        # Each quintuple has ONE aspect + ONE opinion
        if pred_spans and sent_spans:
            # Join all items from predicates and sentiments
            # Sort by index and join with space
            pred_text = ' '.join([word for idx, word in sorted(pred_spans)])
            sent_text = ' '.join([word for idx, word in sorted(sent_spans)])
            
            sentiment_label = determine_sentiment(sent_text, context=pred_text)
            
            triplet = {
                "aspect_term": pred_text,
                "opinion_term": sent_text,
                "aspect_sentiment_label": sentiment_label
            }
            triplets_aste.append(triplet)
    
    if not triplets_aste:
        return None
    
    return {
        "sentence": text,
        "triplets": triplets_aste,
        "source": "cameraCOQE",
        "is_comparative": True
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
        coqe_quintuple_groups = parse_coqe_file(input_file)
        print(f"  Loaded {len(coqe_quintuple_groups)} sentences with comparative opinions")
        
        # Convert to ASTE format
        aste_samples = []
        total_triplets = 0
        for quintuple_group in coqe_quintuple_groups:
            aste_sample = create_aste_format(quintuple_group)
            if aste_sample and aste_sample['triplets']:
                aste_samples.append(aste_sample)
                total_triplets += len(aste_sample['triplets'])
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aste_samples, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(aste_samples)} ASTE samples to {output_file}")
        print(f"  Total triplets: {total_triplets}")
        print(f"  Avg triplets/sample: {total_triplets/len(aste_samples):.2f}" if aste_samples else "  No samples")


if __name__ == '__main__':
    convert_all_coqe_datasets()
