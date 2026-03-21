"""
Convert cameraCOQE ASTE to format compatible with original ASTE training data
Converts to format with entities (type, start, end) and sentiments (type, head, tail)
"""

import json
import os
import re
from nltk import pos_tag
import nltk

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def tokenize_and_get_positions(sentence):
    """
    Simple tokenization by whitespace and punctuation
    Returns: (tokens, [(start, end), ...], pos_tags)
    """
    # Simple whitespace + punctuation split
    tokens = re.findall(r'\b\w+\b|[.,!?;:\'-]', sentence)
    
    # Get POS tags
    pos_tags = pos_tag(tokens)
    
    # Get character positions
    positions = []
    search_start = 0
    for token in tokens:
        # Find token in sentence starting from search_start
        pos = sentence.find(token, search_start)
        if pos == -1:
            # Fallback: search from beginning
            pos = sentence.find(token)
        positions.append((pos, pos + len(token)))
        search_start = pos + len(token)
    
    return tokens, positions, pos_tags


def find_token_indices(tokens, target_text):
    """Find token indices for a given text (may be multiple tokens)"""
    target_tokens = target_text.split()
    indices = []
    
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i+len(target_tokens)] == target_tokens:
            indices.extend(range(i, i + len(target_tokens)))
            return indices
    
    # If exact match fails, try fuzzy match
    for i, token in enumerate(tokens):
        if target_text.lower() in token.lower() or token.lower() in target_text.lower():
            return [i]
    
    return []


def convert_cameracoque_to_aste_format(cameracoque_aste_samples):
    """
    Convert cameraCOQE ASTE format to training-compatible format
    Input: [{"sentence": "...", "triplets": [...]}]
    Output: [{"entities": [...], "sentiments": [...], "pos": [[token, tag], ...]}]
    """
    training_samples = []
    
    for sample in cameracoque_aste_samples:
        sentence = sample['sentence']
        triplets = sample['triplets']
        
        # Tokenize
        tokens, positions, pos_tags = tokenize_and_get_positions(sentence)
        
        # Build entities and sentiments lists
        entities = []
        sentiments = []
        
        # Track unique aspect and opinion terms with their indices
        entity_list = []  # [(type, token_indices, text)]
        
        for triplet in triplets:
            aspect = triplet['aspect_term']
            opinion = triplet['opinion_term']
            sentiment_label = triplet['aspect_sentiment_label']
            
            # Find token indices for aspect
            aspect_indices = find_token_indices(tokens, aspect)
            if aspect_indices:
                start_idx = min(aspect_indices)
                end_idx = max(aspect_indices) + 1
                entity_list.append(('target', start_idx, end_idx, aspect))
            
            # Find token indices for opinion
            opinion_indices = find_token_indices(tokens, opinion)
            if opinion_indices:
                start_idx = min(opinion_indices)
                end_idx = max(opinion_indices) + 1
                entity_list.append(('opinion', start_idx, end_idx, opinion))
        
        # Deduplicate entities by position
        seen = set()
        unique_entities = []
        for ent_type, start, end, text in entity_list:
            key = (ent_type, start, end)
            if key not in seen:
                seen.add(key)
                unique_entities.append((ent_type, start, end))
        
        # Create entities array
        for ent_type, start, end in unique_entities:
            entities.append({
                "type": ent_type,
                "start": start,
                "end": end
            })
        
        # Create sentiments array (pairs of aspect-opinion with sentiment label)
        entity_idx = 0
        target_idx = 0
        opinion_idx = 0
        
        # Map entities by type
        targets = [(i, e) for i, e in enumerate(entities) if e['type'] == 'target']
        opinions = [(i, e) for i, e in enumerate(entities) if e['type'] == 'opinion']
        
        # Create sentiment triplets: (aspect, opinion, label)
        for target_pos, target_ent in targets:
            for opinion_pos, opinion_ent in opinions:
                # Find corresponding triplet to get sentiment
                sentiment_label = 'NEUTRAL'  # default
                
                for triplet in triplets:
                    aspect_text = triplet['aspect_term']
                    opinion_text = triplet['opinion_term']
                    aspect_indices = find_token_indices(tokens, aspect_text)
                    opinion_indices = find_token_indices(tokens, opinion_text)
                    
                    if (aspect_indices and 
                        min(aspect_indices) == target_ent['start'] and
                        max(aspect_indices) + 1 == target_ent['end'] and
                        opinion_indices and
                        min(opinion_indices) == opinion_ent['start'] and
                        max(opinion_indices) + 1 == opinion_ent['end']):
                        
                        # Map sentiment label
                        label = triplet['aspect_sentiment_label'].upper()
                        if label == 'POSITIVE':
                            sentiment_label = 'POSITIVE'
                        elif label == 'NEGATIVE':
                            sentiment_label = 'NEGATIVE'
                        else:
                            sentiment_label = 'NEUTRAL'
                        break
                
                sentiments.append({
                    "type": sentiment_label,
                    "head": target_pos,
                    "tail": opinion_pos
                })
        
        # Create output sample
        output = {
            "entities": entities,
            "sentiments": sentiments,
            "pos": [[token, tag] for token, tag in pos_tags],
            "orig_id": f"cameracoque:{len(training_samples)}"
        }
        
        training_samples.append(output)
    
    return training_samples


# Main conversion
def convert_all_cameracoque():
    """Convert all cameraCOQE splits to training format"""
    
    base_path = '/home/haiyan/DESS-main/DESS-main/Codebase/data'
    output_dir = os.path.join(base_path, 'cameraCOQE')
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(base_path, f'cameraCOQE_ASTE/{split}_aste.json')
        output_file = os.path.join(output_dir, f'{split}_dep_triple_polarity_result.json')
        
        print(f"\nConverting {split}...")
        
        # Load cameraCOQE ASTE format
        with open(input_file, 'r') as f:
            cameracoque_data = json.load(f)
        
        print(f"  Loaded {len(cameracoque_data)} samples")
        
        # Convert to training format
        training_data = convert_cameracoque_to_aste_format(cameracoque_data)
        
        print(f"  Converted to training format")
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Saved to: {output_file}")
        
        # Statistics
        total_entities = sum(len(s['entities']) for s in training_data)
        total_sentiments = sum(len(s['sentiments']) for s in training_data)
        print(f"  Total entities: {total_entities}")
        print(f"  Total sentiments: {total_sentiments}")


if __name__ == '__main__':
    convert_all_cameracoque()
