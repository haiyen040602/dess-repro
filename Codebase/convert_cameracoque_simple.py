"""
Convert cameraCOQE ASTE to training format (simplified - no NLTK dependency)
Format: entities (type, start, end) and sentiments (type, head, tail)
"""

import json
import os
import re


def simple_tokenize(sentence):
    """Simple tokenization without NLTK"""
    # Split by whitespace and keep punctuation
    tokens = sentence.split()
    return tokens


def find_token_indices(tokens, target_text):
    """Find token indices for a given text"""
    target_tokens = target_text.lower().split()
    
    for i in range(len(tokens) - len(target_tokens) + 1):
        tokens_lower = [t.lower() for t in tokens[i:i+len(target_tokens)]]
        if tokens_lower == target_tokens:
            return list(range(i, i + len(target_tokens)))
    
    # Fallback: find first token that matches
    for i, token in enumerate(tokens):
        if target_text.lower() in token.lower() or token.lower() in target_text.lower():
            return [i]
    
    return []


def convert_cameracoque_to_training_format(cameracoque_aste_samples):
    """Convert cameraCOQE ASTE format to training format"""
    training_samples = []
    
    for sample_idx, sample in enumerate(cameracoque_aste_samples):
        sentence = sample['sentence']
        triplets = sample['triplets']
        
        # Tokenize
        tokens = simple_tokenize(sentence)
        
        # Build entities
        entities = []
        entity_map = {}  # (type, start, end) -> index
        
        entity_idx = 0
        for triplet in triplets:
            aspect = triplet['aspect_term']
            opinion = triplet['opinion_term']
            
            # Find aspect tokens
            aspect_indices = find_token_indices(tokens, aspect)
            if aspect_indices:
                start = min(aspect_indices)
                end = max(aspect_indices) + 1
                key = ('target', start, end)
                if key not in entity_map:
                    entity_map[key] = entity_idx
                    entities.append({
                        "type": "target",
                        "start": start,
                        "end": end
                    })
                    entity_idx += 1
            
            # Find opinion tokens
            opinion_indices = find_token_indices(tokens, opinion)
            if opinion_indices:
                start = min(opinion_indices)
                end = max(opinion_indices) + 1
                key = ('opinion', start, end)
                if key not in entity_map:
                    entity_map[key] = entity_idx
                    entities.append({
                        "type": "opinion",
                        "start": start,
                        "end": end
                    })
                    entity_idx += 1
        
        # Build sentiments (aspect-opinion pairs)
        sentiments = []
        
        for triplet in triplets:
            aspect = triplet['aspect_term']
            opinion = triplet['opinion_term']
            sentiment_label = triplet['aspect_sentiment_label'].upper()
            
            # Map to POSITIVE/NEGATIVE/NEUTRAL if needed
            if sentiment_label not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                sentiment_label = 'NEUTRAL'
            
            # Find entity indices
            aspect_indices = find_token_indices(tokens, aspect)
            opinion_indices = find_token_indices(tokens, opinion)
            
            if aspect_indices and opinion_indices:
                aspect_start = min(aspect_indices)
                aspect_end = max(aspect_indices) + 1
                opinion_start = min(opinion_indices)
                opinion_end = max(opinion_indices) + 1
                
                # Find entity index for aspect
                aspect_entity_idx = None
                opinion_entity_idx = None
                
                for i, ent in enumerate(entities):
                    if ent['type'] == 'target' and ent['start'] == aspect_start and ent['end'] == aspect_end:
                        aspect_entity_idx = i
                    if ent['type'] == 'opinion' and ent['start'] == opinion_start and ent['end'] == opinion_end:
                        opinion_entity_idx = i
                
                if aspect_entity_idx is not None and opinion_entity_idx is not None:
                    sentiments.append({
                        "type": sentiment_label,
                        "head": aspect_entity_idx,
                        "tail": opinion_entity_idx
                    })
        
        # Create simple POS tags (all NN for simplicity)
        pos = [[token, 'NN'] for token in tokens]
        
        output = {
            "entities": entities,
            "sentiments": sentiments,
            "pos": pos,
            "orig_id": f"cameracoque:{sample_idx}"
        }
        
        training_samples.append(output)
    
    return training_samples


def convert_all_splits():
    """Convert all cameraCOQE splits"""
    base_path = '/home/haiyan/DESS-main/DESS-main/Codebase/data'
    output_dir = os.path.join(base_path, 'cameraCOQE')
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        input_file = os.path.join(base_path, f'cameraCOQE_ASTE/{split}_aste.json')
        output_file = os.path.join(output_dir, f'{split}_dep_triple_polarity_result.json')
        
        print(f"\n{'='*70}")
        print(f"Converting {split.upper()}")
        print(f"{'='*70}")
        
        # Load
        with open(input_file, 'r') as f:
            cameracoque_data = json.load(f)
        
        print(f"✓ Loaded {len(cameracoque_data)} samples from {input_file}")
        
        # Convert
        training_data = convert_cameracoque_to_training_format(cameracoque_data)
        
        print(f"✓ Converted to training format")
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_file}")
        
        # Statistics
        total_entities = sum(len(s['entities']) for s in training_data)
        total_sentiments = sum(len(s['sentiments']) for s in training_data)
        avg_entities = total_entities / len(training_data) if training_data else 0
        avg_sentiments = total_sentiments / len(training_data) if training_data else 0
        
        print(f"\nStatistics:")
        print(f"  Samples: {len(training_data)}")
        print(f"  Total entities: {total_entities} (avg {avg_entities:.2f}/sample)")
        print(f"  Total sentiments: {total_sentiments} (avg {avg_sentiments:.2f}/sample)")
        
        # Show sample
        if training_data:
            sample = training_data[0]
            print(f"\nFirst sample:")
            print(f"  Entities: {sample['entities']}")
            print(f"  Sentiments: {sample['sentiments']}")
            print(f"  Tokens: {[p[0] for p in sample['pos'][:10]]}")


if __name__ == '__main__':
    convert_all_splits()
    
    print(f"\n{'='*70}")
    print("✅ CONVERSION COMPLETE")
    print(f"{'='*70}")
    print("\nNext steps to train on cameraCOQE:")
    print("1. Update Parameter.py:")
    print("   - Change 'root_path': 'data/cameraCOQE' (instead of data/14res)")
    print("2. Run training:")
    print("   - python3 train.py")
    print("\nYour model will train on cameraCOQE comparative opinions dataset")
