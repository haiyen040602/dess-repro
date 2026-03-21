"""
Deduplicate triplets in cameraCOQE dataset
Remove duplicate (aspect, opinion, label) pairs while keeping sentence context
"""

import json
import os
from collections import defaultdict

def deduplicate_dataset():
    """Remove duplicate triplets from dataset"""
    
    for split in ['train', 'dev', 'test']:
        # Load ASTE format
        aste_path = f'data/cameraCOQE_ASTE/{split}_aste.json'
        with open(aste_path, 'r', encoding='utf-8') as f:
            aste_data = json.load(f)
        
        # Deduplicate
        original_count = sum(len(s.get('triplets', [])) for s in aste_data)
        
        deduplicated = []
        for sample in aste_data:
            seen_triplets = set()
            unique_triplets = []
            
            for triplet in sample.get('triplets', []):
                aspect = triplet.get('aspect_term', '')
                opinion = triplet.get('opinion_term', '')
                label = triplet.get('aspect_sentiment_label', '')
                
                triplet_key = (aspect, opinion, label)
                
                # Only keep if not seen before
                if triplet_key not in seen_triplets:
                    seen_triplets.add(triplet_key)
                    unique_triplets.append(triplet)
            
            # Only keep sample if it has triplets
            if unique_triplets:
                sample['triplets'] = unique_triplets
                deduplicated.append(sample)
        
        new_count = sum(len(s.get('triplets', [])) for s in deduplicated)
        removed = original_count - new_count
        samples_removed = len(aste_data) - len(deduplicated)
        
        # Save deduplicated ASTE
        with open(aste_path, 'w', encoding='utf-8') as f:
            json.dump(deduplicated, f, indent=2, ensure_ascii=False)
        
        print(f"\n{split.upper()}:")
        print(f"  Original triplets:    {original_count}")
        print(f"  After deduplication:  {new_count}")
        print(f"  Triplets removed:     {removed}")
        print(f"  Samples kept:         {len(deduplicated)}")
        
        # Now convert deduplicated ASTE to D2E2S format
        convert_to_d2e2s_format(deduplicated, split)


def convert_to_d2e2s_format(aste_samples, split):
    """Convert deduplicated ASTE to D2E2S training format"""
    import spacy
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("  Installing spaCy model...")
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    
    d2e2s_samples = []
    
    for aste_sample in aste_samples:
        sentence = aste_sample.get('sentence', '')
        tokens = sentence.split()
        
        # NLP processing
        doc = nlp(sentence)
        
        # Extract entities from triplets
        entities_dict = {}  # span_tuple -> entity_type
        entity_list = []
        
        for triplet in aste_sample.get('triplets', []):
            aspect_term = triplet.get('aspect_term', '')
            opinion_term = triplet.get('opinion_term', '')
            label = triplet.get('aspect_sentiment_label', '')
            
            # Find aspect in tokens
            aspect_tokens = aspect_term.split()
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    aspect_span = (i, i + len(aspect_tokens))
                    if aspect_span not in entities_dict:
                        entity_list.append({
                            'type': 'target',
                            'start': aspect_span[0],
                            'end': aspect_span[1]
                        })
                        entities_dict[aspect_span] = 'target'
                    break
            
            # Find opinion in tokens
            opinion_tokens = opinion_term.split()
            for i in range(len(tokens) - len(opinion_tokens) + 1):
                if tokens[i:i+len(opinion_tokens)] == opinion_tokens:
                    opinion_span = (i, i + len(opinion_tokens))
                    if opinion_span not in entities_dict:
                        entity_list.append({
                            'type': 'opinion',
                            'start': opinion_span[0],
                            'end': opinion_span[1]
                        })
                        entities_dict[opinion_span] = 'opinion'
                    break
        
        # Remove duplicates in entity list
        unique_entities = []
        seen = set()
        for ent in entity_list:
            key = (ent['type'], ent['start'], ent['end'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        # Extract sentiments (triplets)
        sentiments = []
        seen_sentiments = set()
        
        for triplet in aste_sample.get('triplets', []):
            aspect_term = triplet.get('aspect_term', '')
            opinion_term = triplet.get('opinion_term', '')
            label = triplet.get('aspect_sentiment_label', '')
            
            # Find indices in entity list
            aspect_tokens = aspect_term.split()
            opinion_tokens = opinion_term.split()
            
            aspect_idx = None
            opinion_idx = None
            
            for i, entity in enumerate(unique_entities):
                entity_span_tokens = tokens[entity['start']:entity['end']]
                if entity['type'] == 'target' and entity_span_tokens == aspect_tokens:
                    aspect_idx = i
                elif entity['type'] == 'opinion' and entity_span_tokens == opinion_tokens:
                    opinion_idx = i
            
            if aspect_idx is not None and opinion_idx is not None:
                sentiment_key = (aspect_idx, opinion_idx, label)
                if sentiment_key not in seen_sentiments:
                    seen_sentiments.add(sentiment_key)
                    sentiments.append({
                        'type': label,
                        'head': aspect_idx,
                        'tail': opinion_idx
                    })
        
        # Extract dependency relations (simplified)
        dependency = []
        for token in doc:
            dep_rel = {
                'index': token.i,
                'word': token.text,
                'pos': token.pos_,
                'dependency': token.dep_
            }
            dependency.append(dep_rel)
        
        # Create D2E2S sample
        if unique_entities and sentiments:
            d2e2s_sample = {
                'entities': unique_entities,
                'sentiments': sentiments,
                'tokens': tokens,
                'pos': [token.pos_ for token in doc],
                'dependency': dependency
            }
            d2e2s_samples.append(d2e2s_sample)
    
    # Save D2E2S format
    output_path = f'data/cameraCOQE/{split}_dep_triple_polarity_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(d2e2s_samples, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved D2E2S format: {output_path} ({len(d2e2s_samples)} samples)")


if __name__ == '__main__':
    print("="*80)
    print("DEDUPLICATING CAMERACOQUE DATASET")
    print("="*80)
    
    deduplicate_dataset()
    
    print("\n" + "="*80)
    print("✅ DEDUPLICATION COMPLETE")
    print("="*80)
