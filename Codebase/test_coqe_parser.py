"""
COQE to ASTE Converter - Fixed parser
"""

import json
import os

def parse_coqe_file(filepath):
    """Parse COQE format file correctly"""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read text+label line
        line = lines[i].rstrip('\r\n')
        i += 1
        
        if not line.strip():
            continue
        
        # Split by last tab (text can contain tabs)
        tab_pos = line.rfind('\t')
        if tab_pos == -1:
            continue
        
        text = line[:tab_pos]
        label_str = line[tab_pos+1:].strip()
        
        try:
            label = int(label_str)
        except:
            continue
        
        # Read annotation line (may span multiple actual lines due to newline in JSON)
        anno_str = ""
        while i < len(lines):
            anno_line = lines[i].rstrip('\r\n')
            anno_str += anno_line
            i += 1
            
            # Check if JSON is complete
            if anno_str.count('[') == anno_str.count(']'):
                break
        
        try:
            anno = json.loads(anno_str)
            subjects, objects, predicates, sentiments, comparators = anno
            
            samples.append({
                'text': text.strip(),
                'label': label,
                'subjects': subjects,
                'objects': objects,
                'predicates': predicates,
                'sentiments': sentiments,
                'comparators': comparators
            })
        except Exception as e:
            print(f"Error parsing anno at line {i}: {e}")
            continue
    
    return samples


def extract_spans(annotations_list):
    """Extract text and indices from annotations like '9&&battery'"""
    spans = []
    for anno in annotations_list:
        if anno and '&&' in anno:
            try:
                idx_str, text = anno.split('&&', 1)
                spans.append({
                    'token_idx': int(idx_str),
                    'text': text
                })
            except:
                pass
    return spans


# Load data
train_file = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE/train.txt'
print(f"Loading {train_file}...")
train_samples = parse_coqe_file(train_file)

print(f"\nTotal samples loaded: {len(train_samples)}")
with_opinions = sum(1 for s in train_samples if s['label'] == 1 and s['predicates'])
print(f"Samples with opinions: {with_opinions}")

# Show examples
print("\n" + "="*80)
print("SAMPLE WITH OPINIONS:")
print("="*80)
for s in train_samples:
    if s['label'] == 1 and s['predicates']:
        print(f"\nText: {s['text']}")
        print(f"Subjects: {s['subjects']}")
        print(f"Objects: {s['objects']}")
        print(f"Predicates: {s['predicates']}")
        print(f"Sentiments: {s['sentiments']}")
        
        pred_spans = extract_spans(s['predicates'])
        sent_spans = extract_spans(s['sentiments'])
        print(f"\nExtracted predicates: {pred_spans}")
        print(f"Extracted sentiments: {sent_spans}")
        break
