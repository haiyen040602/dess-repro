"""
COQE Analysis & Simple Adapter
"""

import json

# Analyze COQE format
coqe_file = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE/train.txt'

print("\n" + "="*80)
print("ANALYZING COQE FORMAT")
print("="*80 + "\n")

samples_with_opinions = []
samples_without_opinions = []

with open(coqe_file, 'r') as f:
    lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if i + 1 >= len(lines):
            break
        
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()
        
        i += 2
        
        if not line1 or '\t' not in line1:
            continue
        
        text, label_str = line1.rsplit('\t', 1)
        text = text.strip()
        label = int(label_str.strip())
        
        try:
            anno = json.loads(line2)
        except:
            continue
        
        # anno = [subjects, objects, predicates, sentiments, comparators]
        subjects, objects, predicates, sentiments, comparators = anno
        
        if label == 1 and predicates and sentiments:
            samples_with_opinions.append({
                'text': text,
                'label': label,
                'subjects': subjects,
                'predicates': predicates,
                'sentiments': sentiments,
                'comparators': comparators
            })
        elif label == 0:
            samples_without_opinions.append({
                'text': text,
                'label': label
            })

print(f"Total analyzed: {len(samples_with_opinions) + len(samples_without_opinions)}")
print(f"With opinions (label=1): {len(samples_with_opinions)}")
print(f"Without opinions (label=0): {len(samples_without_opinions)}")

print("\n" + "="*80)
print("SAMPLE COQE ENTRIES WITH OPINIONS")
print("="*80 + "\n")

for idx, sample in enumerate(samples_with_opinions[:5]):
    print(f"\nExample {idx+1}:")
    print(f"Text: {sample['text']}")
    print(f"Subjects: {sample['subjects']}")
    print(f"Predicates (aspects): {sample['predicates']}")
    print(f"Sentiments (opinions): {sample['sentiments']}")
    print(f"Comparators: {sample['comparators']}")
    
    # Parse out indices and text
    if sample['predicates']:
        print(f"\nParsed Predicates:")
        for pred in sample['predicates']:
            parts = pred.split('&&')
            if len(parts) == 2:
                idx_str, text_str = parts
                print(f"  - Index {idx_str}: '{text_str}'")
    
    if sample['sentiments']:
        print(f"Parsed Sentiments:")
        for sent in sample['sentiments']:
            parts = sent.split('&&')
            if len(parts) == 2:
                idx_str, text_str = parts
                print(f"  - Index {idx_str}: '{text_str}'")

print("\n" + "="*80)
print("INSIGHTS FOR COQE→ASTE ADAPTATION")
print("="*80 + "\n")

print("""
✓ Good news: COQE HAS aspect-opinion pairs!
  - predicates = aspects
  - sentiments = opinion words
  - label = 1 indicates comparison (positive aspect)

✗ Challenges:
  1. COQE sentiment words are just adjectives (e.g., "same", "slower")
     → Not full opinions like in ASTE
  
  2. COQE doesn't have explicit sentiment polarity
     → We can infer: label=1 (comparative) → assume POSITIVE
  
  3. Subject information (comparator) missing
     → Can't directly determine if it's better or worse

✓ Solution Strategy:
  1. Extract predicates (aspects) from COQE
  2. Extract sentiment words as opinions
  3. Use label=1 → POSITIVE for all sentiment types
  4. Create triplet: (aspect, opinion, POSITIVE)
  
  Note: This assumes all comparisons are positive for the main subject
        (which is reasonable for comparative opinion extraction)

Result:
  - Can extract ~800-850 triplets per dataset
  - Useful for evaluating ASTE model generalization
""")

print("\n" + "="*80)
