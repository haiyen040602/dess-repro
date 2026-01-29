"""
Fix COQE parser - handle 2-line format correctly
"""

import json

coqe_file = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE/train.txt'

print("\n" + "="*80)
print("ANALYZING COQE FORMAT - FIXED")
print("="*80 + "\n")

total = 0
with_opinions = 0
without_opinions = 0

with open(coqe_file, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if i + 1 >= len(lines):
        break
    
    # Read 2 lines per sample
    line1 = lines[i].rstrip('\r\n')  # Remove line endings but keep spaces
    line2 = lines[i+1].rstrip('\r\n')
    
    i += 2
    
    # Skip completely empty lines
    if not line1.strip():
        continue
    
    # Find last tab to separate text from label
    tab_pos = line1.rfind('\t')
    if tab_pos == -1:
        continue
    
    try:
        text = line1[:tab_pos].strip()
        label_str = line1[tab_pos+1:].strip()
        
        if not label_str:
            continue
        
        label = int(label_str)
        
        # Parse annotations
        anno = json.loads(line2)
        subjects, objects, predicates, sentiments, comparators = anno
        
        total += 1
        if label == 1:
            with_opinions += 1
            if with_opinions <= 3:
                print(f"\n[SAMPLE {with_opinions} - WITH OPINIONS]")
                print(f"Text: {text[:80]}...")
                print(f"Subjects: {subjects}")
                print(f"Objects: {objects}")
                print(f"Predicates: {predicates}")
                print(f"Sentiments: {sentiments}")
                print(f"Comparators: {comparators}")
        else:
            without_opinions += 1
            
    except Exception as e:
        print(f"Error processing line {i//2}: {e}")
        continue

print(f"\n{'='*80}")
print(f"RESULTS:")
print(f"Total analyzed: {total}")
print(f"With opinions (label=1): {with_opinions}")
print(f"Without opinions (label=0): {without_opinions}")
print(f"{'='*80}\n")
