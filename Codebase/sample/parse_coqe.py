"""
COQE Parser - Custom format parser for semi-JSON format
"""

def parse_coqe_anno_list(anno_str):
    """
    Parse COQE annotation list: [[items];[items];[items];[items];[items]]
    Where items are separated by spaces and formatted as "idx&&text"
    """
    # Remove outer brackets
    anno_str = anno_str.strip()
    if anno_str.startswith('[') and anno_str.endswith(']'):
        anno_str = anno_str[1:-1]
    
    # Split by semicolon to get 5 arrays
    parts = anno_str.split('];[')
    
    result = []
    for part in parts:
        # Clean brackets
        part = part.strip().strip('[]')
        
        if not part or part == '':
            result.append([])
        else:
            # Split items by space (each is like "9&&battery" or "9&&battery 10&&item")
            items = part.split()
            result.append(items)
    
    return result


# Test parsing
test_cases = [
    "[[];[];[];[];[]]",
    "[[11&&S50];[];[9&&battery];[8&&same];[0]]",
    "[[];[];[13&&flash 14&&shutter 15&&speeds];[12&&slower];[-1]]",
]

for test in test_cases:
    print(f"\nInput: {test}")
    try:
        result = parse_coqe_anno_list(test)
        print(f"Output: {result}")
    except Exception as e:
        print(f"Error: {e}")

# Now test full parsing
print("\n" + "="*80)
print("TESTING FULL COQE PARSING")
print("="*80)

def parse_coqe_file(filepath):
    """Parse COQE format file"""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        i += 1
        
        if not line.strip():
            continue
        
        # Split by last tab
        tab_pos = line.rfind('\t')
        if tab_pos == -1:
            continue
        
        text = line[:tab_pos]
        label_str = line[tab_pos+1:].strip()
        
        try:
            label = int(label_str)
        except:
            continue
        
        # Read annotation line
        if i >= len(lines):
            continue
        
        anno_str = lines[i].rstrip('\r\n')
        i += 1
        
        try:
            subjects, objects, predicates, sentiments, comparators = parse_coqe_anno_list(anno_str)
            
            # Only keep samples with opinions (label=1 AND has predicates)
            if label == 1 and predicates:
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
            pass
    
    return samples


# Load and show results
train_file = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE/train.txt'
train_samples = parse_coqe_file(train_file)

print(f"\nTrain samples with opinions: {len(train_samples)}")

# Show first 3 examples
for idx, s in enumerate(train_samples[:3]):
    print(f"\n[SAMPLE {idx+1}]")
    print(f"Text: {s['text'][:80]}")
    print(f"Subjects: {s['subjects']}")
    print(f"Objects: {s['objects']}")
    print(f"Predicates: {s['predicates']}")
    print(f"Sentiments: {s['sentiments']}")
