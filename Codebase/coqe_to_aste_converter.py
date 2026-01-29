"""
COQE to ASTE Format Converter
Chuyển đổi dữ liệu từ Comparative Opinion Extraction sang Aspect-based Sentiment Triplet Extraction
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple

class COQEtoASTE:
    """
    COQE Format Explanation:
    =========================
    Text    label    [subject_list, object_list, predicate_list, sentiment_word_list, [comparator]]
    
    Format: token_index&&token_text (0-indexed)
    
    Example 1:
    "By the way you can use the same battery the S50 comes with."
    Label: 1 (có opinion comparison)
    [
        [11&&S50],                    # Subject (object to compare)
        [],                           # Object (không dùng cho cameras)
        [9&&battery],                 # Predicate (aspect)
        [8&&same],                    # Sentiment word
        [0]                           # Comparator position
    ]
    
    Mapping to ASTE:
    - predicate (aspect) = aspect/target
    - sentiment_word (opinion) = opinion
    - label direction = sentiment polarity
    
    Challenge:
    - COQE chỉ cho predicate + sentiment word (không có full triplet)
    - Cần infer sentiment type từ subject so sánh
    """
    
    def __init__(self, coqe_dir):
        self.coqe_dir = coqe_dir
        self.stats = {
            'total_sentences': 0,
            'with_opinions': 0,
            'without_opinions': 0,
            'converted': 0,
            'skipped': 0,
        }
    
    def parse_coqe_line(self, line: str) -> Tuple[str, int, List]:
        """Parse COQE format line"""
        # Format: text    label    [annotations]
        # Split carefully - text may have tabs, so find the last two fields
        
        line = line.strip()
        if not line:
            return None, None, None
        
        # Try different split strategies
        # Look for pattern: ...    number    [...]
        
        # Find the last occurrence of pattern "number    ["
        import re
        match = re.search(r'\s+(\d+)\s+(\[.*)$', line)
        if not match:
            return None, None, None
        
        text = line[:match.start()].strip()
        label = int(match.group(1))
        
        anno_str = match.group(2).strip()
        try:
            annotations = json.loads(anno_str)
        except Exception as e:
            return text, label, None
        
        return text, label, annotations
    
    def extract_span_from_annotation(self, tokens: List[str], anno_list: List[str]) -> List[Dict]:
        """
        Extract span information from annotation list
        anno_list example: ['11&&S50', '9&&battery']
        Returns: [{'text': 'battery', 'start': 9, 'end': 10, 'type': ...}]
        """
        if not anno_list:
            return []
        
        spans = []
        for anno in anno_list:
            if '&&' not in anno:
                continue
            
            try:
                idx_str, text = anno.split('&&', 1)
                idx = int(idx_str)
                
                # Find actual token span
                # Note: COQE uses token index, need to map to character positions
                if 0 <= idx < len(tokens):
                    spans.append({
                        'text': text,
                        'token_idx': idx,
                    })
            except:
                continue
        
        return spans
    
    def tokenize_preserve_case(self, text: str) -> List[str]:
        """Simple tokenization that preserves case"""
        # Split on whitespace and punctuation but keep them
        tokens = text.split()
        return tokens
    
    def convert_coqe_to_aste(self, text: str, label: int, annotations: List) -> Dict:
        """
        Convert COQE format to ASTE format
        
        ASTE output format:
        {
            "tokens": ["word1", "word2", ...],
            "entities": [
                {"type": "target", "start": 0, "end": 1},
                {"type": "opinion", "start": 2, "end": 3}
            ],
            "sentiments": [
                {"type": "POSITIVE", "head": 0, "tail": 1}
            ]
        }
        """
        
        if not annotations or label == 0:
            return None
        
        # Extract components
        subjects = annotations[0] if len(annotations) > 0 else []
        objects = annotations[1] if len(annotations) > 1 else []
        predicates = annotations[2] if len(annotations) > 2 else []
        sentiment_words = annotations[3] if len(annotations) > 3 else []
        
        # At least predicate and sentiment word must exist
        if not predicates or not sentiment_words:
            return None
        
        tokens = self.tokenize_preserve_case(text)
        
        # Extract predicates (aspects)
        aspects = self.extract_span_from_annotation(tokens, predicates)
        
        # Extract sentiment words (opinions)
        opinions = self.extract_span_from_annotation(tokens, sentiment_words)
        
        if not aspects or not opinions:
            return None
        
        # Build ASTE format
        entities = []
        
        # Add aspects (targets)
        for aspect in aspects:
            entities.append({
                "type": "target",
                "start": aspect['token_idx'],
                "end": aspect['token_idx'] + 1,
            })
        
        # Add opinions
        for opinion in opinions:
            entities.append({
                "type": "opinion",
                "start": opinion['token_idx'],
                "end": opinion['token_idx'] + 1,
            })
        
        # Create triplets: aspect -> opinion
        sentiments = []
        
        # Simple strategy: pair first aspect with first opinion
        # In real scenario, need more sophisticated pairing
        if len(entities) >= 2:
            for i, aspect in enumerate(aspects):
                for j, opinion in enumerate(opinions):
                    sentiments.append({
                        "type": "POSITIVE",  # Inferred from comparative label=1
                        "head": i,  # Index of aspect in entities list
                        "tail": len(aspects) + j,  # Index of opinion
                    })
        
        if not sentiments:
            return None
        
        return {
            "tokens": tokens,
            "entities": entities,
            "sentiments": sentiments,
            "orig_id": f"cameraCOQE_{text[:20]}",
        }
    
    def read_coqe_file(self, filepath: str) -> List[Dict]:
        """Read and convert COQE file"""
        converted = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                # COQE format: line 1 = text + label (tab separated)
                #              line 2 = annotations
                
                if i + 1 >= len(lines):
                    break
                
                line1 = lines[i].strip()
                line2 = lines[i + 1].strip()
                
                i += 2
                
                if not line1 or not line2:
                    continue
                
                # Parse line 1: text\tlabel
                if '\t' not in line1:
                    continue
                
                text, label_str = line1.rsplit('\t', 1)
                text = text.strip()
                
                try:
                    label = int(label_str.strip())
                except:
                    continue
                
                # Parse line 2: annotations
                try:
                    annotations = json.loads(line2)
                except:
                    annotations = None
                
                self.stats['total_sentences'] += 1
                
                if label == 1:
                    self.stats['with_opinions'] += 1
                else:
                    self.stats['without_opinions'] += 1
                
                # Convert to ASTE
                aste_sample = self.convert_coqe_to_aste(text, label, annotations)
                
                if aste_sample:
                    converted.append(aste_sample)
                    self.stats['converted'] += 1
                else:
                    self.stats['skipped'] += 1
        
        return converted
    
    def convert_all(self):
        """Convert all COQE files"""
        splits = ['train', 'dev', 'test']
        results = {}
        
        print("\n" + "="*80)
        print("Converting COQE to ASTE Format")
        print("="*80 + "\n")
        
        for split in splits:
            filepath = f"{self.coqe_dir}/{split}.txt"
            print(f"Processing {split}...")
            
            converted = self.read_coqe_file(filepath)
            results[split] = converted
            
            print(f"  ✓ Converted: {len(converted)} samples")
        
        return results
    
    def print_stats(self):
        """Print conversion statistics"""
        print("\n" + "="*80)
        print("Conversion Statistics")
        print("="*80)
        print(f"Total sentences: {self.stats['total_sentences']}")
        print(f"With opinions: {self.stats['with_opinions']}")
        print(f"Without opinions: {self.stats['without_opinions']}")
        print(f"Successfully converted: {self.stats['converted']}")
        print(f"Skipped: {self.stats['skipped']}")
        
        if self.stats['total_sentences'] > 0:
            conv_rate = (self.stats['converted'] / self.stats['total_sentences']) * 100
            print(f"Conversion rate: {conv_rate:.1f}%")
        print("="*80 + "\n")


def main():
    coqe_dir = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE'
    
    converter = COQEtoASTE(coqe_dir)
    
    # First, understand the data
    print("\n📊 Analyzing COQE format...\n")
    
    with open(f"{coqe_dir}/train.txt", 'r') as f:
        lines = f.readlines()
        print("Sample COQE entries:")
        print("-" * 80)
        
        count = 0
        i = 0
        while i < len(lines) and count < 3:
            if i + 1 < len(lines):
                line1 = lines[i].strip()
                line2 = lines[i + 1].strip()
                
                if line1 and '\t' in line1:
                    text, label_str = line1.rsplit('\t', 1)
                    print(f"\nExample {count+1}:")
                    print(f"  Text: {text[:70]}...")
                    print(f"  Label: {label_str.strip()} (1=comparative, 0=non-comparative)")
                    print(f"  Annotations: {line2[:100]}...")
                    count += 1
            
            i += 2
    
    print("\n" + "-"*80)
    print("\n⚠️  IMPORTANT NOTES ABOUT COQE→ASTE CONVERSION:")
    print("-"*80)
    print("""
1. COQE focuses on comparative statements (e.g., "S50 is better than X")
   - Can have: subject, predicate, sentiment_word, comparator
   - Problem: No explicit "opinion" extracted, only sentiment words

2. ASTE requires full triplets: (aspect, opinion, sentiment_type)
   - We map: predicate → aspect, sentiment_word → opinion
   - We infer: sentiment_type from comparative label

3. Limitations:
   ✗ COQE doesn't have explicit sentiment polarity (POSITIVE/NEGATIVE/NEUTRAL)
   ✗ Some sentences may not have clear aspect-opinion pairs
   ✗ Multi-word aspects/opinions not fully supported

4. Solutions:
   ✓ Use label (1=comparative) → assume POSITIVE sentiment
   ✓ Apply NER/POS tagging to identify multi-word entities
   ✓ Manual annotation for sentiment polarity in subset of data
    """)
    print("-"*80)
    
    # Convert data
    print("\n🔄 Converting COQE to ASTE format...\n")
    results = converter.convert_all()
    converter.print_stats()
    
    # Save results
    print("💾 Saving converted data...\n")
    output_dir = f"{coqe_dir}_ASTE"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for split, data in results.items():
        output_path = f"{output_dir}/{split}_aste.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Saved: {output_path} ({len(data)} samples)")
    
    # Print sample conversion
    print("\n" + "="*80)
    print("Sample Converted Entry")
    print("="*80)
    if results['train']:
        sample = results['train'][0]
        print(json.dumps(sample, indent=2))
    
    print("\n✅ Conversion Complete!")
    print(f"📁 Output saved to: {output_dir}/")


if __name__ == '__main__':
    main()
