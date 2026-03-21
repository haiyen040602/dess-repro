"""
Convert cameraCOQE to the new quintuple training format for D2E2S.

Quintuple format in raw .txt files (per 2 lines):
  Line 1: sentence text <TAB> label (0 or 1)
  Line 2: [subjects];[objects];[aspects];[predicates];[comparator]
  - Each slot: [idx1&&word1 idx2&&word2 ...]  (indices are 1-based)
  - Comparator: plain integer  0=EQUAL, 1=BETTER, -1=WORSE, 2=DIFFERENT

The dep JSON files (*_dep_triple_polarity_result.json) already contain
dependency/POS parsed sentences for comparative (label=1) sentences.

This script:
1. Parses the raw .txt quintuple annotation for all label=1 sentences.
2. Merges it with the dep JSON (which has tokenization, POS, dependency).
3. Emits a new JSON with 4-entity format:
   entities: [{type: subject|object|aspect|predicate, start, end}, ...]
   sentiments: [{type: BETTER|EQUAL|WORSE|DIFFERENT,
                 s: <idx or null>, o: <idx or null>,
                 a: <idx or null>, p: <idx or null>}]
"""

import json
import os
import re

# Comparator integer -> sentiment type string
COMP_LABEL_MAP = {
    '0': 'EQUAL',
    '1': 'BETTER',
    '-1': 'WORSE',
    '2': 'DIFFERENT',
}

# Slot index -> entity type name
ROLE_TYPES = ['subject', 'object', 'aspect', 'predicate']


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_component(component_str):
    """Parse one quintuple slot like '[11&&S50 12&&camera]' → sorted list of
    0-based token indices: [10, 11].
    Returns [] for empty slots.
    """
    s = component_str.strip().lstrip('[').rstrip(']').strip()
    if not s:
        return []
    indices = []
    for item in s.split():
        if '&&' in item:
            try:
                idx = int(item.split('&&', 1)[0])
                indices.append(idx - 1)   # convert 1-based → 0-based
            except ValueError:
                pass
    return sorted(set(indices))


def _parse_quintuple_line(line):
    """Parse the quintuple annotation line into 5 lists of 0-based token indices."""
    parts = line.split(';')
    while len(parts) < 5:
        parts.append('[]')
    s_idxs  = _parse_component(parts[0])
    o_idxs  = _parse_component(parts[1])
    a_idxs  = _parse_component(parts[2])
    p_idxs  = _parse_component(parts[3])
    # Comparator: a bare integer (no &&), strip brackets
    comp_raw = parts[4].strip().lstrip('[').rstrip(']').strip()
    return s_idxs, o_idxs, a_idxs, p_idxs, comp_raw


def _idxs_to_span(indices):
    """Convert a sorted list of 0-based token indices to a (start, end) span tuple.
    Returns None if indices is empty.
    """
    if not indices:
        return None
    return (min(indices), max(indices) + 1)


def _build_raw_map(txt_path):
    """Build a dict mapping normalized sentence text → (label, quintuple_line)."""
    raw_map = {}
    data = open(txt_path, encoding='utf-8').readlines()
    for i in range(0, len(data) - 1, 2):
        sent_line = data[i].rstrip()
        quint_line = data[i + 1].rstrip()
        parts = sent_line.split()
        if not parts:
            continue
        label = parts[-1]
        text = ' '.join(parts[:-1])
        raw_map[text] = (label, quint_line)
    return raw_map


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_split(dep_json_path, txt_path, output_path, fallback_type='EQUAL'):
    """Convert one data split.

    For sentences that have a matching quintuple annotation, the 4-entity
    format is produced.  For dep JSON entries without a matching quintuple
    (possible in partial merges) we fall back to the old target/opinion style
    but still emit under the new schema.
    """
    dep_data = json.load(open(dep_json_path, encoding='utf-8'))
    raw_map  = _build_raw_map(txt_path)

    output = []
    matched = 0
    fallback = 0

    for entry in dep_data:
        tokens = entry['tokens']
        text   = ' '.join(tokens)

        new_entry = {
            'entities': [],
            'sentiments': [],
            'tokens': tokens,
            'pos': entry.get('pos', []),
            'dependency': entry.get('dependency', []),
            'orig_id': entry.get('orig_id', ''),
        }

        label_txt, quint_line = raw_map.get(text, (None, None))

        if quint_line is not None:
            matched += 1
            s_idxs, o_idxs, a_idxs, p_idxs, comp_raw = _parse_quintuple_line(quint_line)
            senti_type = COMP_LABEL_MAP.get(comp_raw, fallback_type)

            spans = {
                'subject':   _idxs_to_span(s_idxs),
                'object':    _idxs_to_span(o_idxs),
                'aspect':    _idxs_to_span(a_idxs),
                'predicate': _idxs_to_span(p_idxs),
            }

            entity_list = []
            role_to_idx = {}   # role name -> index in entity_list
            for role in ROLE_TYPES:
                span = spans[role]
                if span is not None:
                    entity_list.append({
                        'type': role,
                        'start': span[0],
                        'end': span[1],
                    })
                    role_to_idx[role] = len(entity_list) - 1

            sentiment = {
                'type': senti_type,
                's': role_to_idx.get('subject'),
                'o': role_to_idx.get('object'),
                'a': role_to_idx.get('aspect'),
                'p': role_to_idx.get('predicate'),
            }

            new_entry['entities']   = entity_list
            new_entry['sentiments'] = [sentiment]

        else:
            # Fallback: use old entity annotations if available
            fallback += 1
            old_entities = entry.get('entities', [])
            # Remap old type names
            type_remap = {'target': 'aspect', 'opinion': 'predicate',
                          'subject': 'subject', 'object': 'object',
                          'aspect': 'aspect', 'predicate': 'predicate'}
            entity_list = []
            for ent in old_entities:
                new_type = type_remap.get(ent['type'], ent['type'])
                entity_list.append({'type': new_type,
                                    'start': ent['start'],
                                    'end': ent['end']})

            # Build sentiment from old sentiments
            old_sentiments = entry.get('sentiments', [])
            senti_list = []
            for old_s in old_sentiments:
                senti_type = old_s.get('type', fallback_type)
                # head → aspect (index 0 in remap if matched)
                head_i = old_s.get('head'); tail_i = old_s.get('tail')
                new_s = {
                    'type': senti_type,
                    's': None, 'o': None,
                    'a': head_i, 'p': tail_i,
                }
                senti_list.append(new_s)

            new_entry['entities']   = entity_list
            new_entry['sentiments'] = senti_list

        output.append(new_entry)

    print(f"  Entries  : {len(dep_data)}")
    print(f"  Matched  : {matched}  (quintuple annotation found)")
    print(f"  Fallback : {fallback}  (old entity format used)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {output_path}")
    return output


def main():
    base_dep = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE'
    base_txt = base_dep
    output_dir = '/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE_quintuple'

    for split in ('train', 'dev', 'test'):
        print(f"\n=== {split.upper()} ===")
        convert_split(
            dep_json_path=os.path.join(base_dep, f'{split}_dep_triple_polarity_result.json'),
            txt_path     =os.path.join(base_txt, f'{split}.txt'),
            output_path  =os.path.join(output_dir, f'{split}_quintuple.json'),
        )


if __name__ == '__main__':
    main()
