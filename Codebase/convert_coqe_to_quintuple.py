import json
import os
import re
from collections import defaultdict

import spacy
from spacy.tokens import Doc


COMP_LABEL_MAP = {
    "0": "EQUAL",
    "1": "BETTER",
    "-1": "WORSE",
    "2": "DIFFERENT",
}

ROLE_TYPES = ["subject", "object", "aspect", "predicate"]
EMPTY_ANNOTATION = "[[];[];[];[];[]]"
_SPACY_NLP = None


def _normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _parse_component(component_str):
    s = component_str.strip().lstrip("[").rstrip("]").strip()
    if not s:
        return []

    indices = []
    for item in s.split():
        if "&&" not in item:
            continue
        try:
            indices.append(int(item.split("&&", 1)[0]) - 1)
        except ValueError:
            continue
    return sorted(set(indices))


def _parse_quintuple_line(line):
    parts = line.split(";")
    while len(parts) < 5:
        parts.append("[]")

    s_idxs = _parse_component(parts[0])
    o_idxs = _parse_component(parts[1])
    a_idxs = _parse_component(parts[2])
    p_idxs = _parse_component(parts[3])
    comp_raw = parts[4].strip().lstrip("[").rstrip("]").strip()
    return s_idxs, o_idxs, a_idxs, p_idxs, comp_raw


def _idxs_to_span(indices):
    if not indices:
        return None
    return min(indices), max(indices) + 1


def _parse_sentence_header(line):
    line = line.rstrip()
    if "\t" in line:
        text, label = line.rsplit("\t", 1)
        label = label.strip()
        if label in {"0", "1"}:
            return _normalize_text(text), int(label)

    match = re.match(r"^(.*?)(?:\s+)([01])$", line)
    if match:
        return _normalize_text(match.group(1)), int(match.group(2))
    return None


def _parse_raw_blocks(txt_path):
    with open(txt_path, encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file if line.strip()]

    samples = []
    index = 0
    while index < len(lines):
        header = _parse_sentence_header(lines[index])
        if header is None:
            raise ValueError(f"Unexpected raw format at line {index + 1}: {lines[index]}")

        text, label = header
        index += 1
        annotations = []
        while index < len(lines) and _parse_sentence_header(lines[index]) is None:
            annotations.append(lines[index].strip())
            index += 1

        if not annotations:
            annotations = [EMPTY_ANNOTATION]

        samples.append({
            "text": text,
            "label": label,
            "annotations": annotations,
        })

    return samples


def _build_dep_map(dep_json_path):
    dep_map = defaultdict(list)
    if not os.path.exists(dep_json_path):
        return dep_map

    dep_data = json.load(open(dep_json_path, encoding="utf-8"))
    for entry in dep_data:
        norm_text = _normalize_text(" ".join(entry.get("tokens", [])))
        dep_map[norm_text].append(entry)
    return dep_map


def _default_pos(tokens):
    return [[token, "NN"] for token in tokens]


def _default_dependency(tokens):
    dependency = [["ROOT", 1, 1]]
    for token_idx in range(2, len(tokens) + 1):
        dependency.append(["NEXT", token_idx - 1, token_idx])
    return dependency


def _normalize_single_root(dependency, token_count):
    if token_count <= 0 or not dependency:
        return dependency

    normalized = []
    for item in dependency:
        if not isinstance(item, list) or len(item) != 3:
            continue
        rel, head, dep = item
        if not isinstance(head, int) or not isinstance(dep, int):
            continue
        if not (1 <= head <= token_count and 1 <= dep <= token_count):
            continue
        normalized.append([str(rel), head, dep])

    if not normalized:
        return _default_dependency(["x"] * token_count)

    dep_to_item = {}
    for rel, head, dep in normalized:
        dep_to_item[dep] = [rel, head, dep]

    for dep_idx in range(1, token_count + 1):
        if dep_idx not in dep_to_item:
            dep_to_item[dep_idx] = ["dep", 1 if dep_idx != 1 else dep_idx, dep_idx]

    root_candidates = [d for d, (rel, head, _) in dep_to_item.items() if rel.upper() == "ROOT" and head == d]
    main_root = min(root_candidates) if root_candidates else 1

    for dep_idx, item in dep_to_item.items():
        rel, head, dep = item
        if dep_idx == main_root:
            item[0] = "ROOT"
            item[1] = dep_idx
            continue

        if rel.upper() == "ROOT" and head == dep_idx:
            item[0] = "dep"
            item[1] = main_root
        elif head == dep_idx:
            item[1] = main_root

    return [dep_to_item[i] for i in range(1, token_count + 1)]


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def _parse_dependency_with_spacy(tokens):
    if not tokens:
        return []

    nlp = _get_spacy_nlp()
    spaces = [True] * len(tokens)
    spaces[-1] = False
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

    for pipe_name in nlp.pipe_names:
        doc = nlp.get_pipe(pipe_name)(doc)

    dependency = []
    for token in doc:
        dependency.append([token.dep_, token.head.i + 1, token.i + 1])

    return _normalize_single_root(dependency, len(tokens))


def _build_entry(raw_sample, dep_entry, sample_idx, fallback_type):
    text = raw_sample["text"]
    label = raw_sample["label"]

    if dep_entry is not None:
        tokens = dep_entry.get("tokens", text.split())
        pos = dep_entry.get("pos") or _default_pos(tokens)
        dependency = _normalize_single_root(dep_entry.get("dependency", []), len(tokens))
        orig_id = dep_entry.get("orig_id", f"cameraCOQE:{sample_idx}")
    else:
        tokens = text.split()
        pos = _default_pos(tokens)
        try:
            dependency = _parse_dependency_with_spacy(tokens)
        except Exception:
            dependency = _default_dependency(tokens)
        orig_id = f"cameraCOQE:{sample_idx}"

    entry = {
        "entities": [],
        "sentiments": [],
        "tokens": tokens,
        "pos": pos,
        "dependency": dependency,
        "orig_id": orig_id,
        "is_comparative": bool(label),
    }

    entity_map = {}

    def get_entity_index(role, span):
        if span is None:
            return None
        key = (role, span[0], span[1])
        if key not in entity_map:
            entity_map[key] = len(entry["entities"])
            entry["entities"].append({
                "type": role,
                "start": span[0],
                "end": span[1],
            })
        return entity_map[key]

    for annotation in raw_sample["annotations"]:
        s_idxs, o_idxs, a_idxs, p_idxs, comp_raw = _parse_quintuple_line(annotation)
        role_spans = {
            "subject": _idxs_to_span(s_idxs),
            "object": _idxs_to_span(o_idxs),
            "aspect": _idxs_to_span(a_idxs),
            "predicate": _idxs_to_span(p_idxs),
        }

        if annotation == EMPTY_ANNOTATION and not label:
            continue

        if not any(span is not None for span in role_spans.values()) and not comp_raw:
            continue

        entry["sentiments"].append({
            "type": COMP_LABEL_MAP.get(comp_raw, fallback_type),
            "s": get_entity_index("subject", role_spans["subject"]),
            "o": get_entity_index("object", role_spans["object"]),
            "a": get_entity_index("aspect", role_spans["aspect"]),
            "p": get_entity_index("predicate", role_spans["predicate"]),
        })

    return entry


def convert_split(dep_json_path, txt_path, output_path, fallback_type="EQUAL"):
    raw_samples = _parse_raw_blocks(txt_path)
    dep_map = _build_dep_map(dep_json_path)

    output = []
    matched = 0
    fallback = 0
    comparative = 0
    non_comparative = 0

    for sample_idx, raw_sample in enumerate(raw_samples):
        dep_entry = None
        norm_text = _normalize_text(raw_sample["text"])
        if dep_map.get(norm_text):
            dep_entry = dep_map[norm_text].pop(0)
            matched += 1
        else:
            fallback += 1

        if raw_sample["label"]:
            comparative += 1
        else:
            non_comparative += 1

        output.append(_build_entry(raw_sample, dep_entry, sample_idx, fallback_type))

    print(f"  Raw samples      : {len(raw_samples)}")
    print(f"  Comparative      : {comparative}")
    print(f"  Non-comparative  : {non_comparative}")
    print(f"  Matched dep rows : {matched}")
    print(f"  Fallback rows    : {fallback}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"  Saved -> {output_path}")
    return output


def main():
    base_dep = "/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE"
    base_txt = base_dep
    output_dir = "/home/haiyan/DESS-main/DESS-main/Codebase/data/cameraCOQE_quintuple"

    for split in ("train", "dev", "test"):
        print(f"\n=== {split.upper()} ===")
        convert_split(
            dep_json_path=os.path.join(base_dep, f"{split}_dep_triple_polarity_result.json"),
            txt_path=os.path.join(base_txt, f"{split}.txt"),
            output_path=os.path.join(output_dir, f"{split}_quintuple.json"),
        )


if __name__ == "__main__":
    main()
