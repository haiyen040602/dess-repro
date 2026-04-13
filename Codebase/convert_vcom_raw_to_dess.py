import json
import os
import argparse
from collections import Counter


LABEL_MAP = {
    "EQL": "EQUAL",
    "DIF": "DIFFERENT",
    "COM": "DIFFERENT",
    "SUP": "DIFFERENT",
    "COM+": "BETTER",
    "SUP+": "BETTER",
    "COM-": "WORSE",
    "SUP-": "WORSE",
}

VCOM_8_LABELS = {"COM", "COM+", "COM-", "DIF", "EQL", "SUP", "SUP+", "SUP-"}

try:
    from underthesea import pos_tag as uds_pos_tag
    from underthesea import dependency_parse as uds_dependency_parse
except Exception:
    uds_pos_tag = None
    uds_dependency_parse = None


def _parse_indices(items):
    indices = []
    for item in items or []:
        if "&&" not in item:
            continue
        left = item.split("&&", 1)[0].strip()
        if not left.isdigit():
            continue
        indices.append(int(left) - 1)
    return sorted(set(indices))


def _indices_to_span(indices):
    if not indices:
        return None
    return min(indices), max(indices) + 1


def _build_default_pos(tokens):
    return [[tok, "X"] for tok in tokens]


def _build_default_dependency(tokens):
    # Keep the same convention used by current converters/input reader:
    # first edge is ROOT(1->1), then a linear chain 1->2->3... over token indices.
    dependency = [["ROOT", 1, 1]]
    for token_idx in range(2, len(tokens) + 1):
        dependency.append(["NEXT", token_idx - 1, token_idx])
    return dependency


def _build_pos_with_underthesea(tokens):
    if uds_pos_tag is None or not tokens:
        return _build_default_pos(tokens), False

    sentence = " ".join(tokens)
    try:
        tagged = uds_pos_tag(sentence)
    except Exception:
        return _build_default_pos(tokens), False

    flat = []
    for item in tagged:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        word = str(item[0]).strip()
        pos = str(item[1]).strip() if item[1] is not None else "X"
        if not word:
            continue
        parts = word.split()
        for part in parts:
            flat.append((part, pos or "X"))

    if len(flat) != len(tokens):
        return _build_default_pos(tokens), False

    pos_pairs = []
    for i, tok in enumerate(tokens):
        pos_pairs.append([tok, flat[i][1]])
    return pos_pairs, True


def _normalize_dependency_items(dep_items, token_count):
    dep_map = {}

    def parse_int(val):
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.strip().lstrip("-").isdigit():
            try:
                return int(val.strip())
            except Exception:
                return None
        return None

    for fallback_dep_idx, item in enumerate(dep_items or [], start=1):
        dep_idx = None
        head_idx = None
        rel = "dep"

        if isinstance(item, dict):
            dep_idx = parse_int(
                item.get("id")
                or item.get("index")
                or item.get("dependent")
                or item.get("dep")
                or item.get("child")
            )
            head_idx = parse_int(
                item.get("head")
                or item.get("head_index")
                or item.get("governor")
                or item.get("parent")
            )
            rel = str(
                item.get("deprel")
                or item.get("relation")
                or item.get("label")
                or item.get("rel")
                or "dep"
            )
        elif isinstance(item, (list, tuple)):
            # Common underthesea output: (word, head_idx, relation)
            if len(item) >= 3 and isinstance(item[1], int):
                dep_idx = fallback_dep_idx
                head_idx = parse_int(item[1])
                rel = str(item[2]) if item[2] is not None else "dep"

            int_vals = [parse_int(v) for v in item]
            int_vals = [v for v in int_vals if v is not None]
            if dep_idx is None and len(int_vals) >= 2:
                dep_idx = int_vals[0]
                head_idx = int_vals[1]
            str_vals = [str(v) for v in item if isinstance(v, str)]
            if len(item) >= 3 and isinstance(item[1], int) and isinstance(item[2], str):
                pass
            else:
                for sv in str_vals:
                    if not sv.strip().isdigit():
                        rel = sv.strip() or rel
                        break

        if dep_idx is None or dep_idx <= 0 or dep_idx > token_count:
            continue

        if head_idx is None:
            head_idx = dep_idx
        if head_idx == 0:
            head_idx = dep_idx
            rel = "ROOT"
        if head_idx < 1 or head_idx > token_count:
            head_idx = dep_idx

        if dep_idx not in dep_map:
            dep_map[dep_idx] = [rel, head_idx, dep_idx]

    if not dep_map:
        return None

    main_root = None
    for dep_idx in range(1, token_count + 1):
        row = dep_map.get(dep_idx)
        if row and (row[0].upper() == "ROOT" or row[1] == dep_idx):
            main_root = dep_idx
            break
    if main_root is None:
        main_root = 1

    ordered = [["ROOT", 1, 1]]
    for dep_idx in range(2, token_count + 1):
        row = dep_map.get(dep_idx)
        if row is None:
            head = dep_idx - 1 if dep_idx > 1 else dep_idx
            ordered.append(["dep", head, dep_idx])
            continue

        rel, head, dep = row
        if head == dep or rel.upper() == "ROOT":
            head = main_root
            rel = "dep"
        ordered.append([rel, head, dep])

    return ordered


def _build_dependency_with_underthesea(tokens):
    if uds_dependency_parse is None or not tokens:
        return _build_default_dependency(tokens), False

    sentence = " ".join(tokens)
    try:
        dep_items = uds_dependency_parse(sentence)
    except Exception:
        return _build_default_dependency(tokens), False

    normalized = _normalize_dependency_items(dep_items, len(tokens))
    if normalized is None:
        return _build_default_dependency(tokens), False
    return normalized, True


def _build_linguistic_features(tokens, dependency_parser):
    if dependency_parser == "underthesea":
        pos, pos_ok = _build_pos_with_underthesea(tokens)
        dependency, dep_ok = _build_dependency_with_underthesea(tokens)
        return pos, dependency, pos_ok, dep_ok

    return _build_default_pos(tokens), _build_default_dependency(tokens), False, False


def _new_sample(tokens, pos, dependency, orig_id):
    return {
        "entities": [],
        "sentiments": [],
        "tokens": tokens,
        "pos": pos,
        "dependency": dependency,
        "orig_id": orig_id,
        "is_comparative": False,
    }


def _parse_sentence_line(line):
    parts = line.split("\t")
    if len(parts) >= 2:
        tokenized = parts[-1].strip()
    else:
        tokenized = parts[0].strip()

    if not tokenized:
        return []
    return tokenized.split()


def _parse_annotation_line(line):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "label" not in obj:
        return None
    return obj


def _finalize_sample(sample):
    sample["is_comparative"] = bool(sample["sentiments"])
    return sample


def convert_split(
    input_path,
    output_path,
    split_name,
    label_mode="dess4",
    dependency_parser="underthesea",
    strict_dependency=False,
):
    samples = []
    label_counter = Counter()
    skipped_labels = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    current = None
    entity_map = None
    sentence_idx = 0
    uds_pos_used = 0
    uds_dep_used = 0
    uds_dep_fallback = 0

    if dependency_parser == "underthesea" and uds_dependency_parse is None:
        print(
            f"[{split_name}] WARNING: underthesea dependency parser unavailable; "
            "falling back to linear dependencies for all sentences."
        )

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        annotation = _parse_annotation_line(line)
        if annotation is not None:
            if current is None:
                continue

            raw_label = str(annotation.get("label", "")).strip()
            if label_mode == "vcom8":
                mapped = raw_label if raw_label in VCOM_8_LABELS else None
            else:
                mapped = LABEL_MAP.get(raw_label)

            if mapped is None:
                skipped_labels[raw_label] += 1
                continue

            role_spans = {}
            for role in ("subject", "object", "aspect", "predicate"):
                idxs = _parse_indices(annotation.get(role, []))
                max_len = len(current["tokens"])
                idxs = [idx for idx in idxs if 0 <= idx < max_len]
                role_spans[role] = _indices_to_span(idxs)

            # Ignore malformed annotation with no role span at all.
            if not any(role_spans.values()):
                continue

            def get_entity_idx(role_name, span):
                if span is None:
                    return None
                key = (role_name, span[0], span[1])
                if key not in entity_map:
                    entity_map[key] = len(current["entities"])
                    current["entities"].append(
                        {
                            "type": role_name,
                            "start": span[0],
                            "end": span[1],
                        }
                    )
                return entity_map[key]

            current["sentiments"].append(
                {
                    "type": mapped,
                    "s": get_entity_idx("subject", role_spans["subject"]),
                    "o": get_entity_idx("object", role_spans["object"]),
                    "a": get_entity_idx("aspect", role_spans["aspect"]),
                    "p": get_entity_idx("predicate", role_spans["predicate"]),
                }
            )
            label_counter[mapped] += 1
            continue

        # New sentence line.
        if current is not None:
            samples.append(_finalize_sample(current))

        sentence_idx += 1
        tokens = _parse_sentence_line(line)
        pos, dependency, pos_ok, dep_ok = _build_linguistic_features(tokens, dependency_parser=dependency_parser)
        if dependency_parser == "underthesea":
            if pos_ok:
                uds_pos_used += 1
            if dep_ok:
                uds_dep_used += 1
            else:
                uds_dep_fallback += 1
        current = _new_sample(
            tokens=tokens,
            pos=pos,
            dependency=dependency,
            orig_id=f"vcom:{split_name}:{sentence_idx}",
        )
        entity_map = {}

    if current is not None:
        samples.append(_finalize_sample(current))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    comparative_count = sum(1 for s in samples if s["is_comparative"])
    print(f"[{split_name}] sentences={len(samples)} comparative={comparative_count} non_comparative={len(samples)-comparative_count}")
    print(f"[{split_name}] mapped_label_counts={dict(label_counter)}")
    if skipped_labels:
        print(f"[{split_name}] skipped_raw_labels={dict(skipped_labels)}")
    if dependency_parser == "underthesea":
        print(
            f"[{split_name}] pos_underthesea_used={uds_pos_used} "
            f"dep_underthesea_used={uds_dep_used} dep_fallback_linear={uds_dep_fallback}"
        )
        if strict_dependency and uds_dep_fallback > 0:
            raise RuntimeError(
                f"Strict dependency mode failed for split={split_name}: "
                f"{uds_dep_fallback} sentences fell back to linear dependency."
            )



def main():
    parser = argparse.ArgumentParser(description="Convert vcom raw txt to DESS json format")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="dess4",
        choices=["dess4", "vcom8"],
        help="dess4: map to BETTER/EQUAL/WORSE/DIFFERENT, vcom8: keep original 8 labels",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional custom output directory",
    )
    parser.add_argument(
        "--dependency_parser",
        type=str,
        default="underthesea",
        choices=["underthesea", "linear"],
        help="Dependency parser backend for Vietnamese sentences",
    )
    parser.add_argument(
        "--strict_dependency",
        action="store_true",
        default=False,
        help="Fail conversion if any sentence cannot be parsed by dependency parser and falls back",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw", "vcom-raw")
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_name = "vcom_quintuple_8label" if args.label_mode == "vcom8" else "vcom_quintuple"
        out_dir = os.path.join(base_dir, "data", out_name)

    print(f"label_mode={args.label_mode}")
    print(f"output_dir={out_dir}")
    print(f"dependency_parser={args.dependency_parser}")

    for split in ("train", "dev", "test"):
        convert_split(
            input_path=os.path.join(raw_dir, f"{split}.txt"),
            output_path=os.path.join(out_dir, f"{split}_quintuple.json"),
            split_name=split,
            label_mode=args.label_mode,
            dependency_parser=args.dependency_parser,
            strict_dependency=args.strict_dependency,
        )


if __name__ == "__main__":
    main()
