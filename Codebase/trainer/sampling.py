# 枚举span
import random
import torch
from trainer import util

# Index 0 is always the NULL entity (CLS span [0,1]) – used for missing COQE slots
NULL_ENTITY_IDX = 0
NULL_SPAN = (0, 1)  # encoding position of [CLS] token

def pos_entity_sample(sen, context_size, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes):
    """Collect positive entity spans.  Index 0 is reserved for the NULL entity
    (prepended *before* this call in train_create_sample), so real entity spans
    start at index 1 in the combined span list."""
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes, pos_entity_start_masks, pos_entity_end_masks = [], [], [], [], [], []
    for e in sen.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_start_masks.append(create_entity_s_e_mask(*e.span, context_size, 1))
        pos_entity_end_masks.append(create_entity_s_e_mask(*e.span, context_size, 0))
        pos_entity_sizes.append(len(e.tokens))

    entity_types = entity_types + pos_entity_types
    entity_masks = entity_masks + pos_entity_masks
    entity_start_masks = entity_start_masks + pos_entity_start_masks
    entity_end_masks = entity_end_masks + pos_entity_end_masks
    entity_sizes = entity_sizes + pos_entity_sizes

    return pos_entity_spans, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes

def neg_entity_sample(sen, pos_entity_spans,neg_entity_count, max_span_size, token_count,context_size,entity_types, entity_masks, entity_start_masks,entity_end_masks, entity_sizes):
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # Randomly select one part to be a negative sample

    if len(neg_entity_spans) < neg_entity_count:
        neg_entity_count = len(neg_entity_spans) * 10
    else:
        neg_entity_count = len(neg_entity_spans)

    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), int(neg_entity_count)))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_start_masks = [create_entity_s_e_mask(*span, context_size, 1) for span in neg_entity_spans]
    neg_entity_end_masks = [create_entity_s_e_mask(*span, context_size, 0) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    entity_types = entity_types + neg_entity_types
    entity_masks = entity_masks + neg_entity_masks
    entity_start_masks =  entity_start_masks + neg_entity_start_masks
    entity_end_masks = entity_end_masks + neg_entity_end_masks
    entity_sizes = entity_sizes + list(neg_entity_sizes)

    return neg_entity_spans,entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes


def _get_entity_idx(entity, all_spans):
    """Return index into all_spans (which starts with NULL_SPAN at 0)."""
    if entity is None:
        return NULL_ENTITY_IDX
    return all_spans.index(entity.span)


def pos_senti_sample(sen, all_entity_spans, context_size):
    """Create positive quintuple samples.
    all_entity_spans[0] == NULL_SPAN; real entities occupy indices 1+.
    Returns 4-element index tuples (s, o, a, p) referencing all_entity_spans.
    """
    pos_rels, pos_senti_keys, pos_senti_types, pos_senti_masks = [], [], [], []
    for rel in sen.sentiments:
        s_idx = _get_entity_idx(rel.s_entity, all_entity_spans)
        o_idx = _get_entity_idx(rel.o_entity, all_entity_spans)
        a_idx = _get_entity_idx(rel.a_entity, all_entity_spans)
        p_idx = _get_entity_idx(rel.p_entity, all_entity_spans)

        quad = (s_idx, o_idx, a_idx, p_idx)
        pos_rels.append(quad)
        pos_senti_keys.append(quad)
        pos_senti_types.append(rel.sentiment_type)

        # Context mask spans the range between all non-null entity spans
        real_spans = [
            e.span for e in [rel.s_entity, rel.o_entity, rel.a_entity, rel.p_entity]
            if e is not None
        ]
        pos_senti_masks.append(create_senti_mask_multi(real_spans, context_size))

    return pos_senti_keys, pos_senti_types, pos_rels, pos_senti_masks


def neg_senti_sample(all_entity_spans, pos_senti_keys, pos_senti_types):
    """Generate negative 4-tuple quintuple candidates.
    all_entity_spans[0] == NULL_SPAN (index 0 = null).
    Returns a list of (s_idx, o_idx, a_idx, p_idx) index tuples that are
    NOT in pos_senti_keys and have at least one non-null entity.
    """
    neg_quads = []
    n = len(all_entity_spans)
    pos_set = set(pos_senti_keys)
    for i1 in range(n):
        for i2 in range(n):
            for i3 in range(n):
                for i4 in range(n):
                    # At least one slot must be a real entity (not all null)
                    if i1 == 0 and i2 == 0 and i3 == 0 and i4 == 0:
                        continue
                    quad = (i1, i2, i3, i4)
                    if quad not in pos_set:
                        neg_quads.append(quad)
    return neg_quads

def create_entity_sample_mask(entity_masks, entity_types, entity_start_masks, entity_end_masks, entity_sizes, context_size):
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks = torch.stack(entity_end_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
    return entity_sample_masks,entity_types,entity_masks,entity_start_masks,entity_end_masks,entity_sizes

def train_create_sample(sen, neg_entity_count: int, neg_senti_count: int, max_span_size: int, senti_type_count: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)

    # ------------------------------------------------------------------
    # Entity spans.  Index 0 is ALWAYS the null entity (CLS span).
    # Real pos/neg entity spans start from index 1 onwards.
    # ------------------------------------------------------------------
    null_mask = create_entity_mask(*NULL_SPAN, context_size)
    null_start_mask = create_entity_s_e_mask(*NULL_SPAN, context_size, 1)
    null_end_mask = create_entity_s_e_mask(*NULL_SPAN, context_size, 0)

    entity_types = [0]          # None type for null entity
    entity_masks = [null_mask]
    entity_start_masks = [null_start_mask]
    entity_end_masks = [null_end_mask]
    entity_sizes = [1]

    # Positive entity spans (real GT entities, indices 1+)
    pos_entity_spans, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes = \
        pos_entity_sample(sen, context_size, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes)

    # Negative entity spans (non-GT candidate spans, indices after pos spans)
    neg_entity_spans, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes = \
        neg_entity_sample(sen, pos_entity_spans, neg_entity_count, max_span_size, token_count, context_size,
                          entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes)

    entity_sample_masks, entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes = \
        create_entity_sample_mask(entity_masks, entity_types, entity_start_masks, entity_end_masks, entity_sizes, context_size)

    # all_entity_spans: null at index 0, real pos at 1..len(pos), neg after
    all_entity_spans = [NULL_SPAN] + pos_entity_spans + list(neg_entity_spans)

    # Positive quintuple samples
    pos_senti_keys, pos_senti_types, pos_rels, pos_senti_masks = \
        pos_senti_sample(sen, all_entity_spans, context_size)

    # Negative quintuple samples
    neg_quads = neg_senti_sample(all_entity_spans, pos_senti_keys, pos_senti_types)
    neg_senti_count = min(len(neg_quads), max(len(neg_entity_spans), 1))
    neg_quads = random.sample(neg_quads, min(len(neg_quads), neg_senti_count))

    neg_rels = neg_quads
    neg_senti_masks = [
        create_senti_mask_multi(
            [all_entity_spans[idx] for idx in quad if idx != NULL_ENTITY_IDX],
            context_size,
        )
        for quad in neg_quads
    ]
    neg_senti_types = [0] * len(neg_quads)

    rels = pos_rels + neg_rels
    senti_types = [r.index for r in pos_senti_types] + neg_senti_types
    senti_masks = pos_senti_masks + neg_senti_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(entity_start_masks) == len(entity_end_masks)
    assert len(rels) == len(senti_masks) == len(senti_types)

    encodings = torch.tensor(encodings, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    adj = torch.tensor(adj, dtype=torch.float)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        senti_masks = torch.stack(senti_masks)
        senti_types = torch.tensor(senti_types, dtype=torch.long)
        senti_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case: no sentiments
        rels = torch.zeros([1, 4], dtype=torch.long)
        senti_types = torch.zeros([1], dtype=torch.long)
        senti_masks = torch.zeros([1, context_size], dtype=torch.bool)
        senti_sample_masks = torch.zeros([1], dtype=torch.bool)

    # sentiment types → one-hot (exclude 'None' class)
    senti_types_onehot = torch.zeros([senti_types.shape[0], senti_type_count], dtype=torch.float32)
    senti_types_onehot.scatter_(1, senti_types.unsqueeze(1), 1)
    senti_types_onehot = senti_types_onehot[:, 1:]

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks=entity_start_masks, entity_end_masks=entity_end_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, senti_masks=senti_masks, senti_types=senti_types_onehot,
                entity_sample_masks=entity_sample_masks, senti_sample_masks=senti_sample_masks,
                adj = adj)

def create_test_sample(sen, max_span_size: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)
    # Null entity at index 0
    entity_start_masks = [create_entity_s_e_mask(*NULL_SPAN, context_size, 1)]
    entity_end_masks   = [create_entity_s_e_mask(*NULL_SPAN, context_size, 0)]
    entity_spans = [list(NULL_SPAN)]
    entity_masks = [create_entity_mask(*NULL_SPAN, context_size)]
    entity_sizes = [1]
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            entity_spans.append(list(span))
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_start_masks.append(create_entity_s_e_mask(*span, context_size, 1))
            entity_end_masks.append(create_entity_s_e_mask(*span, context_size, 0))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    adj = torch.tensor(adj, dtype=torch.float)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks = torch.stack(entity_end_masks)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks=entity_start_masks, entity_end_masks=entity_end_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks,adj = adj)

def create_entity_index(start, end):
    index = []
    for i in range(start,end):
        index.append(i)
    return index
def create_entity_s_e_mask(start, end, context_size,s_e):
    mask = torch.zeros(context_size, dtype=torch.bool)
    if s_e:
        mask[start] = 1
    else:
        mask[end-1] = 1
    return mask

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def create_senti_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def create_senti_mask_multi(spans, context_size):
    """Context mask spanning the region between the first and last of the given spans.
    `spans` is a list of (start, end) tuples in encoding space (non-null entities only).
    Returns a zero mask if fewer than 2 spans are provided.
    """
    if len(spans) < 2:
        return torch.zeros(context_size, dtype=torch.bool)
    sorted_spans = sorted(spans, key=lambda s: s[0])
    ctx_start = sorted_spans[0][1]   # end of first (leftmost) span
    ctx_end   = sorted_spans[-1][0]  # start of last (rightmost) span
    if ctx_start >= ctx_end:
        return torch.zeros(context_size, dtype=torch.bool)
    return create_entity_mask(ctx_start, ctx_end, context_size)

def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch

def create_eval_sample(sen, max_span_size: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)

    # ------------------------------------------------------------------
    # Index 0 is always the NULL entity (CLS span [0, 1]).
    # All enumerated candidate spans start from index 1.
    # ------------------------------------------------------------------
    entity_start_masks = [create_entity_s_e_mask(*NULL_SPAN, context_size, 1)]
    entity_end_masks   = [create_entity_s_e_mask(*NULL_SPAN, context_size, 0)]
    entity_spans = [list(NULL_SPAN)]
    entity_masks = [create_entity_mask(*NULL_SPAN, context_size)]
    entity_sizes = [1]

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            entity_spans.append(list(span))
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_start_masks.append(create_entity_s_e_mask(*span, context_size, 1))
            entity_end_masks.append(create_entity_s_e_mask(*span, context_size, 0))
            entity_sizes.append(size)

    # create tensors
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    adj = torch.tensor(adj, dtype=torch.float)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    if entity_masks:
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks   = torch.stack(entity_end_masks)
        entity_masks       = torch.stack(entity_masks)
        entity_sizes       = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans       = torch.tensor(entity_spans, dtype=torch.long)
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
        # Mask out the null entity (index 0) so it is not evaluated as a real span
        entity_sample_masks[0] = 0
    else:
        entity_masks        = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes        = torch.zeros([1], dtype=torch.long)
        entity_spans        = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks=entity_start_masks, entity_end_masks=entity_end_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans,
                entity_sample_masks=entity_sample_masks, adj=adj)

