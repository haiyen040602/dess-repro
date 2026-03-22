from trainer.entities import Sentence, Dataset, EntityType, sentimentType
from transformers import PreTrainedTokenizerBase
from trainer.input_reader import JsonInputReader
from typing import List, Tuple, Dict
import torch
import json
from trainer import util
import re
from sklearn.metrics import precision_recall_fscore_support as prfs
import jinja2
import os

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: JsonInputReader, text_encoder: PreTrainedTokenizerBase,
                 sen_filter_threshold: float,
                 predictions_path: str, examples_path: str, example_count: int, epoch: int, dataset_label: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._sen_filter_threshold = sen_filter_threshold

        self._epoch = epoch
        self._dataset_label = dataset_label

        self._predictions_path = predictions_path

        self._examples_path = examples_path
        self._example_count = example_count

        # sentiments
        self._gt_sentiments = []  # ground truth
        self._pred_sentiments = []  # prediction

        # entities
        self._gt_entities = []  # ground truth
        self._pred_entities = []  # prediction

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation
        self._pseudo_quad_type = sentimentType('EXACT_QUAD', 1, 'EQ4', 'Exact-Quadruple')

        self._convert_gt(self._dataset.sentences)


    def eval_batch(self, batch_entity_clf: torch.tensor, batch_senti_clf: torch.tensor,
                   batch_rels: torch.tensor, batch: dict):
        batch_size = batch_senti_clf.shape[0]
        senti_class_count = batch_senti_clf.shape[2]

        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= batch['entity_sample_masks'].long()

        batch_senti_clf = batch_senti_clf.view(batch_size, -1)

        # apply threshold to sentiments
        if self._sen_filter_threshold > 0:
            batch_senti_clf[batch_senti_clf < self._sen_filter_threshold] = 0

        for i in range(batch_size):
            senti_clf = batch_senti_clf[i]
            entity_types = batch_entity_types[i]

            # get predicted sentiment labels and corresponding quintuples
            senti_nonzero = senti_clf.nonzero().view(-1)
            senti_scores = senti_clf[senti_nonzero]

            senti_types = (senti_nonzero % senti_class_count) + 1
            senti_indices = senti_nonzero // senti_class_count

            rels = batch_rels[i][senti_indices]  # shape: [n_pred, 4]

            # spans for all 4 entity slots  →  [n_pred, 4, 2]
            senti_entity_spans = batch['entity_spans'][i][rels].long()

            # entity types for each of the 4 slots  →  [n_pred, 4]
            senti_entity_types = torch.zeros([rels.shape[0], 4], dtype=torch.long)
            if rels.shape[0] != 0:
                senti_entity_types = torch.stack(
                    [entity_types[rels[j]] for j in range(rels.shape[0])]
                )

            sample_pred_sentiments = self._convert_pred_sentiments(
                senti_types, senti_entity_spans, senti_entity_types, senti_scores
            )

            # get entities that are not classified as 'None' and not the null span (idx 0)
            valid_entity_indices = entity_types.nonzero().view(-1)
            # exclude index 0 (null entity)
            valid_entity_indices = valid_entity_indices[valid_entity_indices != 0]
            valid_entity_types = entity_types[valid_entity_indices]
            valid_entity_spans = batch['entity_spans'][i][valid_entity_indices]
            valid_entity_scores = torch.gather(batch_entity_clf[i][valid_entity_indices], 1,
                                               valid_entity_types.unsqueeze(1)).view(-1)

            sample_pred_entities = self._convert_pred_entities(
                valid_entity_types, valid_entity_spans, valid_entity_scores
            )

            self._pred_entities.append(sample_pred_entities)
            self._pred_sentiments.append(sample_pred_sentiments)

    def _convert_pred_entities(self, pred_types: torch.tensor, pred_spans: torch.tensor, pred_scores: torch.tensor):
        converted_preds = []

        for i in range(pred_types.shape[0]):
            label_idx = pred_types[i].item()
            entity_type = self._input_reader.get_entity_type(label_idx)

            start, end = pred_spans[i].tolist()
            score = pred_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_pred_sentiments(self, pred_senti_types: torch.tensor, pred_entity_spans: torch.tensor,
                                pred_entity_types: torch.tensor, pred_scores: torch.tensor):
        """Convert predicted quintuples (s, o, a, p, label) for evaluation."""
        converted_rels = {}
        NULL_SPAN = (0, 1)  # encoding position of CLS = null entity
        expected_roles = ['subject', 'object', 'aspect', 'predicate']

        for i in range(pred_senti_types.shape[0]):
            label_idx = pred_senti_types[i].item()
            pred_senti_type = self._input_reader.get_sentiment_type(label_idx)
            score = pred_scores[i].item()

            entity_tuples = []
            valid_tuple = True
            for j in range(4):  # s, o, a, p
                type_idx = pred_entity_types[i][j].item()
                entity_type = self._input_reader.get_entity_type(type_idx)
                start, end = pred_entity_spans[i][j].tolist()
                if (start, end) != NULL_SPAN and entity_type.identifier != expected_roles[j]:
                    valid_tuple = False
                    break
                entity_tuples.append((start, end, entity_type))

            if not valid_tuple:
                continue
            if (entity_tuples[3][0], entity_tuples[3][1]) == NULL_SPAN:
                continue

            converted_rel = tuple(entity_tuples + [pred_senti_type])
            converted_with_score = tuple(list(converted_rel) + [score])

            if converted_rel not in converted_rels or score > converted_rels[converted_rel][-1]:
                converted_rels[converted_rel] = converted_with_score

        return sorted(converted_rels.values(), key=lambda item: item[-1], reverse=True)

    def _convert_gt(self, sens: List[Sentence]):
        for sen in sens:
            gt_sentiments = sen.sentiments
            gt_entities = sen.entities

            sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
            sample_gt_sentiments = [rel.as_tuple() for rel in gt_sentiments]


            self._gt_entities.append(sample_gt_entities)
            self._gt_sentiments.append(sample_gt_sentiments)

    def compute_scores(self, print_examples: int = 0, print_extra_metrics: bool = False):
        print("Evaluation")

        print("")
        print("--- Entity (s/o/a/p) Span Extraction ---")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities, self._pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Comparative Quintuple Extraction (span-only) ---")
        print("")
        gt, pred = self._convert_by_setting(self._gt_sentiments, self._pred_sentiments, include_entity_types=False)
        senti_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Comparative Quintuple Extraction (spans + entity types) ---")
        print("")
        gt, pred = self._convert_by_setting(self._gt_sentiments, self._pred_sentiments, include_entity_types=True)
        senti_nec_eval = self._score(gt, pred, print_results=True)

        extra_eval = {}
        if print_extra_metrics:
            print("")
            print("--- Comparative Label Prediction (label-only) ---")
            print("")
            label_gt = self._convert_label_only(self._gt_sentiments)
            label_pred = self._convert_label_only(self._pred_sentiments)
            extra_eval['label'] = self._score(label_gt, label_pred, print_results=True)

            print("")
            print("--- Exact Quadruple Extraction (span-only) ---")
            print("")
            t4_gt = self._convert_t4_only(self._gt_sentiments)
            t4_pred = self._convert_t4_only(self._pred_sentiments)
            extra_eval['exact_quadruple'] = self._score(t4_gt, t4_pred, print_results=True)

        # Optionally print examples (text + TP/FP/FN) to console
        if print_examples and print_examples > 0:
            n = print_examples
            print('\n' + '-' * 40)
            print(f'Printing {n} example(s) per category (entities / sentiments)')
            print('-' * 40)

            entity_examples = []
            senti_examples = []
            senti_examples_nec = []

            for i, doc in enumerate(self._dataset.sentences):
                entity_examples.append(self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                             include_entity_types=True, to_html=self._entity_to_html))
                senti_examples.append(self._convert_example(doc, self._gt_sentiments[i], self._pred_sentiments[i],
                                                            include_entity_types=False, to_html=self._senti_to_html))
                senti_examples_nec.append(self._convert_example(doc, self._gt_sentiments[i], self._pred_sentiments[i],
                                                                include_entity_types=True, to_html=self._senti_to_html))

            def strip_html(s: str) -> str:
                return re.sub('<[^<]+?>', '', s)

            def print_examples_list(examples, title: str):
                print(f"\n*** {title} ***")
                for ex in examples[:n]:
                    print('\nTEXT:')
                    print(strip_html(ex['text']))
                    print('\nTP:')
                    for tp in ex['tp']:
                        print(f"  - {strip_html(tp[0])} \t| type: {tp[1]} \t| score: {tp[2]:.4f}")
                    print('FN:')
                    for fn in ex['fn']:
                        print(f"  - {strip_html(fn[0])} \t| type: {fn[1]}")
                    print('FP:')
                    for fp in ex['fp']:
                        print(f"  - {strip_html(fp[0])} \t| type: {fp[1]} \t| score: {fp[2]:.4f}")

            print_examples_list(entity_examples, 'Entity Extraction Examples')
            print_examples_list(senti_examples, 'Quintuple Extraction Examples (span-only)')
            print_examples_list(senti_examples_nec, 'Quintuple Extraction Examples (with entity types)')

            print('\n' + '-' * 40)

        return ner_eval, senti_eval, senti_nec_eval, extra_eval

    def _convert_label_only(self, sentiments: List[List[Tuple]]):
        converted = []
        for sample in sentiments:
            converted.append([(sentiment[4],) for sentiment in sample])
        return converted

    def _convert_t4_only(self, sentiments: List[List[Tuple]]):
        converted = []
        for sample in sentiments:
            converted_sample = []
            for sentiment in sample:
                quad = tuple((sentiment[j][0], sentiment[j][1]) for j in range(4))
                converted_sample.append(tuple(list(quad) + [self._pseudo_quad_type]))
            converted.append(converted_sample)
        return converted

    def _remove_overlapping(self, entities, sentiments):
        non_overlapping_entities = []
        non_overlapping_sentiments = []

        for entity in entities:
            if not self._is_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        for rel in sentiments:
            # Check first two non-null entity spans for overlap
            real_spans = [e for e in rel[:4] if e[0] != 0 or e[1] != 1]
            overlap = False
            for idx_a in range(len(real_spans)):
                for idx_b in range(idx_a + 1, len(real_spans)):
                    if self._check_overlap(real_spans[idx_a], real_spans[idx_b]):
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                non_overlapping_sentiments.append(rel)

        return non_overlapping_entities, non_overlapping_sentiments

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1, e2):
        if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
            return False
        else:
            return True

    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        assert len(gt) == len(pred)

        def convert(t):
            if not include_entity_types:
                if type(t[0]) == int:  # entity: (start, end, type)
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # quintuple: (s, o, a, p, senti_type)
                    c = [
                        (t[j][0], t[j][1], self._pseudo_entity_type)
                        for j in range(4)
                    ] + [t[4]]
            else:
                if type(t[0]) == int:  # entity: (start, end, type[, score])
                    c = list(t[:3])
                else:  # quintuple: (s, o, a, p, senti_type[, score])
                    c = list(t[:5])

            if include_score and len(t) > len(c):
                c.append(t[-1])

            return tuple(c)

        converted_gt, converted_pred = [], []
        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred
    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                # Use last element as the type (works for entities AND quintuples)
                t = s[-1]
                if s in sample_gt:
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        # results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)
        # return results_str

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def store_predictions(self):
        predictions = []
        NULL_SPAN = (0, 1)

        for i, doc in enumerate(self._dataset.sentences):
            tokens = doc.tokens
            pred_entities = self._pred_entities[i]
            pred_sentiments = self._pred_sentiments[i]

            # convert entities
            converted_entities = []
            for entity in pred_entities:
                entity_span = entity[:2]
                span_tokens = util.get_span_tokens(tokens, entity_span)
                if not span_tokens:
                    continue
                entity_type = entity[2].identifier
                converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
                converted_entities.append(converted_entity)
            converted_entities = sorted(converted_entities, key=lambda e: e['start'])

            # convert quintuples  (s, o, a, p, senti_type [, score])
            converted_sentiments = []
            role_keys = ['s', 'o', 'a', 'p']
            for sentiment in pred_sentiments:
                senti_type = sentiment[4].identifier
                role_indices = []
                all_found = True
                for j, key in enumerate(role_keys):
                    espan = sentiment[j]
                    if (espan[0], espan[1]) == NULL_SPAN:
                        role_indices.append(None)
                        continue
                    span_tokens = util.get_span_tokens(tokens, espan[:2])
                    if not span_tokens:
                        role_indices.append(None)
                        continue
                    e_dict = dict(type=espan[2].identifier,
                                  start=span_tokens[0].index,
                                  end=span_tokens[-1].index + 1)
                    try:
                        role_indices.append(converted_entities.index(e_dict))
                    except ValueError:
                        # entity was not in the converted list (filtered above)
                        role_indices.append(None)

                converted_sentiment = dict(type=senti_type)
                for j, key in enumerate(role_keys):
                    converted_sentiment[key] = role_indices[j]
                converted_sentiments.append(converted_sentiment)

            doc_predictions = dict(tokens=[t.phrase for t in tokens],
                                   entities=converted_entities,
                                   sentiments=converted_sentiments)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)
        try:
            print(f"Saved predictions: {self._predictions_path % (label, epoch)}")
        except Exception:
            pass

    def store_examples(self):

        entity_examples = []
        senti_examples = []
        senti_examples_nec = []

        for i, doc in enumerate(self._dataset.sentences):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # sentiments
            # without entity types
            senti_example = self._convert_example(doc, self._gt_sentiments[i], self._pred_sentiments[i],
                                                include_entity_types=False, to_html=self._senti_to_html)
            senti_examples.append(senti_example)

            # with entity types
            senti_example_nec = self._convert_example(doc, self._gt_sentiments[i], self._pred_sentiments[i],
                                                    include_entity_types=True, to_html=self._senti_to_html)
            senti_examples_nec.append(senti_example_nec)

        label, epoch = self._dataset_label, self._epoch

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % ('entities', label, epoch),
                             template='entity_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('entities', label, epoch)}")
        except Exception:
            pass

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('entities_sorted', label, epoch),
                             template='entity_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('entities_sorted', label, epoch)}")
        except Exception:
            pass

        # sentiments
        # without entity types
        self._store_examples(senti_examples[:self._example_count],
                             file_path=self._examples_path % ('rel', label, epoch),
                             template='sentiment_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('rel', label, epoch)}")
        except Exception:
            pass

        self._store_examples(sorted(senti_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('senti_sorted', label, epoch),
                             template='sentiment_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('senti_sorted', label, epoch)}")
        except Exception:
            pass

        # with entity types
        self._store_examples(senti_examples_nec[:self._example_count],
                             file_path=self._examples_path % ('senti_nec', label, epoch),
                             template='sentiment_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('senti_nec', label, epoch)}")
        except Exception:
            pass

        self._store_examples(sorted(senti_examples_nec[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % ('senti_nec_sorted', label, epoch),
                             template='sentiment_examples.html')
        try:
            print(f"Saved examples: {self._examples_path % ('senti_nec_sorted', label, epoch)}")
        except Exception:
            pass

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)

    def _convert_example(self, sen: Sentence, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        encoding = sen.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            # strip score for _score call (last element)
            pred_s = [p[:-1] for p in pred]
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            # entity tuple: (start, end, entity_type)
            # quintuple tuple: (s, o, a, p, senti_type)
            type_verbose = s[2].verbose_name if type(s[0]) == int else s[4].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(sen.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _senti_to_html(self, sentiment: Tuple, encoding: List[int]):
        """Render a quintuple (s, o, a, p, senti_type) as an HTML snippet."""
        NULL_SPAN = (0, 1)
        ROLE_LABELS = ['subj', 'obj', 'aspect', 'pred']
        ROLE_CSS = ['head', 'tail', 'aspect', 'pred']

        # Sort non-null entities by their position in the text
        entity_slots = [
            (j, sentiment[j]) for j in range(4)
            if (sentiment[j][0], sentiment[j][1]) != NULL_SPAN
        ]
        entity_slots_sorted = sorted(entity_slots, key=lambda x: x[1][0])

        # Build annotated HTML by inserting span tags at entity boundaries
        html = self._text_encoder.decode(encoding)
        # Fallback: just list the entities
        parts = []
        for j, e in entity_slots_sorted:
            etext = self._text_encoder.decode(encoding[e[0]:e[1]])
            parts.append('<span class="%s"><span class="type">%s</span>%s</span>' %
                         (ROLE_CSS[j], ROLE_LABELS[j] + ':' + e[2].verbose_name, etext))
        html = ' … '.join(parts)
        return self._prettify(html)