from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset
from typing import List
from trainer import sampling


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid
        self._index = index
        self._span_start = span_start
        self._span_end = span_end
        self._phrase = phrase

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def phrase(self):
        return self._phrase

    @property
    def index(self):
        return self._index

    @property
    def span(self):
        return self._span_start, self._span_end

class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset
        self._entity_type = entity_type
        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __repr__(self):
        return self._phrase

    def __str__(self):
        return self._phrase

class sentimentType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def symmetric(self):
        return self._symmetric

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name
_NULL_ENTITY_TYPE = EntityType('None', 0, 'None', 'No Entity')

# Null span in encoding space: position 0 = [CLS] token
NULL_SPAN = (0, 1)


def _entity_as_span_tuple(entity):
    """Return (span_start, span_end, entity_type); uses NULL_SPAN for absent entities."""
    if entity is None:
        return (NULL_SPAN[0], NULL_SPAN[1], _NULL_ENTITY_TYPE)
    return (entity.span_start, entity.span_end, entity.entity_type)


class Sentiment:
    """Represents a COQE comparative quintuple: (subject, object, aspect, predicate, label)."""

    def __init__(self, rid: int, sentiment_type: sentimentType,
                 s_entity,   # Subject   – may be None
                 o_entity,   # Object    – may be None
                 a_entity,   # Aspect    – may be None
                 p_entity):  # Predicate – may be None
        self._rid = rid
        self._sentiment_type = sentiment_type
        self._s_entity = s_entity
        self._o_entity = o_entity
        self._a_entity = a_entity
        self._p_entity = p_entity

    def as_tuple(self):
        return (
            _entity_as_span_tuple(self._s_entity),
            _entity_as_span_tuple(self._o_entity),
            _entity_as_span_tuple(self._a_entity),
            _entity_as_span_tuple(self._p_entity),
            self._sentiment_type,
        )

    @property
    def sentiment_type(self):
        return self._sentiment_type

    @property
    def s_entity(self):
        return self._s_entity

    @property
    def o_entity(self):
        return self._o_entity

    @property
    def a_entity(self):
        return self._a_entity

    @property
    def p_entity(self):
        return self._p_entity

class Sentence:
    def __init__(self, sen_id: int, tokens: List[Token], entities: List[Entity], sentiments: List[Sentiment],
                 encoding: List[int], adj, is_comparative: bool = None):
        self._sen_id = sen_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities
        self._sentiments = sentiments

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding
        self._adj = adj
        self._is_comparative = bool(len(sentiments)) if is_comparative is None else bool(is_comparative)

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @property
    def sen_id(self):
        return self._sen_id

    @property
    def entities(self):
        return self._entities

    @property
    def sentiments(self):
        return self._sentiments

    @property
    def adj(self):
        return self._adj

    @property
    def is_comparative(self):
        return self._is_comparative


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'
    def __init__(self, label, senti_types, entity_types, neg_entity_count,
                 neg_senti_count, max_span_size):
        self._label = label
        self._senti_types = senti_types
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._neg_senti_count = neg_senti_count
        self._max_span_size = max_span_size
        self._mode = Dataset.TRAIN_MODE

        self._sentences = OrderedDict()
        self._entities = OrderedDict()
        self._sentiments = OrderedDict()
        # current ids
        self._sen_id = 0
        self._rid = 0
        self._eid = 0
        self._tid = 0

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def create_sentiment(self, sentiment_type, s_entity, o_entity, a_entity, p_entity) -> Sentiment:
        sentiment = Sentiment(self._rid, sentiment_type, s_entity, o_entity, a_entity, p_entity)
        self._sentiments[self._rid] = sentiment
        self._rid += 1
        return sentiment

    def create_sentence(self, tokens, entity_mentions, sentiments, sen_encoding, adj, is_comparative=None) -> Sentence:
        sentence = Sentence(self._sen_id, tokens, entity_mentions, sentiments, sen_encoding, adj, is_comparative)
        self._sentences[self._sen_id] = sentence
        self._sen_id += 1
        # print(self._sen_id,len(self._sentences))
        return sentence

    def __len__(self):
        return len(self._sentences)

    @property
    def sentences(self):
        return list(self._sentences.values())

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def sentence_count(self):
        return len(self._sentences)

    @property
    def entity_count(self):
        return len(self._entities)

    @property
    def sentiment_count(self):
        return len(self._sentiments)

    def __getitem__(self, index: int):
        sen = self._sentences[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.train_create_sample(sen, self._neg_entity_count, self._neg_senti_count,
                                                self._max_span_size, len(self._senti_types))
        else:
            return sampling.create_eval_sample(sen, self._max_span_size)

