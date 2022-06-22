from collections import defaultdict
from copy import deepcopy
from itertools import permutations
from typing import List
import json
from html import unescape
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration, 
                          BertForSequenceClassification)

import torch
from .base import Reranker, Query, Text
from .similarity import SimilarityMatrixProvider
from pygaggle.model import (BatchTokenizer,
                            LongBatchEncoder,
                            QueryDocumentBatch,
                            DuoQueryDocumentBatch,
                            QueryDocumentBatchTokenizer,
                            SpecialTokensCleaner,
                            T5BatchTokenizer,
                            T5DuoBatchTokenizer,
                            greedy_decode)

from ebert.emb_input_transformers import EmbInputBertForSequenceClassification, EmbInputBertModel
from ebert.embeddings import load_embedding, MappedEmbedding
from ebert.mappers import load_mapper


__all__ = ['MonoT5',
           'DuoT5',
           'UnsupervisedTransformerReranker',
           'MonoBERT',
           'QuestionAnsweringTransformerReranker']


class MonoT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monot5-base-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 't5-base',
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            input_ids = batch.output['input_ids'].to(self.device)
            attn_mask = batch.output['attention_mask'].to(self.device)
            _, batch_scores = greedy_decode(self.model,
                                            input_ids,
                                            length=1,
                                            attention_mask=attn_mask,
                                            return_last_logits=True)

            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = batch_scores[:, [6136, 1176]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score
        return texts


class DuoT5(Reranker):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/duot5-base-msmarco',
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 't5-base',
                      *args, batch_size: int = 8, **kwargs) -> T5DuoBatchTokenizer:
        return T5DuoBatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        doc_pairs = list(permutations(texts, 2))
        scores = defaultdict(float)
        batch_input = DuoQueryDocumentBatch(query=query, doc_pairs=doc_pairs)
        for batch in self.tokenizer.traverse_duo_query_document(batch_input):
            input_ids = batch.output['input_ids'].to(self.device)
            attn_mask = batch.output['attention_mask'].to(self.device)
            _, batch_scores = greedy_decode(self.model,
                                            input_ids,
                                            length=1,
                                            attention_mask=attn_mask,
                                            return_last_logits=True)

            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = batch_scores[:, [6136, 1176]]
            batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
            batch_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.doc_pairs, batch_probs):
                scores[doc[0].metadata['docid']] += score
                scores[doc[1].metadata['docid']] += (1 - score)

        for text in texts:
            text.score = scores[text.metadata['docid']]
        return texts


class UnsupervisedTransformerReranker(Reranker):
    methods = dict(max=lambda x: x.max().item(),
                   mean=lambda x: x.mean().item(),
                   absmean=lambda x: x.abs().mean().item(),
                   absmax=lambda x: x.abs().max().item())

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: BatchTokenizer,
                 sim_matrix_provider: SimilarityMatrixProvider,
                 method: str = 'max',
                 clean_special: bool = True,
                 argmax_only: bool = False):
        assert method in self.methods, 'inappropriate scoring method'
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = LongBatchEncoder(model, tokenizer)
        self.sim_matrix_provider = sim_matrix_provider
        self.method = method
        self.clean_special = clean_special
        self.cleaner = SpecialTokensCleaner(tokenizer.tokenizer)
        self.device = next(self.model.parameters(), None).device
        self.argmax_only = argmax_only

    @torch.no_grad()
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        encoded_query = self.encoder.encode_single(query)
        encoded_documents = self.encoder.encode(texts)
        texts = deepcopy(texts)
        max_score = None
        for enc_doc, text in zip(encoded_documents, texts):
            if self.clean_special:
                enc_doc = self.cleaner.clean(enc_doc)
            matrix = self.sim_matrix_provider.compute_matrix(encoded_query,
                                                             enc_doc)
            score = self.methods[self.method](matrix) if matrix.size(1) > 0 \
                else -10000
            text.score = score
            max_score = score if max_score is None else max(max_score, score)
        if self.argmax_only:
            for text in texts:
                if text.score != max_score:
                    text.score = max_score - 10000
        return texts


class MonoBERT(Reranker):
    def __init__(self,
                 model: PreTrainedModel = None,
                 tokenizer: PreTrainedTokenizer = None):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model[0].parameters(), None).device
        self.ebert = self.get_ebert()
        self.ent_preamb = ['ENTITY/', 'ENTITIY/']


    @staticmethod
    def get_model(pretrained_model_name_or_path: # str = 'models/bert-large-uncased',
                  str =  'models/monobert-large-msmarco',
                  *args, device: str = None, **kwargs) -> BertForSequenceClassification:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        print("Loading regular BERT)")
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, *args, **kwargs).to(device).eval()
        print("Loading EBERT embeddings")
        embeddings = load_embedding(pretrained_model_name_or_path)
        return (model ,embeddings )

    @staticmethod
    def get_ebert():
        # wiki_emb = load_embedding("resources/wikipedia2vec.pkl")
        # mapper = load_mapper("mappers/wikipedia2vec-base-cased.monobert-base-cased.linear.npy")
        wiki_emb_path = "resources/wikipedia2vec/wikipedia-20190701/wikipedia2vec_500.pkl"
        mapper_path = "mappers/wikipedia2vec-500-cased.monobert-base-cased.linear.npy"
        # wiki_emb_path = "/scratch/gerritse/pygaggle/resources/wikipedia2vec/wikipedia-20190701/wikipedia2vec_500.pkl"
        # mapper_path = "/scratch/gerritse/pygaggle/mappers/wikipedia2vec-500-cased.bert-large-uncased.linear"
        print("Loading ", wiki_emb_path, " and ", mapper_path)
        wiki_emb = load_embedding(wiki_emb_path)
        mapper = load_mapper(mapper_path)
        return (wiki_emb, mapper)

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 'bert-large-uncased',
                      *args, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    def ebert_tokenizer(self, words, label, tokenized, concat):
        try:
            words = json.loads(words)['contents']
        except (ValueError, TypeError) as e:
            words = words
        words = unescape(words)
        # print(words)
        words = words.split()
        for word in words:
            if self.ent_preamb[0] in word or self.ent_preamb[1] in word:
                tokenized['token_type_ids'] += 2 * [label]
                tokenized['input_ids'] += ['/', word]
                if concat:
                    #other_tokens = ['/'] + self.tokenizer.tokenize(word[len(self.ent_preamb):])
                    other_tokens = self.tokenizer.tokenize(word[len(self.ent_preamb):])
                    token_count = len(other_tokens)
                    tokenized['token_type_ids'] += token_count * [label]
                    tokenized['input_ids'] += other_tokens
            else:
                other_tokens = self.tokenizer.tokenize(word)
                token_count = len(other_tokens)
                tokenized['token_type_ids'] += token_count * [label]
                tokenized['input_ids'] += other_tokens
        return tokenized

    def tokenize_with_entities(self, query, passage, concat = False, pad = False,  max_length = 512):
        # Tokenizes words while keeping entities as entities. If concat = true, it splits the entity into tokens
        # Example: [CLS] The ENTITY/Eiffel_Tower is located in [MASK] becomes with concat = True
        # ['[CLS]', 'the', 'ENTITY/Eiffel_Tower', '/', 'e', '##iff', '##el', '_', 'tower', 'is', 'located', 'in', '[MASK]', '.', '[SEP]']
        # And with concat = False: ['[CLS]', 'the', 'ENTITY/Eiffel_Tower', 'is', 'located', 'in', '[MASK]', '.', '[SEP]']
        tokenized = {}
        tokenized['input_ids'] = ['[CLS]']
        tokenized['token_type_ids'] = [0]

        tokenized = self.ebert_tokenizer(query, 0, tokenized, concat)

        tokenized['input_ids'] += ['[SEP]']
        tokenized['token_type_ids'] += [0]

        tokenized = self.ebert_tokenizer(passage, 1, tokenized, concat)

        tokenized['input_ids'] = tokenized['input_ids'][0:max_length-1]
        tokenized['token_type_ids'] = tokenized['token_type_ids'][0:max_length-1]
        tokenized['input_ids'] += ['[SEP]']
        tokenized['token_type_ids'] += [1]
        if pad:
            left = max_length - len(tokenized['input_ids'])
            tokenized['token_type_ids'] += left * [1]
            tokenized['input_ids'] += left * ['[PAD]']
        tokenized['token_type_ids']  = torch.tensor([tokenized['token_type_ids']])
        return(tokenized)

    def vectorize(self, tokens, model_emb, mapper, wiki_emb):
        # Looks up the vectors for the tokens. If a word is an Entity, it uses the EBERT Mapper, otherwise, it just uses Bert
        vectors = []
        # print("Vectorizing tokens", tokens)
        for tok in tokens:
            if self.ent_preamb[0] in tok or self.ent_preamb[1] in tok:
                if tok in wiki_emb:
                    vectors.append(mapper.apply(wiki_emb[tok]))
                else:
                    #print(f"{tok} not embedded in wikipedia2vec, added _ instead")
                    vectors.append(model_emb['_'])
            else:
                vectors.append(model_emb[tok])
        return torch.tensor([vectors])

    @torch.no_grad()
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        model, model_emb = self.model
        wiki_emb, mapper = self.ebert
        for text in texts:
            # ret = self.tokenizer.encode_plus(query.text,
            #                                  text.text,
            #                                  max_length=512,
            #                                  truncation=True,
            #                                  return_token_type_ids=True,
            #                                  return_tensors='pt')
            # print("Gonna tokenize", text.text)
            ret = self.tokenize_with_entities(query.text, text.text)
	   
            #print(ret)
            input_ids = self.vectorize(ret['input_ids'], model_emb, mapper, wiki_emb).to(self.device)
            tt_ids = ret['token_type_ids'].to(self.device)
            #tmp = model(input_ids, token_type_ids=tt_ids.long()
            #print(tt_ids.shape)
            #print(input_ids.shape)
            output, = model(inputs_embeds = input_ids.float(), token_type_ids=tt_ids.long(), return_dict = False)
            #print(output)
            #prediction_logits = output.logits

            # print(tmp)
            # tmp = tmp[0]
            # print("shape of tmp", tmp.shape)
            # output = language_model(inputs_embeds = tmp, token_type_ids=tt_ids.long())
            # output = output[0]
            
            if output.size(1) > 1:
                text.score = torch.nn.functional.log_softmax(output, 1)[0, -1].item()
            else:
                text.score = output.item()
        return texts


class QuestionAnsweringTransformerReranker(Reranker):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            ret = self.tokenizer.encode_plus(query.text,
                                             text.text,
                                             max_length=512,
                                             truncation=True,
                                             return_tensors='pt',
                                             return_token_type_ids=True)
            input_ids = ret['input_ids'].to(self.device)
            tt_ids = ret['token_type_ids'].to(self.device)
            start_scores, end_scores = self.model(input_ids,
                                                  token_type_ids=tt_ids,
                                                  return_dict=False)
            start_scores = start_scores[0]
            end_scores = end_scores[0]
            start_scores[(1 - tt_ids[0]).bool()] = -5000
            end_scores[(1 - tt_ids[0]).bool()] = -5000
            smax_val, smax_idx = start_scores.max(0)
            emax_val, emax_idx = end_scores.max(0)
            text.score = max(smax_val.item(), emax_val.item())
        return texts
