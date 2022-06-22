import os
from os import path

import pandas as pd
from pyserini.search import SimpleSearcher
import json

import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from datasets import Dataset
from datasets import load_dataset
from datasets import load_metric


from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import EMBERT
from ebert.embeddings import load_embedding, MappedEmbedding

from datetime import datetime
from tqdm import trange


def linecounter(filename):
    cnt = 0
    with open(filename) as file:
        for line in file:
            cnt += 1
    return cnt


def entity_converter(word, reverse = False, nospace = True, concatenate = True):
    # Input: an entity string in dbpedia format
    # Output: string in wikipedia2vec format or readible format
    # Getting entities in the right format
    returnword = ''
    if reverse:
        word = word.replace("<dbpedia:", "")
        word = word.replace(">", "")
        if nospace:
            returnword = word
        else:
            word = word.replace("_", " ")
            returnword = word
    else:
        word = word.replace("<dbpedia:", "ENTITY/")
        word = word.replace(">", "")
        returnword = word
    if concatenate:
        returnword += returnword.replace("ENTITY/", " ")
    return returnword

for foldnr in range(1, 6):
    print(f"Ranking for Fold {foldnr}")
    print("Loading Pandas")
    #path_to_set = 'traintest/babysample.tsv'
    #path_to_set = 'traintest/smallsample.tsv'
    path_to_set = f'traintest/dbpedia_crossfolds/Fold{foldnr}_train.txt'
    LEN_TRAIN = linecounter(path_to_set)

    #path_to_eval = 'traintest/babysample.tsv'
    #path_to_eval = 'traintest/smalleval.tsv'
    path_to_eval = 'traintest/dbpediababyeval.tsv'
    LEN_EVAL = linecounter(path_to_eval)

    path_to_big_eval = 'traintest/dbpediasmallereval.tsv'
    LEN_BIG_EVAL = linecounter(path_to_big_eval)

    #output_file = 'traintest/smallvectors.tsv'

    index_path = 'indexes/lucene-index-dbpedia_annotated_full'
    #index_path = 'indexes/lucene-index-dbpedia_annotated_small'

    print("Loading searcher")
    searcher = SimpleSearcher(index_path)

    #queries_path = 'data/dbpedia-topics/queries_dbpedia.tsv'
    queries_eval_path = 'dbpedia/queries_dbpedia_annotated_rel.tsv'
    queries_train_path = 'data/DBpedia-Entity/queries_linked_rel.tsv'
    print(f"Going to train on {path_to_set} \n with index {index_path} and queries {queries_train_path} {queries_eval_path}")

    queries_eval = {}
    with open(queries_eval_path) as f:
        for line in f:
            splitted = line.split('\t')
            key = splitted[0]
            value = " ".join(splitted[1:])
            queries_eval[int(key)] = value
    
    queries_train = {}
    with open(queries_train_path) as f:
        for line in f:
            splitted = line.split('\t')
            key = splitted[0]
            value = " ".join(splitted[1:])
            queries_train[key] = value

    # Initializing all Bert stuff

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reranker =  EMBERT()
    model, model_emb = reranker.model
    wiki_emb, mapper = reranker.ebert
    


    # Function to map query-passage-label pairs to things which can be eaten by the trainer function
    def create_input(batch, train = False):
        inputs_embeds, tt_ids, labels = [], [], []
        for q in batch:
            #query, passage, label = q['query'], q['passage'], q['label']
            qid, pid, label = q['qid'], q['pid'], q['label']
            doc = searcher.doc(str(pid))
            try:
                #passage = doc.raw()
                passage = json.loads(doc.raw())['contents']
            except AttributeError:
                passage = entity_converter(pid)
            if train:
                query = queries_train[qid]
            else:
                query = queries_eval[int(qid)]
            #print(query, passage, label)
            ret = reranker.tokenize_with_entities(query, passage, pad = True)
            input_id = reranker.vectorize(ret['input_ids'], model_emb, mapper, wiki_emb)
            tt_id = ret['token_type_ids'].long()
            inputs_embeds.append(torch.squeeze(input_id).type(torch.float32))
            tt_ids.append(torch.squeeze(tt_id))
            labels.append(torch.tensor(label))
        #print(inputs_embeds)
        inputs_embeds = torch.stack(inputs_embeds).to(device)
        tt_ids = torch.stack(tt_ids).to(device)
        labels = torch.stack(labels).to(device)
        return({"inputs_embeds": inputs_embeds, "token_type_ids" : tt_ids, "labels" : labels})

    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 64
    accumulation_steps = 8

    #EVAL_EVERY = 512
    #EVAL_EVERY = 1024
    EVAL_EVERY = 128
    BIG_EVAL_EVERY = 16

    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-6)
    from transformers import get_linear_schedule_with_warmup
    num_warmup_steps = 4000
    num_train_steps = LEN_TRAIN*2
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    
    PATH = 'output/testmodel_acc_batch_600k_64_e6model.pt'
    if path.exists(PATH):
        checkpoint = torch.load(PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']  
        print("Succesfully loaded from checkpoint!")


    writer = SummaryWriter()
    counter = 0
    model.train()


    with open(path_to_set) as file:
        line = file.readline()
        batch = []
        #while line:
        for i in trange(LEN_TRAIN, desc="Training", unit="passages"):
            
            counter += 1
            qid, pid, label = line.split()
            label = int(label)
            if label > 0:
                label = 1
            tup = {'qid' : qid, 'pid' : pid, 'label' : label}
            batch.append(tup)
            line = file.readline()
            #print(counter)
            if len(batch) == BATCH_SIZE or not line:
                batch_tens = create_input(batch, train = True)
                outputs = model(**batch_tens)
                loss = outputs.loss
                loss = loss / accumulation_steps 
                writer.add_scalar("Loss/train", loss, counter)
                loss.backward()

                if (counter) % BATCH_SIZE*accumulation_steps == 0:

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                batch = []
                #print(datetime.now(), " now at file", counter)
            if counter % (EVAL_EVERY*BATCH_SIZE) == 0 or not line: 
                if counter % (BIG_EVAL_EVERY*EVAL_EVERY*BATCH_SIZE) == 0:
                    now_path_to_eval = path_to_big_eval
                    NOW_LEN_EVAL = LEN_BIG_EVAL
                    writername = "Accuracy/bigtest"
                else:
                    now_path_to_eval = path_to_eval
                    NOW_LEN_EVAL = LEN_EVAL
                    writername = "Accuracy/test"
                #print("Evaluating")
                metric = load_metric("accuracy")
                model.eval()
                with open(now_path_to_eval) as evalfile:
                    evalline = evalfile.readline()
                    evalbatch = []
                    for i in trange(NOW_LEN_EVAL, desc="Evaluating", unit="passages"):
                        qid, positive, negative = evalline.split()
                        tup1 = {'qid' : qid, 'pid' : positive, 'label' : 1}
                        tup2 = {'qid' : qid, 'pid' : negative, 'label' : 0}
                        evalbatch += [tup1, tup2]
                        evalline = evalfile.readline()
                        if len(evalbatch) == EVAL_BATCH_SIZE or not evalline:
                            #print(datetime.now(), "create batch")
                            batch_tens = create_input(evalbatch)
                            #print(datetime.now(),"Put in model")
                            with torch.no_grad():
                                outputs = model(**batch_tens)
                            logits = outputs.logits
                            #print(datetime.now(),"Predict")
                            #print(logits)
                            #print("Logits", logits)
                            predictions = torch.argmax(logits, dim=1)
                            #print("Pred", predictions)
                            #print(list(zip(predictions, batch_tens["labels"])))
                            metric.add_batch(predictions=predictions, references=batch_tens["labels"])
                            evalbatch = []
                    #print(metric)
                    score = metric.compute()
                    #print(score)
                    score = score['accuracy']
                    #print(score)
                    writer.add_scalar(writername, score, counter)
                    writer.flush()
                model.train()

    writer.flush()
    writer.close()

    savepath = f"output/dbpedia_acc_batch_64_e6_fold{foldnr}_annotated"
    print(f"Saved model to {savepath}")
    PATH = savepath+"model.pt"

    model.save_pretrained(savepath)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)