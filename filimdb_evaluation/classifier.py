from typing import List, Any
from random import random

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def count_labels(labels: List):
    return {
        unique_label: sum(1 for label in labels if label == unique_label)
        for unique_label in set(labels)
    }

def make_texts_better(texts):
    for i in range(0, len(texts)):
        texts[i] = texts[i].strip().lower()
       # texts[i] = ' '.join(texts[i])                    # строчка для отладки функции
        texts[i] = ' '.join(texts[i].split('<br />'))
        texts[i] = ' , '.join(texts[i].split(','))
        texts[i] = ' . '.join(texts[i].split('.'))
        texts[i] = ' " '.join(texts[i].split('"'))
        texts[i] = ' : '.join(texts[i].split(':'))
        texts[i] = ' ; '.join(texts[i].split(';'))
        texts[i] = ' ! '.join(texts[i].split('!'))
        texts[i] = ' ( '.join(texts[i].split('('))
        texts[i] = ' ) '.join(texts[i].split(')'))
        texts[i] = ' / '.join(texts[i].split('/'))
        texts[i] = ' - '.join(texts[i].split('-'))

def make_tokenized(texts):
    for i in range(0, len(texts)):
        texts[i] = texts[i].split(' ')
        texts[i] = list(filter(None, texts[i]))
        
def shuffle_data(docs):
    total_texts = docs.size
    idxs = np.arange(total_texts)
    np.random.shuffle(idxs)
    
    for i in range(0, total_texts):
        np.random.shuffle(docs[i])
        
    docs_ids = np.empty(total_texts, dtype=np.ndarray)
    sh_docs = docs[idxs] # возьмем документы в порядке, заданном индексами в idxs
    for i in range(0, total_texts): # idxs[i] - настоящий индекс текущего документа
        docs_ids[i] = np.ones(sh_docs[i].size, dtype=int) * idxs[i]

    sh_docs = np.concatenate(sh_docs) # список индексов нграмм всех документов в одномерном массиве
    docs_ids = np.concatenate(docs_ids) # список индексов документов
    
    return sh_docs, docs_ids

def batch_generator(grams, probs, docs_ids, neg_samples, nb = 2, batch_size = 100):
    cur_batch = 0
    grams_sz = grams.size
    neg_sz = grams_sz * nb
    
    sh_idxs = np.arange(grams_sz)
    np.random.shuffle(sh_idxs)
    
    for i in range(0, grams_sz, batch_size):
        sz = min(batch_size, grams_sz - i)
        idxs = sh_idxs[i : i + sz]
        yield grams[idxs], docs_ids[idxs], np.ones(sz)   # заменить ли на 1 вместо массива единиц, и 0 соотв-но
        for k in range(nb):
            yield neg_samples[cur_batch : cur_batch + sz], docs_ids[idxs], np.zeros(sz) # nb раз создает негативные батчи
            cur_batch += sz

def sigmoid(x):
    sgm = 1. / (1 + np.exp(-x))
    n = (sgm <= 0.0).astype(int) * 0.00000001
    sgm += n
    n = (sgm >= 1.0).astype(int) * 0.00000001
    sgm -= n
    return sgm


class Doc2Vec:
    def __init__(self, vocab_size, docs_size, embed_size=500):
        self.word_embs = np.random.uniform(-0.001, 0.001, (vocab_size, embed_size))
        self.doc_embs = np.random.uniform(-0.001, 0.001, (docs_size, embed_size))
    
    def train(self, words_idxs, docs_idxs, labels, eta=0.1):
                    
        words_batch = self.word_embs[words_idxs]
        docs_batch = self.doc_embs[docs_idxs]
        
        scal_pr = np.sum(words_batch * docs_batch, axis=1)
        sigm = sigmoid(scal_pr)
       # loss = np.sum(- labels * np.log(sigm) - (1 - labels) * np.log(1 - sigm))
        
        koefs = np.diagflat(- labels + sigm)
        grad_w = np.dot(koefs, docs_batch)
        grad_d = np.dot(koefs, words_batch)
        
        words_batch -= eta * grad_w
        docs_batch -= eta * grad_d
        
        self.word_embs[words_idxs] = words_batch
        self.doc_embs[docs_idxs] = docs_batch
        
        #return loss

    def classify(self, words_idxs, docs_idxs):
        words_batch = self.word_embs[words_idxs]
        docs_batch = self.doc_embs[docs_idxs]
        
        scal_pr = np.sum(words_batch * docs_batch, axis=1)
        sigm = sigmoid(scal_pr)
        labels = (sigm >= 0.5).astype(int)
             
        return labels    




def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    doc2vec = pretrain_params
    
    train_labels = np.array(train_labels)
    train_labels = (train_labels == 'pos').astype(int)
    
    classifier = LogisticRegression(solver='lbfgs', random_state=0)
    classifier.fit(doc2vec.doc_embs[0:train_labels.size], train_labels)
    
    return doc2vec, classifier


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    all_texts = texts_list
    total_texts = np.sum(len(text) for text in all_texts)
    
    for texts in all_texts:
        make_texts_better(texts)
    for texts in all_texts:
        make_tokenized(texts)
       
    ngrams = {}
    for texts in all_texts:
        for text in texts:
            # униграммы
            for ugram in text:
                if not (ugram in ngrams):
                    ngrams[ugram] = 1
                else:
                    ngrams[ugram] += 1                  
    # биграммы
    for texts in all_texts:
        for text in texts:
            for j in range(1, len(text)):
                tupl = (text[j - 1], text[j])
                if not (tupl in ngrams):
                    ngrams[tupl] = 1
                else:
                    ngrams[tupl] += 1
    # триграммы
    for texts in all_texts:
        for text in texts:
            for j in range(2, len(text)):
                tupl = (text[j - 2], text[j - 1], text[j])
                if not (tupl in ngrams):
                    ngrams[tupl] = 1
                else:
                    ngrams[tupl] += 1

    words = list(ngrams.keys())
    for key in words:
         if ngrams[key] <= 4:
            del ngrams[key]
            
    sorted_grams = sorted(ngrams, key=lambda x: int(ngrams[x]), reverse=True) 
    for item in sorted_grams[:27]:
        del ngrams[item]
    
    voc = np.empty(len(ngrams), dtype=object)
    probs = np.empty(len(ngrams), dtype=float)
    n = 0
    for key in ngrams.keys():
        voc[n] = key
        probs[n] = ngrams[key]
        ngrams[key] = n
        n += 1
        
    probs = probs ** 0.75
    sum = np.sum(probs)
    probs = probs / sum
    
    num_texts = []
    for texts in all_texts:
        num_texts.append(len(texts))
    for i in range(1, len(all_texts)):
        num_texts[i] += num_texts[i - 1]
    one_all_texts = []
    for texts in all_texts:
        one_all_texts.extend(texts)
    
    docs = np.empty(total_texts, dtype=np.ndarray)
    for i in range(0, total_texts):
        docgrams = []
        text = one_all_texts[i]
        for word in text:
            if word in ngrams:
                docgrams.append(ngrams[word])
        for j in range(1, len(text)):
            tupl = (text[j - 1], text[j])
            if tupl in ngrams:
                docgrams.append(ngrams[tupl])
        for j in range(2, len(text)):
            tupl = (text[j - 2], text[j - 1], text[j])
            if tuple in ngrams:
                docgrams.append(ngrams[tupl])
        docs[i] = np.array(docgrams, dtype=int)
        
    doc2vec = Doc2Vec(probs.size, total_texts)
    max_epoch = 9
    for epoch in range(max_epoch):
        print('epoch', epoch+1)
        nb = 1
        sh_docs, docs_ids = shuffle_data(docs)
        neg_samples = np.random.choice(len(probs), size=sh_docs.size*nb, p=probs)
       
        batch_gen = batch_generator(sh_docs, probs, docs_ids, neg_samples, nb)
        cnt = 0
        for batch in batch_gen:    
            doc2vec.train(batch[0], batch[1], batch[2], eta=0.05)
    
    return doc2vec


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    doc2vec, classifier = params
    texts_sz = [0, 15000, 10000, 25000, 2000, 8599]
    idx = texts_sz.index(len(texts))
    for i in range(1, 6):
        texts_sz[i] += texts_sz[i - 1]

    preds = classifier.predict(doc2vec.doc_embs[texts_sz[idx - 1] : texts_sz[idx]])
    preds = preds.astype(object)
    for i in range(preds.size):
        if preds[i] == 1:
            preds[i] = 'pos'
        else:
            preds[i] = 'neg'
    
    return preds