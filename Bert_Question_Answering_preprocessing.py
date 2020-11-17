#!/usr/bin/env python
# coding: utf-8

# https://medium.com/@eyfydsyd97/bert-for-question-answer-fine-tuning%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-by-pytorch-fbe15fdef330
# 
# https://github.com/younggeun-kim/BERT_Question_Answering_PyTorch
# 
# https://github.com/kushalj001/pytorch-question-answering/blob/master/1.%20DrQA.ipynb

# In[1]:


import torch
import numpy as np
import pandas as pd
import pickle
import re, os, string, typing, gc, json
import spacy
from collections import Counter
nlp = spacy.load('en')


# In[2]:


def load_json(path):
    '''
    Loads the JSON file of the Squad dataset.
    Returns the json object of the dataset.
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("Length of data: ", len(data['data']))
    print("Data Keys: ", data['data'][0].keys())
    print("Title: ", data['data'][0]['title'])
    
    return data


# ![image.png](attachment:image.png)

# In[1]:


def parse_data(data:dict) -> list:
    """
    Squad 데이터 세트의 JSON 파일을
    key와 values을 입력하고 dictionry를 반환합니다.
    컨텍스트, query 및 레이블 트리플렛이 각 사전의 키가됩니다.
    """
    
    data = data['data']
    qa_list = []

    for paragraphs in data:

        for para in paragraphs['paragraphs']:
            context = para['context']

            for qa in para['qas']:
                
                id = qa['id']
                question = qa['question']
                
                for ans in qa['answers']:
                    answer = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(answer)
                    
                    qa_dict = {}
                    qa_dict['id'] = id
                    qa_dict['context'] = context
                    qa_dict['question'] = question
                    qa_dict['label'] = [ans_start, ans_end]

                    qa_dict['answer'] = answer
                    qa_list.append(qa_dict)    

    
    return qa_list


# In[4]:


def filter_large_examples(df):
    """
    임의의 값 이상의 컨텍스트 길이, 쿼리 길이 및 answer 길이가있는 예제의 ID를 반환합니다.
    그런 다음 이러한 ID를 데이터 프레임에서 삭제할 수 있습니다.
    이것은 QANet에서 명시 적으로 언급되지만 다른 모델에서도 가능합니다.
    """

    ctx_lens = []
    query_lens = []
    ans_lens = []
    for index, row in df.iterrows():
        ctx_tokens = [w.text for w in nlp(row.context, disable=['parser','ner','tagger'])]
        if len(ctx_tokens)>400:
            ctx_lens.append(row.name)

        query_tokens = [w.text for w in nlp(row.question, disable=['parser','tagger','ner'])]
        if len(query_tokens)>50:
            query_lens.append(row.name)

        ans_tokens = [w.text for w in nlp(row.answer, disable=['parser','tagger','ner'])]
        if len(ans_tokens)>30:
            ans_lens.append(row.name)

        assert row.name == index
    
    return set(ans_lens + ctx_lens + query_lens)


# In[5]:


def gather_text_for_vocab(dfs:list):
    """
    문맥과 질문에서 텍스트를 수집하여 어휘를 만듭니다.
    
    : param dfs : SQUAD 데이터 세트의 데이터 프레임 목록.
    : returns : 컨텍스트 및 질문 목록
    """
    
    text = []
    total = 0
    for df in dfs:
        unique_contexts = list(df.context.unique())
        unique_questions = list(df.question.unique())
        total += df.context.nunique() + df.question.nunique()
        text.extend(unique_contexts + unique_questions)
    
    assert len(text) == total
    
    return text


# In[6]:


def build_word_vocab(vocab_text):
    """
    주어진 텍스트에서 word-level의 어휘를 만듭니다.
    
    : param list vocab_text : 컨텍스트 및 질문 목록
    : return
        dict word2idx : 단어를 인덱스로 매핑하는 단어
        dict idx2word : 정수 대 단어 매핑
        list word_vocab : 빈도별로 정렬 된 단어 목록
    """
    
    words = []
    for sent in vocab_text:
        for word in nlp(sent, disable=['parser','tagger','ner']):
            words.append(word.text)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    #word_vocab = list(set(word_vocab).intersection(set(glove_words)))
    print(f"glove-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word:idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v:k for k,v in word2idx.items()}
    
    
    return word2idx, idx2word, word_vocab


# In[7]:


def build_char_vocab(vocab_text):
    """
    주어진 텍스트에서 char-level의 어휘를 만듭니다.
    
    : param list vocab_text : 컨텍스트 및 질문 목록
    : return
        dict char2idx : 문자와 단어의 인덱스 매핑
        list char_vocab : 빈도별로 정렬 된 문자 목록
    """
    
    chars = []
    for sent in vocab_text:
        for ch in sent:
            chars.append(ch)
            
    char_counter = Counter(chars)
    char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
    print(f'raw-char-vocab: {len(char_vocab)}')
    high_freq_char = [char for char, count in char_counter.items() if count>=20]
    char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
    print(f'char-vocab-intersect: {len(char_vocab)}')
    char_vocab.insert(0, '<unk>')
    char_vocab.insert(1, '<pad>')
    char2idx = {char:idx for idx, char in enumerate(char_vocab)}
    print(f'char2idx-length: {len(char2idx)}')
    
    return char2idx, char_vocab


# In[8]:


def context_to_ids(text, word2idx):
    """
    각 단어를 매핑하여 컨텍스트 텍스트를 해당 ID로 변환합니다.
    word2idx 사용. 입력 텍스트는 먼저 spacy 토크 나이저를 사용하여 토큰 화됩니다.
    
    : param str text : 변환 할 컨텍스트 텍스트
    : return list context_ids : 매핑 된 ID 목록
    
    : raises assertion error : 상태 체크
    """
    
    context_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    context_ids = [word2idx[word] for word in context_tokens]
    
    assert len(context_ids) == len(context_tokens)
    return context_ids

def question_to_ids(text, word2idx):
    """
    각 단어를 매핑하여 질문 텍스트를 해당 ID로 변환합니다.
    """
    
    question_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    question_ids = [word2idx[word] for word in question_tokens]
    
    assert len(question_ids) == len(question_tokens)
    return question_ids


# In[9]:


def test_indices(df, idx2word):
    """
    위에서 언급 한 테스트를 수행합니다. 
    이 방법은 각 예제의 context_ids와 관련된 답변의 시작과 끝을 가져옵니다.
    
    : param 데이터 프레임 df : SQUAD df
    : returns
        list start_value_error : example idx 텍스트에서 시작 idx를 찾을 수 없을 때 
        list end_value_error : example idx 텍스트에서 종료 idx를 찾을 수 없을 때 
        list assert_error : 어설 션 오류에 실패한 예. 대다수는 위의 오류로 인한 것입니다.
    """
    
    start_value_error = []
    end_value_error = []
    assert_error = []
    for index, row in df.iterrows():

        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser','tagger','ner'])]

        start_token = answer_tokens[0]
        end_token = answer_tokens[-1]
        
        context_span  = [(word.idx, word.idx + len(word.text)) 
                         for word in nlp(row['context'], disable=['parser','tagger','ner'])]

        starts, ends = zip(*context_span)

        answer_start, answer_end = row['label']

        try:
            start_idx = starts.index(answer_start)
        except:
            start_value_error.append(index)
        try:
            end_idx  = ends.index(answer_end)
        except:
            end_value_error.append(index)

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except:
            assert_error.append(index)


    return start_value_error, end_value_error, assert_error


# In[10]:


def get_error_indices(df, idx2word):
    """
    test_indices함수에서 오류를 가져와 에러 반환
    """
    
    start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    err_idx = set(err_idx)
    print(f"Error indices: {len(err_idx)}")
    
    return err_idx


# In[11]:


def index_answer(row, idx2word):
    """
    데이터 프레임의 행 또는 하나의 training example를 가져와 계산하여
    answer의 시작 및 끝 위치의 튜플을 반환합니다.
    """
    
    context_span = [(word.idx, word.idx + len(word.text)) for word in nlp(row.context, disable=['parser','tagger','ner'])]
    starts, ends = zip(*context_span)
    
    answer_start, answer_end = row.label
    start_idx = starts.index(answer_start)
 
    end_idx  = ends.index(answer_end)
    
    ans_toks = [w.text for w in nlp(row.answer,disable=['parser','tagger','ner'])]
    ans_start = ans_toks[0]
    ans_end = ans_toks[-1]
    assert idx2word[row.context_ids[start_idx]] == ans_start
    assert idx2word[row.context_ids[end_idx]] == ans_end
    
    return [start_idx, end_idx]

