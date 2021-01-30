#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import pandas as pd
import gensim
from gensim.models import KeyedVectors
import tensorflow
import keras
import pickle
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Bidirectional, LSTM, Masking, Dense, Input, TimeDistributed, Activation, Lambda, Dropout
from keras.optimizers import RMSprop
#from keras_contrib.layers import CRF
#from keras_contrib.losses import crf_loss
#from keras_contrib.metrics import crf_viterbi_accuracy
from keras import backend as K
from tqdm import tqdm
from sklearn.metrics import f1_score
import re


# In[ ]:


keras.__version__


# In[ ]:


# 超参量
SAVE_PATH = 'config.pkl'
RNN_UNITS = 300
BATCH_SIZE = 32
EMBED_DIM = 300
EPOCHS = 20


# In[4]:


# train_set = open('msrseg/msr_training.utf8', 'r', encoding='utf-8')
train_set = pd.read_csv('msrseg/msr_training.utf8', encoding= 'utf8', header=None)  # 不把第一行作为列属性，且pd读出来就是数据帧，就是字符串
test_set = pd.read_csv('msrseg/msr_test_gold.utf8', encoding='utf8', header=None)
print(train_set.head())
print(test_set.head())
#train_set = train_set.values  #转成二维的nparray[[s1],[s2],]


# In[ ]:


# 将句子转换成词序列
def get_word(sentence):
    word_list = []
    sentence = re.sub("[+\.\!\/_,$%^*(+\“\”\‘\’]+|[+——！，。？、；：《》【】~@#￥%……&*（）]", "",sentence)  # 去掉所有除空格外的标点符号
    sentence = sentence.split() #去掉空格
    # sentence.append('<EOS>')  # 加入结束符<EOS>
    return sentence


# In[ ]:


def read_file(file):
    word, content = [], []
    maxlen = 0

    for i in range(len(file)):  # 记得加range！！
        line = file.loc[i,0]   # 用loc来访问dataframe
        line = line.strip('\n') #去掉换行符
        line = line.strip(' ')  #去掉开头和结尾的空格
        
        word_list = get_word(line)        #获得字列表：去掉标点，（不添加<EOS>结束符
        
        maxlen = max(maxlen, len(word_list))
        word.extend(word_list)            #每一个单元是1个词，且不加<EOS>符号
        content.append(word_list)         # 每一个单元是一行里面的各个词（分好）
    return word, content, maxlen  #word是单列表，content和label是双层列表


# In[ ]:


# process data: padding
def process_data(word_list, vocab, MAXLEN):
    # vocab to idx dictionary:
    vocab2idx = {word: idx for idx, word in enumerate(vocab)}
    # x: get every idx of every word, map to idx in vocab, set to <EOS> if not in vocab(<EOS> not included in vocab)
    x = [[vocab2idx.get(word, 1) for word in s] for s in word_list]
    
    # y: get next word idx
    y = []
    for i in range(len(word_list)):
        temp = []
        for j in range(len(word_list[i])):
            if j == len(word_list[i]) - 1:
                temp.append(1)  # 1 means <EOS>
            else:
                temp.append(x[i][j+1])
        y.append(temp)
    
    # padding of x, default is 0(symbolizes <PAD>). padding includes:over->cutoff, less->padding. default: left_padding
    x = pad_sequences(x, maxlen=MAXLEN, value=0, padding='post', truncating='post')
    # padding of y, default is 0. right padding
    y = pad_sequences(y, maxlen=MAXLEN, value=0, padding='post', truncating='post')
    # one-hot of y
    y = to_categorical(y, len(vocab))

    return x, y


# In[ ]:


def load_data():
    train_word, train_content, _ = read_file(train_set)
    test_word, test_content, maxlen = read_file(test_set)
    
    vocab = list(set(train_word + test_word))   # 合并，构成大词表
    special_chars = ['<PAD>', '<EOS>']   #特殊词表示：PAD表示padding，EOS表示句子结尾
    vocab = special_chars + vocab
    
    # save initial config data
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump((vocab), f)
    
    # process data: padding
    print('maxlen is %d' % maxlen)
    return train_content, test_content, vocab, maxlen


# In[ ]:


word2vec_model_path = 'sgns.wiki.word.bz2'  #词向量位置
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')


# In[ ]:


def make_embeddings_matrix(word2vec_model, vocab):
    word2vec_dict = {}    # 字对字向量
    vocab2idx = {char: idx for idx, char in enumerate(vocab)}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    embeddings_matrix = np.zeros((len(vocab), EMBED_DIM))# form huge matrix
    for i in tqdm(range(2, len(vocab))):
        word = vocab[i]
        if word in word2vec_dict.keys():    # 如果word在词向量列表中，更新权重；否则，赋值为全0（默认）
            word_vector = word2vec_dict[word]
            embeddings_matrix[i] = word_vector
    return embeddings_matrix


# In[ ]:


train_content, test_content, vocab, maxlen = load_data()
# change maxlen
maxlen = 50
embeddings_matrix = make_embeddings_matrix(word2vec_model, vocab)
# input layer
inputs = Input(shape=(maxlen, ), dtype='int32')
# masking layer 屏蔽层
# x = Masking(mask_value=0)(inputs)
# embedding layer: map the word to it's weights(with embedding-matrix)
x = Embedding(len(vocab), EMBED_DIM, weights=[embeddings_matrix], input_length=maxlen, trainable=True)(inputs)
# LSTM layer
x = LSTM(RNN_UNITS, input_shape=(maxlen, EMBED_DIM), return_sequences=True)(x)
# Dropout: 正则化，防止过拟合.argument means percentage
# x = Dropout(0.5)(x)
# 一维展开，全连接
x = TimeDistributed(Dense(len(vocab)))(x)
# 激活函数：softmax
outputs = Activation('softmax')(x)
# model
model = Model(inputs=inputs, outputs=outputs)
# print arguments of each layer
model.summary()
# target_function: includes optimizer, function_type, metrics
RMSPROP = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


def build_test_data(word_idx, maxlen, index):
    result = [0] * maxlen
    result[index] = word_idx
    result = np.array(result) 
    # print(result.shape)
    return result


# In[ ]:


# train
def train(start1, end1, epochs1, batch_size1):
    model.load_weights('model1.h5')
    maxlen = 50   # maxlen取50，直接截断
    start = start1
    EPOCHS = epochs1
    TRAIN_BATCH = 200
    while start < len(train_content):
        print(start)
        file = open('temp1.txt', 'a')
        file.write(str(start))
        file.write('\n')
        file.close()
        if start == end1:
            break
        if start % 10000 == 0:
            model.save_weights('model1.h5')
        if start+TRAIN_BATCH <= len(train_content):
            train_x, train_y = process_data(train_content[start: start+TRAIN_BATCH], vocab, maxlen)
        else:
            train_x, train_y = process_data(train_content[start: ], vocab, maxlen)
        
        model.fit(train_x, train_y, batch_size=batch_size1, epochs=EPOCHS, verbose=0, validation_split=0.1)
        start += TRAIN_BATCH
    model.save_weights('model1.h5')


# In[ ]:


vocab2idx = {word: idx for idx, word in enumerate(vocab)}


# In[ ]:


def test1(start, test_num1):
    model.load_weights('model1.h5')
    file = open('result1.txt', 'a')
    
    i = 0
    j = 0
    TEST_NUM = test_num1
    for i in range(start, start + TEST_NUM):
        if len(test_content[i])==0:
            continue
        sentence = [test_content[i][0]]
        word_idx = vocab2idx.get(test_content[i][0])
        test_x = build_test_data(word_idx, maxlen, 0)
        for j in range(0, 49):
            temp = []
            temp.append(test_x)
            temp = np.array(temp)
            next_word = model.predict(temp, batch_size=1)  # 输入得是numpy数组，不能是list
            index = np.argmax(next_word[0][j])
            if index == 1 or index == 0:   # means predict <EOS>
                print(i,j, index)
                file.write(str(i))
                file.write(' ')
                file.write(str(j))
                file.write(' ')
                file.write(str(index))
                file.write('\n')
                break
            sentence.append(vocab[index])
            test_x = build_test_data(index, maxlen, j+1)
        for word in sentence:
            file.write(word)
            file.write(' ')
        file.write('<EOS>')
        file.write('\n')
        print(sentence)   
    file.close()


# In[ ]:


train(start1=0, end1=90000, epochs1=15, batch_size1=32)


# In[ ]:


test1(start=0, test_num1=3985)

