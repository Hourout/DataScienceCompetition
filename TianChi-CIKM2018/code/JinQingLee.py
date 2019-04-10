import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='-1'
import io
import sys
import csv
import time
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import  KFold
import tensorflow as tf

MAX_LEN = 30
N_EPOCH = 10
N_FOLD = 5


def read_data(sp_train_path, en_train_path, test_path, untrain_path):
    """数据读入"""
    start = time.time()
    train_sp = pd.read_table(sp_train_path, header=None, names=['sp1', 'en1', 'sp2', 'en2', 'label'])
    train_en = pd.read_table(en_train_path, header=None, names=['en1', 'sp1', 'en2', 'sp2', 'label'])
    test = pd.read_table(test_path, header=None, names=['sp1', 'sp2'])
    untrain = pd.read_table(untrain_path, header=None, names=['q1', 'q2'])
    
    train = pd.concat([train_sp[['sp1', 'sp2', 'label']], train_en[['sp1', 'sp2', 'label']]]).reset_index(drop=True)
    train['sp1'] = train.sp1.map(lambda x:clean_str_stem(x))
    train['sp2'] = train.sp2.map(lambda x:clean_str_stem(x))
    test['sp1'] = test.sp1.map(lambda x:clean_str_stem(x))
    test['sp2'] = test.sp2.map(lambda x:clean_str_stem(x))
    untrain = untrain.q1.map(lambda x:clean_str_stem(x))

    sp = pd.concat([train.sp1, train.sp2, test.sp1,test.sp2, untrain]).reset_index(drop=True)
    print("数据读入完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return sp, train, test

def clean_str_stem(stri):
    """将所有的大写字母转换为小写字母"""
    text = stri.lower()
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r'\?', " ? ", text)
    text = re.sub(r'？', " ? ", text)
    text = re.sub(r'¿', " ¿ ", text)
    text = re.sub(r'¡', " ¡ ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r'"', " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\*+", "*", text)
    text = re.sub(r"[`|´]", " ", text)
    text = re.sub(r"[&|#|}]", " ", text)
    return text

def load_vectors(sp_word2vec_path):
    """载入词向量"""
    start = time.time()
    fin = io.open(sp_word2vec_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_vec_dict = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word_vec_dict[tokens[0]] = list(map(float, tokens[1:]))
    print("载入词向量完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return word_vec_dict

def wor2vec_obtain(data, sp_word2vec_path):
    """获取词向量字典和词索引字典"""
    start = time.time()
    word_vector_dict = load_vectors(sp_word2vec_path)
    word_index_dict = {v: k + 1 for k, v in enumerate(word_vector_dict.keys())}
    stops = set(stopwords.words("spanish"))
    for word in data:
        segList = word.strip().split(' ')
        for each in segList:
            if (each not in stops)&(each not in word_index_dict.keys()):
                word_index_dict[each] = len(word_index_dict)+1
                word_vector_dict[each] = np.random.random((300,)).tolist()
    print("获取词向量字典和词索引字典完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return word_index_dict, word_vector_dict

def text2index(word_index_dict, data, maxlen=30, verbose=True):
    """词转换为序列索引"""
    start = time.time()
    stops = set(stopwords.words("spanish"))
    not_exit_word = []
    new_data = []
    for word in data:
        new_word = []
        segList = word.strip().split(' ')
        for each in segList:
            if each not in stops:
                try:
                    new_word.append(word_index_dict[each])
                except:
                    not_exit_word.append(each)
                    new_word.append(0)
        new_data.append(new_word)
    if len(not_exit_word)>0:
        print("不存在单词", len(not_exit_word), not_exit_word)
    new_data = tf.keras.preprocessing.sequence.pad_sequences(new_data, padding='post', maxlen=maxlen)
    if verbose:
        print("词转换为序列索引完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return new_data

def make_sample(train, word_index_dict, file_per_sample=50000):
    """构造样本并写入本地文件"""
    start = time.time()
    f = open('./data/sample/make_sample_1.csv', "a+")
    writer = csv.writer(f)
    no_dict = defaultdict(list)
    for i, j in zip(train[train.label==0].sp1, train[train.label==0].sp2):
        if j not in no_dict[i]:
            no_dict[i].append(j)
    G = nx.Graph()
    for (i, j) in zip(train[train.label==1].sp1, train[train.label==1].sp2):
        G.add_edge(i, j)
    n = len(list(nx.connected_components(G)))
    sample_num = 0
    label_1 = 0
    label_0 = 0
    for record, i in enumerate(nx.connected_components(G)):
        for j in list(itertools.combinations(i, 2)):
            t = np.concatenate((text2index(word_index_dict, [j[0]], maxlen=30, verbose=False).reshape((-1)),
                                text2index(word_index_dict, [j[1]], maxlen=30, verbose=False).reshape((-1)), np.array([1], 'int')), axis=0)
            writer.writerow(t.tolist())
            label_1 += 1
            sample_num += 1
            if sample_num%file_per_sample==0:
                f.close()
                f = open('./data/sample/make_sample_'+str(sample_num//file_per_sample)+'.csv', "a+")
                writer = csv.writer(f)
            if j[0] in no_dict.keys():
                for k in no_dict[j[0]]:
                    t = np.concatenate((text2index(word_index_dict, [j[1]], maxlen=30, verbose=False).reshape((-1)),
                                        text2index(word_index_dict, [k], maxlen=30, verbose=False).reshape((-1)), np.array([0], 'int')), axis=0)
                    writer.writerow(t.tolist())
                    label_0 += 1
                    sample_num += 1
                    if sample_num%file_per_sample==0:
                        f.close()
                        f = open('./data/sample/make_sample_'+str(sample_num//file_per_sample)+'.csv', "a+")
                        writer = csv.writer(f)
        sys.stdout.write("make sample to percent: %.2f %% write sample num %.0f p_n_rate %.2f \r" % (record/n*100, sample_num, label_1/(label_1+label_0)))
        sys.stdout.flush()
    for i, j, k in zip(train.sp1, train.sp2, train.label):
        t = np.concatenate((text2index(word_index_dict, [i], maxlen=30, verbose=False).reshape((-1)),
                            text2index(word_index_dict, [j], maxlen=30, verbose=False).reshape((-1)), np.array([k], 'int')), axis=0)
        writer.writerow(t.tolist())
        if sample_num%file_per_sample==0:
            f.close()
            f = open('./data/sample/make_sample_'+str(sample_num//file_per_sample)+'.csv', "a+")
            writer = csv.writer(f)
    f.close()
    print("构造样本并写入本地文件完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))

def create_word_vector_matrix(word_index_dict, word_vector_dict, embed_dim=300):
    """构造词索引字典与词向量字典对应的矩阵"""
    start = time.time()
    word_count = len(word_index_dict)+1
    index_embed_matrix = np.zeros((word_count, embed_dim))
    for word, index in word_index_dict.items():
        index_embed_matrix[index, :] = word_vector_dict[word]
    print("构造词索引字典与词向量字典对应的矩阵完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return index_embed_matrix

def transform(sp_train_path, en_train_path, test_path, untrain_path, sp_word2vec_path):
    """数据转化操作"""
    start = time.time()
    sp, train, test = read_data(sp_train_path, en_train_path, test_path, untrain_path)
    word_index_dict, word_vector_dict = wor2vec_obtain(sp, sp_word2vec_path)
    index_embed_matrix = create_word_vector_matrix(word_index_dict, word_vector_dict, embed_dim=300)
    np.save('./data/train_sp1_sequence.npy', text2index(word_index_dict, train.sp1, maxlen=30))
    np.save('./data/train_sp2_sequence.npy', text2index(word_index_dict, train.sp2, maxlen=30))
    np.save('./data/train_label.npy', train.label.values)
    np.save('./data/test_sp1_sequence.npy', text2index(word_index_dict, test.sp1, maxlen=30))
    np.save('./data/test_sp2_sequence.npy', text2index(word_index_dict, test.sp2, maxlen=30))
    np.save('./data/index_embed_matrix.npy', index_embed_matrix)
    print("数据转换操作完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return train, test

def unlabel_transform(word_index_dict, untrain_path, param=0.8, param_mean=0.8):
    """未标注数据筛选样本"""
    start = time.time()
    untrain = pd.read_table(untrain_path, header=None, names=['q1', 'q2'])
    untrain = untrain.q1.map(lambda x:clean_str_stem(x))
    model =  decomposable_attention()
    score = pd.DataFrame()
    for model_index in range(5):
        model.load_weights('./model/CIKM_dec_Attention_classify_{}.hdf5'.format(model_index))
        if model_index==0:
            n = 5
            t = pd.DataFrame()
            k = len(untrain)
            for i in range(len(untrain)):
                if i==len(untrain)-1:
                    break
                t = pd.concat([t, pd.DataFrame({'sp1':untrain[i], 'sp2':untrain[i+1:]})])
                if i%n==0:
                    train_left = text2index(word_index_dict, t.sp1, maxlen=30, verbose=False)
                    train_right = text2index(word_index_dict, t.sp2, maxlen=30, verbose=False)
                    t['model_'+str(model_index)] = model.predict([train_left, train_right]).reshape((-1))
                    t = t[t['model_'+str(model_index)]>param]
                    score = pd.concat([score, t])
                    t = pd.DataFrame()
                    n = round(n+5+n/10)
                sys.stdout.write("make sample to percent: %.2f %% \r" % (i/k*100))
                sys.stdout.flush()
        if model_index>0:
            train_left = text2index(word_index_dict, score.sp1, maxlen=30, verbose=False)
            train_right = text2index(word_index_dict, score.sp2, maxlen=30, verbose=False)
            score['model_'+str(model_index)] = model.predict([train_left, train_right]).reshape((-1))
    score['label'] = score.iloc[:, 2:].mean(axis=1)
    score = score[score['label']>param_mean]
    score['label'] = 1
    score = score.drop(['model_0', 'model_1', 'model_2', 'model_3', 'model_4'], axis=1).reset_index(drop=True)
    print("未标注数据筛选样本完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return score

def distances(q1, q2, metric='cosine'):
    """文本相似度距离"""
    text = [q1, q2]
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(text)
    t = pairwise_distances(tokenizer.texts_to_matrix(text), metric='cosine').max()
    return t

def feature(train, test):
    """特征工程"""
    start = time.time()
    for data in [train, test]:
        data['q1_num'] = data.q1.map(lambda x: len(x.split(' ')))
        data['q2_num'] = data.q2.map(lambda x:len(x.split(' ')))
        data['q1_q2_diff_len'] = np.abs(data.q1_num - data.q2_num)
        data['q1_q2_minmax_rate'] = np.min(data.q1_num, data.q2_num)/np.max(data.q1_num, data.q2_num)
        data['same_num'] = data.apply(lambda x:len(set(x.q1.split(' '))&set(x.q2.split(' '))), axis=1)
        data['same_q1_rate'] = data.same_num/data.q1_num
        data['same_q2_rate'] = data.same_num/data.q2_num
        for i in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev',
                  'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                  'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            data['distances_'+i] = data.apply(lambda x:distances(x.q1, x.q2, metric=i), axis=1)
    print("特征工程处理完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return train, test


def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = tf.keras.layers.Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = tf.keras.layers.Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = tf.keras.layers.Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = tf.keras.layers.Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= tf.keras.layers.Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = tf.keras.layers.Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = tf.keras.layers.TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = tf.keras.layers.Dot(axes=-1)([input_1, input_2])
    w_att_1 = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=1),
                                     output_shape=unchanged_shape)(attention)
    w_att_2 = tf.keras.layers.Permute((2,1))(tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x, axis=2),
                                             output_shape=unchanged_shape)(attention))
    in1_aligned = tf.keras.layers.Dot(axes=1)([w_att_1, input_1])
    in2_aligned = tf.keras.layers.Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding='./data/index_embed_matrix.npy',
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933
    inputs = Input(shape=(60,))
#     q1 = inputs[:, :30]
#     q2 = inputs[:, 30:]
#     print(q1.shape)
#     print(q2.shape)
#     q1 = Input(name='q1',shape=(maxlen,))
#     q2 = Input(name='q2',shape=(maxlen,))
    
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, mask_zero=False)
    q1_embed = embedding(tf.keras.layers.Lambda(lambda x: x[:, :30])(inputs))
    q2_embed = embedding(tf.keras.layers.Lambda(lambda x: x[:, 30:])(inputs))
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                tf.keras.layers.Dense(projection_hidden, activation=activation),
                tf.keras.layers.Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            tf.keras.layers.Dense(projection_dim, activation=None),
            tf.keras.layers.Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare
    q1_combined = tf.keras.layers.Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = tf.keras.layers.Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compare_layers = [
        tf.keras.layers.Dense(compare_dim, activation=activation),
        tf.keras.layers.Dropout(compare_dropout),
        tf.keras.layers.Dense(compare_dim, activation=activation),
        tf.keras.layers.Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [tf.keras.layers.GlobalAvgPool1D(), tf.keras.layers.GlobalMaxPool1D()])
    
    # Classifier
    merged = tf.keras.layers.Concatenate()([q1_rep, q2_rep])
    dense = tf.keras.layers.BatchNormalization()(merged)
    dense = tf.keras.layers.Dense(dense_dim, activation=activation)(dense)
    dense = tf.keras.layers.Dropout(dense_dropout)(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(dense_dim, activation=activation)(dense)
    dense = tf.keras.layers.Dropout(dense_dropout)(dense)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def to_tensor(line):
    parsed_line = tf.decode_csv(line, [[0]]*61, field_delim=',')
    label = parsed_line[-1]
    del parsed_line[-1]
    features = parsed_line
    return features, tf.reshape(label, [-1])

def train_func():
    t = pd.Series(['./data/sample/'+i for i in os.listdir('./data/sample') if i[:4]=='make'])
    skf = KFold(n_splits=5, shuffle=False, random_state=27)
    model = decomposable_attention()
    model.save_weights('./model/CIKM_dec_Attention_classify_random.hdf5', save_format='h5')
    for k, (tr, va) in enumerate(skf.split(t)):
        train_dataset = tf.data.TextLineDataset(t[tr].tolist()).map(to_tensor).shuffle(buffer_size=10000).batch(128).repeat()
        vaild_dataset = tf.data.TextLineDataset(t[va].tolist()).map(to_tensor).batch(128).repeat()
        model.load_weights('./model/CIKM_dec_Attention_classify_random.hdf5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/CIKM_dec_Attention_classify_{}.hdf5'.format(k), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4)
        callbacks_list = [checkpoint, early]
        model.fit(train_dataset, epochs=N_EPOCH, verbose=1, steps_per_epoch=int(2200000/128*0.8),
                  validation_data=vaild_dataset, validation_steps=int(2200000/128*0.2), callbacks=callbacks_list)

if __name__ == "__main__":
    #数据处理并存到本地
    train, test = transform(sp_train_path='./data/cikm_spanish_train_20180516.txt',
                            en_train_path='./data/cikm_english_train_20180516.txt',
                            test_path='./data/cikm_test_a_20180516.txt',
                            untrain_path='./data/cikm_unlabel_spanish_train_20180516.txt',
                            sp_word2vec_path='./data/wiki.es.vec')
    # 数据增强并写入本地csv文件，要在。/data下创建。/data/sample文件夹
    sp, train, _ = read_data(sp_train_path='./data/cikm_spanish_train_20180516.txt', en_train_path='./data/cikm_english_train_20180516.txt',
                     test_path='./data/cikm_test_a_20180516.txt', untrain_path='./data/cikm_unlabel_spanish_train_20180516.txt')
    word_index_dict, _ = wor2vec_obtain(sp, './data/wiki.es.vec')
    make_sample(train, word_index_dict)
    #模型训练5折交叉
    train_func()
