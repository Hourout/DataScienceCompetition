import os
os.environ["CUDA_VISIBLE_DEVICES"] ='-1'
import sys
import csv
import time
import zipfile

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import  KFold
from sklearn.model_selection import  StratifiedKFold


def upzip_file(path):
    """解压缩文件"""
    start = time.time()
    with zipfile.ZipFile(path, 'r') as zin:
        zin.extractall(os.path.expanduser('./data'))
    print("数据集解压完成 run time: %d min %.2f s" % divmod((time.time() - start), 60))

def read_data(train_path, test_path):
    """数据读入预处理"""
    start = time.time()
    train = pd.read_csv(train_path, dtype={'file_id':np.uint16, 'label':np.uint8, 'api':'category', 'tid':np.uint16})
    train = train.sort_values(['file_id', 'tid', 'index']).drop(['return_value', 'index'], axis=1).reset_index(drop=True)
    test = pd.read_csv(test_path, dtype={'file_id':np.uint16, 'api':'category', 'tid':np.uint16})
    test = test.sort_values(['file_id', 'tid', 'index']).drop(['return_value', 'index'], axis=1).reset_index(drop=True)
    api_index_dict = {v: k+1 for k, v in enumerate(pd.concat([train.api, test.api]).drop_duplicates().tolist())}
    print('api count number: ', len(api_index_dict))
    train['api'] = [api_index_dict[v] for v in train.api]
    test['api'] = [api_index_dict[v] for v in test.api]
    train = train.groupby(['file_id', 'label', 'tid']).apply(lambda x:x.api.tolist()).rename('sequences').reset_index().drop(['tid'], axis=1)
    test = test.groupby(['file_id', 'tid']).apply(lambda x:x.api.tolist()).rename('sequences').reset_index().drop(['tid'], axis=1)
    print("数据读入预处理完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))
    return train, test

def make_train_sample(train, max_len=300, p_dict={0:3, 1:10, 2:5, 3:3, 4:30, 5:5}, sample_file_num=100):
    """构造训练样本并写入本地文件"""
    start = time.time()
    if not os.path.exists('./data/sample'):
        os.makedirs('./data/sample')
    print(train.shape)
    print(train.label.value_counts())
    l = train.shape[0]
    sample_num = 0
    f = open('./data/sample/make_train_sample.csv', "a+")
    writer = csv.writer(f)
    for k, (file_id, label, sequences) in enumerate(zip(train.file_id, train.label, train.sequences)):
        if len(sequences)>max_len:
            rd = np.random.choice(np.arange(len(sequences)-max_len+1), min(int(np.ceil(len(sequences)-max_len+1)), p_dict[label]), replace=False).tolist()
            for i in rd:
                writer.writerow(sequences[i:i+max_len]+[label])
                sample_num += 1
        else:
            writer.writerow(sequences+[0]*(max_len-len(sequences))+[label])
            sample_num += 1
        sys.stdout.write("make sample to percent: %.2f %% write sample num %.0f \r" % (k/l*100, sample_num))
        sys.stdout.flush()
    f.close()
    t = pd.read_csv('./data/sample/make_train_sample.csv', header=None).sample(frac=1, random_state=27).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=sample_file_num, random_state=27, shuffle=True)
    for k, (_, index) in enumerate(skf.split(t.iloc[:, :max_len], t[max_len])):
        t.loc[index].to_csv('./data/sample/make_train_sample_'+str(k+1)+'.csv', index=False, header=None)
    os.remove('./data/sample/make_train_sample.csv')
    print("构造训练样本并写入本地文件完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))

def make_test_sample(test, max_len=300):
    """构造测试样本并写入本地文件"""
    start = time.time()
    print(test.shape)
    l = test.shape[0]
    sample_num = 0
    f = open('./data/make_test_sample.csv', "a+")
    writer = csv.writer(f)
    for k, (file_id, sequences) in enumerate(zip(test.file_id, test.sequences)):
        if len(sequences)>max_len:
            rd = np.random.choice(np.arange(len(sequences)-max_len+1), 1, replace=False).tolist()
            for i in rd:
                writer.writerow(sequences[i:i+max_len]+[file_id])
                sample_num += 1
        else:
            writer.writerow(sequences+[0]*(max_len-len(sequences))+[file_id])
            sample_num += 1
        sys.stdout.write("make sample to percent: %.2f %% write sample num %.0f \r" % (k/l*100, sample_num))
        sys.stdout.flush()
    f.close()
    print("构造测试样本并写入本地文件完成 run time: %d min %.2f sec" % divmod((time.time() - start), 60))

def tsnet(sequence_length=300, embed_in_dim=312, embed_out_dim=300, dense_dim=300, lr=1e-3, activation='elu'):
    """时序网络模型"""
    inputs = tf.keras.Input(shape=(sequence_length,), dtype='int32')
    embedding = tf.keras.layers.Embedding(embed_in_dim, embed_out_dim, input_length=sequence_length)(inputs)
    embedding = tf.keras.layers.Reshape((sequence_length, embed_out_dim, 1))(embedding)
    print(embedding.shape)
    
    stage1 = tf.keras.layers.Conv2D(32, (3, embed_out_dim))(embedding)
    stage1 = tf.keras.layers.BatchNormalization()(stage1)
    stage1 = tf.keras.layers.Activation(activation)(stage1)
    print(stage1.shape)
    stage1 = tf.keras.layers.Conv2D(32, (3, 1))(stage1)
    stage1 = tf.keras.layers.BatchNormalization()(stage1)
    stage1 = tf.keras.layers.Activation(activation)(stage1)
    stage1 = tf.keras.layers.GlobalMaxPool2D()(stage1)
    print(stage1.shape)
    
    stage2 = tf.keras.layers.Conv2D(32, (3, embed_out_dim))(embedding)
    stage2 = tf.keras.layers.BatchNormalization()(stage2)
    stage2 = tf.keras.layers.Activation(activation)(stage2)
    stage2 = tf.keras.layers.Conv2D(32, (5, 1))(stage2)
    stage2 = tf.keras.layers.BatchNormalization()(stage2)
    stage2 = tf.keras.layers.Activation(activation)(stage2)
    stage2 = tf.keras.layers.GlobalMaxPool2D()(stage2)
    
    
    combined = tf.keras.layers.Concatenate()([stage1, stage2])    
    
    dense = tf.keras.layers.BatchNormalization()(combined)
    dense = tf.keras.layers.Dense(dense_dim, activation=activation)(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(dense_dim, activation=activation)(dense)
    predictions = tf.keras.layers.Dense(6, activation='softmax')(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def to_tensor(line):
    line = tf.decode_csv(line, [[0]]*301, field_delim=',')
    return line[:-1], tf.one_hot(line[-1], 6)

def train_func(n_splits=5, epochs=10):
    """训练模型"""
    if not os.path.exists('./model'):
        os.makedirs('./model')
    t = pd.Series(['./data/sample/'+i for i in os.listdir('./data/sample') if i[:4]=='make'])
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=27)
    model = tsnet()
    model.save_weights('./model/tsnet_random.hdf5', save_format='h5')
    for k, (tr, va) in enumerate(skf.split(t)):
        train_dataset = tf.data.TextLineDataset(t[tr].tolist()).map(to_tensor).shuffle(buffer_size=20000).batch(128).repeat()
        vaild_dataset = tf.data.TextLineDataset(t[va].tolist()).map(to_tensor).batch(128).repeat()
        model.load_weights('./model/tsnet_random.hdf5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/tsnet_{}.hdf5'.format(k+1), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        callbacks_list = [checkpoint, earlystop]
        model.fit(train_dataset, epochs=epochs, verbose=1, steps_per_epoch=int(1312007/128*0.03),
                  validation_data=vaild_dataset, validation_steps=int(1312007/128*0.002), callbacks=callbacks_list)

def test_func():
    """预测结果"""
    pre = pd.read_csv('./data/make_test_sample.csv', usecols=[300], names=['file_id'])
    for k in range(5):
        model = tf.keras.models.load_model('./model/tsnet_{}.hdf5'.format(k+1))
        test_dataset = tf.data.TextLineDataset('./data/make_test_sample.csv').map(to_tensor).batch(128).repeat()
        pred['model_{}'.format(k+1)] = model.predict(test_dataset, steps=int(np.ceil(441748/128)))

