import os
os.environ["CUDA_VISIBLE_DEVICES"] ='-1'
import sys
import time
import PIL.Image

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer


def read_sample(train_path, test_path):
    """读入数据"""
    a = time.time()
    with open(train_path) as f:
        train = pd.DataFrame([line.replace("\n","").split(",") for line in f][1:], columns=['id','article','word','label']).drop(['article'], axis=1)
    with open(test_path) as f:
        test = pd.DataFrame([line.replace("\n","").split(",") for line in f][1:], columns=['id','article','word']).drop(['article'], axis=1)
    print('读入数据完成 run time %d min %.2f sec' %divmod((time.time()-a), 60))
    return train, test

# def wordcount(series):
#     """词计数"""
#     a = time.time()
#     word_count_dict = defaultdict(lambda: 0)
#     for sequence in series:
#         for word in set(sequence):
#             word_count_dict[word] += 1
#     print('词计数完成 run time %d min %.2f sec' %divmod((time.time()-a), 60))
#     return word_count_dict

def make_sample(train, test):
    """构造训练测和试样本并写入本地文件"""
    a = time.time()
    vectorizer = TfidfVectorizer()
    train_all = np.round(vectorizer.fit_transform(train.word)*255).astype('uint8')
    test_all = np.round(vectorizer.transform(test.word)*255).astype('uint8')
    for path in ['./data/train_sample', './data/test_sample']:
        if tf.gfile.Exists(path): tf.gfile.DeleteRecursively(path)
        tf.gfile.MakeDirs(path)
    for k, (ids, feature, label) in enumerate(zip(train.id, train_all, train.label)):
        PIL.Image.fromarray(feature.toarray()[0].reshape((769, 1138)), 'L').save('./data/train_sample/'+label+'_'+ids+'.png')
        sys.stdout.write("make train sample to percent: %.2f %% write sample num %.0f \r" % (k/102277*100, k))
        sys.stdout.flush()
    for k, (ids, feature) in enumerate(zip(train.id, train_all)):
        PIL.Image.fromarray(feature.toarray()[0].reshape((769, 1138)), 'L').save('./data/test_sample/'+ids+'.png')
        sys.stdout.write("make test sample to percent: %.2f %% write sample num %.0f \r" % (k/102277*100, k))
        sys.stdout.flush()
    print('构造训练和测试样本并写入本地文件完成 run time %d min %.2f sec' %divmod((time.time()-a), 60))

def get_model(lr=1e-3):
    inputs = tf.keras.Input((769, 1138, 1))
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv1)
    conv1 = tf.keras.layers.MaxPool2D()(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv1)
    conv1 = tf.keras.layers.MaxPool2D()(conv1)
    dense = tf.keras.layers.Flatten()(conv1)
    dense = tf.keras.layers.Dense(2048, activation='relu')(dense)
    dense = tf.keras.layers.Dense(1024, activation='relu')(dense)
    predictions = tf.keras.layers.Dense(19, activation='softmax')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


class f1_score(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        print(' — val_f1: %f' %(_val_f1))

def to_tensor(filename, label):
    image = tf.image.decode_png(tf.read_file(filename))/255
    return image, tf.one_hot(label, 19)

def train_func(epochs=10, bacth=64):
    if not tf.gfile.Exists('./model'): tf.gfile.MakeDirs('./model')
    t = pd.DataFrame({'image':tf.gfile.Glob('./data/train_sample/*.png')})
    t['label'] = t.image.map(lambda x:int(x.split('/')[3].split('_')[0])).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    model = get_model()
    model.save_weights('./model/daguan_random.hdf5', save_format='h5')
    for k, (tr, va) in enumerate(skf.split(t.image, t.label)):
        train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(t.image[tr].tolist()), tf.constant(t.label[tr].tolist()))).map(to_tensor).shuffle(bacth*3).batch(bacth).repeat()
        vaild_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(t.image[va].tolist()), tf.constant(t.label[va].tolist()))).map(to_tensor).batch(bacth).repeat()
        model.load_weights('./model/daguan_random.hdf5')
        f1 = f1_score()
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/daguan_{}.hdf5'.format(k+1), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_lr=1e-5)
        model.fit(train_dataset, epochs=epochs, verbose=1, steps_per_epoch=int(102277/bacth*0.8),
                  validation_data=vaild_dataset, validation_steps=int(102277/bacth*0.2), callbacks=[checkpoint, early, lr_reduce, f1])


# train, test = read_sample('./data/train_set.csv', './data/test_set.csv')
# make_sample(train, test)
# train_func()

