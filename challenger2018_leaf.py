import os
import time
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


ht = 224
wd = 224
label_char_dict = {'苹果健康':0, '苹果黑星病一般':1, '苹果黑星病严重':2, '苹果灰斑病':3, '苹果雪松锈病一般':4, '苹果雪松锈病严重':5,
                   '樱桃健康':6, '樱桃白粉病一般':7, '樱桃白粉病严重':8, '玉米健康':9, '玉米灰斑病一般':10, '玉米灰斑病严重':11,
                   '玉米锈病一般':12, '玉米锈病严重':13, '玉米叶斑病一般':14, '玉米叶斑病严重':15, '玉米花叶病毒病':16, '葡萄健康':17,
                   '葡萄黑腐病一般':18, '葡萄黑腐病严重':19, '葡萄轮斑病一般':20, '葡萄轮斑病严重':21, '葡萄褐斑病一般':22,
                   '葡萄褐斑病严重':23, '柑桔健康':24, '柑桔黄龙病一般':25, '柑桔黄龙病严重':26, '桃子健康':27, '桃疮痂病一般':28,
                   '桃疮痂病严重':29, '辣椒健康':30, '辣椒疮痂病一般':31, '辣椒疮痂病严重':32, '马铃薯健康':33, '马铃薯早疫病一般':34,
                   '马铃薯早疫病严重':35, '马铃薯晚疫病一般':36, '马铃薯晚疫病严重':37, '草莓健康':38, '草莓叶枯病一般':39,
                   '草莓叶枯病严重':40, '番茄健康':41, '番茄白粉病一般':42, '番茄白粉病严重':43, '番茄疮痂病一般':44,
                   '番茄疮痂病严重':45, '番茄早疫病一般':46, '番茄早疫病严重':47, '番茄晚疫病菌一般':48, '番茄晚疫病菌严重':49,
                   '番茄叶霉病一般':50, '番茄叶霉病严重':51, '番茄斑点病一般':52, '番茄斑点病严重':53, '番茄斑枯病一般':54,
                   '番茄斑枯病严重':55, '番茄红蜘蛛损伤一般':56, '番茄红蜘蛛损伤严重':57, '番茄黄化曲叶病毒病一般':58,
                   '番茄黄化曲叶病毒病严重':59, '番茄花叶病毒病':60}
label_multy_dict = {j:i for i in [[0,1,2,3,4,5], [6,7,8], [9,10,11,12,13,14,15,16], [17,18,19,20,21,22,23], [24,25,26],
                                  [27,28,29], [30,31,32], [30,31,32], [33,34,35,36,37], [38,39,40],
                                  [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]] for j in i}

def upzip_file(path):
    """解压缩文件"""
    start = time.time()
    with zipfile.ZipFile(path, 'r') as zin:
        zin.extractall(os.path.expanduser('/www/liyanpeng/data'))
    print("数据集解压完成 run time: %d min %.2f s" % divmod((time.time() - start), 60))

def data_preprocessing(train_old_path, valid_old_path, new_path):
    """数据预处理"""
    start = time.time()
    if tf.gfile.Exists(new_path): tf.gfile.DeleteRecursively(new_path)
    tf.gfile.MakeDirs(new_path)
    for (state, old_path) in [('train',train_old_path), ('valid', valid_old_path)]:
        for i in tf.gfile.ListDirectory(old_path):
            tf.gfile.Rename(old_path+'/'+i, old_path+'/'+i.replace(' ', ''), overwrite=True)
        for (char, label) in label_char_dict.items():
            if '健康' in char:
                list_path = os.listdir(old_path+'/'+char)
                for img in list_path:
                    tf.gfile.Copy(old_path+'/'+char+'/'+img, new_path+'/'+str(label)+'_'+state+'_'+img, overwrite=True)
            elif '一般' in char:
                list_path = os.listdir(old_path+'/'+char[:-2]+'/一般')
                for img in list_path:
                    tf.gfile.Copy(old_path+'/'+char[:-2]+'/一般/'+img, new_path+'/'+str(label)+'_'+state+'_'+img, overwrite=True)
            elif '严重' in char:
                list_path = os.listdir(old_path+'/'+char[:-2]+'/严重')
                for img in list_path:
                    tf.gfile.Copy(old_path+'/'+char[:-2]+'/严重/'+img, new_path+'/'+str(label)+'_'+state+'_'+img, overwrite=True)
            else:
                list_path = os.listdir(old_path+'/'+char)
                for img in list_path:
                    tf.gfile.Copy(old_path+'/'+char+'/'+img, new_path+'/'+str(label)+'_'+state+'_'+img, overwrite=True)
    for i in os.listdir(new_path):
        file = i.split('.')[-1]
        if (file=='JPG')|(file=='png'):
            tf.gfile.Rename(new_path+'/'+i, new_path+'/'+i[:-3]+'jpg', overwrite=True)
        elif file=='lnk':
            tf.gfile.Remove(new_path+'/'+i)
    print("数据预处理完成 run time: %.2f min %.2f s" % divmod((time.time() - start), 60))

def get_model(lr=1e-3):
    inputs = tf.keras.Input(shape=(ht, wd, 3))
    base_model = tf.keras.applications.DenseNet121(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg')
    x = tf.keras.layers.Dense(1024, activation='relu')(base_model.output)
    predictions = tf.keras.layers.Dense(61, activation='softmax', name='predictions')(x)
    sigmoid_loss = tf.keras.layers.Dense(61, activation='sigmoid', name='sigmoid_loss')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[predictions, sigmoid_loss])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={'predictions':'categorical_crossentropy', 'sigmoid_loss':'binary_crossentropy'},
                  loss_weights={'predictions':1, 'sigmoid_loss':0.3},
                  metrics={'predictions':'categorical_accuracy', 'sigmoid_loss':'binary_accuracy'})
    return model

def image_aug(filename, label, sig_label, if_train=True):
    image = tf.image.decode_jpeg(tf.read_file(filename))
    if if_train:
        image = tf.image.random_flip_left_right(image)
        if np.random.random()>0.5:
            image = tf.random_crop(image, size=[int(ht/3), int(wd/3), 3])
#         if np.random.random()>0.5:
#             image = tf.image.transpose_image(image)
#         if np.random.random()>0.5:
#             image = tf.image.random_brightness(image, 0.5)
#         if np.random.random()>0.5:
#             image = tf.image.random_contrast(image, 0.5, 2)
#         if np.random.random()>0.5:
#             image = tf.image.random_hue(image, 0.25)
#         if np.random.random()>0.5:
#             image = tf.image.random_saturation(image, 2., 4.)
    image = tf.image.resize_images(image, [ht, wd])
    return image, {'predictions':tf.one_hot(label, 61), 'sigmoid_loss':sig_label}

def train_func(epochs=10, batch=64):
    if not tf.gfile.Exists('./model'): tf.gfile.MakeDirs('./model')
    t = pd.DataFrame({'image':tf.gfile.Glob('/www/liyanpeng/data/sample/*.jpg')}).sample(frac=1, random_state=27).reset_index(drop=True)
    t['label'] = t.image.map(lambda x:int(x.split('/')[5].split('_')[0])).values
    t['sig_label'] = t.label.map(lambda x:[1 if i in label_multy_dict[x] else 0 for i in range(61)])
    model = get_model()
    model.save_weights('./model/leaf_random.hdf5', save_format='h5')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    for k, (tr, va) in enumerate(skf.split(t.image, t.label)):
        train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(t.image[tr].tolist()), tf.constant(t.label[tr].tolist()),
            tf.constant(t.sig_label[tr].tolist()))).map(lambda x,y,z: image_aug(x, y, z, True)).shuffle(batch*3).batch(batch).prefetch(batch).repeat()
        vaild_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(t.image[va].tolist()), tf.constant(t.label[va].tolist()),
            tf.constant(t.sig_label[va].tolist()))).map(lambda x,y,z: image_aug(x, y, z, False)).batch(batch).prefetch(batch).repeat()
        model.load_weights('./model/leaf_random.hdf5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/leaf_{}.hdf5'.format(k+1), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_lr=1e-5)
        model.fit(train_dataset, epochs=epochs, verbose=1, steps_per_epoch=int(35000/batch*0.8),
                  validation_data=vaild_dataset, validation_steps=int(35000/batch*0.2), callbacks=[checkpoint, early, lr_reduce])
