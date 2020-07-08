import numpy as np
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Activation, Layer
from keras.layers import Conv2D, MaxPooling2D, Flatten, MaxPool2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image

# ------------------------------------------------------------------------
# 从图片文件读取数据


def get_img_data(nImgCount=1, basepath='.\\train', cata='train'):
    # 逐个加载图片
    labels = np.zeros((nImgCount), dtype='int32')
    size = 128
    train_imgs = np.zeros((nImgCount, size, size, 3), dtype='float32')
    for index in range(1, nImgCount+1):
        img_filepath = basepath + '\\' + cata + str(index) + '.jpg'
        img = image.load_img(img_filepath, target_size=(128, 128))
        img_tensor = image.img_to_array(img)
        #img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        train_imgs[index-1, :, :, :] = img_tensor
        labels[index-1] = (index+1) % 2 + 1
    #print('==labels.shape=', labels.shape)
    #print('==labels =', labels)
    labels = np_utils.to_categorical(labels, 3)
    # print(labels.shape)
    return (train_imgs, labels)

# -----------------------------------------------------------------------


def MiniVGG(input_tensor, input_shape=None, output_shape=None):
    # 定义输入层
    # Determine proper input shape
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # 至此，得到输入img_input

    # region 定义隐含层
    conv1_1 = Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='Block1_Conv1')(img_input)  # (?, 28, 28, 32)
    conv1_2 = Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='Block1_Conv2')(conv1_1)  # (?, 28, 28, 32)
    maxpool_1 = MaxPool2D(pool_size=(2, 2),
                          strides=(2, 2),
                          padding='same',
                          name='Block1_MaxPool')(conv1_2)  # (?, 14, 14, 32)
    conv2_1 = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='Block2_Conv1')(maxpool_1)  # (?, 28, 28, 32)
    conv2_2 = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='Block2_Conv2')(conv2_1)  # (?, 28, 28, 32)
    maxpool_2 = MaxPool2D(pool_size=(2, 2),
                          strides=(2, 2),
                          padding='same',
                          name='Block2_MaxPool')(conv2_2)  # (?, 14, 14, 32)
    # endregion 定义隐含层
    # region 定义输出层
    flattened = Flatten(input_shape=(-1, maxpool_2.shape[1],
                                     maxpool_2.shape[2],
                                     maxpool_2.shape[3]))(maxpool_2)

    fc1 = Dense(128, activation='relu', name='FC1')(flattened)

    out_labels = Dense(output_shape, activation='softmax', name='output')(fc1)
    # endregion 定义输出层
    return out_labels


(x_train, y_train) = get_img_data(nImgCount=40, basepath='.\\train', cata='train')
print('y_train.shape=', y_train.shape)
(x_test, y_test) = get_img_data(nImgCount=20, basepath='.\\test', cata='test')
print('y_test.shape=', y_test.shape)
print(y_test)

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
#print('img_rows=', img_rows)
#print('img_cols=', img_cols)
#x_train = x_train.reshape(-1, img_rows, img_cols, 3).astype('float32')
#x_test = x_test.reshape(-1, img_rows, img_cols, 3).astype('float32')
input_shape = (img_rows, img_cols, 3)
#print('input_shape=', input_shape)
output_shape = 3
img_input = Input(shape=input_shape)
out_labels = MiniVGG(img_input, input_shape, output_shape)
print('=================\n', out_labels.shape)
my_model = Model(inputs=img_input, outputs=out_labels)
my_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])  # optimizer='rmsprop'
# 训练
my_model.fit(x_train, y_train, batch_size=3, epochs=10)

loss1, accuracy1 = my_model.evaluate(x_test, y_test)
print('loss1, accuracy1:', loss1, accuracy1)

# my_model.save('my_dogs_vs_cats_model.h5')
