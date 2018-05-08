#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
@date: 2018.04.02
@author: zq
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from time import time

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet import init
from mxnet import image


# sns.set(context='notebook', style='white', palette='deep')


# 1.Load data
train = pd.read_csv("../dataset/mnist/train.csv")
test = pd.read_csv("../dataset/mnist/test.csv")


y_train = train["label"]
x_train = train.drop(labels=["label"], axis=1)

del train
# g = sns.countplot(y_train)
# y_train.value_counts()


# 2.Check for null and missing values
print(x_train.isnull().any().describe())
print(test.isnull().any().describe())

# there is no missing values in the train and test dataset. go ahead safely.


# try use all gpu
def try_all_gpu():
    ctx_list = []
    try:
        for i in range(8):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list


# try use one gpu
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


# testset accuracy
def evaluate_accuracy(net, test_iter, ctx=mx.gpu()):
    acc = nd.array([0.])
    n = 0
    for i, (data, label) in enumerate(test_iter):
        label = label.as_in_context(mx.gpu())
        acc += nd.sum(net(data).argmax(axis=1)==label).copyto(mx.cpu())
        n += label.shape[0]
        acc.wait_to_read()
    return acc.asscalar() / n


# 3. define loss function, learning_rate, epochs, weight_decay
softmaxCrossentroy = gluon.loss.SoftmaxCrossEntropyLoss()
learning_rate = 0.0003
epochs = 35
verbose_epoch = 0
weight_decay = 1e-4
ctx = try_gpu()
# ctx_list = try_all_gpu()


# 4.Normalization
x_train = x_train / 255.0
test = test / 255.0

# 5.transfer to mxnet.ndarray
x_train = x_train.as_matrix().reshape(-1, 1, 28, 28)
y_train = y_train.as_matrix()
test = test.as_matrix().reshape(-1, 1, 28, 28)

x_train = nd.array(x_train, ctx=ctx)
y_train = nd.array(y_train, ctx=ctx)
x_test = nd.array(test, ctx=ctx)
del test

# plt.imshow(x_train[4][0][:,:])
# plt.show()


# define vgg_block
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu')
        )
        out.add(nn.MaxPool2D(pool_size=2, strides=2))
        return out


# define vgg_stack
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out


# 6. define net
def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=20, kernel_size=5, strides=1, padding=0, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2, padding=0),
            nn.Conv2D(channels=50, kernel_size=3, strides=1, padding=0, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2, padding=0),
            nn.Conv2D(channels=50, kernel_size=3, strides=1, padding=1, activation='relu'),
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            # nn.Dropout(0.25),
            nn.Dense(10)
        )
    net.initialize(init=init.Xavier(), ctx=ctx)

    net2 = nn.Sequential()
    with net2.name_scope():
        net2.add(
            nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            # nn.MaxPool2D(pool_size=2, strides=2, padding=0),

            nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2, padding=0),

            nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),

            nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),

            nn.Flatten(),
            nn.Dense(256),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.Dropout(0.25),
            nn.Dense(256),
            nn.BatchNorm(axis=1),
            nn.Activation(activation='relu'),
            nn.Dropout(0.25),
            nn.Dense(10)
        )
    net2.initialize(init=init.Xavier(), ctx=ctx)

    net3 = nn.Sequential()
    architecture = ((1, 20), (2, 30), (2, 50))
    with net3.name_scope():
        net3.add(
            vgg_stack(architecture),
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            nn.Dropout(0.25),
            nn.Dense(128, activation='relu'),
            nn.Dropout(0.25),
            nn.Dense(10)
        )
    net3.initialize(init=init.Xavier(), ctx=ctx)

    net2.hybridize()
    return net2

net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        # nn.MaxPool2D(pool_size=2, strides=2, padding=0),

        nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2, padding=0),

        nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),

        nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),

        nn.Flatten(),
        nn.Dense(256),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.25),
        nn.Dense(256),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.Dropout(0.25),
        nn.Dense(10)
    )
net.initialize(init=init.Xavier(), ctx=ctx)
# print(net)


# Augmentation train data


def validation(net, val_data, ctx):

    test_loss = 0.
    test_acc = 0.

    for data, label in val_data:
        label = label.as_in_context(ctx)
        output = net(data)
        loss = softmaxCrossentroy(output, label)

        test_loss += loss.mean().asscalar()
        test_acc += nd.mean(output.argmax(axis=1)==label).asscalar()

    return test_loss, test_acc


# 7.define train process
def train(net, x_train, y_train, x_test, epochs, verbose_epoch, learning_rate, weight_decay):
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(x_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    # dataset_test = gluon.data.ArrayDataset(x_test, y_test)
    # data_iter_test = gluon.data.DataLoader(dataset_test, batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate,
                       'wd': weight_decay})
    total_train_loss = []
    total_test_loss = []
    for epoch in range(epochs):
        tic = time()        # 开始时刻
        train_loss = .0
        train_acc = .0
        test_loss = 0.
        if x_test is not None:
            test_loss = .0
        for data, label in data_iter_train:
            with autograd.record():
                # print('label.shape.lens: ', len(label.shape))
                # data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                output = net(data)
                loss = softmaxCrossentroy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()
            train_acc += nd.mean(output.argmax(axis=1)==label).asscalar()

        # test_acc = evaluate_accuracy(net, data_iter_test)

        # for data, label in data_iter_test:
        #     label = label.as_in_context(ctx)
        #     output = net(data)
        #     loss = softmaxCrossentroy(output, label)
        #     test_loss += loss.mean().asscalar()
        if epoch > verbose_epoch:
            total_train_loss.append(train_loss)
            total_test_loss.append(test_loss)
            print("Epoch: %d, train_loss: %f, train_acc: %f, time: %f" %(epoch,
                 train_loss/len(data_iter_train), train_acc/len(data_iter_train), time()-tic))
    return total_train_loss, total_test_loss


def train2(X_train, Y_train, X_test, Y_test, epochs, verbose, learning_rate, weight_decay, batch_size):
    # 用于画曲线图
    train_loss_list = []
    train_acc_list = []
    if X_test is not None:
        test_loss_list = []
        test_acc_list = []

    dataset_train = gluon.data.ArrayDataset(X_train, Y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

    dataset_test = gluon.data.ArrayDataset(X_test, Y_test)
    data_iter_test = gluon.data.DataLoader(dataset_test, batch_size, shuffle=False)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True, ctx=mx.gpu(0))  # 每轮的训练重新初始化权重
    # net.initialize(ctx=ctx)
    for epoch in range(epochs):
        tic = time()
        train_loss = 0.
        train_acc = 0.
        test_loss = 0.
        test_acc = 0.
        for data, label in data_iter_train:
            with autograd.record():
                label = label.as_in_context(ctx)
                output = net(data)
                loss = softmaxCrossentroy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()
            train_acc += nd.mean(output.argmax(axis=1)==label).asscalar()

        test_loss, test_acc = validation(net, data_iter_test, ctx)

        train_loss_list.append(train_loss/len(data_iter_train))
        train_acc_list.append(train_acc/len(data_iter_train))
        test_loss_list.append(test_loss / len(data_iter_test))
        test_acc_list.append(test_acc / len(data_iter_test))

        if epoch > verbose_epoch and X_test is not None:
            print('Epoch %d, train-loss: %.3f, train-acc: %.3f, test loss: %.3f, test_acc: %.3f, time: %.1f' % (epoch,
                train_loss/len(data_iter_train), train_acc/len(data_iter_train), test_loss/len(data_iter_test), test_acc/len(data_iter_test), time()-tic))


    # 画出loss和acc曲线图
    x_axis = np.arange(1, 36)
    plt.plot(x_axis, train_loss_list, 'r')
    plt.plot(x_axis, test_loss_list, 'g')
    plt.legend(['train-loss', 'test-loss'])
    plt.show()

    if X_test is not None:
        return train_loss/len(data_iter_train), test_loss/len(data_iter_test)
    else:
        return train_loss/len(data_iter_train)


# 8.K-fold corss validation
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay, batch_size):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i*fold_size:(test_i+1)*fold_size, :]
        y_val_test = y_train[test_i*fold_size:(test_i+1)*fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i*fold_size:(i+1)*fold_size,:]
                y_cur_fold = y_train[i*fold_size:(i+1)*fold_size]

                if not val_train_defined:
                    x_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    x_val_train = nd.concat(x_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        # net.collect_params().initialize(force_reinit=True)
        # net = get_net()
        train_loss, test_loss = train2(x_val_train, y_val_train, X_val_test, y_val_test, epochs, verbose_epoch,
              learning_rate, weight_decay, batch_size)
        train_loss_sum += train_loss
        print('[%d fold validation] train-loss: %.3f,  test-loss: %.3f' % (test_i, train_loss, test_loss))
        test_loss_sum += test_loss

        if test_i == 0:
            break
    return (train_loss_sum / k, test_loss_sum / k)


# 9.inference
def learn(test_data):
    # batch_size = 100
    # dataset_test = gluon.data.ArrayDataset(test_data)
    # data_iter_test = gluon.data.DataLoader(dataset_test, batch_size)
    output = net(test_data)
    output = output.argmax(axis=1)
    output = output.asnumpy()
    output = output.astype(np.uint8)
    output = pd.Series(output, name='Label')

    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), output], axis=1)
    submission.to_csv("cnn_mnist_test.csv", index=False)
    print('-------predict over----------')


if __name__ == '__main__':
    k = 10
    batch_size = 50
    avg_train_loss, avg_test_loss= k_fold_cross_valid(k, epochs, verbose_epoch, x_train, y_train, learning_rate, weight_decay, batch_size)
    print('%d-fold validation: Avg train loss: %f, Avg test loss: %f' %(k, avg_train_loss, avg_test_loss))
    print('------begin to predict-------')
    learn(x_test)


# define activation function
def relu(x):
    return np.maximum(x, 0)


# define softmax_loss
def softmax_loss(x, y):
    x = x - np.max(x, axis=1, keepdims=True)    # 减去均值
    probs = np.exp(x)/np.sum(np.exp(x), axis=0)  # 计算概率
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y.astype('int64')])) / N
    dx = probs.copy()       # gradient of the loss to x
    dx[np.arange(N), y.astype('int64')] -= 1
    dx /= N
    return loss, dx



