

import random
import numpy as np
#首先定义激活函数备用：  

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):
        return output * (1 - output)


class FC_Layer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))


def forward(self, input_array):

        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)


def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad
```

#最后就是将一层一层的Layerd堆起来组成网络了，神经网络类：

class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
`
#训练方法：
def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], 
                    data_set[d], rate)

def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)            

#预测方法：

def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
    output = sample
    for layer in self.layers:
        layer.forward(output)
        output = layer.output
    return output

#工具函数：

#计算误差项
def calc_gradient(self, label):
    delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    
#更新权重和偏置
def update_weight(self, rate):
    for layer in self.layers:
        layer.update(rate)

#损失函数
def loss(self, output, label):
    return 0.5 * ((label - output) * (label - output)).sum()

def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
    self.predict(sample_feature)
    self.calc_gradient(sample_label)

        # 检查梯度
    epsilon = 10e-4
    for fc in self.layers:
        for i in range(fc.W.shape[0]):
            for j in range(fc.W.shape[1]):
                fc.W[i,j] += epsilon
                output = self.predict(sample_feature)
                err1 = self.loss(sample_label, output)
                fc.W[i,j] -= 2*epsilon
                output = self.predict(sample_feature)
                err2 = self.loss(sample_label, output)
                expect_grad = (err1 - err2) / (2 * epsilon)
                fc.W[i,j] += epsilon
                print ('weights(%d,%d): expected - actural %.4e - %.4e' %(
                           i, j, expect_grad, fc.W_grad[i,j]))

