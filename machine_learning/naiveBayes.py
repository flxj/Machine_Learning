# -*- coding: utf-8 -*-
##################本程序实现一个简单的朴素贝叶斯垃圾邮件过滤器########
import os
import jieba
import pickle
#import shutil
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.externals import joblib


file_path=os.path.join('/')#该路径用于存储原始邮件文本,以及由此构建的词表
new_file_path=os.path.join('/')#该路径用于存储后续收集到的新邮件样本
train_data_path=os.path.join('/')#用于存储处理好的训练数据
test_data_path=os.path.join('/')#测试文本

"""原始邮件格式：第一行为标记，自第二行及之后内容为邮件文本"""

###############################数据预处理##########################
def Punctuation():
    s='·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？\，\。\、\`\~\!\@\#\$\%\^\&\*\(\)\_\+\-\=\[\]\{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?'
    return set(s.split('\\'))
def word_segmentation(file,skip=False):
    punctuation=Punctuation()
    words=set()

    if skip :
        next(file)
    line=file.readline() 
    while line:
        tmp=[s for s in jieba.lcut(line) if len(s)>0 and s not in punctuation ]
        #words|=set(tmp)
        for word in tmp:
            words.add(word)
        line=file.readline()
    return words
    
    pass
"""读取某个目录下的邮件文件，构建训练集"""
def get_vocabulary():
    #punctuation=Punctuation()
    m=0
    vocabulary=set()
    path = Path(file_path)
    files = path.iterdir()
    for item in files:
        if item.is_file():
            m+=1
            with open(item, 'r') as f:
                vocabulary|=word_segmentation(f,skip=True)
                """
                lines = f.readlines()
                for i in range(1,len(lines)):
                    words=[s for s in jieba.lcut(lines[i]) if len(s)>0 and s not in punctuation ]
                    #vocabulary|=set(words)
                    for word in words:
                        vocabulary.add(word)
                """
    print('Vocabulary creation completed')
    with open(os.path.join(file_path,'/vocabulary/vocabulary.txt'),'wb') as f:
        pickle.dump(list(vocabulary),f)
    print('Save  vocabulary completed')
    return list(vocabulary),m,len(vocabulary)

def get_train_data():
    #punctuation=Punctuation()
    vocabulary,m,n=get_vocabulary()
    X=np.zeros((m, n))
    Y=np.zeros(m)
    k=0
    basepath = Path(file_path)
    files = basepath.iterdir()
    for item in files:
        if item.is_file():
            #file_words=set()
            with open(item, 'r') as f:
                line = f.readline()
                if '垃圾' in line or 'spam' in line :
                    Y[k]=1.
                file_words=word_segmentation(f,skip=True)
            if file_words:
                for i in range(n):
                    if vocabulary[i] in file_words:
                        X[k][i]=1.
            k+=1
    return X,Y
def save_train_data(X,Y):
    #X,Y=get_train_data()
    np.savetxt(os.path.join(train_data_path,'/trainX.txt'),X,delimiter=',')
    np.savetxt(os.path.join(train_data_path,'/trainY.txt'),Y,delimiter=',')
    print('Training data saving is completed')
                
"""加载数据用于训练模型"""
def load_train_data():
    X=np.loadtxt(os.path.join(train_data_path,'/trainX.txt'),delimiter=',')
    Y=np.loadtxt(os.path.join(train_data_path,'/trainY.txt'),delimiter=',')
    return X,Y

"""数据预处理函数用于将一封邮件转换为训练数据格式"""
def data_preprocess(file_name):
    #加载词表
    with open(os.path.join(file_path,'/vocabulary/vocabulary.txt'),'rb') as f:
        vocabulary=pickle.load(f)
    with open(file_name,'rb') as f:
        words=word_segmentation(f)
        X=np.zeros(len(vocabulary))
        for i in range(len(vocabulary)):
            if vocabulary[i] in words:
                vocabulary[i]=1.
    return X
        
###############################模型训练#############################
class NaiveBayes(object):
    def __init__(self):
        #记录类别值
        self.class_=None
        #类先验概率
        self.prior_=None
        #所有体征的类条件概率对数值
        self.likelihood_=None
        #各类对应的样本数目
        self.class_count_=None
    
    def fit(self,X,Y):
        self.class_=np.unique(Y)
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        #初始化类先验概率，全为0
        #self.prior_=np.zeros(n_classes,dtype=np.float64)
        #初始化对数似然，全为0
        self.likelihood_=np.zeros((n_classes,n_features))
        #初始化类别计数，全为0
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        
        #计算属性
        for y_i in self.class_:
            i=self.likelihood_.searchsorted(y_i)
            X_i = X[Y == y_i, :] #找出训练数据X中所有第y_i类样本
            N_i = X_i.shape[0] #第y_i类样本的个数
        
            #更新样本个数
            self.class_count_[i] += N_i
            
            #计算似然
            _likelihood=self.likelihood_.searchsorted[i,:]
            _likelihood+=np.sum(X_i,axis=0)/N_i
        #更新类先验   
        if self.prior_ is None:   
            self.prior_= self.class_count_ / self.class_count_.sum()   
        return self
        
        
    def _joint_log_likelihood(self, X):
        """计算非归一化的对数后验概率.
           对X的每一行，对所有可能类别，计算log P(c) + log P(x|c)，
           结果形如 [n_classes, n_samples].
           返回值是其转置[ n_samples，n_classes]. 
        """  
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.prior_[i]) #第i类先验对应的对数log P(i)
            P_ji=np.dot(np.log(self.likelihood_[i,:]),X.T)#计算Sum log P(x|i)
            joint_log_likelihood.append(jointi + P_ji) #将log P(i) + Sum log P(x|i)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
    
    def predict(self, X):
        """
        返回预测分类结果.
        ----------
        输入形如 [n_samples, n_features]：待预测样本按行排列
        -------
        输出形如 [n_samples]：预测结果列向量
        """
        #就是取对数似然最大的类别
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        返回对测试向量X计算的对数概率估计
        ----------
        输入形如 [n_samples, n_features]    
        -------
        输出形如 [n_samples, n_classes]
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        返回估计概率值.
        """
        return np.exp(self.predict_log_proba(X))
    
    def model_save(self,path,name):
        joblib.dump(self,os.path.join(path,name))
        print('model saved')

RE_TRAIN=0.5
train_model_path=os.path.join('/')#用于存储训练好的模型
"""该函数用于模型训练与保存"""
def train():
    old,new=0,0
    
    old_path = Path(file_path)
    old_files = old_path.iterdir()
    
    new_path=Path(new_file_path)
    new_files = new_path.iterdir()
    
    for item in old_files:
        if item.is_file():
            old+=1
    for item in new_files:
        if item.is_file():
            old+=1
    #重新训练模型
    if new*RE_TRAIN>=old:
        pass
    #开始预处理训练数据
    trainX,trainY=get_train_data()
    """
    save_train_data(trainX,trainY)
    trainX,trainY=load_train_data()
    """
    model=NaiveBayes()
    model.fit(trainX,trainY)
    model.model_save(train_model_path,'naivebayes.pkl')
    print('The model training is completed')

"""该函数用于分类预测"""  
def predict(file_):
    predict_data=data_preprocess(file_)
    #加载模型
    model=joblib.load(os.path.join(train_model_path,'naivebayes.pkl'))
    res=model.predict(predict_data)
    return res
    
    
    



