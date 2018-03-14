# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:27:30 2018

@author: shrey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from math import sqrt
from sklearn.metrics import mean_squared_error

def pca(data,std=True):# method for PCA
    x_array=np.array(data)

    data_x=x_array.astype(float)

    if std==True:
        
            data_mean=data_x.mean(axis=0)
    
            data_mc=data_x-data_mean
            pca_data=data_mc
    else:
        pca_data=data_x
    
    data_cov=np.cov(pca_data.T)
        
    e_val,e_vec=np.linalg.eig(data_cov) 
 
    aa =e_val.argsort()[::-1]   
    e_val=e_val[aa]
    e_vec=e_vec[:,aa]
    scores = np.matmul(pca_data, e_vec[:,[0,1]])

    var_sum=e_val.sum()
    
    e_var=e_val/var_sum
    plt.plot(e_var)
    return e_val,e_vec,scores,aa

data=pd.read_excel('/home/shrey/Desktop/project/project_data/data_english/total_diary_4579.ko.en.xlsx',sep=',',header=2)

mig_data=data[['Become sensitive to light','Light / Sound Sensitive','Migraine presence','stress','Sleeping too much','Lack of sleep','Exercise','Not to Exercise',
'Physical fatigue','Weather / temperature changes','Excessive sunlight','noise','Excessive drinking','Irregular eating (fasting, etc.)','Excessive caffeine',
'Excessive smoking','Chocolate Cheesecake','Travel']]

mig_data=mig_data.reset_index(drop=True)

mig_data=mig_data[:4579]

mig_data.dropna(inplace=True)

plt.scatter(mig_data['Migraine presence'],np.zeros((1098)))

dat=np.array(mig_data)

e_val,e_vec,scores,aa=pca(dat,std=False)

learning_rate = 0.005
num_steps = 1000
batch_size = 100

display_step = 1000
examples_to_show = 10

num_hidden_1 = 20
num_hidden_2 = 10
num_input = 18

X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    error=[]
    for i in range(1, num_steps+1):
        #train_x=np.load('train_x.npy')
        train_x=dat
        j=0
        while j<len(train_x):
           start=j
           end=j+batch_size
           batch_x=train_x[start:end]
           _,l=sess.run([optimizer,loss],feed_dict={X:batch_x})
           j+=batch_size
           
        print('loss at ',i,' epoch is',l)
        error.append(l)  
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'iteration')
    ax.set_ylabel(r'error')  
    ax.plot(error)     
    ax.show()
        
            


