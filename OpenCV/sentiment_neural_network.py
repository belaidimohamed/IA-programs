import tensorflow as tf
import os
import numpy as np
os.chdir(r'C:\Users\user16\Downloads\Video\deep learning')
from create_sentiment_features_set import create_feature_sets_and_labels
train_x,train_y,test_x,test_y=create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1=500
n_nodes_hl2=500  #nodes of hidden layers
n_nodes_hl3=500

n_classes=2
batch_size=100  # number of features to work on

x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')

def neural_network_model(data):
    hiddenlayer_1={'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hiddenlayer_2={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hiddenlayer_3={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    outputlayer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(data,hiddenlayer_1["weights"]),hiddenlayer_1['biases'])
    l1=tf.nn.relu(l1) #activation function

    l2=tf.add(tf.matmul(l1,hiddenlayer_2["weights"]),hiddenlayer_2['biases'])
    l2=tf.nn.relu(l2)

    l3=tf.add(tf.matmul(l2,hiddenlayer_3["weights"]),hiddenlayer_3['biases'])
    l3=tf.nn.relu(l3)

    output=tf.matmul(l3,outputlayer["weights"])+outputlayer['biases']
    return output
def train_neural_network(x):
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    epchos=11
    with tf.Session() as sess :
        sess.run(tf.initialize_all_variables())
        for i in range(epchos) :
            epchos_loss=0

            j=0
            while j<len(train_x):
                start=j
                end=j+batch_size
                batch_x=np.array(train_x[start:end])
                batch_y=np.array(train_y[start:end])
                print(batch_x,'-----',batch_y)
                _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epchos_loss+=c
                j+=batch_size
            print("Epoch",i,'completed out of ',epchos,'loss:',epchos_loss)

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy',accuracy.eval({x:test_x,y:test_y}))
train_neural_network(x)


