# import tensorflow as tf
# hello=tf.constant("shit man")
# sess=tf.Session()
# print(sess.run(hello))
import tensorflow as tf 
import os
os.chdir(r'C:\Users\user16\Downloads\Video\deep learning')
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('tmp/data/',one_hot=True)

n_nodes_hl1=500
n_nodes_hl2=500  #nodes of hidden layers
n_nodes_hl3=500

n_classes=10
batch_size=100  # number of features to work on

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural_network_model(data):	
    hiddenlayer_1={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
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
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x1,y1=mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,cost],feed_dict={x:x1,y:y1})
                epchos_loss+=c
            print("Epoch",i,'completed out of ',epchos,'loss:',epchos_loss)
        
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
train_neural_network(x)


    