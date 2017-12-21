import numpy as np
import sys
import random
import tensorflow as tf
from inp_data import *

ntrain, ntest, train_in, train_out, test_in, test_out = read_raw_data()

print(ntrain, ntest)
print(train_in.shape, train_out.shape)
print(test_in.shape, test_out.shape)



inp_size = train_in.shape[1]
out_size = train_out.shape[1]

nhidden1 = 256
nhidden2 = 128


# define placeholders
x = tf.placeholder(tf.float32, [None, inp_size])
y = tf.placeholder(tf.float32, [None, out_size])

# hyperparameters
epochs = 5000
batch_size = 128
learning_rate = 1e-4


seed = 258


# weights and bias (2 layers)
weights = {
    'hidden1': tf.Variable(tf.random_normal([inp_size, nhidden1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([nhidden1, nhidden2], seed=seed)),
    'output': tf.Variable(tf.random_normal([nhidden2, out_size], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([nhidden1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([nhidden2], seed=seed)),
    'output': tf.Variable(tf.random_normal([out_size], seed=seed))
}


# define neural net
hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer_1 = tf.nn.relu(hidden_layer_1)
hidden_layer_1 = tf.nn.dropout(hidden_layer_1, keep_prob=0.8)
hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['hidden2']), biases['hidden2'])
hidden_layer_2 = tf.nn.relu(hidden_layer_2)
hidden_layer_2 = tf.nn.dropout(hidden_layer_2, keep_prob=0.8)
output_layer = tf.matmul(hidden_layer_2, weights['output']) + biases['output']



# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))


# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# train the neural network
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train_in.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x = train_in[i*total_batch:(i+1)*total_batch,:]
            batch_y = train_out[i*total_batch:(i+1)*total_batch,:] 
  
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print("\nTraining complete!")
    

    # find predictions on validation set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print("Validation Accuracy:", accuracy.eval({x: test_in.reshape(-1, inp_size), y: test_out}))
    
 
    saver = tf.train.Saver()
    saver.save(sess, "ckpt/weights", meta_graph_suffix='meta', write_meta_graph=True)

