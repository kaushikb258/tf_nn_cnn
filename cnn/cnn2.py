import tensorflow as tf
import numpy as np
from read_data import *
import progressbar

img_size = 224
ntrain, nval, x_train, y_train, x_val, y_val = read_imgs(img_size)

print(ntrain, nval)



# Network Parameters
num_classes = 2
learning_rate = 1e-4
batch_size = 16
epochs = 100


# placeholders
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
kp =  tf.placeholder(tf.float32)


# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # 3x3 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    # fully connected, N*N*384 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.random_normal([1*256, 64])),
    # 512 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([64, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bfc1': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}



# Create the convolutional neural network
conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1,2,2,1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, biases['bc1'])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1,2,2,1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, biases['bc2'])
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1,2,2,1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, biases['bc3'])
conv3 = tf.nn.relu(conv3)
conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1,2,2,1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, biases['bc4'])
conv4 = tf.nn.relu(conv4)
conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

fc1 = tf.reshape(conv4, [-1, weights['wfc1'].get_shape().as_list()[0]])
fc1 = tf.matmul(fc1, weights['wfc1'])
fc1 = tf.add(fc1, biases['bfc1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob=kp)

output_layer = tf.matmul(fc1, weights['out'])
output_layer = tf.add(output_layer, biases['out'])


# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))


# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# train the convolutional neural network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(x_train.shape[0]/batch_size)
        bar = progressbar.ProgressBar()
        for i in bar(range(total_batch)):

            batch_x = np.zeros((batch_size,img_size,img_size,3))
            batch_y = np.zeros((batch_size,2))

            batch_x[:,:,:,:] = x_train[i*batch_size:(i+1)*batch_size,:,:,:]
            batch_y[:,:] = y_train[i*batch_size:(i+1)*batch_size,:] 
  
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y, kp: 1.0})
            
            avg_cost += c / total_batch
            
        print("Epoch:", (epoch+1), "cost =", "{:.3f}".format(avg_cost))
    
        # find predictions on validation set
        if (epoch > 1 and epoch%10 == 0): 
          ncorrect = 0
          for i in range(x_val.shape[0]):
           xx = np.zeros((1,img_size,img_size,3))
           yy = np.zeros((1,2))
           xx[0,:,:,:] = x_val[i,:,:,:]
           yy[0,:] = y_val[i,:] 

           classification = sess.run(tf.nn.softmax(output_layer), feed_dict={x: xx, kp: 1.0})
           if (np.argmax(classification) == np.argmax(yy[0,:])):     
             ncorrect += 1         

           # find predictions on validation set
           #pred_temp = tf.equal(tf.argmax(tf.nn.softmax(output_layer), 1), tf.argmax(yy, 1))
           #accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
           #ncorrect += accuracy.eval({x: xx, kp: 1.0})
     
          print("validation: number correct/total: ", int(ncorrect), "/", nval)
 
    print("\nTraining complete!")

    saver = tf.train.Saver()
    saver.save(sess, "ckpt/weights", meta_graph_suffix='meta', write_meta_graph=True)



