import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import random, os
import cv2
import sys
import itertools


img_files = glob('/home/scg/PycharmProjects/untitled44/train_test/*/*.bmp')

train_data_path = []
test_data_path = []

def Path_load(g_path):
     digit_path = glob(g_path)
     digit_path.sort()
     temp = glob(g_path)
     train_data_path.append(temp[0:1600])
     test_data_path.append(temp[1600:2000])
     print((train_data_path))
     print("\n")
     return train_data_path, test_data_path


#IMAGES = tf.app.images.IMAGES
#IMAGES.height = 32
#IMAGES.width = 32
#IMAGES.depth = 3

#def convert_images(sess):
#    value = tf.read_file(train_data_path)
#    decoded_image = tf.image.decode_bmp(value, channels=IMAGES.depth)
#    resized_image = tf.image.resize_images(decoded_image, IMAGES.height, IMAGES.width)
#    resized_image = tf.cast(resized_image, tf.uint8)

#    plt.imshow(np.reshape(IMAGES.data, [32, 32, IMAGES.depth]))
#    plt.show()


def main():
    for i in range(10):
        g_path = '/home/scg/PycharmProjects/untitled44/train_test/%d/*.bmp'%i
        train_path, test_path = Path_load(g_path)

    train_data_path = list(itertools.chain.from_iterable(train_path))
    test_data_path = list(itertools.chain.from_iterable(test_path))
    print(len(train_data_path))
    print(len(test_data_path))
    random.shuffle(train_data_path)
    batch_size = 128
    class_num = 10

    #train
    for i in range(int(len(train_data_path)/batch_size)):
        images = [cv2.imread(k) for k in train_data_path[batch_size*i: batch_size*(i+1)]]
        labels = [label[48] for label in train_data_path[batch_size*i: batch_size*(i+1)]]

        labels_one_hot = np.zeros((batch_size,class_num),dtype=np.uint8)
        for k in range(len(labels)):
            print(labels[k])
            labels_one_hot[k][int(labels[k])] = 1
        print(labels_one_hot[0])

    np.save("train_data.npy", images)
    np.save("train_label.npy", labels)




    #test
    for n in range(int(len(test_data_path)/batch_size)):
        images1 = [cv2.imread(k) for k in test_data_path[batch_size*n: batch_size*(n+1)]]
        labels1 = [label[48] for label in test_data_path[batch_size*n: batch_size*(n+1)]]

    labels_one_hot = np.zeros((batch_size, class_num), dtype=np.uint8)
    for z in range(len(labels)):
        print(labels[z])
        labels_one_hot[z][int(labels[z])] = 1
    print(labels_one_hot[2])

    np.save("test_data.npy", images1)
    np.save('test_label.npy', labels1)

    #test_n = 0
    #plt.title(labels[test_n])
    #plt.imshow(images[test_n, :, 0])
    #plt.show()



    x = tf.placeholder(tf.float32, [None, 1024])
    x_img = tf.reshape(x, [-1, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    L1 = tf.nn.conv2d(x_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    L2_flat = tf.reshape(L2, [-1, 8*8*64])
    W3 = tf.get_variable("W3", shape=[8*8*64, 10], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L2_flat, W3) + b




    max_epochs = 20
    network = SimpleConvNET()
    iters_num = 10000
    train_size = images.shape[0]
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    print(iter_per_epoch)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = images[batch_mask]
        t_batch = labels[batch_mask]
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)
        for key in ('W1', 'b1', 'W2', 'b2'): network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % iter_per_epoch == 0:
            print(images.shape)
            train_acc = network.accuracy(images, labels)
            test_acc = network.accuracy(images1, labels1)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))






   # learning_rate = 0.001
   # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
   # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #training_epochs = 10
    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())

    #for epoch in range(training_epochs):
     #   avg_cost = 0
      #  total_batch = int(len(train_data_path) / batch_size)

       # for a in range(total_batch):
        #    start = ((a + 1)*batch_size)-batch_size
         #   end = ((a + 1)*batch_size)
          #  x_batch = images[start:end]
           # y_batch = labels[start:end]
            #feed_dict = {x: x_batch, y: y_batch}
            #c, _=sess.run([cost, optimizer], feed_dict=feed_dict)
            #avg_cost += c/total_batch

        #print('Epoch is: ', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))
    #print('FINISHED')

    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print('Accuracy is: ', sess.run(accuracy, feed_dict={x:test_input, y:test_label}))


if __name__ == '__main__':
        main()







