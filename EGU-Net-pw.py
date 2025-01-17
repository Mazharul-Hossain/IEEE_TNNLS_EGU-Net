# import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio
import scipy.io as sio
from tf_utils import random_mini_batches, convert_to_one_hot
from tensorflow.python.framework import ops


def create_placeholders(n_x1, n_x2, n_y):
    keep_prob = tf.placeholder(tf.float32)
    isTraining = tf.placeholder_with_default(True, shape=())
    x_pure = tf.placeholder(tf.float32, [None, n_x1], name="x_pure")
    x_mixed = tf.placeholder(tf.float32, [None, n_x2], name="x_mixed")
    y = tf.placeholder(tf.float32, [None, n_y], name="Y")
    return x_pure, x_mixed, y, isTraining, keep_prob


def initialize_parameters():
    tf.set_random_seed(1)

    x_w1 = tf.get_variable("x_w1", [224, 256], initializer=tf.contrib.layers.xavier_initializer())
    x_b1 = tf.get_variable("x_b1", [256], initializer=tf.zeros_initializer())

    x_w2 = tf.get_variable("x_w2", [256, 128], initializer=tf.contrib.layers.xavier_initializer())
    x_b2 = tf.get_variable("x_b2", [128], initializer=tf.zeros_initializer())

    x_w3 = tf.get_variable("x_w3", [128, 32], initializer=tf.contrib.layers.xavier_initializer())
    x_b3 = tf.get_variable("x_b3", [32], initializer=tf.zeros_initializer())

    x_w4 = tf.get_variable("x_w4", [32, 5], initializer=tf.contrib.layers.xavier_initializer())
    x_b4 = tf.get_variable("x_b4", [5], initializer=tf.zeros_initializer())

    x_dew1 = tf.get_variable("x_dew1", [5, 32], initializer=tf.contrib.layers.xavier_initializer())
    x_deb1 = tf.get_variable("x_deb1", [32], initializer=tf.zeros_initializer())

    x_dew2 = tf.get_variable("x_dew2", [32, 128], initializer=tf.contrib.layers.xavier_initializer())
    x_deb2 = tf.get_variable("x_deb2", [128], initializer=tf.zeros_initializer())

    x_dew3 = tf.get_variable("x_dew3", [128, 256], initializer=tf.contrib.layers.xavier_initializer())
    x_deb3 = tf.get_variable("x_deb3", [256], initializer=tf.zeros_initializer())

    x_dew4 = tf.get_variable("x_dew4", [256, 224], initializer=tf.contrib.layers.xavier_initializer())
    x_deb4 = tf.get_variable("x_deb4", [224], initializer=tf.zeros_initializer())

    return {"x_w1": x_w1,
            "x_b1": x_b1,
            "x_w2": x_w2,
            "x_b2": x_b2,
            "x_w3": x_w3,
            "x_b3": x_b3,
            "x_w4": x_w4,
            "x_b4": x_b4,
            "x_dew1": x_dew1,
            "x_deb1": x_deb1,
            "x_dew2": x_dew2,
            "x_deb2": x_deb2,
            "x_dew3": x_dew3,
            "x_deb3": x_deb3,
            "x_dew4": x_dew4,
            "x_deb4": x_deb4}


def my_network(x_pure, x_mixed, parameters, isTraining, keep_prob, momentum=0.9):
    with tf.name_scope("x_layer_1"):
        x_pure_z1 = tf.matmul(x_pure, parameters['x_w1']) + parameters['x_b1']
        x_pure_z1_bn = tf.layers.batch_normalization(x_pure_z1, axis=1, momentum=momentum, training=isTraining,
                                                     name='l1')
        x_pure_z1_do = tf.nn.dropout(x_pure_z1_bn, keep_prob)
        x_pure_a1 = tf.nn.tanh(x_pure_z1_do)

        x_mixed_z1 = tf.matmul(x_mixed, parameters['x_w1']) + parameters['x_b1']
        x_mixed_z1_bn = tf.layers.batch_normalization(x_mixed_z1, axis=1, momentum=momentum, training=isTraining,
                                                      name='l1', reuse=True)
        x_mixed_z1_do = tf.nn.dropout(x_mixed_z1_bn, keep_prob)
        x_mixed_a1 = tf.nn.tanh(x_mixed_z1_do)

    with tf.name_scope("x_layer_2"):
        x_pure_z2 = tf.matmul(x_pure_a1, parameters['x_w2']) + parameters['x_b2']
        x_pure_z2_bn = tf.layers.batch_normalization(x_pure_z2, axis=1, momentum=momentum, training=isTraining,
                                                     name='l2')
        x_pure_a2 = tf.nn.tanh(x_pure_z2_bn)

        x_mixed_z2 = tf.matmul(x_mixed_a1, parameters['x_w2']) + parameters['x_b2']
        x_mixed_z2_bn = tf.layers.batch_normalization(x_mixed_z2, axis=1, momentum=momentum, training=isTraining,
                                                      name='l2', reuse=True)
        x_mixed_a2 = tf.nn.tanh(x_mixed_z2_bn)

    with tf.name_scope("x_layer_3"):
        x_pure_z3 = tf.matmul(x_pure_a2, parameters['x_w3']) + parameters['x_b3']
        x_pure_z3_bn = tf.layers.batch_normalization(x_pure_z3, axis=1, momentum=momentum, training=isTraining,
                                                     name='l3')
        x_pure_a3 = tf.nn.relu(x_pure_z3_bn)

        x_mixed_z3 = tf.matmul(x_mixed_a2, parameters['x_w3']) + parameters['x_b3']
        x_mixed_z3_bn = tf.layers.batch_normalization(x_mixed_z3, axis=1, momentum=momentum, training=isTraining,
                                                      name='l3', reuse=True)
        x_mixed_a3 = tf.nn.relu(x_mixed_z3_bn)

    with tf.name_scope("x_layer_4"):
        x_pure_z4 = tf.add(tf.matmul(x_pure_a3, parameters['x_w4']), parameters['x_b4'])
        abundances_pure = tf.nn.softmax(x_pure_z4)

        x_mixed_z4 = tf.add(tf.matmul(x_mixed_a3, parameters['x_w4']), parameters['x_b4'])
        abundances_mixed = tf.nn.softmax(x_mixed_z4)

    with tf.name_scope("x_de_layer_1"):
        x_mixed_de_z1 = tf.matmul(abundances_mixed, parameters['x_dew1']) + parameters['x_deb1']
        x_mixed_de_z1_bn = tf.layers.batch_normalization(x_mixed_de_z1, axis=1, momentum=momentum, training=isTraining)
        x_mixed_de_a1 = tf.nn.sigmoid(x_mixed_de_z1_bn)

    with tf.name_scope("x_de_layer_2"):
        x_mixed_de_z2 = tf.matmul(x_mixed_de_a1, parameters['x_dew2']) + parameters['x_deb2']
        x_mixed_de_z2_bn = tf.layers.batch_normalization(x_mixed_de_z2, axis=1, momentum=momentum, training=isTraining)
        x_mixed_de_a2 = tf.nn.sigmoid(x_mixed_de_z2_bn)

    with tf.name_scope("x_de_layer_3"):
        x_mixed_de_z3 = tf.matmul(x_mixed_de_a2, parameters['x_dew3']) + parameters['x_deb3']
        x_mixed_de_z3_bn = tf.layers.batch_normalization(x_mixed_de_z3, axis=1, momentum=momentum, training=isTraining)
        x_mixed_de_a3 = tf.nn.sigmoid(x_mixed_de_z3_bn)

    with tf.name_scope("x_de_layer_4"):
        x_mixed_de_z4 = tf.matmul(x_mixed_de_a3, parameters['x_dew4']) + parameters['x_deb4']
        x_mixed_de_z4_bn = tf.layers.batch_normalization(x_mixed_de_z4, axis=1, momentum=momentum, training=isTraining)
        x_mixed_de_a4 = tf.nn.sigmoid(x_mixed_de_z4_bn)

    l2_loss = tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2']) + tf.nn.l2_loss(
        parameters['x_w3']) + tf.nn.l2_loss(parameters['x_w4']) + tf.nn.l2_loss(parameters['x_dew1']) + tf.nn.l2_loss(
        parameters['x_dew2']) + tf.nn.l2_loss(
        parameters['x_dew3']) + tf.nn.l2_loss(parameters['x_dew4'])

    return x_pure_z4, abundances_mixed, x_mixed_de_a4, l2_loss, abundances_pure, abundances_mixed


def my_network_optimization(y_est, y_re, r1, r2, l2_loss, reg, learning_rate, global_step):
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_est, labels=y_re)) \
               + reg * l2_loss + 1 * tf.reduce_mean(tf.pow(r1 - r2, 2))

    with tf.name_scope("optimization"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        optimizer = tf.group([optimizer, update_ops])

    return cost, optimizer


def train_my_network(x_pure_set, x_mixed_set, y_train, y_test, learning_rate_base=0.1, beta_reg=0.005, num_epochs=200,
                     minibatch_size=8000, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1
    (m, n_x1) = x_pure_set.shape
    (m1, n_x2) = x_mixed_set.shape
    (m, n_y) = y_train.shape

    costs = []
    costs_dev = []
    train_acc = []
    val_acc = []
    plot_epoch = []
    plot_lr = []

    x_train_pure, x_train_mixed, y, isTraining, keep_prob = create_placeholders(n_x1, n_x2, n_y)

    parameters = initialize_parameters()

    with tf.name_scope("network"):
        x_pure_layer, x_mixed_layer, x_mixed_de_layer, l2_loss, abundances_pure, abundances_mixed = my_network(
            x_train_pure, x_train_mixed, parameters, isTraining, keep_prob)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate_base, global_step, m / minibatch_size, 0.99)

    with tf.name_scope("optimization"):
        cost, optimizer = my_network_optimization(x_pure_layer, y, x_mixed_de_layer, x_train_mixed, l2_loss, beta_reg,
                                                  learning_rate, global_step)

    with tf.name_scope("metrics"):

        accuracy = tf.losses.absolute_difference(labels=y, predictions=abundances_pure)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Do the training loop
        for epoch in range(1, num_epochs + 1):
            epoch_cost = 0.  # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m1 / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x_pure_set, x_mixed_set, y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (batch_x1, batch_x2, batch_y) = minibatch
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy],
                                                            feed_dict={x_train_pure: batch_x1, x_train_mixed: batch_x2,
                                                                       y: batch_y, isTraining: True, keep_prob: 0.9})
                epoch_cost += minibatch_cost
                epoch_acc += minibatch_acc

            epoch_cost_f = epoch_cost / (num_minibatches + 1)
            epoch_acc_f = epoch_acc / (num_minibatches + 1)

            if print_cost is True and epoch % 5 == 0:

                re, abund, epoch_cost_dev, epoch_acc_dev, lr = sess.run(
                    [x_mixed_de_layer, abundances_pure, cost, accuracy, learning_rate],
                    feed_dict={x_train_pure: x_mixed_set,
                               x_train_mixed: x_mixed_set, y: y_test,
                               isTraining: True, keep_prob: 1})

                costs.append(epoch_cost_f)
                train_acc.append(epoch_acc_f)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)
                plot_epoch.append(epoch)
                plot_lr.append(lr)

                if epoch % 20 == 0:
                    print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (
                        epoch, epoch_cost_f, epoch_cost_dev, epoch_acc_f, epoch_acc_dev))

        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # plot the cost
        axes[0].plot(plot_epoch, np.squeeze(costs))
        axes[0].plot(plot_epoch, np.squeeze(costs_dev))
        axes[0].set_ylabel('cost')
        axes[0].set_xlabel('iterations (per tens)')
        axes[0].set_title("Training-Val Cost")

        # plot the learning rate
        axes[1].plot(plot_epoch, plot_lr)
        axes[1].set_ylabel('Learning rate')
        axes[1].set_xlabel('iterations (per tens)')
        axes[1].set_title("Training-LR")

        # plot the accuracy
        axes[2].plot(plot_epoch, np.squeeze(train_acc))
        axes[2].plot(plot_epoch, np.squeeze(val_acc))
        axes[2].set_ylabel('accuracy')
        axes[2].set_xlabel('iterations (per tens)')
        axes[2].set_title("Training-Val Absolute Error")

        plt.savefig('Training-Val-Report.png', bbox_inches='tight')
        plt.show()
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        return parameters, val_acc, re.squeeze(), abund


def main():
    Pure_TrSet = scio.loadmat('Data/Pure_TrSet.mat')
    Mixed_TrSet = scio.loadmat('Data/Mixed_TrSet.mat')

    TrLabel = scio.loadmat('Data/TrLabel.mat')
    TeLabel = scio.loadmat('Data/TeLabel.mat')

    Pure_TrSet = Pure_TrSet['Pure_TrSet']
    Mixed_TrSet = Mixed_TrSet['Mixed_TrSet']
    TrLabel = TrLabel['TrLabel']
    TeLabel = TeLabel['TeLabel']

    Y_train = TrLabel
    Y_test = TeLabel

    parameters, val_acc, high_res, abund = train_my_network(Pure_TrSet, Mixed_TrSet, Y_train, Y_test)
    sio.savemat('abund.mat', {'abund': abund})
    sio.savemat('hi_res.mat', {'hi_res': high_res})


if __name__ == "__main__":
    main()
