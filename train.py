from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_config', 'Model string.')  # 'gcn_config', 'gcn_cheby', 'dense'

flags.DEFINE_string('learning_rate', '0.0025', 'Initial learning rate.')
flags.DEFINE_string('adj_index', '56','adj_index')

flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
lr = float(FLAGS.learning_rate)
adj_index = int(FLAGS.adj_index)
print("lr: ", lr)
print("adj index: ",adj_index)


pretrain = True
if not pretrain:
    adj = np.load('adj_'+str(int(FLAGS.adj_index))+'.npy')
    adj = adj - np.identity(adj.shape[0])
    print("num of edges * 2:", np.count_nonzero(adj))
    print(adj)
    adj = sp.coo_matrix(adj, dtype=float)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn_config':
    support = np.array(preprocess_adj(adj), dtype=float)
    print("number of edges * 2 + diag", np.count_nonzero(adj.toarray()))
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': support,
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
}

model = model_func(placeholders=placeholders, input_dim=features[2][1], logging=False)
mytrainer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.learning_rate))
loss = model.loss
opt_op = None


print("cosnstruct pretrain")
grads = mytrainer.compute_gradients(loss)
opt_op = mytrainer.apply_gradients(grads)


# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())


# Define model evaluation function
def evaluate(features, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

cost_val = []
total_time = 0
total_iter = 0
# Train model
# cur_adj_out = sess.run(model.vars['gcn_config/graphconvolution_1_adj_vars/adj:0'])
# w1 = sess.run(model.vars['gcn_config/graphconvolution_1_vars/weights_0:0'])
print("pretrain or retrain withouting ADMM")
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([opt_op, loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    total_time = total_time + (time.time() - t)
    total_iter = total_iter + 1
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break
# cur_adj_in = sess.run(model.vars['gcn_config/graphconvolution_1_adj_vars/adj:0'])
# print("is equal of two adj", isequal(cur_adj_in, cur_adj_out))
# w2 = sess.run(model.vars['gcn_config/graphconvolution_1_vars/weights_0:0'])
# print("is equal of two w", isequal(w1, w2)

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, y_test, test_mask, placeholders)
print("Finally Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

print("average train time: ", total_time/total_iter)
#Test sparse adj
cur_adj1 = sess.run(model.vars['gcn/graphconvolution_1_adj_vars/adj:0'])
cur_adj2 = sess.run(model.vars['gcn/graphconvolution_2_adj_vars/adj:0'])
print("finish L1 training, num of edges *2 + diag in adj1:", np.count_nonzero(cur_adj1))
print("finish L1 training, num of edges * 2 + diag in adj2:", np.count_nonzero(cur_adj2))
print("adj1", cur_adj1)
print("symmetry result adj1: ", testsymmetry(cur_adj1))
print("symmetry result adj2: ", testsymmetry(cur_adj2))
print("is equal of two adj", isequal(cur_adj1, cur_adj2))


