import tensorflow as tf
import numpy as np
import importlib
import argparse
import h5py
import os,sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from tensorflow.contrib.data import Iterator
from datagenerator import PointCloudDataGenerator
import tf_util

train_tfrecord_path = 'data/modelnet40_cls_train_1024.tfrecords'
eval_tfrecord_path = 'data/modelnet40_cls_test_1024.tfrecords'
NUM_TRAIN_EXAMPLES = 9840
NUM_EVAL_EXAMPLES = 2468

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg_fast', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--measure_time', default='true', help='Whether to measure running time [default: true]')
parser.add_argument('--overwrite_log', default='false', help='Whether to overwrite log [default: false]')
parser.add_argument('--resume', default='false', help='Whether to resume from checkpoints in log_dir')
parser.add_argument('--resume_epoch', default='None', help='epoch to resume')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MEASURE_TIME = FLAGS.measure_time
OVERWRITE_LOG = FLAGS.overwrite_log
RESUME = FLAGS.resume == 'true'
RESUME_EPOCH = FLAGS.resume_epoch

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

if RESUME:
  assert os.path.exists(LOG_DIR)
  
if not RESUME:
  if os.path.exists(LOG_DIR) and OVERWRITE_LOG == 'false': LOG_DIR = LOG_DIR +'_'
  if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

  os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
  os.system('cp %s %s' % (os.path.realpath(__file__), LOG_DIR)) # bkp of train procedure
  os.system('cp utils/pointnet_util_fast.py %s' % (LOG_DIR)) 
  os.system('cp utils/tf_util.py %s' % (LOG_DIR)) 
  
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
                      BASE_LEARNING_RATE,  # Base learning rate.
                      batch * BATCH_SIZE,  # Current index into the dataset.
                      DECAY_STEP,          # Decay step.
                      DECAY_RATE,          # Decay rate.
                      staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate        

def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch*BATCH_SIZE,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay

best_accuracy = 0
best_epoch = 0
def train():
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      train_data = PointCloudDataGenerator(train_tfrecord_path, "training", BATCH_SIZE)
      eval_data = PointCloudDataGenerator(eval_tfrecord_path, "inference", BATCH_SIZE)

      # create an reinitializable iterator given the dataset structure
      iterator = Iterator.from_structure(train_data.data.output_types,
                                          train_data.data.output_shapes)
      next_batch = iterator.get_next()
      train_data_init_op = iterator.make_initializer(train_data.data)
      eval_data_init_op = iterator.make_initializer(eval_data.data)
      
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
      is_training_pl = tf.placeholder(tf.bool, shape=())
      
      # Note the global_step=batch parameter to minimize. 
      # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
      batch = tf.get_variable(
        'batch', [],
        initializer=tf.constant_initializer(0), trainable=False)#tf.Variable(0)
      bn_decay = get_bn_decay(batch)
      tf.summary.scalar('bn_decay', bn_decay)

      # Get model and loss 
      pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
      loss = MODEL.get_loss(pred, labels_pl, end_points)
      tf.summary.scalar('loss', loss)

      correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
      accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
      tf.summary.scalar('accuracy', accuracy)
      
      # Get training operator
      learning_rate = get_learning_rate(batch)
      tf.summary.scalar('learning_rate', learning_rate)
      if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
      elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=batch)
      
      # Add ops to save and restore all the variables.
      saver = tf.train.Saver()
          
      # Create a session
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True
      config.log_device_placement = False
      sess = tf.Session(config=config)

      # Add summary writers
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                sess.graph)
      test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

      # Init variables
      init = tf.global_variables_initializer()
      # To fix the bug introduced in TF 0.12.1 as in
      # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
      #sess.run(init)
      sess.run(init, {is_training_pl: True})

      ops = {'pointclouds_pl': pointclouds_pl,
              'labels_pl': labels_pl,
              'is_training_pl': is_training_pl,
              'pred': pred,
              'loss': loss,
              'train_op': train_op,
              'merged': merged,
              'step': batch}

      start_epoch = 0
      if RESUME:
        saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt-"+RESUME_EPOCH))
        log_string('Resume from %s' % os.path.join(LOG_DIR, "model.ckpt-"+RESUME_EPOCH))
        start_epoch = int(RESUME_EPOCH) + 1
      
      for epoch in range(start_epoch, MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
          
        if MEASURE_TIME == 'true':
          start_time = time.time()
        
        sess.run(train_data_init_op)
        train_one_epoch(sess, ops, train_writer, next_batch)
        
        if MEASURE_TIME == 'true':
          duration = time.time() - start_time
          log_string('train one epoch time: %f secs' % duration)
          
          start_time = time.time()
        
        sess.run(eval_data_init_op)
        eval_one_epoch(sess, ops, test_writer, epoch, next_batch)
        
        if MEASURE_TIME == 'true':
          duration = time.time() - start_time
          log_string('test one epoch time: %f secs' % duration)
        
        # Save the variables to disk.
        if epoch % 10 == 0:
          save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
          log_string("Model saved in file: %s" % save_path)
        log_string("")
        
      train_writer.close()
      test_writer.close()
      global best_accuracy, best_epoch
      log_string("Best accuracy %f at epoch %d" % (best_accuracy, best_epoch))
      

def train_one_epoch(sess, ops, train_writer, next_batch):
  """ ops: dict mapping from string to tf ops """
  is_training = True
  
  num_batches = NUM_TRAIN_EXAMPLES // BATCH_SIZE
  
  total_correct = 0
  total_seen = 0
  loss_sum = 0
    
  for batch_idx in range(num_batches):
    points_batch, label_batch = sess.run(next_batch)
    label_batch = np.squeeze(label_batch)
    feed_dict = {ops['pointclouds_pl']: points_batch,
                  ops['labels_pl']: label_batch,
                  ops['is_training_pl']: is_training,}
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
        ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == label_batch)
    total_correct += correct
    total_seen += BATCH_SIZE
    loss_sum += loss_val
      
  log_string('mean loss: %f' % (loss_sum / float(num_batches)))
  log_string('accuracy: %f' % (total_correct / float(total_seen)))
        
        
def eval_one_epoch(sess, ops, test_writer, epoch, next_batch):
  """ ops: dict mapping from string to tf ops """
  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  total_seen_class = [0 for _ in range(NUM_CLASSES)]
  total_correct_class = [0 for _ in range(NUM_CLASSES)]
  
  num_batches = NUM_EVAL_EXAMPLES // BATCH_SIZE
  
  for batch_idx in range(num_batches):
    points_batch, label_batch = sess.run(next_batch)
    label_batch = np.squeeze(label_batch)

    feed_dict = {ops['pointclouds_pl']: points_batch,
                  ops['labels_pl']: label_batch,
                  ops['is_training_pl']: is_training}
    summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
        ops['loss'], ops['pred']], feed_dict=feed_dict)
    test_writer.add_summary(summary, step)

    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == label_batch)
    total_correct += correct
    total_seen += BATCH_SIZE
    loss_sum += (loss_val*BATCH_SIZE)
    for i in xrange(BATCH_SIZE):
      l = label_batch[i]
      total_seen_class[l] += 1
      total_correct_class[l] += (pred_val[i] == l)
  
  eval_accuracy = total_correct / float(total_seen)
  eval_mean_loss = loss_sum / float(total_seen)
  eval_avg_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class,dtype=np.float))
  log_string('eval mean loss: %f' % eval_mean_loss)
  log_string('eval accuracy: %f'% eval_accuracy)
  log_string('eval avg class acc: %f' % eval_avg_class_acc)
  
  global best_accuracy, best_epoch
  if eval_accuracy > best_accuracy:
    best_accuracy = eval_accuracy
    best_epoch = epoch
    log_string('new best accuracy')


if __name__ == "__main__":
  train()
  LOG_FOUT.close()
