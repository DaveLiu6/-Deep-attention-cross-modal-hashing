
from models import *
from data_processing import *
from Custom_function import *

import time

# environmental setting: setting the following parameters based on your experimental environment.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #按照PCI_BUS_ID顺序从0开始排列GPU设备
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# # 对需要进行限制的GPU进行设置
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])


# ===================================================== Data processing ================================================

# images, tags, labels = loading_data_Mir25K(opt.load_data_path)
#
# # ydim: 1000
# ydim = tags.shape[1]
# # num_class: 21
# num_class = labels.shape[1]
#
# X, Y, L = split_data(images, tags, labels)


# MTR25K_dataset = np.load(opt.pre_dataset)
#
#
#
# mean, mean_pixel_ = mean_piexl()
#
#
# # training
# train_L = MTR25K_dataset['train_L']  # labels
# train_x = MTR25K_dataset['train_x']  # images
# train_y = MTR25K_dataset['train_y']  # texts/ tags
#
# query_L = MTR25K_dataset['query_L']  # labels
# query_x = MTR25K_dataset['query_x']  # images
# query_y = MTR25K_dataset['query_y']  # texts/ tags
#
# retrieval_L = MTR25K_dataset['retrieval_L']  # labels
# retrieval_x = MTR25K_dataset['retrieval_x']  # images
# retrieval_y = MTR25K_dataset['retrieval_y']  # texts/ tags



image_input = layers.Input(shape=(224, 224, 3), dtype=tf.float32)
feature = resnet50(num_classes=5, include_top=False)
# feature.build((None, 224, 224, 3))  # when using subclass model
feature.load_weights(opt.load_pre_weiht_Resnet_path + 'pretrain_weights.ckpt')
feature.trainable = False

images, tags, labels = loading_data_NUS(opt.load_data_path)

# ydim: 1000
ydim = tags.shape[1]
# num_class: 21
num_class = labels.shape[1]

X, Y, L = split_data(images, tags, labels)


mean, mean_pixel_ = mean_piexl()

# training
train_L = L['train']  # labels
train_x = X['train']  # images
train_y = Y['train']  # texts/ tags

query_L = L['query']  # labels
query_x = X['query']  # images
query_y = Y['query']  # texts/ tags

retrieval_L = L['retrieval']  # labels
retrieval_x = X['retrieval']  # images
retrieval_y = Y['retrieval']  # texts/ tags


print('...loading and splitting data finished\n')

# num_train: 10500
num_train = train_x.shape[0]
train_L = train_L.astype(np.float32)

var = {}
var['batch_size'] = opt.batch_size
var['X'] = np.random.randn(num_train, opt.bit)
var['Y'] = np.random.randn(num_train, opt.bit)
var['B'] = np.sign(var['X'] + var['Y'])

ph = {}
ph['lr'] = layers.Input(dtype=tf.float32, shape=(), name='lr')
ph['S_x'] = layers.Input(dtype=tf.float32, shape=[num_train], name='pS_x')
ph['S_y'] = layers.Input(dtype=tf.float32, shape=[num_train], name='pS_y')
ph['X'] = layers.Input(dtype=tf.float32, shape=[opt.bit], name='pX')
ph['Y'] = layers.Input(dtype=tf.float32, shape=[opt.bit], name='pY')
ph['B_batch'] = layers.Input(dtype=tf.float32, shape=[opt.bit], name='pB_batch')
ph['X_label'] = layers.Input(dtype=tf.float32, shape=[num_class], name='pX_label')
ph['Y_label'] = layers.Input(dtype=tf.float32, shape=[num_class], name='pY_label')

# ======================================================================================================================

# ==================================================  construct model  =================================================


# feature.summary()

my_model = MCDH_adv_and_att(feature, opt.emb_dim, opt.num_label, ydim, opt.bit, lambd=opt.lambd)

# ======================================================================================================================

# ================================================= training ===========================================================


max_mapi2t = 0.
max_mapt2i = 0.


# FEATURE_I = np.random.randn(opt.TRAINING_SIZE, opt.emb_dim).astype(np.float32)
# FEATURE_T = np.random.randn(opt.TRAINING_SIZE, opt.emb_dim).astype(np.float32)
#
# U = np.random.randn(opt.TRAINING_SIZE, opt.bit).astype(np.float32)
# V = np.random.randn(opt.TRAINING_SIZE, opt.bit).astype(np.float32)
#
FEATURE_MAP = np.random.randn(opt.num_label, opt.emb_dim).astype(np.float32)
# CODE_MAP = np.random.randn(opt.num_label, opt.bit).astype(np.float32)

i = 10

for t in range(i):

    print(t)
    print('\n')
    print('\n')
    print('****************************************************************')
    print('Run start time:', time.asctime(time.localtime(time.time())))
    print("The dataset for this training is :", opt.Dataset)
    print("Binary hash length:", opt.bit, "bit")
    print('...training procedure starts')
    print('****************************************************************\n')

    for epoch in range(opt.MAX_ITER):
        print('****************************************************************')
        print('Run start time:', time.asctime(time.localtime(time.time())))


        var['X'], var['Y'] = training_net(my_model, var, ph, train_x, train_y, train_L, mean_pixel_, FEATURE_MAP, epoch)
        var['B'] = np.sign(var['X'] + var['Y'])
        #if (epoch % 10 == 9) or (epoch == MAX_ITER - 1) or (epoch == 0):
        mapi2t, mapt2i, mapi2i, mapt2t = evaluation(my_model, query_x,
                                                    query_y,query_L, retrieval_x, retrieval_y, retrieval_L, opt.bit, FEATURE_MAP, mean)

        print('...epoch: %3d, test: map(i->t): %3.4f, map(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
        print('...epoch: %3d, test: map(t->t): %3.4f, map(i->i): %3.4f' % (epoch + 1, mapt2t, mapi2i))
        print('Run start time:', time.asctime(time.localtime(time.time())))
        print('****************************************************************\n')
        print('\n')

        if mapi2t >= max_mapi2t and mapt2i >= max_mapt2i:
            max_mapi2t = mapi2t
            max_mapt2i = mapt2i

    print('****************************************************************')
    print('...training procedure finished')
    print('Max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    print('Run end time:', time.asctime(time.localtime(time.time())))
    print('****************************************************************\n')

#=======================================================================================================================

result = {}
result['mapi2t'] = mapi2t
result['mapt2i'] = mapt2i

