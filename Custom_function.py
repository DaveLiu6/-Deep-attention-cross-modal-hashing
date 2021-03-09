
import numpy as np
import tensorflow as tf
from config import opt
from models import LearningRateExponentialDecay
from data_processing import calc_map
from tensorflow.keras.layers import *
from tensorflow.keras import layers, Sequential, Model

def split_data(images, tags, labels):
    # QUERY_SIZE : 2100
    # DATABASE_SIZE : 193734
    # TRAINING_SIZE : 10500
    np.random.seed(0)
    index_all = np.random.permutation(opt.QUERY_SIZE + opt.DATABASE_SIZE)
    ind_Q = index_all[0: opt.QUERY_SIZE]
    ind_T = index_all[opt.QUERY_SIZE: opt.TRAINING_SIZE + opt.QUERY_SIZE]
    ind_R = index_all[opt.QUERY_SIZE: opt.DATABASE_SIZE + opt.QUERY_SIZE]

    X = {}
    X['query'] = images[ind_Q, :, :, :]
    X['train'] = images[ind_T, :, :, :]
    X['retrieval'] = images[ind_R, :, :, :]

    Y = {}
    Y['query'] = tags[ind_Q, :]
    Y['train'] = tags[ind_T, :]
    Y['retrieval'] = tags[ind_R, :]

    L = {}
    L['query'] = labels[ind_Q, :]
    L['train'] = labels[ind_T, :]
    L['retrieval'] = labels[ind_R, :]

    return X, Y, L


def calc_similarity(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)

    return Sim


def jaccard_similarity(a, b):
    a = (a).astype(np.float32)
    b = (b).astype(np.float32)

    x = tf.reduce_sum(a, axis=-1)
    x = tf.expand_dims(x, axis=0)
    num1 = b.shape[0]
    x = tf.tile(x, multiples=[num1, 1])
    x = tf.transpose(x)
    #print(x)
    y = tf.reduce_sum(b, axis=-1)
    y = tf.expand_dims(y, axis=0)
    num2 = a.shape[0]
    y = tf.tile(y, multiples=[num2, 1])

    #print(y)

    z = (np.dot(a, tf.transpose(b))).astype(np.float32)
    #print(z)

    #jacc = tf.convert_to_tensor(z / (x + y - z), dtype=tf.float32)
    #jacc = tf.convert_to_tensor(tf.divide(z, tf.subtract(tf.add(x, y), z)), dtype=tf.float32)
    jacc = (np.divide(z, np.subtract(np.add(x, y), z))).astype(np.float32)
    #print(jacc)

    return jacc

# ======================================================= training ===============================================================

def training_net(my_model, var, ph, train_x, train_y, train_L, mean_pixel_, FEATURE_MAP, epoch):
    X = var['X']
    Y = var['Y']
    batch_size = var['batch_size']
    num_train = train_x.shape[0]
    # learningrate = 1e-4
    # if 50 < epoch <= 80:
    #     learningrate = 1e-5
    # elif epoch > 80:
    #     learningrate = 1e-7
    learningrate = 5e-5
    if epoch > 20:
        learningrate = LearningRateExponentialDecay(learningrate, 1, 0.98, epoch)
    for iter in range(int(num_train / batch_size + 1)):
        index = np.random.permutation(num_train)
        ind = index[0: batch_size]
        sample_L = train_L[ind, :]
        image = train_x[ind, :, :, :].astype(np.float32)
        image = image - mean_pixel_


        text = train_y[ind, :]
        text = tf.convert_to_tensor(text.reshape([text.shape[0], 1, 1, text.shape[1]]), dtype=tf.float32)
        S = calc_similarity(sample_L, train_L)
        S1 = jaccard_similarity(sample_L, train_L)
        ph['S_x'] = S
        ph['Y'] = tf.convert_to_tensor(var['Y'], dtype=tf.float32)
        ph['B_batch'] = var['B'][ind, :]
        ph['S_y'] = S
        ph['X'] = tf.convert_to_tensor(var['X'], dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate= learningrate)
        with tf.GradientTape(persistent=True) as tape:
            #cur_x = model1(image, training=True)
            cur_x, cur_y = my_model(image, text, FEATURE_MAP)

            X[ind, :] = cur_x
            # cur_y = model2(text, training=True)
            Y[ind, :] = cur_y

            # =============================== Adversial and Attention ====================================

            # FEATURE_I[ind, :] = f_x
            # FEATURE_T[ind, :] = f_y
            # U[ind, :] = cur_x
            # V[ind, :] = cur_y

            # TRAIN TXT DISCRIMINATOR

            # D_text_real = my_model.dis_txt(f_y)
            # D_text_real = -tf.reduce_mean(D_text_real)
            #
            # var_D_txt = my_model.txt_discriminator.trainable_variables
            # # gradientD1 = tape.gradient(loss, var_D_txt)
            # # optimizer.apply_gradients(zip(gradientD1, var_D_txt))
            #
            # # TRAIN WITH FAKE
            #
            # D_text_fake = my_model.dis_txt(f_x)
            # D_text_fake = tf.reduce_mean(D_text_fake)
            #
            # # gradientD2 = tape.gradient(loss, var_D_txt)
            # # optimizer.apply_gradients(zip(gradientD2, var_D_txt))
            #
            #
            # # Train with gradient penalty
            # alpha = np.random.rand(opt.batch_size, opt.emb_dim).astype(np.float32)
            # interpolates = alpha * f_y + (1 - alpha) * f_x
            # disc_interpolates = my_model.dis_txt(interpolates)



            # ============================================================================================

            # ======================================== image_net =========================================
            theta_xy = 1.0 / 2 * tf.matmul(cur_x, tf.transpose(a=ph['Y']))
            logloss_xy = tf.reduce_mean(
                input_tensor=tf.multiply((-tf.multiply(ph['S_x'], theta_xy) + tf.math.log(1.0 + tf.exp(theta_xy))), tf.exp(S1)))
            quantization_x = tf.reduce_mean(input_tensor=tf.pow(1 / 2.0 * (ph['B_batch'] - cur_x), 2))
            loss_xy_object = opt.coe_logloss * logloss_xy + opt.coe_quantization * quantization_x
            #=============================================================================================

            # ======================================== txt_net ===========================================

            theta_yx = 1.0 / 2 * tf.matmul(cur_y, tf.transpose(a=ph['X']))
            logloss_yx = tf.reduce_mean(
                input_tensor=tf.multiply((-tf.multiply(ph['S_y'], theta_yx) + tf.math.log(1.0 + tf.exp(theta_yx))), tf.exp(S1)))

            # ph['B_batch']: (batch_size, bit)
            quantization_y = tf.reduce_mean(input_tensor=tf.pow(1 / 2.0 * (ph['B_batch'] - cur_y), 2))

            loss_yx_object = opt.coe_logloss * logloss_yx + opt.coe_quantization * quantization_y

            loss = loss_xy_object + loss_yx_object
            model_var = my_model.Img_model.trainable_variables + my_model.Txt_model.trainable_variables \
                        + my_model.img_hash_model.trainable_variables + my_model.txt_hash_model.trainable_variables

            # ============================================================================================
        gradient1 = tape.gradient(loss, model_var)
        optimizer.apply_gradients(zip(gradient1, model_var))
        # FEATURE_MAP = update_feature_map(FEATURE_I, FEATURE_T, train_L)

    print("...epoch: {0}".format(epoch + 1))
    print("learning_rate: {0}".format(learningrate))
    #print('learningrate:{0:.4f}'.format(learningrate))
    print("loss_xy: {0:.4f}, logloss_xy: {1:.4f}, quantization_x: {2:.4f}".format(loss_xy_object, logloss_xy, quantization_x))

    #print("text epoch: {0}".format(epoch + 1))
    print("loss_yx: {0:.4f}, logloss_yx: {1:.4f}, quantization_y: {2:.4f}".format(loss_yx_object, logloss_yx, quantization_y))

    return X, Y


# ===============================================================================================================================

def generate_code(my_model, X, Y, bit, FEATURE_MAP, mean_pixel):
    numx_data = X.shape[0]
    numy_data = Y.shape[0]
    index_x = np.linspace(0, numx_data - 1, numx_data).astype(int)
    index_y = np.linspace(0, numy_data - 1, numy_data).astype(int)
    B_x = np.zeros([numx_data, bit], dtype=np.float32)
    B_y = np.zeros([numy_data, bit], dtype=np.float32)
    for iter in range(int(numx_data / opt.batch_size + 1)):
        ind_x = index_x[iter * opt.batch_size: min((iter + 1) * opt.batch_size, numx_data)]
        mean_pixel_ = np.repeat(mean_pixel[:, :, :, np.newaxis], len(ind_x), axis=3)
        image = X[ind_x, :, :, :].astype(np.float32) - mean_pixel_.astype(np.float32).transpose(3, 0, 1, 2)

        ind_y = index_y[iter * opt.batch_size: min((iter + 1) * opt.batch_size, numy_data)]
        text = Y[ind_y, :]
        text = tf.convert_to_tensor(text.reshape([text.shape[0], 1, 1, text.shape[1]]), dtype=tf.float32)
        cur_x, cur_y = my_model(image, text, FEATURE_MAP)

        cur_x = tf.squeeze(cur_x)
        cur_y = tf.squeeze(cur_y)

        B_x[ind_x, :] = cur_x
        B_y[ind_y, :] = cur_y

    B_x = np.sign(B_x)
    B_y = np.sign(B_y)

    return B_x, B_y


def evaluation(my_model, query_x, query_y, query_L, retrieval_x, retrieval_y,retrieval_L, bit, FEATURE_MAP, _meanpix):

    qBX, qBY = generate_code(my_model, query_x, query_y, bit, FEATURE_MAP,_meanpix)
    rBX, rBY = generate_code(my_model, retrieval_x, retrieval_y, bit, FEATURE_MAP, _meanpix)
    # qBX = generate_image_code(query_x, bit, _meanpix)
    # qBY = generate_text_code(query_y, bit)
    # rBX = generate_image_code(retrieval_x, bit, _meanpix)
    # rBY = generate_text_code(retrieval_y, bit)

    mapi2t = calc_map(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map(qBY, rBX, query_L, retrieval_L)

    mapi2i = calc_map(qBX, rBX, query_L, retrieval_L)
    mapt2t = calc_map(qBY, rBY, query_L, retrieval_L)
    return mapi2t, mapt2i, mapi2i, mapt2t


def mean_piexl():
    # mean_pixel: (224, 224, 3)
    # mean_pixel_: (batch_size, 224, 224, 3)

    mean = [[[123.68, 116.779, 103.939]]]
    mean1 = np.repeat(mean, 224, axis=0)
    mean_pixel1 = np.repeat(mean1, 224, axis=1)
    # net = img_net_structure(image_input, num_class, bit)
    mean_pixel_ = mean_pixel1[:, :, :, np.newaxis].transpose(3, 0, 1, 2)
    # mean_pixel_ = np.repeat(mean_pixel_.astype(np.float32), opt.batch_size, axis=0)
    mean_pixel_ = np.repeat(mean_pixel_.astype(np.float32), opt.batch_size, axis=0)
    return mean_pixel1, mean_pixel_

def update_feature_map(FEAT_I, FEAT_T, Lab, mode='average'):

    if mode is 'average':

        feature_map_I = tf.matmul(tf.transpose(Lab), FEAT_I) / tf.expand_dims(tf.reduce_sum(Lab, axis=0), -1)
        feature_map_T = tf.matmul(tf.transpose(Lab), FEAT_T) / tf.expand_dims(tf.reduce_sum(Lab, axis=0), -1)

    else:
        assert mode is 'max'
        feature_map_I = tf.reduce_max((tf.expand_dims(tf.transpose(Lab), axis=-1) * FEAT_I), axis=1)[0]
        feature_map_T = tf.reduce_max((tf.expand_dims(tf.transpose(Lab), axis=-1) * FEAT_T), axis=1)[0]

    FEATURE_MAP = (feature_map_I + feature_map_T) / 2

    # normalization
    FEATURE_MAP = FEATURE_MAP / tf.sqrt(tf.reduce_sum(FEATURE_MAP ** 2, axis=-1, keepdims=True))
    return FEATURE_MAP

