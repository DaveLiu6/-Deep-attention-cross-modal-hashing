

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class MCDH_adv_and_att(Model):

    def __init__(self, feature_extractor, emb_dim, num_label, ydim, bit, lambd=0.8, **kwargs):
        super(MCDH_adv_and_att, self).__init__(**kwargs)
        self.model_name = 'MCDH'
        self.bit = bit
        self.ydim = ydim
        self.lambd = lambd


        self.feature_extractor = feature_extractor

        self.Img_model = Sequential([feature_extractor,
                                     layers.Conv2D(4096, kernel_size=2, padding='valid', activation='relu'),
                                     layers.MaxPool2D(pool_size=[2, 2], strides=2),
                                     layers.Conv2D(2048, kernel_size=2, padding='valid', activation='relu'),
                                     layers.Conv2D(emb_dim, kernel_size=2, padding='valid', activation='relu'),
                                     ])

        self.Txt_model = Sequential([layers.Conv2D(8192, kernel_size=1, activation='relu'),
                                     layers.Conv2D(4096, kernel_size=1, activation='relu'),
                                     layers.Conv2D(emb_dim, kernel_size=1, activation='relu')])

        self.img_hash_model = Sequential([layers.Conv2D(bit, kernel_size=1, activation='tanh')])
        self.txt_hash_model = Sequential([layers.Conv2D(bit, kernel_size=1, activation='tanh')])

        # self.img_classifier = Sequential([layers.Conv2D(num_label,kernel_size=1, activation='sigmoid')])
        # self.txt_classifier = Sequential([layers.Conv2D(num_label, kernel_size=1, activation='sigmoid')])
        #
        # self.img_discriminator = Sequential([layers.Conv2D(emb_dim, kernel_size=1, activation='relu'),
        #                                      layers.Conv2D(256, kernel_size=1, activation='relu'),
        #                                      layers.Conv2D(1, kernel_size=1)])
        #
        # self.txt_discriminator = Sequential([layers.Conv2D(emb_dim, kernel_size=1, activation='relu'),
        #                                      layers.Conv2D(256, kernel_size=1, activation='relu'),
        #                                      layers.Conv2D(1, kernel_size=1)])







    def call(self, img_input, txt_input, feature_map=None):

        f_x = self.Img_model(img_input)
        f_y = self.Txt_model(txt_input)

        # normalization
        # f_x = f_x / tf.math.sqrt(tf.math.reduce_sum(f_x ** 2))
        # f_y = f_y / tf.math.sqrt(tf.math.reduce_sum(f_y ** 2))

        # attention

        if feature_map is not None:

            # img attention
            mask_img = tf.sigmoid(5 * tf.matmul(tf.squeeze(f_x), tf.transpose(feature_map)))  # size: (batch, num_label)
            mask_f_x = tf.matmul(mask_img, feature_map) / tf.expand_dims(tf.reduce_sum(mask_img, axis=1), -1)   # size: (batch, emb_dim)
            mask_f_x = self.lambd * f_x + (1 - self.lambd) * tf.expand_dims(tf.expand_dims(mask_f_x, axis=1), axis=1)

            # txt attention
            mask_txt = tf.sigmoid(5 * tf.matmul(tf.squeeze(f_y), tf.transpose(feature_map)))  # size: (batch, num_label)
            mask_f_y = tf.matmul(mask_txt, feature_map) / tf.expand_dims(tf.reduce_sum(mask_txt, axis=1), -1)  # size: (batch, emb_dim)
            mask_f_y = self.lambd * f_y + (1 - self.lambd) * tf.expand_dims(tf.expand_dims(mask_f_y, axis=1), axis=1)

        else:
            mask_f_x, mask_f_y = f_x, f_y


        # x_class = tf.squeeze(self.img_classifier(mask_f_x))
        # y_class = tf.squeeze(self.txt_classifier(mask_f_y))

        x_code = tf.squeeze(self.img_hash_model(mask_f_x))
        y_code = tf.squeeze(self.txt_hash_model(mask_f_y))




        # h_x = self.Hash_model(f_x)
        # h_y = self.Hash_model(f_y)

        return x_code, y_code
        # return f_x, f_y


    def dis_img(self, f_x):

        is_img = self.img_discriminator(tf.expand_dims(tf.expand_dims(f_x, axis=1), axis=1))

        return  tf.squeeze(is_img)

    def dis_txt(self, f_y):

        inouts = tf.expand_dims(tf.expand_dims(f_y, axis=1), axis=1)
        is_txt = self.txt_discriminator(inouts)

        return tf.squeeze(is_txt)

    def generate_img_code(self, x, feature_map=None):
        f_x = self.Img_model(x)
        f_x = f_x / tf.math.sqrt(tf.reduce_sum(f_x ** 2))

        # attention
        if feature_map is not None:
            mask_img = tf.sigmoid(5 * tf.matmul(tf.squeeze(f_x), feature_map.transpose()))  # size: (batch, num_label)
            mask_f_x = tf.matmul(mask_img, feature_map) / tf.expand_dims(tf.reduce_sum(mask_img, axis=1),
                                                                         -1)  # size: (batch, emb_dim)
            f_x = self.lambd * f_x + (1 - self.lambd) * tf.expand_dims(tf.expand_dims(mask_f_x, axis=-1), axis=-1)

        code = self.Hash_model(f_x)

        return code

    def generate_txt_code(self, y, feature_map=None):
        f_y = self.Txt_model(y)
        f_y = f_y / tf.math.sqrt(tf.reduce_sum(f_y ** 2))

        # attention
        if feature_map is not None:
            mask_txt = tf.sigmoid(5 * tf.matmul(tf.squeeze(f_y), feature_map.transpose()))  # size: (batch, num_label)
            mask_f_y = tf.matmul(mask_txt, feature_map) / tf.expand_dims(tf.reduce_sum(mask_txt, axis=1),
                                                                         -1)  # size: (batch, emb_dim)
            f_y = self.lambd * f_y + (1 - self.lambd) * tf.expand_dims(tf.expand_dims(mask_f_y, axis=-1), axis=-1)

        code = self.Hash_model(f_y)

        return code



