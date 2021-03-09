
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, models


class MCDH(Model):
    def __init__(self, feature_extractor, emb_dim, ydim, bit, **kwargs):
        super(MCDH, self).__init__(**kwargs)
        self.model_name = 'MCDH'
        self.bit = bit
        self.ydim = ydim

        self.Img_model = Sequential([feature_extractor,
                                     layers.Conv2D(4096, kernel_size=2, padding='valid', activation='relu'),
                                     layers.MaxPool2D(pool_size=[2, 2], strides=2),
                                     layers.Conv2D(2048, kernel_size=2, padding='valid', activation='relu'),
                                     layers.Conv2D(emb_dim, kernel_size=2, padding='valid', activation='relu'),
                                     ])

        self.Txt_model = Sequential([layers.Conv2D(4096, kernel_size=1, activation='relu'),
                                     layers.Conv2D(2048, kernel_size=1, activation='relu'),
                                     layers.Conv2D(emb_dim, kernel_size=1, activation='relu')])

        self.Hash_model = Sequential([layers.Conv2D(bit, kernel_size=1, activation='tanh')])

    def call(self, img_input, txt_input):

        f_x = self.Img_model(img_input)
        f_y = self.Txt_model(txt_input)

        h_x = self.Hash_model(f_x)
        h_y = self.Hash_model(f_y)

        return h_x, h_y







