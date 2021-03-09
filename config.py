
import warnings

class DefaultConfig(object):



    load_data_path = 'Data_path'


    pre_dataset = 'Pre_path'



    Dataset = 'NUS WIDE'
    TRAINING_SIZE = 10500
    QUERY_SIZE = 2100
    DATABASE_SIZE = 193734


    emb_dim = 512

    num_label = 24

    batch_size = 32
    MAX_ITER = 50
    bit = 64

    coe_logloss = 1.0
    coe_quantization = 1.0
    coe_label = 1.0

    lambd = 0.99






opt = DefaultConfig()
