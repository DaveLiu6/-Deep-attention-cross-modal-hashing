
import scipy.io as sio
import tensorflow as tf
import h5py


def loading_data_NUS(data_path):
    # path = './data'
    #path = '/Extra/Datasets/nus-wide-tc21/'
    path = data_path
    # load original image : (195834, 3, 244, 244)
    IALL = h5py.File(path + 'nus-wide-tc21-iall.mat', 'r')
    images = IALL['IAll'][:].transpose(0, 3, 2, 1)

    # load text feature : (195834, 1000)
    YALL = sio.loadmat(path + 'nus-wide-tc21-yall.mat')
    tags = YALL['YAll'][:]

    # load label : (195834, 24)
    LALL = sio.loadmat(path + 'nus-wide-tc21-lall.mat')
    labels = LALL['LAll'][:]

    IALL.close()
    return images, tags, labels

def loading_data_NUS_no_zero(data_path):

    #path = '/Extra/Datasets/nus-wide-tc21/'
    path = data_path
    # load original image : (195834, 3, 244, 244)
    IALL = h5py.File(path + 'nus-wide-tc21-iall.mat', 'r')
    images = IALL['IAll'][:].transpose(0, 3, 2, 1)

    # load text feature : (195834, 1000)
    YALL = sio.loadmat(path + 'nus-wide-tc21-yall.mat')
    tags = YALL['YAll'][:]

    # load label : (195834, 24)
    LALL = sio.loadmat(path + 'nus-wide-tc21-lall.mat')
    labels = LALL['LAll'][:]

    No_zero_id = []
    toal_tags = tags.shape[0]
    for i in range(toal_tags):
        if tf.reduce_sum(tags[i, :]) != 0:
            No_zero_id.append(i)

    images = images[No_zero_id, :]
    labels = labels[No_zero_id, :]
    tags = tags[No_zero_id, :]
    IALL.close()
    return images, tags, labels



if __name__ == '__main__':
    images, tags, labels = loading_data_NUS_no_zero()
    print('images: %s' % str(images.shape))
    print('text: %s' % str(tags.shape))
    print('labels: %s' % str(labels.shape))
