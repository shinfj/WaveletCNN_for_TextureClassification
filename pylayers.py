# coding:utf-8
import numpy as np
import caffe
import functions


class GCN(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        batch_size = len(bottom[0].data)
        for index in range(batch_size):
            img = bottom[0].data[index].transpose(1,2,0)
            img_vec = img.flatten('F')
            img_vec = img_vec.reshape(1, len(img_vec))
            img_gcn = functions.misc.global_contrast_normalize(img_vec)
            img_pp = img_gcn.reshape(img.shape[0], img.shape[1], img.shape[2], order='F')
            img_pp = img_pp.transpose(2, 0, 1)
            top[0].data[index] = img_pp

    def backward(self, top, propagate_down, bottom):
        pass


class GramMatrix(caffe.Layer):
    def setup(self, bottom, top):
        bt_shape = bottom[0].data.shape
        top[0].reshape(bt_shape[0], 1, bt_shape[1], bt_shape[1])

    def reshape(self, bottom, top):
        bt_shape = bottom[0].data.shape
        top[0].reshape(bt_shape[0], 1, bt_shape[1], bt_shape[1])

    def forward(self, bottom, top):
        batch_size = len(bottom[0].data)
        for index in range(batch_size):
            activations = bottom[0].data[index]
            G, F = functions.misc.gram_matrix(activations[None, :, :, :])
            top[0].data[index] = G[None, :, :]

    def backward(self, top, propagate_down, bottom):
        M = bottom[0].data.shape[2] * bottom[0].data.shape[3]
        diff_tmp = top[0].diff / M
        batch_size = len(top[0].diff)
        for index in range(batch_size):
            activations = bottom[0].data[index]
            G, F = functions.misc.gram_matrix(activations[None, :, :, :])
            top_tmp = diff_tmp[index][0] + diff_tmp[index][0].T
            bottom[0].diff[index] = np.dot(top_tmp, F).reshape(bottom[0].diff.shape[1], bottom[0].diff.shape[2], bottom[0].diff.shape[2])


class WaveletHaarLevelLayer(caffe.Layer):
    def setup(self, bottom, top):
        try:
            self.level = int(self.param_str)   # string style of number
        except ValueError:
            raise ValueError("param_str must be string")

        # check if the number of tops is equal to that of self.level
        if len(top) != self.level:
            raise Exception(" number of tops must be equal to that of levels")

        bt_shape = bottom[0].data.shape
        for i in range(self.level):
            div = 2**(i+1)
            top[i].reshape(bt_shape[0], 4*bt_shape[1], int(bt_shape[2]/div), int(bt_shape[3]/div))

    def reshape(self, bottom, top):
        bt_shape = bottom[0].data.shape
        for i in range(self.level):
            div = 2**(i+1)
            top[i].reshape(bt_shape[0], 4*bt_shape[1], int(bt_shape[2]/div), int(bt_shape[3]/div))

    def forward(self, bottom, top):
        batch_size = len(bottom[0].data)
        for index in range(batch_size):
            for i in range(len(bottom[0].data[0])):
                wavelets = functions.wavelet_haar.WaveletTransform(bottom[0].data[index][i], self.level)
                for j in range(self.level):
                    top[j].data[index][4*i] = wavelets["LL_"+str(j+1)]
                    top[j].data[index][4*i+1] = wavelets["LH_"+str(j+1)]
                    top[j].data[index][4*i+2] = wavelets["HL_"+str(j+1)]
                    top[j].data[index][4*i+3] = wavelets["HH_"+str(j+1)]

    def backward(self, top, propagate_down, bottom):
        pass
