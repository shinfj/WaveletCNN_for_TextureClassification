# coding: utf-8
'''
Usage:
    (for train) python run_waveletcnn.py -p train [-g 0] -d path/to/training/dataset
    (for test)  python run_waveletcnn.py -p test [-g 0] -i path/to/trained/model -t path/to/target/image/file
This program try to train/test wavelet CNNs with 4-level decomposition.
'''

import os
import sys
import argparse
import numpy as np
import caffe
import functions


parser = argparse.ArgumentParser(description='Clasify the input image into the correct category.')
parser.add_argument('--phase', '-p', type=str, required=True,
                    help='Run Wavelet CNN on train/test mode (input train or test)')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
# for Train phase
parser.add_argument('--dataset', '-d', default='Dataset', type=str,
                    help='dataset directory path for training')
# for Test phase
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--target_image', '-t', default=None, type=str,
                    help='target image path')
args = parser.parse_args()

base_dir = os.getcwd()
sys.path.append(base_dir)

if args.gpu < 0:
    caffe.set_mode_cpu()
else:
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()


if args.phase == "train":
    functions.misc.rewrite_data('models/WaveletCNN_4level.prototxt', args.dataset)
    Netsolver = os.path.join(base_dir, 'models/solver_WaveletCNN_4level.prototxt')
    solver = caffe.SGDSolver(Netsolver)
    solver.solve()
elif args.phase == "test":
    net = caffe.Net('models/WaveletCNN_4level_deploy.prototxt', args.initmodel, caffe.TEST)
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    #load the image in the data layer
    image = caffe.io.load_image(args.target_image)
    min_length = min(image.shape[:2])
    crop_length = int(min_length * 0.6)     # crop image with 60% length of shorter edge
    cropped_imgs = functions.misc.random_crop(image, (crop_length, crop_length), 1)   # shape is N x H x W x C
    cropped_im = cropped_imgs[0]
    resized_im = caffe.io.resize_image(cropped_im, (224, 224), interp_order=3)

    net.blobs['data'].data[...] = transformer.preprocess('data', resized_im)
    out = net.forward()
    # the output probability vector for the first image in the batch
    output_prob = out['prob'][0]

    labels = 'Dataset/labels.txt'   # Path to the text file containing a label name per each line.
    label_list = np.loadtxt(labels, str, delimiter='\t')

    # top-5 for the probability
    top_idx = output_prob.argsort()[::-1][:5]
    print(args.target_image + ':')
    # print zip(label_list[top_idx], output_prob[top_idx])

    for i in range(len(top_idx)):
        print(label_list[top_idx][i] + ': ' + str(round(output_prob[top_idx][i]*100, 3)))
