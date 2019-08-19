# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
caffe_root='/home/dtt/arm_ziliao/caffe/'
#os.chdir(caffe_root)
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

import argparse
import configparser
import io
import os
import sys
import logging
import time
import yaml
from collections import defaultdict

import cv2

from caffe2.python import core
from caffe2.python import workspace
from caffe2.python import model_helper,brew
from caffe2.python.predictor.predictor_exporter import prepare_prediction_net


def CreateCaffeNetFromCaffe2Net(caffe2Net, caffeNet):
    with myutils.NamedCudaScope(0):
        workspace.ResetWorkspace("/opt/yolov3caffe2")
        predict_net = prepare_prediction_net(caffe2Net, "minidb")
        net_proto = predict_net.Proto()
        print(net_proto)
        img = cv2.imread('/opt/tmp.jpg')
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        sized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        npar = np.array(sized)
        pp = np.ascontiguousarray(np.transpose(npar, [2, 0, 1])).reshape(1, 3, sized.shape[0], sized.shape[1]).astype(
            np.float32) / 255.0

        workspace.CreateNet(predict_net, overwrite=True)

        workspace.FeedBlob('gpu_0/data', pp)
        workspace.RunNet(predict_net.Proto().name)


        # # print(net_proto)
        # # print predict_net.Proto()
        fp = open(caffeNet, 'w')
        fp.write("name: \"ResNetCaffe\"\n")
        fp.write("input: \"gpu_0/data\"\n")
        fp.write('input_dim: 1\n')
        fp.write('input_dim: 3\n')
        fp.write('input_dim: 224\n')
        fp.write('input_dim: 224\n')
        fp.write('\n')
        conv_id = 0
        bn_id = 0
        pool_id =0
        relu_id =0
        softmax_id = 0
        sum_id = 0
        fc_id =0
        def writeConv():
            fp.write("layer {\n")
            fp.write("  type: \"Convolution\"\n")
            fp.write("  name: \"layer-conv_{}\"\n".format(conv_id))

            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))

            fp.write("  convolution_param {\n")
            blob = workspace.FetchBlob(op.output[0])
            num_output = blob.shape[1]
            fp.write("      num_output: {}\n".format(num_output))
            for arg in op.arg:
                if arg.name == "kernel":
                    fp.write("      kernel_size: {}\n".format(arg.i))
                if arg.name == "pad":
                    fp.write("      pad: {}\n".format(arg.i))
                if arg.name == "stride":
                    fp.write("      stride: {}\n".format(arg.i))
            if len(op.input) == 3:
                fp.write("      bias_term: true\n")
            else:
                fp.write("      bias_term: false\n")

            fp.write("  }\n")
            fp.write("}\n")
        def writeBN():
            fp.write("layer {\n")
            fp.write("  type: \"BatchNorm\"\n")
            fp.write("  name: \"layer-bn_{}\"\n".format(bn_id))

            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))

            fp.write("  batch_norm_param {\n")
            fp.write("      use_global_stats: true\n")

            fp.write("  }\n")
            fp.write("}\n")
            #Scale
            fp.write("layer {\n")
            fp.write("  type: \"Scale\"\n")
            fp.write("  name: \"layer-scale_{}\"\n".format(bn_id))

            fp.write("  bottom: \"{}\"\n".format(op.output[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))

            fp.write("  scale_param {\n")
            fp.write("      bias_term: true\n")

            fp.write("  }\n")
            fp.write("}\n")
        def wrieteSoftMax():
            fp.write("layer {\n")
            fp.write("  type: \"Softmax\"\n")
            fp.write("  name: \"layer-softmax_{}\"\n".format(0))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("}\n")
        def writeRelu():
            fp.write("layer {\n")
            fp.write("  type: \"ReLU\"\n")
            fp.write("  name: \"layer-relu_{}\"\n".format(relu_id))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("}\n")
        def wrieteFC():
            fp.write("layer {\n")
            fp.write("  type: \"InnerProduct\"\n")
            fp.write("  name: \"layer-FC_{}\"\n".format(fc_id))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("  inner_product_param {\n")
            blob = workspace.FetchBlob(op.output[0])
            num_output = blob.shape
            fp.write("      num_output: {}\n".format(num_output[1]))
            fp.write("  }\n")
            fp.write("}\n")
        def wrieteSum():
            fp.write("layer {\n")
            fp.write("  type: \"Eltwise\"\n")
            fp.write("  name: \"layer-Eltwise_{}\"\n".format(sum_id))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  bottom: \"{}\"\n".format(op.input[1]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("}\n")
        def writeMaxPool():
            fp.write("layer {\n")
            fp.write("  type: \"Pooling\"\n")
            fp.write("  name: \"layer-pool_{}\"\n".format(pool_id))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("  pooling_param {\n")
            for arg in op.arg:
                if arg.name == "kernel":
                    fp.write("      kernel_size: {}\n".format(arg.i))
                if arg.name == "stride":
                    fp.write("      stride: {}\n".format(arg.i))
            fp.write("      pool: MAX\n")
            fp.write("  }\n")
            fp.write("}\n")
        def writeAveragePool():
            fp.write("layer {\n")
            fp.write("  type: \"Pooling\"\n")
            fp.write("  name: \"layer-avpool_{}\"\n".format(pool_id))
            fp.write("  bottom: \"{}\"\n".format(op.input[0]))
            fp.write("  top: \"{}\"\n".format(op.output[0]))
            fp.write("  pooling_param {\n")
            for arg in op.arg:
                if arg.name == "kernel":
                    fp.write("      kernel_size: {}\n".format(arg.i))
                if arg.name == "stride":
                    fp.write("      stride: {}\n".format(arg.i))
            fp.write("      pool: AVE\n")
            fp.write("  }\n")
            fp.write("}\n")
        for op in net_proto.op:
            if op.type == "Conv":
                writeConv()
                conv_id +=1
            elif op.type == "FC":
                wrieteFC()
                fc_id +=1
            elif op.type == "SpatialBN":
                writeBN()
                bn_id +=1
            elif op.type == "Relu":
                writeRelu()
                relu_id +=1
            elif op.type == "Sum":
                wrieteSum()
                sum_id +=1
            elif op.type == "MaxPool":
                writeMaxPool()
                pool_id +=1
            elif op.type == "Softmax":
                wrieteSoftMax()
            elif op.type == "AveragePool":
                writeAveragePool()
                pool_id +=1
            else:
                print("Unknow layertype:{}, model convert failed.".format(op.type))
                exit()

        fp.close()

def initCaffeWeigthsWithCaffe2Weigths(caffe2Net, caffeNet, caffeModel):
    with myutils.NamedCudaScope(0):
        workspace.ResetWorkspace("/opt/yolov3caffe2")
        predict_net = prepare_prediction_net(caffe2Net, "minidb")
        net_proto = predict_net.Proto()
        print(net_proto)
        img = cv2.imread('/opt/tmp.jpg')
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        sized = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        npar = np.array(sized)
        pp = np.ascontiguousarray(np.transpose(npar, [2, 0, 1])).reshape(1, 3, sized.shape[0], sized.shape[1]).astype(
            np.float32) / 255.0

        workspace.CreateNet(predict_net, overwrite=True)

        workspace.FeedBlob('gpu_0/data', pp)
        workspace.RunNet(predict_net.Proto().name)

        net = caffe.Net(caffeNet, caffe.TEST)
        params = net.params
        bn_id = 0
        conv_id = 0
        fc_id = 0
        for op in net_proto.op:
            if op.type == "Conv":
                param_layer_name = "layer-conv_{}".format(conv_id)
                conv_param = params[param_layer_name]
                conv_weight = conv_param[0].data
                conv_blob = workspace.FetchBlob(op.input[1])
                conv_param[0].data[...] = np.reshape(conv_blob, conv_weight.shape);
                if len(op.input) == 3:
                    conv_bias = conv_param[1].data
                    bias_blob = workspace.FetchBlob(op.input[2])
                    conv_param[1].data[...] = np.reshape(bias_blob, conv_bias.shape)
                conv_id += 1
            elif op.type == "SpatialBN":
                bn_layer_name = "layer-bn_{}".format(bn_id)
                scale_layer_name = "layer-scale_{}".format(bn_id)
                bn_param = params[bn_layer_name]
                scale_param = params[scale_layer_name]
                running_mean = bn_param[0].data
                running_var = bn_param[1].data
                scale_weight = scale_param[0].data
                scale_bias = scale_param[1].data
                scale_blob = workspace.FetchBlob(op.input[1])
                bias_blob = workspace.FetchBlob(op.input[2])
                mean_blob = workspace.FetchBlob(op.input[3])
                var_blob = workspace.FetchBlob(op.input[4])
                bn_param[0].data[...] = np.reshape(mean_blob,running_mean.shape)
                bn_param[1].data[...] = np.reshape(var_blob,running_var.shape)
                scale_param[0].data[...] = np.reshape(scale_blob,scale_weight.shape)
                scale_param[1].data[...] = np.reshape(bias_blob,scale_bias.shape)

                bn_id += 1
            elif op.type == "FC":
                """
                weight = fc_param[0].data
                bias = fc_param[1].data
                fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
                fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
                """
                layer_name = "layer-FC_{}".format(fc_id)
                layer_param = params[layer_name]
                weight = layer_param[0].data
                bias = layer_param[1].data
                weight_blob = workspace.FetchBlob(op.input[1])
                bias_blob = workspace.FetchBlob(op.input[2])
                layer_param[0].data[...] = np.reshape(weight_blob, weight.shape)
                layer_param[1].data[...] = np.reshape(bias_blob, bias.shape)
                fc_id += 1
            else:
                print "Layey name:{}, type:{}, has no weights, skipping this layer".format(op.name, op.type)
        print("Save converted caffe model to {}".format(caffeModel))
        net.save(caffeModel)


def saveCaffeModel():
    pass


caffe2Net = "/opt/fogtrain/fogmodel_final.model"
caffeNet = "/opt/fogtrain/fogmodel.prototxt"
caffeModel ="/opt/fogtrain/fogmodel.caffemodel"
CreateCaffeNetFromCaffe2Net(caffe2Net=caffe2Net, caffeNet=caffeNet)
initCaffeWeigthsWithCaffe2Weigths(caffe2Net, caffeNet, caffeModel)
