from ctypes import *
import math
import random
import csv
import numpy as np
#from operator import itemgetter, attrgetter
#from numpy  import *

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/userr/ncku/LeGAN_Log_Time/darknet_function/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

train_network = lib.train_network
train_network.argtypes = [c_char_p, c_int]
train_network.restype =POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def takeSecond(elem):
    return elem[1]

def detect(net, meta, image, thresh=0, hier_thresh=.5, nms=0):


    im = load_image(image, 0, 0)

    num = c_int(0)

    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]


    all_box =[]
    for j in range(num):
        prob_array = np.array([dets[j].prob[:meta.classes]])
        objectness_array = dets[j].objectness
        #print(prob_array)
        #prob_array = np.array(prob_array)
        #max_index = np.argmax(prob_array)

        box = dets[j].bbox
        boxs = [box.x, box.y, box.w, box.h,objectness_array]
        boxs.extend(prob_array.tolist()[0])
        #boxs = np.array([objectness_array,box.x, box.y, box.w, box.h, for a in prob_array])
        #print(boxs)
        boxs = np.asarray(boxs)
        all_box.append(boxs)


    np.array(all_box)
    return all_box
#
# def darknet_exe():
#     net = load_net(str.encode("/home/mmdb/Desktop/darknet_function/truedata/cfg/yolov3.cfg"), str.encode("/home/mmdb/Desktop/darknet_function/truedata/cfg/weights/yolov3_10000.weights"), 0)
#     meta = load_meta(str.encode("/home/mmdb/Desktop/darknet_function/truedata/cfg/obj.data"))
#     all_box = detect(net, meta, str.encode("/home/mmdb/Desktop/darknet_function/73.jpg"))
#
#     return all_box
#
# if __name__ == "__main__":
#     #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
#     #im = load_image("data/eagle.jpg", 0, 0)
#     #meta = load_meta("cfg/imagenet1k.data")
#     #r = classify(net, meta, im)
#     #print r[:10]
#     net = load_net(str.encode("/home/mmdb/Desktop/bounding_box/truedata/cfg/yolov3.cfg"), str.encode("/home/mmdb/Desktop/bounding_box/truedata/cfg/weights/yolov3_10000.weights"), 0)
#     #im = load_image(str.encode("/home/mmdb/Desktop/rpidarknet_(copy)/75.jpg"),0,0)
#     #print(c_char_p(s.encode('utf-8')))
#
#     #print(predict_image)
#     meta = load_meta(str.encode("/home/mmdb/Desktop/bounding_box/truedata/cfg/obj.data"))
#     all_box = detect(net, meta, str.encode("/home/mmdb/Desktop/bounding_box/73.jpg"))
#
#     for i in range(14):
#         print(len(all_box[i]))
