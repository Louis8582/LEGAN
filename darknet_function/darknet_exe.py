import sys, string, os
sys.path.insert(0, 'python')
import string
import darknet as dn
import subprocess

#os.system("/home/userr/ncku/LeGAN_Log_Time/truedata_darknet/darknet detector test %s %s %s %s %s %s " % ("truedata/cfg/obj.data","truedata/cfg/yolov3.cfg","truedata/cfg/weights/yolov3_10000.weights","/home/userr/ncku/LeGAN_Log_Time/truedata_darknet/80.jpg","< data / train.txt >","result.txt"))

def train_network(darknet_path, obj_data_path, data_cfg, weight, epoch):
    os.system(darknet_path +" detector train %s %s %s %d " % (obj_data_path, data_cfg, weight, epoch))

def change_loss_batch(loss_batch, data_cfg):

    loss_batch_str = "loss_batch="+str(loss_batch)+"\n"
    cfg_file = open(data_cfg, "r+")
    line_list =[]
    for line in cfg_file:
        line_list.append(line)
    cfg_file.close()

    line_list[7] = loss_batch_str
    cfg_file = open(data_cfg, "w+")
    for line in line_list:
        cfg_file.write(line)
    cfg_file.close()


def loss_calculate( image_path_list, darknet_path, obj_data_path, data_cfg, weight):

    text_file = open("/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/loss.txt", "w+")

    for path in image_path_list:
        text_file.write(path+"\n")
    # text_file.write(image_path)
    text_file.close()



    cmd = darknet_path + " detector loss" + " " + obj_data_path + " " + data_cfg + " " + weight + " 0"

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out)

    output_array = out.split("Loss:")
    loss = output_array[1].split('\n')[0]

    return float(loss)
    #return cmd
def get_bounding_box(picture_path, obj_data_path, data_cfg, weight):
    net = dn.load_net(str.encode(data_cfg), str.encode(weight), 0)
    meta = dn.load_meta(str.encode(obj_data_path))
    all_box = dn.detect(net, meta, str.encode(picture_path))

    return all_box


#def get_bounding_box():
    #net = dn.load_net(str.encode("/home/userr/ncku/LeGAN_Log_Time/bounding_box/truedata/cfg/yolov3.cfg"), str.encode("/home/userr/ncku/LeGAN_Log_Time/bounding_box/truedata/cfg/weights/yolov3_10000.weights"), 0)
    #meta = dn.load_meta(str.encode("/home/userr/ncku/LeGAN_Log_Time/bounding_box/truedata/cfg/obj.data"))
    #all_box = dn.detect(net, meta, str.encode("/home/userr/ncku/LeGAN_Log_Time/bounding_box/73.jpg"))

    #return all_box
