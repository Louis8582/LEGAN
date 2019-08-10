import darknet_exe as dn

darknet_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/darknet"
obj_data_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/obj.data"
cfg_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/yolov3.cfg"
weight_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/darknet53.conv.74"
so_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/libdarknet.so"
epoch = 1000

# ------------ train network ------------
dn.train_network(darknet_path, obj_data_path, cfg_path, weight_path, epoch)

# ------------ get bounding box ------------
#all_box = dn.get_bounding_box(image_path, obj_data_path, cfg_path, weight_path)

# ------------ loss calculate ------------
# loss = dn.loss_calculate(image_path, darknet_path, obj_data_path, cfg_path, weight_path)
# print(loss)
