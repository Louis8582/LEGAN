#!/usr/bin/env python
# coding: utf-8

# In[1]:


import darknet_function.python.darknet as dn
import subprocess
import sys, string, os
import time
import numpy as np
from os import walk
# os.environ['CUDA_VISIBLE_DEVICES']='3'

def train_network(darknet_path, obj_data_path, data_cfg, weight, epoch):
    os.system(darknet_path + " -i 1"+" detector train %s %s %s %d " % (obj_data_path, data_cfg, weight, epoch))
    
    
def change_loss_parameters(loss_batch,loss_subdiv, data_cfg):

    loss_batch_str = "loss_batch="+str(loss_batch)+"\n"
    loss_subdiv_str = "loss_subdivisions="+str(loss_subdiv)+"\n"
    cfg_file = open(data_cfg, "r+")
    line_list =[]
    for line in cfg_file:
        line_list.append(line)
    cfg_file.close()

    line_list[7] = loss_batch_str
    line_list[8] = loss_subdiv_str
    cfg_file = open(data_cfg, "w+")
    for line in line_list:
        cfg_file.write(line)
    cfg_file.close()    

def loss_calculate( image_path_list, darknet_path, obj_data_path, data_cfg, weight):
    text_file = open("/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/loss.txt", "w+")

    for path in image_path_list:
        text_file.write(path+"\n")
    text_file.close()
    cmd = darknet_path + " -i 1" + " detector loss" + " " + obj_data_path + " " + data_cfg + " " + weight + " 0"

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = str(out, 'utf-8')
    output_array = out.split("Loss:")
    loss = output_array[1].split('\n')[0]

    return float(loss)

def get_bounding_box(picture_path, obj_data_path, data_cfg, weight):
    net = dn.load_net(str.encode(data_cfg), str.encode(weight), 0)
    meta = dn.load_meta(str.encode(obj_data_path))
    all_box = dn.detect(net, meta, str.encode(picture_path))

    return all_box


# In[2]:


darknet_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function"
obj_data_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/obj.data"
cfg_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/yolov3.cfg"
weight_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/weights/yolov3_10000.weights"


# In[3]:


def IoU(target,compare):
    target_l = target[0] - target[2]/2
    target_r = target[0] + target[2]/2
    target_d = target[1] - target[3]/2
    target_t = target[1] + target[3]/2
    
    compare_l = compare[0] - compare[2]/2
    compare_r = compare[0] + compare[2]/2
    compare_d = compare[1] - compare[3]/2
    compare_t = compare[1] + compare[3]/2
    
    
    inter_l = np.max([target_l,compare_l])
    inter_r = np.min([target_r,compare_r])
    inter_d = np.max([target_d,compare_d])
    inter_t = np.min([target_t,compare_t])
    
    interArea = (inter_r-inter_l) * (inter_t-inter_d)
    if ((target_r - compare_l > 0) and (compare_r - target_l > 0)) :
        interArea = interArea
    else:
        interArea = 0
    
    totalArea = (target_r-target_l) * (target_t-target_d) +                 (compare_r-compare_l) * (compare_t-compare_d) - interArea
    
    return np.max([interArea / (totalArea + 1e-23),0])


# In[1]:


#tmp = "/home/mmdb/Desktop/Data/Train_HD/10/"
tmp = "/home/userr/ncku/LeGAN_Log_Time/BeanData/Train_data/40_pages/"
# tmp = "/home/louis/Documents/darknet/Data/Train_HD/30_All_labelImg_original/"
image_path = []
for i in range(40):
    image_path.append(tmp + str(i+1))
print(image_path)


# In[5]:


p = 0
print(p)
GDV = get_bounding_box(image_path[p]+".jpg", obj_data_path, cfg_path, weight_path)
print(image_path[p]+".jpg")
GDV = np.asarray(GDV)
# all_box = get_bounding_box("/home/louis/Documents/darknet/Data/Train_HD/30_All_labelImg/1"+".jpg", obj_data_path, cfg_path, weight_path)


# # 確定 Class and Confidence

# In[6]:


important_dim = [1,2,3,4,5,6,7,8,9,10,11,12,13]
unimportant_dim = []
for i in range(14):
    if (i in important_dim) == False:
        unimportant_dim.extend([i])
print(important_dim)
print(unimportant_dim)


# In[7]:


import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

img = cv2.imread(image_path[p]+".jpg",cv2.IMREAD_GRAYSCALE)
print(image_path[p]+".jpg")
img_color = cv2.imread(image_path[p]+".jpg")
print(img_color)
HoughCircles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                            param1=200,param2=10,minRadius=20,maxRadius=30)
HoughCircles = HoughCircles[0,:,:3]
print(HoughCircles.shape)
for circle in HoughCircles:
    position = int(circle[0]),int(circle[1])
    size = int(circle[2])
    cv2.circle(img_color,position, 5, (0, 0, 255), -1)
    
    
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.savefig("HoughCircle")


# In[8]:


def check_point_num(bb,hough):
    left_x = bb[0] - bb[2]/2
    right_x = bb[0] + bb[2]/2
    down_y = bb[1] - bb[3]/2
    top_y = bb[1] + bb[3]/2
    if (left_x <= hough[0] and right_x >= hough[0]) and (down_y <= hough[1] and top_y >= hough[1]):
        return True
    else:
        return False


# In[9]:


min_point_threshold = 1
max_point_threshold = 1

one_point_dim = []
for i,bb in enumerate(GDV):
    count = 0
    for hough in HoughCircles:
        if check_point_num(bb,hough):
            count = count + 1
        if count > max_point_threshold:
            break
    if count >= min_point_threshold and count <= max_point_threshold:
        one_point_dim.extend([i])
    
print(len(one_point_dim))
one_point_GDV = GDV[one_point_dim,:]
print(one_point_GDV.shape)


# In[10]:


bean_dim = []

for hough in HoughCircles:
    tmp = []
    for i,bb in enumerate(one_point_GDV):
        if check_point_num(bb,hough):
            tmp.extend([i])
    bean_dim.append(tmp)
    
#print(bean_dim)


# In[11]:


confident_threshold_table = [12.71,4.303,3.128,2.766,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,            2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045,2.042]


# In[12]:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x,axis=1).reshape(-1,1))
    return e_x / np.sum(e_x,axis=1).reshape(-1,1)


# In[13]:


# def hypothesis_class(GDV,table):
#     if GDV.shape[0] <= 30 :
#         confident_threshold = confident_threshold_table[GDV.shape[0] - 1]
#     else:
#         confident_threshold = 1.96
        
#     GDV_class_prob = GDV[:,5:]
    
#     class_prob_mean = np.mean(GDV_class_prob,axis=0)
#     class_prob_std = np.std(GDV_class_prob,axis=0)
    
#     pi = np.ones([14]) * 3.141
#     Gaussian = np.exp(-1*np.divide(np.power(GDV_class_prob - class_prob_mean,2),2*np.power(class_prob_std,2)))
    
#     Gaussian_multi_class_prob = Gaussian * GDV_class_prob
#     argMax = np.argmax(Gaussian_multi_class_prob,axis=1)
# #     argMax = np.argmax(GDV_class_prob,axis=1)

#     GDV_BBT = np.append(GDV[:,:5],argMax.reshape(-1,1),axis=1)
#     return GDV_BBT


# In[14]:


def hypothesis_class(GDV,table,important_dim,unimportant_dim):
    if GDV.shape[0] <= 30 :
        confident_threshold = confident_threshold_table[GDV.shape[0] - 1]
    else:
        confident_threshold = 1.96
        
    GDV_class_prob = GDV[:,5:]
    
    class_prob_mean = np.mean(GDV_class_prob,axis=0)
    class_prob_std = np.std(GDV_class_prob,axis=0)
    
    
    t_hypo = (GDV_class_prob - class_prob_mean) / (class_prob_std / np.sqrt(GDV.shape[0]))
    dim = np.where(np.abs(t_hypo) > confident_threshold)
    
    normal_dim = []
    for i in range(t_hypo.shape[0]):
        if (i in np.unique(dim[0])) == False:
            normal_dim.extend([i])


    normal_GDV_class_prob = GDV_class_prob[normal_dim,:]
    normal_class_prob_mean = np.mean(normal_GDV_class_prob,axis=0)
    normal_class_prob_std = np.std(normal_GDV_class_prob,axis=0)

    pi = np.ones([14]) * 3.141
    Gaussian = np.exp(-1*np.divide(np.power(normal_GDV_class_prob - normal_class_prob_mean,2),                                   2*np.power(normal_class_prob_std,2)))
#     Gaussian = Gaussian / (np.sqrt(pi * 2)*class_prob_std)
    
    
    normal_GDV_class_prob[:,important_dim] = normal_GDV_class_prob[:,important_dim] * (2-Gaussian[:,important_dim])
    normal_GDV_class_prob[:,unimportant_dim] = normal_GDV_class_prob[:,unimportant_dim] * Gaussian[:,unimportant_dim]
    
    GDV_class_prob[normal_dim,:] = normal_GDV_class_prob

#     Gaussian_multi_class_prob = Gaussian * GDV_class_prob
    argMax = np.argmax(GDV_class_prob,axis=1)
#    argMax = np.argmax(GDV_class_prob,axis=1)

    GDV_BBT = np.append(GDV[:,:5],argMax.reshape(-1,1),axis=1)
    return GDV_BBT


# In[15]:


hypo_class_BBT = []
for i,dim in enumerate(bean_dim):
    group = one_point_GDV[dim,:]
    tmp_BBT = hypothesis_class(group,confident_threshold_table,important_dim,unimportant_dim)
    #print(tmp_BBT[0])
#     for j in range(len(tmp_BBT)):
    
#         if int(tmp_BBT[j][5]) != 0:
#             tmp_BBT[j][5] = 1
    if i == 0:
        argMax = np.argmax(tmp_BBT[:,4],axis=0)
        #print(argMax)
        hypo_class_BBT = tmp_BBT[argMax,:].reshape(1,-1)
        #print(tmp_BBT[argMax,:])
    else: 
        argMax = np.argmax(tmp_BBT[:,4],axis=0)
        hypo_class_BBT = np.append(hypo_class_BBT,tmp_BBT[argMax,:].reshape(1,-1),axis=0)

print(hypo_class_BBT.shape)


# # 確定 confident

# In[16]:


bbox = hypo_class_BBT
#bbox =group
print(bbox.shape)
dim = np.where(bbox[:,4]>0.0005)
bbox = bbox[dim[0],:]

print(hypo_class_BBT.shape)


# In[17]:


tag_txt = []
for i in range(14):
    tag_txt.append(str(i))


# In[2]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
label=[]
with open(image_path[p]+".txt") as f:
    print(image_path[p]+".txt")
#with open("/home/louis/Documents/darknet/Data/Train_HD/30_All_labelImg/1"+".txt") as f:
#with open("/home/mmdb/Desktop/darknet/TestData/1"+".txt") as f:
    lines = f.readlines()
    for index,line in enumerate(lines):
#         if index < 10:
        data = line.split(' ')
        label.append(data)
# print(label)

img = cv2.imread(image_path[p]+".jpg")
print(image_path[p]+".txt")
#img = cv2.imread("/home/mmdb/Desktop/darknet/TestData/1"+".jpg")


for obj in label:
    if int(obj[0]) != 0:
        #print(obj)
        obj[0] = 1
        obj[4] = obj[4].split('\n')[0]
        obj = np.asarray(obj).astype(np.float)
        obj[1:] = (obj[1:]*608).astype(np.int)



        left_top = (int(obj[1]-obj[3]/2),int(obj[2]-obj[4]/2))
        right_down = (int(obj[1]+obj[3]/2),int(obj[2]+obj[4]/2))
        cv2.rectangle(img, left_top, right_down, (255,255,255), 5)

        textbox_left_top = (int(obj[1]-obj[3]/2),int(obj[2]-obj[4]/2) - 30)
        textbox_right_down = (int(obj[1]+obj[3]/2),int(obj[2]-obj[4]/2))
        cv2.rectangle(img, textbox_left_top, textbox_right_down, (255,255,255), -1)

        textbox = (int(obj[1]-15),int(obj[2]-obj[4]/2) - 5)
        #print(tag_txt[int(obj[0])])
        cv2.putText(img,tag_txt[int(obj[0])], textbox,cv2.FONT_HERSHEY_TRIPLEX,
                  1, (0,0,0), 2, cv2.LINE_AA)
        

count = 0
for j in range(bbox.shape[0]):
    #print(bbox[j])
    if int(bbox[j,5]) != 0:
        
        count = count + 1
    #         if count == 3:
        left_top = (int(bbox[j,0]-bbox[j,2]/2),int(bbox[j,1]-bbox[j,3]/2))
        right_down = (int(bbox[j,0]+bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2))
        cv2.rectangle(img, left_top, right_down, (0,0,255), 4)

        textbox_left_top = (int(bbox[j,0]-bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2))
        textbox_right_down = (int(bbox[j,0]+bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2)+30)
        cv2.rectangle(img, textbox_left_top, textbox_right_down, (0,0,255), -1)

        x = int(bbox[j,0]-bbox[j,2]/2)
        y = int(bbox[j,1]+bbox[j,3]/2 + 20)
        textbox = (x,y)
        #print(type(tag_txt[int(bbox[j,5])]))
        #cv2.putText(img,tag_txt[int(bbox[j,5])], textbox,cv2.FONT_HERSHEY_TRIPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img,tag_txt[int(bbox[j,5])], textbox,cv2.FONT_HERSHEY_TRIPLEX,1, (0,0,0), 2, cv2.LINE_AA)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.savefig("/home/mmdb/Desktop/darknet/Fig/Hypoth_Class_Confidence/Data_" + str(p+1))
#cv2.imwrite("/home/mmdb/Desktop/darknet/Fig/old_hypoth418(2label)/Data_" + str(p+1) + ".jpg", img)


# In[19]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
label=[]
with open(image_path[p]+".txt") as f:
    print(image_path[p]+".txt")
#with open("/home/louis/Documents/darknet/Data/Train_HD/30_All_labelImg/1"+".txt") as f:
#with open("/home/mmdb/Desktop/darknet/TestData/1"+".txt") as f:
    lines = f.readlines()
    for index,line in enumerate(lines):
#         if index < 10:
        data = line.split(' ')
        label.append(data)
# print(label)

img = cv2.imread(image_path[p]+".jpg")
print(image_path[p]+".txt")
#img = cv2.imread("/home/mmdb/Desktop/darknet/TestData/1"+".jpg")


for obj in label:
    if int(obj[0]) != 0:
        #print(obj)
        obj[4] = obj[4].split('\n')[0]
        obj = np.asarray(obj).astype(np.float)
        obj[1:] = (obj[1:]*608).astype(np.int)



        left_top = (int(obj[1]-obj[3]/2),int(obj[2]-obj[4]/2))
        right_down = (int(obj[1]+obj[3]/2),int(obj[2]+obj[4]/2))
        cv2.rectangle(img, left_top, right_down, (255,255,255), 5)

      #  textbox_left_top = (int(obj[1]-obj[3]/2),int(obj[2]-obj[4]/2) - 30)
     #   textbox_right_down = (int(obj[1]+obj[3]/2),int(obj[2]-obj[4]/2))
    #    cv2.rectangle(img, textbox_left_top, textbox_right_down, (255,255,255), -1)

     #   textbox = (int(obj[1]-15),int(obj[2]-obj[4]/2) - 5)
        #print(tag_txt[int(obj[0])])
    #    cv2.putText(img,str(1), textbox,cv2.FONT_HERSHEY_TRIPLEX,
      #            1, (0,0,0), 2, cv2.LINE_AA)
        

count = 0
for j in range(bbox.shape[0]):
    if int(bbox[j,5]) != 0:
        #print(bbox[j,5])
        count = count + 1
#         if count == 3:
        left_top = (int(bbox[j,0]-bbox[j,2]/2),int(bbox[j,1]-bbox[j,3]/2))
        right_down = (int(bbox[j,0]+bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2))
        cv2.rectangle(img, left_top, right_down, (0,0,255), 4)

#       textbox_left_top = (int(bbox[j,0]-bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2))
#      textbox_right_down = (int(bbox[j,0]+bbox[j,2]/2),int(bbox[j,1]+bbox[j,3]/2)+30)
#     cv2.rectangle(img, textbox_left_top, textbox_right_down, (0,0,255), -1)

#    x = int(bbox[j,0]-bbox[j,2]/2)
 #   y = int(bbox[j,1]+bbox[j,3]/2 + 20)
  #  textbox = (x,y)
    #print(type(tag_txt[int(bbox[j,5])]))
    #cv2.putText(img,tag_txt[int(bbox[j,5])], textbox,cv2.FONT_HERSHEY_TRIPLEX,1, (0,0,0), 2, cv2.LINE_AA)
   # cv2.putText(img,str(1), textbox,cv2.FONT_HERSHEY_TRIPLEX,1, (0,0,0), 2, cv2.LINE_AA)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.savefig("/home/mmdb/Desktop/darknet/Fig/Hypoth_Class_Confidence/Data_" + str(p+1))
#cv2.imwrite("/home/mmdb/Desktop/darknet/Fig/old_hypoth418(2label)/Data_no_" + str(p+1) + ".jpg", img)


# In[20]:


def DIQC(bbox1,bbox2):
    print(bbox1)
    print(bbox2)
    flag = 0
    bbox1_left = int((bbox1[0] - bbox1[2]/2))
    bbox1_right = int((bbox1[0] + bbox1[2]/2))
    bbox1_down = int((bbox1[1] - bbox1[3]/2))
    bbox1_top = int((bbox1[1] + bbox1[3]/2))
    
    bbox2_left = int((bbox2[0] - bbox2[2]/2))
    bbox2_right = int((bbox2[0] + bbox2[2]/2))
    bbox2_down = int((bbox2[1] - bbox2[3]/2))
    bbox2_top = int((bbox2[1] + bbox2[3]/2))
    
    
    if (bbox1_right >= bbox2_left and bbox1_left <= bbox2_right):
        flag = flag + 1
    if (bbox1_top >= bbox2_down and bbox1_down <= bbox2_top):
        flag = flag + 1
    
    if flag == 2:
        return 1
    else:
        return 0


# In[21]:


count = 0
iou = 0
tmp_count = 0
for j in range(bbox.shape[0]):
    if int(bbox[j,5]) != 0:
        bbox1 = bbox[j,:].astype(np.int)
        tmp_count = tmp_count + 1
#         print(bbox1)
        for obj in label:
            tmp_obj = np.asarray([float(ob) * 608 for ob in obj])
#             print(tmp_obj)
            bbox2 = tmp_obj[1:].astype(np.int)
#             print(bbox2)
            if IoU(bbox1,bbox2) > 0.1:
                count = count + 1
#                 print(IoU(bbox1,bbox2))
                iou = iou + IoU(bbox1,bbox2)
                break
print(count)
print(iou)
print(iou / count)


# In[22]:


right_count = 0
fail_count = 0
count = 0
print(len(label))
for j in range(bbox.shape[0]):
    if int(bbox[j,5]) != 0:
        count += 1
        bbox1 = bbox[j,:].astype(np.int)
        flag = 0
        for obj in label:
            tmp_obj = np.asarray([float(ob) * 608 for ob in obj])
            bbox2 = tmp_obj[1:].astype(np.int)
            if IoU(bbox1,bbox2) > 0.3:
                flag = 1
                break
        if flag == 1:
            right_count += 1
        else:
            fail_count += 1
print(count)
print(right_count)
print(fail_count)


# In[23]:


normal_count = 0

for obj in label:
    tmp_obj = np.asarray([float(ob) * 608 for ob in obj])
    bbox2 = tmp_obj[1:].astype(np.int)
    flag = 0
    for j in range(bbox.shape[0]):
        if int(bbox[j,5]) != 0:
            bbox1 = bbox[j,:].astype(np.int)
            if IoU(bbox1,bbox2) > 0.3:
                flag = 1
                break
    if flag == 0:
        normal_count += 1
        
print(normal_count)


# In[ ]:


print("   Def_pre_Def \t,Def_pre_Nor \t,Nor_pre_Def")
print("\t" + str(right_count) + "\t     " + str(normal_count) + "\t\t     " + str(fail_count))

