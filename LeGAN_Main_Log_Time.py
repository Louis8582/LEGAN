#!/usr/bin/env python
# coding: utf-8

# In[2]:

#import random
import heapq
import numpy as np
import pickle
import os
import cv2
import shutil
os.environ['CUDA_VISIBLE_DEVICES']='0'
np.set_printoptions(suppress=True)

# In[3]:


n = 20

nnn=0 
# In[4]:


import heapq
class Generate_Model:
    def __init__(self):
        self.gener_W = []
        self.gener_B = []
        self.gener_W.append(np.random.uniform(-1,1,[input_dim,hidden1_dim]).astype(np.float64))
        #print("self.gener_W[0]:",self.gener_W[0].shape,"[input_dim,hidden1_dim])",input_dim," , ",hidden1_dim)
        self.gener_B.append(np.random.uniform(-1,1,[hidden1_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden1_dim,hidden2_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden2_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden2_dim,hidden3_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden3_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden3_dim,hidden4_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden4_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden4_dim,hidden5_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden5_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden5_dim,hidden6_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden6_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden6_dim,output_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[output_dim]).astype(np.float64))
        #print("self.gener_W[0]):",self.gener_W[0].shape)
        self.obj_num = 11
    
    def set_model(self,gener_W,gener_B):  #change the neural network parameters
        self.gener_W = gener_W
        self.gener_B = gener_B
        
    def sigmoid(self,x):
        sigm = 1. / (1. + np.exp(-x))
        return sigm
    def tanh(self,x):
        return np.tanh(x)
    def relu(self,x):
        relu = x * (x>0)
        return relu
    def softmax_Class(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=3).reshape([-1,11,11,1])
    def softmax_Cell(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape([-1,1])
    
    def run(self,data):  # neural network forward 
        #print("data\n",data)
        self.gener_Layer_output = []
        
        self.gener_Layer_output.append(np.add(np.matmul(data,self.gener_W[0]),self.gener_B[0]))
        self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
        for i in range(5):
            self.gener_Layer_output.append(np.add(np.matmul(self.gener_Layer_output[-1],self.gener_W[i+1]),self.gener_B[i+1]))
            self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
        self.gener_Layer_output.append(np.add(np.matmul(self.gener_Layer_output[-1],self.gener_W[i+2]),self.gener_B[i+2]))
        self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
            
        self.output_reshape = self.gener_Layer_output[-1].reshape(-1,11,11,generator_output_dim)
        self.output_BB = self.output_reshape[:,:,:,:2]
        self.output_Cell = self.softmax_Cell(self.output_reshape[:,:,:,3].reshape(-1,11*11))
        self.output_Class = self.softmax_Class(self.output_reshape[:,:,:,3:])
        
    def get_new_obj_XY_Class(self): # get the new object class and coordinates in cell space
        for i in range(self.output_reshape.shape[0]):
            nlarg_Cell = heapq.nlargest(self.obj_num, range(len(self.output_Cell[i])), self.output_Cell[i].take)
            output_Class_MaxIndex = np.argmax(self.output_Class,axis=3).reshape([-1,11,11,1])
                
            tmp_obj_XY = self.output_BB[i].reshape(-1,11*11,2)[:,nlarg_Cell,:]
            #print("tmp_obj_XY:",tmp_obj_XY)
            tmp_obj_Class = output_Class_MaxIndex[0].reshape(-1,11*11,1)[:,nlarg_Cell,:]
            if i == 0:
                self.new_obj_XY = tmp_obj_XY
                self.new_obj_Class = tmp_obj_Class
            else:
                self.new_obj_XY = np.append(self.new_obj_XY,tmp_obj_XY,axis=0)
                self.new_obj_Class = np.append(self.new_obj_Class,tmp_obj_Class,axis=0)

    def get_new_obj_tmp_Yolo_output(self,yolo_trainY):  # get the new object ouput (x,y,class)
        #print(yolo_trainY.shape)
        for i in range(self.output_reshape.shape[0]):
            nlarg_Cell = heapq.nlargest(self.obj_num, range(len(self.output_Cell[i])), self.output_Cell[i].take)
            #print("nlarg_Cell:\n",nlarg_Cell)
            #chage the coordinates to image space
            ten_digit = np.divide(nlarg_Cell,10).astype(int)
            #print("ten_digit:\n",ten_digit)
            digit = np.mod(nlarg_Cell,10)
            new_obj_Yolo_X = np.divide(digit,10) + np.divide(self.new_obj_XY[i,:,0],10)
            new_obj_Yolo_Y = np.divide(ten_digit,10) + np.divide(self.new_obj_XY[i,:,1],10)
            #print("new_obj_Yolo_XY\n",new_obj_Yolo_X*608,"\n",new_obj_Yolo_Y*608)
            #print("shape:",new_obj_Yolo_Y.shape)
            
            int_11=9
            our_work_address_xy=np.arange(33,608,67)
            create64 = np.arange(0,81,1)
            np.random.shuffle(create64)
            
            create64=create64[0:11]
            
            new_obj_Yolo_X = our_work_address_xy[create64%int_11]/608
            #print(create64%8)
            create64=create64/int_11
            create64=create64.astype(int)
            new_obj_Yolo_Y = our_work_address_xy[create64]/608
            #print(create64)
            #print("QQ:")
            #print(new_obj_Yolo_X)
            #print(new_obj_Yolo_Y)
            
            tmp_new_obj_tmp_Yolo_output = np.append(new_obj_Yolo_X.reshape(-1,self.obj_num,1),new_obj_Yolo_Y.reshape(-1,self.obj_num,1),axis=2)
            
            grid_x, grid_y = np.where(yolo_trainY[i,:,:,5] == 1) #get the cell dimension when confident == 1 
            obj = yolo_trainY[i,grid_x,grid_y,:]
            obj_index = np.argsort(obj,axis=0)
            obj = obj[obj_index[:,0]]
            obj = obj[self.new_obj_Class[i].reshape(1,-1)]

            tmp_new_obj_tmp_Yolo_output = np.append(obj[:,:,0].reshape([-1,self.obj_num,1]),tmp_new_obj_tmp_Yolo_output,axis=2)
            tmp_new_obj_tmp_Yolo_output = np.append(tmp_new_obj_tmp_Yolo_output,obj[:,:,3:],axis=2)

            #print("tmp_new_obj_tmp_Yolo_outputQ\n",tmp_new_obj_tmp_Yolo_output*608)
            if i == 0:
                self.new_obj_tmp_Yolo_output = tmp_new_obj_tmp_Yolo_output
            else:
                self.new_obj_tmp_Yolo_output = np.append(self.new_obj_tmp_Yolo_output,tmp_new_obj_tmp_Yolo_output,axis=0)
        #print(self.new_obj_tmp_Yolo_output)
        
                
    def get_obj_img(self,yolo_trainX,yolo_trainY):  # get the object image with bouding box label 
        self.obj_img = []
        for j in range(yolo_trainY.shape[0]):
            grid_x, grid_y = np.where(yolo_trainY[j,:,:,5] == 1)
            obj = yolo_trainY[j,grid_x,grid_y,:]
            obj_index = np.argsort(obj,axis=0)
            obj = obj[obj_index[:,0]]
            img_space_obj_Y = obj[:,1:5] * 608
            x_left = (img_space_obj_Y[:,0] - img_space_obj_Y[:,2]/2).astype(int)
            x_right = (img_space_obj_Y[:,0] + img_space_obj_Y[:,2]/2).astype(int)
            y_down = (img_space_obj_Y[:,1] - img_space_obj_Y[:,3]/2).astype(int)
            y_top = (img_space_obj_Y[:,1] + img_space_obj_Y[:,3]/2).astype(int)
            for i in range(y_down.shape[0]):
                if i == 0:
                    tmp_img_space_obj_X = yolo_trainX[j,y_down[i]:y_top[i],x_left[i]:x_right[i],:]
                    self.obj_img.append(tmp_img_space_obj_X)
                else:
                    tmp_img_space_obj_X = yolo_trainX[j,y_down[i]:y_top[i],x_left[i]:x_right[i],:]
                    self.obj_img.append(tmp_img_space_obj_X)
              
    def get_new_img(self):  # get the new image
        for j in range(self.output_reshape.shape[0]):
            tmp_img = np.zeros((608, 608, 3), np.uint8)
            tmp_img[:,:,0] = 105
            tmp_img[:,:,1] = 152
            tmp_img[:,:,2] = 143
            #tmp_img[:,:,0] = 255
            #tmp_img[:,:,1] = 255
            #tmp_img[:,:,2] = 255
            #grid_x, grid_y = np.where(yolo_trainY[j,:,:,5] == 1)
            #obj = yolo_trainY[j,grid_x,grid_y,:]
            #obj_index = np.argsort(obj,axis=0)
            #obj = obj[obj_index[:,0]]
            
            new_img_space_obj_Y = (self.new_obj_tmp_Yolo_output[j,:,1:6] * 608).astype(int).reshape(self.obj_num,5)
            #print("new_img_space_obj_\nY",new_img_space_obj_Y)
            #print("new_obj_tmp_Yolo_output\n",self.new_obj_tmp_Yolo_output[j])
            #print("new_obj_tmp_Yolo_output\n",self.new_obj_tmp_Yolo_output[j])
            #print("new_img_space_obj_Y\n",new_img_space_obj_Y)
            #exit()
            new_img_obj_class = (self.new_obj_Class[j,:,0]).astype(int)
            #print("new_img_space_obj_Y\n",new_img_space_obj_Y)
            for i in range(new_img_space_obj_Y.shape[0]):
                #print("new_img_obj_class[i]:",new_img_obj_class[i])
                #print("j",j)
                #print("self.obj_img:",self.obj_img[j*11+new_img_obj_class[i]])
                # the each image has 11 object so need to times 11
                img_Y = self.obj_img[j*11+new_img_obj_class[i]]
                w = img_Y.shape[1]
                h = img_Y.shape[0]
                
                Y_down = new_img_space_obj_Y[i,1]-int(h/2)
                Y_top = new_img_space_obj_Y[i,1]+int(h/2)
                if Y_down < 0:
                    Y_top = Y_top-Y_down
                    self.new_obj_tmp_Yolo_output[j,i,2] = ((self.new_obj_tmp_Yolo_output[j,i,2]*608).astype(int)-Y_down)/608
                    Y_down = 0
                if Y_top > 607:
                    Y_down = Y_down-(Y_top-607)
                    self.new_obj_tmp_Yolo_output[j,i,2] = ((self.new_obj_tmp_Yolo_output[j,i,2]*608).astype(int)-(Y_top-607))/608    
                    Y_top = 607            
                
                X_left = new_img_space_obj_Y[i,0]-int(w/2)
                X_right = new_img_space_obj_Y[i,0]+int(w/2)
                if X_left < 0:
                    X_right = X_right-X_left
                    self.new_obj_tmp_Yolo_output[j,i,1] = ((self.new_obj_tmp_Yolo_output[j,i,1]*608).astype(int)-X_left)/608
                    X_left = 0
                    
                if X_right > 607:
                    X_left = X_left-(X_right-607)
                    self.new_obj_tmp_Yolo_output[j,i,1] = ((self.new_obj_tmp_Yolo_output[j,i,1]*608).astype(int)-(X_right-607))/608
                    X_right = 607
                   
                if w % 2 != 0:
                    X_right = X_right + 1
                if h % 2 != 0:
                    Y_top = Y_top + 1
                #global nnn
                #print("Y_down",Y_down," X_left:",X_left)
                tmp_img[Y_down:Y_top, X_left:X_right,:] = img_Y
            #cv2.imwrite('Q_Q'+str(nnn)+'.jpg', tmp_img)
            #nnn=nnn+1
            #print("new_obj_tmp_Yolo_output\n",self.new_obj_tmp_Yolo_output[j])
            if j == 0:
                self.new_img = tmp_img.reshape(-1,608,608,3)
            else:
                self.new_img = np.append(self.new_img,tmp_img.reshape(-1,608,608,3),axis=0)
        #exit()
                
    def get_new_obj_Yolo_output(self): # get the new image label [x,y,w,h,confident,class]
        self.new_obj_Yolo_output = np.ones([self.new_img.shape[0],11,11,6])
        self.new_obj_Yolo_output[:,:,:,5] = np.multiply(self.new_obj_Yolo_output[:,:,:,5],0)
        for i in range(train_dataX.shape[0]):
            x = np.multiply(self.new_obj_tmp_Yolo_output[i,:,1].reshape(-1),10).astype(int)
            y = np.multiply(self.new_obj_tmp_Yolo_output[i,:,2].reshape(-1),10).astype(int)
            self.new_obj_Yolo_output[i,x,y,:] = self.new_obj_tmp_Yolo_output[i]


# In[5]:


class GA():
    def __init__(self,darknet_path, obj_data_path, cfg_path, weight_path):
        self.Generation_num = 20
        self.darknet_path = darknet_path
        self.obj_data_path = obj_data_path
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.bitNum = 16
        
    def set_weight_path(self,newModel_path):
        self.weight_path = newModel_path
    
    def set_gener_model(self):
        self.gener_model = []
        for i in range(self.Generation_num):
            self.gener_model.append(Generate_Model())
            
    def normalize(self,x):
        if np.min(x) == np.max(x):
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))
    
    def softmax(self,x):
        return np.exp(x) * (x > 0)/ np.sum(np.exp(x) * (x > 0))
    
    def float_to_bit(self,array):
        tmp_bit = np.zeros([array.shape[0],array.shape[1],self.bitNum])
        tmp_array = array

        first_dim,second_dim = np.where(tmp_array[:,:]<0)
        tmp_bit[first_dim,second_dim,0] = 1
        tmp_array = np.abs(tmp_array)
        for i in range(self.bitNum-1):
            tmp_array = tmp_array*2
            first_dim,second_dim = np.where(tmp_array>1)
            tmp_bit[first_dim,second_dim,i+1] = 1
#             tmp_array = np.mod(tmp_array,1)
            tmp_array[first_dim,second_dim] = tmp_array[first_dim,second_dim]-1
        return tmp_bit
    
    def bit_to_float(self ,array):
        tmp_float = np.zeros([array.shape[0],array.shape[1],1])
        tmp_array = array

        for i in range(self.bitNum-1):
            first_dim,second_dim = np.where(tmp_array[:,:,i+1]==1)
            tmp_float[first_dim,second_dim,0] = tmp_float[first_dim,second_dim,0] + 1/np.power(2,i+1)

        first_dim,second_dim = np.where(tmp_array[:,:,0]==1)
        tmp_float[first_dim,second_dim,0] = tmp_float[first_dim,second_dim,0] * -1

        return tmp_float.reshape(array.shape[0],array.shape[1])

    def run_gener_model(self,genera_input,img,label):
        for i in range(self.Generation_num):
            self.gener_model[i].run(genera_input)
            self.gener_model[i].get_new_obj_XY_Class()
            self.gener_model[i].get_new_obj_tmp_Yolo_output(label)
            self.gener_model[i].get_obj_img(img,label)
            self.gener_model[i].get_new_img()
            self.gener_model[i].get_new_obj_Yolo_output()

    def DIQC(self,bbox1,bbox2):
        flag = 0
        bbox1_left  = int((bbox1[0] - bbox1[2]/2) * 608)
        bbox1_right = int((bbox1[0] + bbox1[2]/2) * 608)
        bbox1_down  = int((bbox1[1] - bbox1[3]/2) * 608)
        bbox1_top   = int((bbox1[1] + bbox1[3]/2) * 608)

        bbox2_left  = int((bbox2[0] - bbox2[2]/2) * 608)
        bbox2_right = int((bbox2[0] + bbox2[2]/2) * 608)
        bbox2_down  = int((bbox2[1] - bbox2[3]/2) * 608)
        bbox2_top   = int((bbox2[1] + bbox2[3]/2) * 608)


        if (bbox1_right > bbox2_left and bbox1_left < bbox2_right):
            flag = flag + 1
        if (bbox1_top > bbox2_down and bbox1_down < bbox2_top):
            flag = flag + 1

        if flag == 2:
            return 1
        else:
            return 0 

    def calculate_Adaptability(self,iteration,GA_epochs):
        Adaptability = []
        print("calculate_Adaptability")
        for i in range(self.Generation_num):
            img = self.gener_model[i].new_img
            Yolo_ans = self.gener_model[i].new_obj_Yolo_output
            """
            # if DIQC return 1 then will delete that image because its not reasonable
            quality_index = []
            overLapping_count = 0
            for label_index in range(Yolo_ans.shape[0]):
                overLapping = 0
                yolo_output = Yolo_ans[label_index]
                first_dim,second_dim = np.where(yolo_output[:,:,5]==1)
                obj_info = yolo_output[first_dim,second_dim,:]
                for k1 in range(obj_info.shape[0]):
                    for k2 in range(obj_info.shape[0]):
                        if k1 != k2:
                            overLapping = overLapping + self.DIQC(obj_info[k1,1:5],obj_info[2,1:5])

                #for k in range(obj_info.shape[0]):
                #    for j in range(obj_info.shape[0]-(k+1)):
                #        j = j+k+1
                #        overLapping = overLapping + self.DIQC(obj_info[k,1:5],obj_info[j,1:5])                

                overLapping_count += overLapping
                print("overLapping_count:",overLapping_count)
                if  overLapping == 0:
                    quality_index.append(label_index)
                else :
                    quality_index.append(label_index)

            print("quality_index:",quality_index)
            img = img[quality_index]
            Yolo_ans = Yolo_ans[quality_index]
            """
            if i == 0:
                path = "/home/userr/ncku/LeGAN_Log_Time/Data/GA/" + str(n) + "_pages/" + str(iteration) + "_iteration/"
                if os.path.isdir(path) == False:
                    os.mkdir(path) # make the new file
            
            path = "/home/userr/ncku/LeGAN_Log_Time/Data/GA/" + str(n) + "_pages/" + str(iteration) + "_iteration/" + str(GA_epochs) + "epochs_" + str(i) + "chromosome"   
            
            if os.path.isdir(path):  # if has the file then remove it 
                shutil.rmtree(path)
            os.mkdir(path) # make the new file
            for k in range(Yolo_ans.shape[0]):
                first_dim,second_dim = np.where(Yolo_ans[k,:,:,5])
                obj_info = Yolo_ans[k,first_dim,second_dim,:]
                # write the text of label because the yolo calculate loss need to load text type file
                with open(path+'/'+str(k)+'.txt','w') as f:
                    #for j in range(obj_info.shape[0]-1,-1,-1):
                    for j in range(obj_info.shape[0]):
                        #print("j:",j)
                        obj = obj_info[j]
                        """
                        line = ''
                        for data in range(len(obj) - 1):
                            if data < len(obj) - 2:
                                line = line + str(obj[data])[:7] + ' '
                            else:
                                line = line + str(obj[data])[:7] + '\n'
                        """
                        is_overLapping=0;
                        #if obj_info.shape[0] != j:
                        for chk in range(obj_info.shape[0]):
                            if chk != j:
                                is_overLapping = is_overLapping + self.DIQC(obj[1:5],obj_info[chk,1:5])
                                #print("chk:",chk,"  OL:",is_overLapping)
                        if is_overLapping == 0:
                            line='%d %.6f %.6f %.6f %.6f\n'%(obj[0],obj[1],obj[2],obj[3],obj[4])
                            f.write(line)

#             print(img.shape)
            # create new image because the yolo calculate loss need
            for j in range(img.shape[0]):
                cv2.imwrite(path + '/' + str(j)+'.jpg',img[j])
            
            # the function loss_calculate need the data list to load data
            loss_path_list = []
            for j in range(img.shape[0]):
                loss_path_list.append(path + '/' + str(j)+'.jpg')
            loss = 0
            #exit()
            if(len(loss_path_list) > 0):
                subdiv = len(loss_path_list)/5
                if(np.mod(subdiv,1))!=0:
                    change_loss_parameters(len(loss_path_list),int(subdiv)+1, cfg_path)
                else:
                    change_loss_parameters(len(loss_path_list),int(subdiv), cfg_path)
                loss = loss_calculate(loss_path_list, self.darknet_path, self.obj_data_path,                                              self.cfg_path, self.weight_path)
                #loss = loss - overLapping_count * 5
                #loss=1000
            # if all image is reasonable then Adaptability is 0,so it will hard to crossover 
            if img.shape[0] > 0:
                #Adaptability.append(loss)
                Adaptability.append(loss/img.shape[0])
            else :
                Adaptability.append(0)
            if GA_epochs == 0:
                best_model_index = np.argmax(Adaptability)
                self.best_model_W = self.gener_model[best_model_index].gener_W
                self.best_model_B = self.gener_model[best_model_index].gener_B
                self.best_model_Adaptability = Adaptability[best_model_index]
            else:
                best_model_index = np.argmax(Adaptability)
                if best_model_index > self.best_model_Adaptability:
                    self.best_model_W = self.gener_model[best_model_index].gener_W
                    self.best_model_B = self.gener_model[best_model_index].gener_B
                    self.best_model_Adaptability = Adaptability[best_model_index]
        #print(Adaptability)
        print('calculate_Adaptability_done')
        return Adaptability        
    
    def get_adaptability_probability(self,k,GA_epochs):
        self.adaptability = self.calculate_Adaptability(k,GA_epochs)
        print('adaptability: ',self.adaptability)
        self.adaptability_normalize = self.normalize(np.asarray(self.adaptability))
        self.adaptability_softmax = self.softmax(self.adaptability_normalize)
        tmp = []
        for i in range(self.adaptability_softmax.shape[0]):
            tmp.append(np.sum(self.adaptability_softmax[:i+1]))
        self.adaptability_probability = np.asarray(tmp)
        
    def run_gener_model_weight(self): # The crossover part for generator model's weight
        
        self.new_gener_model_weight=[]

        for k in range(int(self.Generation_num/2)):
            #print(k)
            choose_random = np.random.rand(1)
            choose_gener = np.where(self.adaptability_probability >= choose_random)[0]
            if choose_gener.shape[0] == 0:
                choose_gener = [0]
            father_model = self.gener_model[choose_gener[0]]
            choose_random = np.random.rand(1)
            choose_gener = np.where(self.adaptability_probability >= choose_random)[0]
            if choose_gener.shape[0] == 0:
                choose_gener = [0]
            mother_model = self.gener_model[choose_gener[0]]

            new1_gener_W = []
            new2_gener_W = []
            for i in range(len(father_model.gener_W)):
                father_generW_bit = self.float_to_bit(father_model.gener_W[i])
                mother_generW_bit = self.float_to_bit(mother_model.gener_W[i])

                father_generW_bit = father_generW_bit.reshape(father_generW_bit.shape[0],-1)
                mother_generW_bit = mother_generW_bit.reshape(mother_generW_bit.shape[0],-1)

                tmp1_gener_W = np.zeros(father_generW_bit.shape)
                tmp2_gener_W = np.zeros(father_generW_bit.shape)

                choose_mating_index = np.random.randint(1,mother_generW_bit.shape[1]-1)
                father_head =  father_generW_bit[:,:choose_mating_index]
                father_tail =  father_generW_bit[:,choose_mating_index:]

                mother_head =  mother_generW_bit[:,:choose_mating_index]
                mother_tail =  mother_generW_bit[:,choose_mating_index:]
                tmp1_gener_W[:,:] = np.append(father_head,mother_tail,axis=1)
                tmp2_gener_W[:,:] = np.append(mother_head,father_tail,axis=1)

                tmp1_gener_W = tmp1_gener_W.reshape(father_model.gener_W[i].shape[0],father_model.gener_W[i].shape[1],self.bitNum)
                tmp2_gener_W = tmp2_gener_W.reshape(father_model.gener_W[i].shape[0],father_model.gener_W[i].shape[1],self.bitNum)

                mutation_probability = np.random.rand(1) 
                if mutation_probability < 0.2:
                    mutation_index =  np.random.randint(1,self.bitNum)
                    zero_first_dim,zero_second_dim = np.where(tmp1_gener_W[:,:,mutation_index] == 0.)
                    one_first_dim,one_second_dim = np.where(tmp1_gener_W[:,:,mutation_index] == 1.)

                    tmp1_gener_W[zero_first_dim,zero_second_dim,mutation_index] = 1.
                    tmp1_gener_W[one_first_dim,one_second_dim,mutation_index] = 0.

                    zero_first_dim,zero_second_dim = np.where(tmp2_gener_W[:,:,mutation_index] == 0.)
                    one_first_dim,one_second_dim = np.where(tmp2_gener_W[:,:,mutation_index] == 1.)
                    tmp2_gener_W[zero_first_dim,zero_second_dim,mutation_index] = 1.
                    tmp2_gener_W[one_first_dim,one_second_dim,mutation_index] = 0.

                new1_gener_W.append(self.bit_to_float(tmp1_gener_W))
                new2_gener_W.append(self.bit_to_float(tmp2_gener_W))
            self.new_gener_model_weight.append(new1_gener_W)
            self.new_gener_model_weight.append(new2_gener_W)

        print('run_gener_model_weight_done')
        
    def run_gener_model_bias(self): # The crossover part for generator model's bias
        self.new_gener_model_bias=[]
        for k in range(int(self.Generation_num/2)):
            #print(k)
            choose_random = np.random.rand(1)
            choose_gener = np.where(self.adaptability_probability >= choose_random)[0]
            father_model = self.gener_model[choose_gener[0]]
            choose_random = np.random.rand(1)
            choose_gener = np.where(self.adaptability_probability >= choose_random)[0]
            mother_model = self.gener_model[choose_gener[0]]

            new1_gener_B = []
            new2_gener_B = []
            for i in range(len(father_model.gener_W)):
                father_generB_bit = self.float_to_bit(father_model.gener_B[i].reshape(1,-1))
                mother_generB_bit = self.float_to_bit(mother_model.gener_B[i].reshape(1,-1))

                father_generB_bit = father_generB_bit.reshape(father_generB_bit.shape[0],-1)
                mother_generB_bit = father_generB_bit.reshape(father_generB_bit.shape[0],-1)

                tmp1_gener_B = np.zeros(father_generB_bit.shape)
                tmp2_gener_B = np.zeros(father_generB_bit.shape)

                choose_mating_index = np.random.randint(1,mother_generB_bit.shape[1]-1)
                father_head =  father_generB_bit[:,:choose_mating_index]
                father_tail =  father_generB_bit[:,choose_mating_index:]

                mother_head =  mother_generB_bit[:,:choose_mating_index]
                mother_tail =  mother_generB_bit[:,choose_mating_index:]
                tmp1_gener_B[:,:] = np.append(father_head,mother_tail,axis=1)
                tmp2_gener_B[:,:] = np.append(mother_head,father_tail,axis=1)

                tmp1_gener_B = tmp1_gener_B.reshape(1,father_model.gener_B[i].shape[0],self.bitNum)
                tmp2_gener_B = tmp2_gener_B.reshape(1,father_model.gener_B[i].shape[0],self.bitNum)

                mutation_probability = np.random.rand(1) 
                if mutation_probability < 0.2:
                    mutation_index =  np.random.randint(1,self.bitNum)
                    zero_first_dim,zero_second_dim = np.where(tmp1_gener_B[:,:,mutation_index] == 0.)
                    one_first_dim,one_second_dim = np.where(tmp1_gener_B[:,:,mutation_index] == 1.)

                    tmp1_gener_B[0,zero_second_dim,mutation_index] = 1.
                    tmp1_gener_B[0,one_second_dim,mutation_index] = 0.

                    zero_first_dim,zero_second_dim = np.where(tmp2_gener_B[:,:,mutation_index] == 0.)
                    one_first_dim,one_second_dim = np.where(tmp2_gener_B[:,:,mutation_index] == 1.)
                    tmp2_gener_B[0,zero_second_dim,mutation_index] = 1.
                    tmp2_gener_B[0,one_second_dim,mutation_index] = 0.

                new1_gener_B.append(self.bit_to_float(tmp1_gener_B).reshape(-1))
                new2_gener_B.append(self. bit_to_float(tmp1_gener_B).reshape(-1))
            self.new_gener_model_bias.append(new1_gener_B)
            self.new_gener_model_bias.append(new2_gener_B)
        print('run_gener_model_bias_done')
    
    def change_mode(self):
        for i in range(self.Generation_num):
            self.gener_model[i].set_model(self.new_gener_model_weight[i],self.new_gener_model_bias[i])
    
    def get_target_model(self):
        adaptability = np.asarray(self.adaptability)
        model_index = np.argmax(adaptability)
        #print(model_index)
        return self.gener_model[model_index]


# In[6]:


class LBIG:
    def __init__(self):
        self.gener_W = []
        self.gener_B = []
        self.gener_W.append(np.random.uniform(-1,1,[input_dim,hidden1_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden1_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden1_dim,hidden2_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden2_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden2_dim,hidden3_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden3_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden3_dim,hidden4_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden4_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden4_dim,hidden5_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden5_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden5_dim,hidden6_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[hidden6_dim]).astype(np.float64))
        self.gener_W.append(np.random.uniform(-1,1,[hidden6_dim,output_dim]).astype(np.float64))
        self.gener_B.append(np.random.uniform(-1,1,[output_dim]).astype(np.float64))
        self.obj_num = 11
        
    def set_model(self,gener_W,gener_B):
        self.gener_W = gener_W
        self.gener_B = gener_B
        
    def sigmoid(self,x):
        sigm = 1. / (1. + np.exp(-x))
        return sigm
    def tanh(self,x):
        return np.tanh(x)
    def relu(self,x):
        relu = x * (x>0)
        return relu
    def softmax_Class(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=3).reshape([-1,11,11,1])
    def softmax_Cell(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape([-1,1])
    
    def run(self,data):
        self.gener_Layer_output = []
        self.gener_Layer_output.append(np.add(np.matmul(data,self.gener_W[0]),self.gener_B[0]))
        self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
        for i in range(5):
            self.gener_Layer_output.append(np.add(np.matmul(self.gener_Layer_output[-1],self.gener_W[i+1]),self.gener_B[i+1]))
            self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
        self.gener_Layer_output.append(np.add(np.matmul(self.gener_Layer_output[-1],self.gener_W[i+2]),self.gener_B[i+2]))
        self.gener_Layer_output[-1] = self.sigmoid(self.gener_Layer_output[-1])
            
        self.output_reshape = self.gener_Layer_output[-1].reshape(-1,11,11,generator_output_dim)
        self.output_BB = self.output_reshape[:,:,:,:2]
        self.output_Cell = self.softmax_Cell(self.output_reshape[:,:,:,3].reshape(-1,11*11))
        self.output_Class = self.softmax_Class(self.output_reshape[:,:,:,3:])
    
    # it's like generator but the quantity will change with density
    def get_new_obj_XY_Class(self,density):
        for i in range(self.output_reshape.shape[0]):
            nlarg_Cell = heapq.nlargest(self.obj_num*density, range(len(self.output_Cell[i])), self.output_Cell[i].take)
            output_Class_MaxIndex = np.argmax(self.output_Class,axis=3).reshape([-1,11,11,1])
            if density == 1:
                tmp_obj_XY = self.output_BB[i].reshape(-1,11*11,2)[:,nlarg_Cell,:]
            else:
                tmp_obj_XY = np.ones([1,11*density,2]) * 0.5 # the coordinates in cell space is always in the center
#                 tmp_obj_XY = self.output_BB[i].reshape(-1,11*11,2)[:,nlarg_Cell,:]
    
            tmp_obj_Class = output_Class_MaxIndex[0].reshape(-1,11*11,1)[:,nlarg_Cell,:]
            if i == 0:
                self.new_obj_XY = tmp_obj_XY
                self.new_obj_Class = tmp_obj_Class
            else:
                #new_obj_XY = [img1_density*obj_num,img2_density*obj_num,.....]
                self.new_obj_XY = np.append(self.new_obj_XY,tmp_obj_XY,axis=0)
                self.new_obj_Class = np.append(self.new_obj_Class,tmp_obj_Class,axis=0)
                
    def get_new_obj_tmp_Yolo_output(self,yolo_trainY,density):
        for i in range(self.output_reshape.shape[0]):
            nlarg_Cell = heapq.nlargest(self.obj_num*density, range(len(self.output_Cell[i])), self.output_Cell[i].take)
            ten_digit = np.divide(nlarg_Cell,10).astype(int)
            digit = np.mod(nlarg_Cell,10)
            new_obj_Yolo_X = np.divide(digit,10) + np.divide(self.new_obj_XY[i,:,0],10)
            new_obj_Yolo_Y = np.divide(ten_digit,10) + np.divide(self.new_obj_XY[i,:,1],10)
            #tmp_new_obj_tmp_Yolo_output = np.append(new_obj_Yolo_X.reshape(-1,self.obj_num*density,1),new_obj_Yolo_Y.reshape(-1,self.obj_num*density,1),axis=2)
            int_11=9
            our_work_address_xy=np.arange(33,608,67)
            create64 = np.arange(0,81,1)
            np.random.shuffle(create64)
            #print("self.obj_num*density",self.obj_num*density)
            create64=create64[0:self.obj_num*density]
            
            new_obj_Yolo_X = our_work_address_xy[create64%int_11]/608
            #print(create64%8)
            create64=create64/int_11
            create64=create64.astype(int)
            new_obj_Yolo_Y = our_work_address_xy[create64]/608
            tmp_new_obj_tmp_Yolo_output = np.append(new_obj_Yolo_X.reshape(-1,self.obj_num*density,1),new_obj_Yolo_Y.reshape(-1,self.obj_num*density,1),axis=2)
            

            for density_time in range(density):
                grid_x, grid_y = np.where(yolo_trainY[i,density_time,:,:,5] == 1)
                if density_time == 0:
                    tmp_obj = yolo_trainY[i,density_time,grid_x,grid_y,:]
                    obj_index = np.argsort(tmp_obj,axis=0) # sorting is more easily to make the new image,because the index is equal the class
                    obj = tmp_obj[obj_index[:,0]]
                else:
                    tmp_obj = yolo_trainY[i,density_time,grid_x,grid_y,:]
                    obj_index = np.argsort(tmp_obj,axis=0)
                    obj = np.append(obj,tmp_obj[obj_index[:,0]],axis=0)
            # random to choose the obj,because when density != 0 the same class object quatity is density
            if i == 0 :
                self.index = np.random.randint(density,size=(self.obj_num*density,1)).reshape(-1,self.obj_num*density,1)
            else:
                self.index = np.append(self.index,np.random.randint(density,size=(self.obj_num*density,1)).reshape(-1,self.obj_num*density,1),axis=0)
        
            # get obj by the class and the random,times obj_num because the one image has obj_num objs
            obj = obj[(self.new_obj_Class[i]+self.index[i]*self.obj_num).reshape(-1)].reshape(1,self.obj_num*density,6)
            tmp_new_obj_tmp_Yolo_output = np.append(obj[:,:,0].reshape([-1,self.obj_num*density,1]),tmp_new_obj_tmp_Yolo_output,axis=2)
            tmp_new_obj_tmp_Yolo_output = np.append(tmp_new_obj_tmp_Yolo_output,obj[:,:,3:].reshape([-1,self.obj_num*density,3]),axis=2)
            if i == 0:
                self.new_obj_tmp_Yolo_output = tmp_new_obj_tmp_Yolo_output
            else:
                self.new_obj_tmp_Yolo_output = np.append(self.new_obj_tmp_Yolo_output,tmp_new_obj_tmp_Yolo_output,axis=0)

    def get_obj_img(self,yolo_trainX,yolo_trainY,density):
        self.obj_img = []
        for j in range(self.output_reshape.shape[0]):
            for density_time in range(density):
                grid_x, grid_y = np.where(yolo_trainY[j,density_time,:,:,5] == 1)
                
                obj = yolo_trainY[j,density_time,grid_x,grid_y,:]
                obj_index = np.argsort(obj,axis=0)
                obj = obj[obj_index[:,0]]
                
                img_space_obj_Y = obj[:,1:6] * 608
                
                x_left = (img_space_obj_Y[:,0] - img_space_obj_Y[:,2]/2).astype(int)
                x_right = (img_space_obj_Y[:,0] + img_space_obj_Y[:,2]/2).astype(int)
                y_down = (img_space_obj_Y[:,1] - img_space_obj_Y[:,3]/2).astype(int)
                y_top = (img_space_obj_Y[:,1] + img_space_obj_Y[:,3]/2).astype(int)
                
                # obj_img(img1_density1_obj,img1_density2_obj,....,img2_density1_obj,img2_density2_obj,......)
                for i in range(y_down.shape[0]):
                    self.obj_img.append(yolo_trainX[j,density_time,y_down[i]:y_top[i],x_left[i]:x_right[i],:])
    
    def get_new_img(self,density):
        for j in range(self.output_reshape.shape[0]):
            tmp_img = np.zeros((608, 608, 3), np.uint8)
            tmp_img[:,:,0] = 105
            tmp_img[:,:,1] = 152
            tmp_img[:,:,2] = 143
            #tmp_img[:,:,0] = 255
            #tmp_img[:,:,1] = 255
            #tmp_img[:,:,2] = 255
            
            
            new_img_space_obj_Y = (self.new_obj_tmp_Yolo_output[j,:,1:6]*608).astype(int).reshape(self.obj_num*density,5)
            new_img_obj_class = (self.new_obj_Class[j,:,0]).astype(int)

            for density_time in range(density):
                for i in range(int(new_img_space_obj_Y.shape[0]/density)):
                    # self.obj_num * density * j to calculate the image
                    # self.new_obj_Class[j,i+density_time*self.obj_num] to get class number
                    # self.index[j,i+density_time*self.obj_num]*self.obj_num to calcualte the random in obj_img
                    img = self.obj_img[(self.obj_num*density*j+self.new_obj_Class[j,i+density_time*self.obj_num] + self.index[j,i+density_time*self.obj_num]*self.obj_num)[0]]
                    
                    
                    w = img.shape[1]
                    h = img.shape[0]

                    Y_down = new_img_space_obj_Y[i+density_time*self.obj_num,1]-int(h/2)
                    Y_top = new_img_space_obj_Y[i+density_time*self.obj_num,1]+int(h/2)
                    
                    if Y_down < 0:
                        Y_top = Y_top-Y_down
                        self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,2] = ((self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,2]*608).astype(int)-Y_down)/608
                        Y_down = 0
                    if Y_top > 607:
                        Y_down = Y_down-(Y_top-607)
                        self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,2] = ((self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,2]*608).astype(int)-(Y_top-607))/608    
                        Y_top = 607            

                    X_left = new_img_space_obj_Y[i+density_time*self.obj_num,0]-int(w/2)
                    X_right = new_img_space_obj_Y[i+density_time*self.obj_num,0]+int(w/2)
                    if X_left < 0:
                        X_right = X_right-X_left
                        self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,1] = ((self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,1]*608).astype(int)-X_left)/608
                        X_left = 0

                    if X_right > 607:
                        X_left = X_left-(X_right-607)
                        self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,1] = ((self.new_obj_tmp_Yolo_output[j,i+density_time*self.obj_num,1]*608).astype(int)-(X_right-607))/608
                        X_right = 607

                    if w % 2 != 0:
                        X_right = X_right + 1
                    if h % 2 != 0:
                        Y_top = Y_top + 1
                        
                    tmp_img[Y_down:Y_top, X_left:X_right,:] = img

            if j == 0:
                self.new_img = tmp_img.reshape(-1,608,608,3)
            else:
                self.new_img = np.append(self.new_img,tmp_img.reshape(-1,608,608,3),axis=0)

                
    def get_new_obj_Yolo_output(self):
        self.new_obj_Yolo_output = np.ones([self.new_img.shape[0],11,11,6])
        self.new_obj_Yolo_output[:,:,:,5] = np.multiply(self.new_obj_Yolo_output[:,:,:,5],0)

        for i in range(self.new_obj_tmp_Yolo_output.shape[0]):
            x = np.multiply(self.new_obj_tmp_Yolo_output[i,:,1].reshape(-1),10).astype(int)
            y = np.multiply(self.new_obj_tmp_Yolo_output[i,:,2].reshape(-1),10).astype(int)
            self.new_obj_Yolo_output[i,y,x,:] = self.new_obj_tmp_Yolo_output[i]


# In[7]:


def DIQC(bbox1,bbox2):
    flag = 0
    bbox1_left  = int((bbox1[0] - bbox1[2]/2) * 608)
    bbox1_right = int((bbox1[0] + bbox1[2]/2) * 608)
    bbox1_down  = int((bbox1[1] - bbox1[3]/2) * 608)
    bbox1_top   = int((bbox1[1] + bbox1[3]/2) * 608)
     
    bbox2_left  = int((bbox2[0] - bbox2[2]/2) * 608)
    bbox2_right = int((bbox2[0] + bbox2[2]/2) * 608)
    bbox2_down  = int((bbox2[1] - bbox2[3]/2) * 608)
    bbox2_top   = int((bbox2[1] + bbox2[3]/2) * 608)
    
    
    if (bbox1_right > bbox2_left and bbox1_left < bbox2_right):
        flag = flag + 1
    if (bbox1_top > bbox2_down and bbox1_down < bbox2_top):
        flag = flag + 1
    
    if flag == 2:
        return 1
    else:
        return 0


# In[8]:


def DESIMQC(quilty_array):
    
    flag = 1
    for i in range(quilty_array.shape[0]):
        if quilty_array[i] > 0.5:
            flag = 0
            break
    return flag
            


# In[9]:


generator_output_dim = 14


# In[10]:


density_list = [1,2,4,6]


# In[11]:


import numpy as np
input_dim = 11*density_list[-1]*3
hidden1_dim = np.power(2,int(np.log2(input_dim) + 1) + 2)
hidden2_dim = int(hidden1_dim / 2)
hidden3_dim = int(hidden2_dim / 2)
hidden4_dim = 300
hidden5_dim = hidden4_dim * 4
hidden6_dim = hidden5_dim * 4
output_dim = 11*11*(14)

print("hidden3_dim:",hidden3_dim)
print("output_dim:",output_dim)


# In[12]:


yolo_trainX = np.load('/home/userr/ncku/LeGAN_Log_Time/Data/CASE_Data/Bean_dataX.npy')
yolo_trainY = np.load('/home/userr/ncku/LeGAN_Log_Time/Data/CASE_Data/Bean_dataY_transpose.npy')
print("yolo_trainX.shape:",yolo_trainX.shape)
print("yolo_trainY.shape:",yolo_trainY.shape)


# train_dataX = train_dataX[35:64,:,:]
train_dataX = yolo_trainY[:n,:]
yolo_trainX = yolo_trainX[:n,:]
yolo_trainY = yolo_trainY[:n,:]
print("train_dataX.shape:",train_dataX.shape)
print("yolo_trainX.shape:",yolo_trainX.shape)
#print("\n\n",train_dataX,"\n\n")

"""
for iii in range(yolo_trainY.shape[0]):

    #print(yolo_trainY[iii])
    objj = yolo_trainY[iii]
    with open('Q_Q'+str(iii)+'_2.txt','w') as f:
        for jjj in range(objj.shape[0]):
            objjj=objj[jjj]
            print(objjj)
            line='%d %.6f %.6f %.6f %.6f\n'%(objjj[0],objjj[1],objjj[2],objjj[3],objjj[4])
            f.write(line)

    cv2.imwrite('Q_Q'+str(iii-1)+'_3.jpg', yolo_trainX[iii-1])
"""
"""
for iii in range(yolo_trainY.shape[0]):
    print(iii)
    cv2.imwrite('Q_Q'+str(iii)+'_3.jpg', yolo_trainX[iii])
exit() 
"""
"""
cv2.imwrite('Q_Q'+str(0)+'_2.jpg', yolo_trainX[0])
cv2.imwrite('Q_Q'+str(1)+'_2.jpg', yolo_trainX[1])
cv2.imwrite('Q_Q'+str(2)+'_2.jpg', yolo_trainX[2])
cv2.imwrite('Q_Q'+str(3)+'_2.jpg', yolo_trainX[3])
cv2.imwrite('Q_Q'+str(4)+'_2.jpg', yolo_trainX[4])
exit()  
"""
# In[13]:


tmp = np.asarray([])
for i in range(train_dataX.shape[0]):
    grid_x, grid_y = np.where(train_dataX[i,:,:,5] == 1)
    if i == 0:
        tmp = train_dataX[i,grid_x,grid_y,0:3].reshape(1,-1,3)
    else:
        tmp = np.append(tmp,train_dataX[i,grid_x,grid_y,0:3].reshape(1,-1,3),axis=0)
#print(tmp.shape)
generate_input = tmp
img_obj_info = tmp


# In[14]:


tmp = np.asarray([])
for i in range(generate_input.shape[0]):
    #print("train_dataX[i,:,:,5]:",train_dataX[i,:,:,:] )
    grid_x, grid_y = np.where(train_dataX[i,:,:,5] == 1)
    if i == 0:
        tmp  = np.tile(train_dataX[i,grid_x,grid_y,:3].reshape(1,-1),density_list[-1])
    else:
        tmp  = np.append(tmp,np.tile(train_dataX[i,grid_x,grid_y,:3].reshape(1,-1),density_list[-1]),axis = 0)

#print("tmp.shape:",tmp.shape)
generate_input = tmp
#print("generate_input:\n",generate_input[0][0:20])
#print("generate_input.shape:\n",generate_input.shape)
img = yolo_trainX
label = yolo_trainY
#print(img.shape)
#print(label.shape)


# In[15]:


import darknet_function.python.darknet as dn
import subprocess
import sys, string, os
import time
from os import walk

def train_network(darknet_path, obj_data_path, data_cfg, weight, epoch):
    print(darknet_path + " -i 0"+" detector train %s %s %s %d " % (obj_data_path, data_cfg, weight, epoch))
    os.system(darknet_path + " -i 0"+" detector train %s %s %s %d " % (obj_data_path, data_cfg, weight, epoch))
    
    
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
    cmd = darknet_path + " -i 0" + " detector loss" + " " + obj_data_path + " " + data_cfg + " " + weight + " 0"
    print(cmd)
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True,  stderr=subprocess.DEVNULL)
    (out, err) = proc.communicate()
    out = str(out, 'utf-8')
#     print("out :",out)
#     time.sleep(5)
    output_array = out.split("Loss:")
    if len(output_array) > 1:
        loss = output_array[1].split('\n')[0]
    else:
        loss = 0
    
    print("loss_calculate funtion loss=:",loss)
    return float(loss)

def get_bounding_box(picture_path, obj_data_path, data_cfg, weight):
    net = dn.load_net(str.encode(data_cfg), str.encode(weight), 0)
    meta = dn.load_meta(str.encode(obj_data_path))
    all_box = dn.detect(net, meta, str.encode(picture_path))

    return all_box


# In[16]:


# mage_path = "/home/mmdb/Desktop/darknet_function/truedata/loss/3.jpg"
darknet_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/darknet"
obj_data_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/obj.data"
cfg_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/yolov3.cfg"
# init_weight_path = "/home/userr/ncku/LeGAN/darknet_function/truedata/darknet53.conv.74"
#init_weight_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/weights/LD_train/yolov3_Basic.weights"
init_weight_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/yolov3_160000.weights"



so_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/libdarknet.so"



model_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/weights/"

#train_path = "/home/userr/ncku/LeGAN_Log_Time/BeanData/Train_data/" + str(n) + "_pages/"

GA_epochs = 1
epoch = 10000

# train_network(darknet_path, obj_data_path, cfg_path, init_weight_path, epoch)


# In[17]:


# train_network(darknet_path, obj_data_path, cfg_path, init_weight_path, epoch)
# aaaa


# In[18]:


# yolo_trainData_path = "/home/userr/ncku/LeGAN/darknet_function/truedata/cfg/"
# filepath = "/home/userr/ncku/LeGAN/Data/LBIG_darknet_noZero/"
# for root, dirs, files in walk(filepath):
#     if len(dirs) > 0:
#         dir_list = dirs

# with open(yolo_trainData_path + "train.txt","w") as f:
#     for one_dir in dir_list:
#         print(one_dir)
#         for root, dirs, files in walk(filepath + one_dir + "/"):
#             for file in files:
#                 fileType = os.path.splitext(filepath + one_dir + "/" + file)[1]
#                 fileName = os.path.splitext(filepath + one_dir + "/" + file)[0]
#                 if fileType == '.jpg' :
#                     print(filepath + one_dir + "/" + file)
#                     f.write(filepath + one_dir + "/" + file+"\n")


# In[19]:


# yolo_trainData_path = "/home/userr/ncku/Defect_Bean_Batch_GPU0/darknet_function/truedata/cfg/"
# savepath = "/home/userr/ncku/Defect_Bean_Batch_GPU0/BeanData/"

# with open(yolo_trainData_path +'train.txt','w') as f:
#     for k in range(n):
#         dataLine = savepath + "Train_data/"+str(n)+"_pages/"+ str(k)+'.jpg\n'
#         f.write(dataLine)


# In[20]:


# yolo_trainData_path = "/home/userr/ncku/Defect_Bean_Batch_GPU0/darknet_function/truedata/cfg/"
# loss_path_list = []
# with open(yolo_trainData_path +'train.txt','r') as f:
#     lines = f.readlines()
#     for line in lines:
#         loss_path_list.append(line.split('\n')[0])
# #         aaa
# #         dataLine = savepath + "Train_data/"+str(n)+"_pages/"+ str(k)+'.jpg\n'
# #         f.write(dataLine)
# print(loss_path_list)


# In[21]:


# yolo_trainData_path = "/home/userr/ncku/Defect_Bean_Batch_GPU0_train_allData/darknet_function/truedata/cfg/"
# filepath = "/home/userr/ncku/Defect_Bean_Batch_GPU0_train_allData/Data/LBIG_darknet/"

# with open(yolo_trainData_path + "train.txt","w") as f:
#     for i in range(8):
#         for root, dirs, files in walk(filepath + str(i) + "_times/"):
#             for file in files:
#                 fileType = os.path.splitext(filepath + str(i) + "_times/" + file)[1]
#                 fileName = os.path.splitext(filepath + str(i) + "_times/" + file)[0]
#                 if fileType == '.jpg' :
#                     print(filepath + str(i) + "_times/" + file)
#                     f.write(filepath + str(i) + "_times/" + file+"\n")


# In[22]:


# import numpy as np
# original_LBIG = "/home/userr/ncku/Defect_Bean_Batch_GPU0/Data/LBIG_darknet/"
# deleteZero_LBIG = "/home/userr/ncku/Defect_Bean_Batch_GPU0/Data/LBIG_darknet_noZero/"

# for i in range(20):
#     for root, dirs, files in walk(original_LBIG + str(i) + "_times/"):
#         for file in files:
#             fileType = os.path.splitext(original_LBIG + str(i) + "_times/" + file)[1]
#             fileName = os.path.splitext(original_LBIG + str(i) + "_times/" + file)[0]
#             if fileType == '.txt':
#                 with open(original_LBIG + str(i) + "_times/" + file,"r") as rf:
#                     lines = rf.readlines()
#                 with open(deleteZero_LBIG + str(i) + "_times/" + file,"w") as wf:
#                     for line in lines:
#                         if int(line[0]) != 0:
#                             wf.write(line)
#                         else:
#                             if np.random.rand() > 0.5:
#                                 wf.write(line)


# In[23]:


ga = GA(darknet_path, obj_data_path, cfg_path, init_weight_path)


# In[24]:


# image_path = "/home/userr/ncku/Defect_Bean/BeanData/Test_data/5_pages/1.jpg"
# loss = loss_calculate(image_path, darknet_path, obj_data_path, cfg_path, weight_path)
# print(loss)


# In[25]:


# all_box = get_bounding_box(image_xpath, obj_data_path, cfg_path, weight_path)


# In[26]:


import time
k = 0
yolo_loss_array = np.zeros([2])
quality_array = np.zeros([3])
LBIG_Time = []
GA_adaptability = []


# In[27]:


while (1):
    print('k ',k)
    if k == 0:
#         train_network(darknet_path, obj_data_path, cfg_path, init_weight_path, epoch)
#         newModel_path = model_path + "model_"+str(k)+".weights"
        newModel_path = model_path + "LD_train/yolov3_Basic.weights"
#         os.rename(model_path+"yolov3_final.weights",newModel_path)
        
        test_path = "/home/userr/ncku/LeGAN_Log_Time/BeanData/Test_data/" + str(n) + "_pages/"
        test_path_list = []
        for i in range(n):
            test_path_list.append(test_path + str(i) + ".jpg")
        
        subdiv = len(test_path_list)/5
        if(np.mod(subdiv,1))!=0:
            change_loss_parameters(len(test_path_list),int(subdiv)+1, cfg_path)
        else:
            change_loss_parameters(len(test_path_list),int(subdiv), cfg_path)

        loss = loss_calculate(test_path_list, darknet_path, obj_data_path, cfg_path, newModel_path)
        
        print('k = 0 tims loss: ',loss)
        yolo_loss_array[0] = loss
    else :
        train_network(darknet_path, obj_data_path, cfg_path, newModel_path, epoch)
        newModel_path = model_path + "model_"+str(k)+".weights"
        #rename yolov3_final.weights to model_k.wieghts
        os.rename(model_path+"yolov3_final.weights",newModel_path)
        loss = loss_calculate(loss_path_list, darknet_path, obj_data_path, cfg_path, newModel_path)
        
        print('k = ' + str(k) + " times loss " + str(loss))
        yolo_loss_array[0] = loss
       
    startTime = time.time()
    tmp_GA_adaptability = []
    ga.set_gener_model()  # initialize chromosomes(generator model parameters)
    ga.set_weight_path(newModel_path) # load yolov3 model parameters
    for i in range(GA_epochs): # GA basis optimizer for GAN
#         if i % 10 == 0 and i != 0:
        print('GA_epochs ',i)
        print("#1")
        ga.run_gener_model(generate_input,yolo_trainX,yolo_trainY)
        print("#2")
        ga.get_adaptability_probability(k,i)
        print("#3")
        tmp_GA_adaptability.append(ga.adaptability)
        print("#4")
        ga.run_gener_model_weight()
        print("#5")
        ga.run_gener_model_bias()
        print("#6")
        ga.change_mode()
        print("#7")
    GA_adaptability.append(tmp_GA_adaptability)
    
    generator = ga.get_target_model()
    
    #print("generator:",generator)
    gener_LBIG = LBIG()
    # load the generator model which is the best chromsome from GA
    gener_LBIG.set_model(generator.gener_W,generator.gener_B)
    #print("generator.gener_W:",generator.gener_W)
    #print("generator.gener_B:",generator.gener_B)

    # set LBIG input data(need to consider density)
    gener_data_input = np.zeros([len(density_list),yolo_trainY.shape[0],11*density_list[-1]*3])
    gener_data_img = np.zeros([len(density_list),yolo_trainX.shape[0],density_list[-1],608,608,3])
    obj_info = np.zeros([len(density_list),yolo_trainY.shape[0],density_list[-1],11,11,6])
    for density in range(len(density_list)):
        print('density ',density)
        for gener_times in range(img_obj_info.shape[0]):
            for i in range(density_list[density]):
                choose_picture_info = np.random.randint(img_obj_info.shape[0])
                tmp = img_obj_info[choose_picture_info]

                gener_data_input[density,gener_times,i*33:(i+1)*33] = tmp.reshape(-1)
                gener_data_img[density,gener_times,i] = yolo_trainX[choose_picture_info]
                obj_info[density,gener_times,i] = yolo_trainY[choose_picture_info]

    generate_img_data = []
    generate_yolo_output_data = []
    for density in range(len(density_list)):
        gener_LBIG.run(gener_data_input[density])
        gener_LBIG.get_new_obj_XY_Class(density_list[density])
        gener_LBIG.get_new_obj_tmp_Yolo_output(obj_info[density,:],density_list[density])
        gener_LBIG.get_obj_img(gener_data_img[density,:],obj_info[density,:],density_list[density])
        gener_LBIG.get_new_img(density_list[density])
        gener_LBIG.get_new_obj_Yolo_output()

        # because when density == 0,the new obj coordinates in cell space is real outpu
        # so need to consier the reasonable.When density != 0,the new obj coordinates in cell space
        # is always in center.
        if density == 0:
            generate_img_data = gener_LBIG.new_img
            generate_yolo_output_data = gener_LBIG.new_obj_Yolo_output
            
            quality_index = []
            for label_index in range(generate_yolo_output_data.shape[0]):
                overLapping = 0
                quality_target = generate_yolo_output_data[label_index]
                first_dim,second_dim = np.where(quality_target[:,:,5]==1)
                quality_target_obj_info = quality_target[first_dim,second_dim,:]
                for i in range(quality_target_obj_info.shape[0]):
                    for j in range(quality_target_obj_info.shape[0]-(i+1)):
                        j = j+i+1
                        overLapping = overLapping + DIQC(quality_target_obj_info[i,1:5],quality_target_obj_info[j,1:5])
                if  overLapping == 0:
                    quality_index.append(label_index)
            print('quality_index ',quality_index)

            generate_img_data = generate_img_data[quality_index]
            generate_yolo_output_data = generate_yolo_output_data[quality_index]
        else:
            if generate_img_data.shape[0] == 0:
                generate_img_data = gener_LBIG.new_img
                generate_yolo_output_data = gener_LBIG.new_obj_Yolo_output
            else:
                generate_img_data = np.append(generate_img_data,gener_LBIG.new_img,axis=0)
                generate_yolo_output_data = np.append(generate_yolo_output_data,gener_LBIG.new_obj_Yolo_output,axis=0)
    print(generate_yolo_output_data.shape)

    # img = {img[:n],generate_img_data},because if data have a long distance between yolo_trainX and generate_img_data
    # then model will more easily become nan in optimization
    img = np.append(img[:yolo_trainX.shape[0]],generate_img_data,axis=0)
    label = np.append(label[:yolo_trainX.shape[0]],generate_yolo_output_data,axis=0)
    
    # AIQC
    quality_index = []
    noQuality_index = []
    for label_index in range(label.shape[0]):
        overLapping = 0
        yolo_output = label[label_index]
        first_dim,second_dim = np.where(yolo_output[:,:,5]==1)
        if len(first_dim) <= 11 :
            obj_info = yolo_output[first_dim,second_dim,:]
            for i in range(obj_info.shape[0]):
                for j in range(obj_info.shape[0]-(i+1)):
                    j = j+i+1
                    overLapping = overLapping + DIQC(obj_info[i,1:5],obj_info[j,1:5])
            if  overLapping == 0:
                quality_index.append(label_index)
            else:
                noQuality_index.append(label_index)
        else:
            quality_index.append(label_index)
    print('quality_index ',quality_index)

    img = img[quality_index]
    label = label[quality_index]
#     generate_input = generate_input[quality_index]
    
    
    # save the LBIG img and label
    yolo_train_path = "/home/userr/ncku/LeGAN_Log_Time/darknet_function/truedata/cfg/train.txt"
    LBIG_path = "/home/userr/ncku/LeGAN_Log_Time/Data/LBIG_darknet/" + str(k) + "_times"
#     LBIG_path = "/home/userr/ncku/LeGAN/Data/LBIG_noQuality/" + str(k) + "_times"
    if os.path.isdir(LBIG_path):
        shutil.rmtree(LBIG_path)
    os.mkdir(LBIG_path)

    with open(yolo_train_path,'w') as yf:
        for i in range(label.shape[0]):
            first_dim,second_dim = np.where(label[i,:,:,5])
            obj_info = label[i,first_dim,second_dim,:]
            yf.write(LBIG_path+'/'+str(i)+'.jpg'+'\n')
            with open(LBIG_path+'/'+str(i)+'.txt','w') as f:
                for j in range(obj_info.shape[0]):
                    obj = obj_info[j]
                    is_overLapping=0;
                    #if obj_info.shape[0] != j:
                    """
                    for chk in range(obj_info.shape[0]):
                        if chk != j:
                            is_overLapping = is_overLapping + DIQC(obj[1:5],obj_info[chk,1:5])
                            #print("chk:",chk,"  OL:",is_overLapping)
                    if is_overLapping == 0:
                        line='%d %.6f %.6f %.6f %.6f\n'%(obj[0],obj[1],obj[2],obj[3],obj[4])
                        f.write(line)
                    """
                    line='%d %.6f %.6f %.6f %.6f\n'%(obj[0],obj[1],obj[2],obj[3],obj[4])
                    f.write(line)

    for j in range(img.shape[0]):
        cv2.imwrite(LBIG_path + '/' + str(j)+'.jpg',img[j])
    endTime = time.time()
    LBIG_Time.append([endTime - startTime,img.shape[0]])
    
    # DMQC
    loss = 0
    loss_path_list =[]
    for j in range(img.shape[0]):
        loss_path_list.append(LBIG_path + '/' + str(j)+'.jpg')
           
    subdiv = len(loss_path_list)/5
    if(np.mod(subdiv,1))!=0:
        change_loss_parameters(len(loss_path_list),int(subdiv)+1, cfg_path)
    else:
        change_loss_parameters(len(loss_path_list),int(subdiv), cfg_path)
        
    loss = loss_calculate(loss_path_list, darknet_path, obj_data_path,  cfg_path, newModel_path)
        
    yolo_loss_array[1] = loss
    quality_array[k%3] = (yolo_loss_array[1] - yolo_loss_array[0]) / yolo_loss_array[1]
    print('quality_array ',quality_array)
    
    if k >= 2:
        flag = DESIMQC(quality_array)
        if flag:
            print("find best Model")
            break
    
    print('LBIG_Time ',LBIG_Time)
    print('yolo_loss_array ',yolo_loss_array)
#     print(quailty_array)
    k = k+1
#     np.save('Data/LBIG/LBIG_img_' + str(k),generate_img_data)
#     np.save('Data/LBIG/LBIG_L_' + str(k),generate_yolo_output_data)
#     np.save('Data/LBIG_generate_input_' + str(k),generate_input)


# In[ ]:


with open('/home/userr/ncku/LeGAN_Log_Time/Data/LBIG_image_time/Image_Num_and_Time', 'wb') as plkfile:
    pickle.dump(LBIG_Time,plkfile)
with open('/home/userr/ncku/LeGAN_Log_Time/Data/LBIG_image_time/GA_optimize_loss', 'wb') as plkfile:
    pickle.dump(GA_adaptability,plkfile)


# In[ ]:


import xlwt
def write_Excel(LBIG_Time):
    #建立Workbook物件
    book = xlwt.Workbook(encoding="utf-8")
    #使用Workbook裡的add_sheet函式來建立Worksheet
    sheet1 = book.add_sheet("Sheet1")
    #使用Worksheet裡的write函式將值寫入
    sheet1.write(0,0,'Time')
    sheet1.write(0,1,'Sample')
    time = 0
    sample = 0
    for i in range(len(LBIG_Time)):
        time = time + float(LBIG_Time[i][0])
        sample = sample + float(LBIG_Time[i][1])
        sheet1.write(i+1,0,time)
        sheet1.write(i+1,1,sample)
    #將Workbook儲存為原生Excel格式的檔案
    book.save("/home/userr/ncku/LeGAN_Log_Time/Data/LBIG_image_time/Image_Num_and_Time.xls")
    


# In[ ]:


write_Excel(LBIG_Time)


# In[ ]:


aaaa


# In[ ]:


# endTime = time.time()
# print("time ",endTime-startTime)


# In[ ]:


#jupyter nbconvert --to script main_darknet.ipynb


# In[ ]:


weight_path = "/home/userr/ncku/Defect_Bean_Batch_GPU0/darknet_function/truedata/cfg/weights/model_3.weights"
image_path = ["/home/userr/ncku/Defect_Bean_Batch_GPU0/BeanData/Original/Test_LD/46.jpg"]
loss = loss_calculate(image_path, darknet_path, obj_data_path, cfg_path, weight_path)
print(loss)


# In[ ]:


all_box = get_bounding_box(image_path[0], obj_data_path, cfg_path, weight_path)


# In[ ]:


import numpy as np

bbox = np.asarray([])
for index,bbox_class in enumerate(all_box):
    bbox_class = np.asarray(bbox_class)
    dim = np.where(bbox_class[:,4] > 0.0001)
    if bbox.shape[0]==0:
        bbox = np.append(bbox_class[dim],np.ones([len(dim[0]),1]) * index,axis=1)
    else:
        bbox = np.append(bbox,np.append(bbox_class[dim],np.ones([len(dim[0]),1]) * index,axis=1),axis=0)
print(bbox.shape)


# In[ ]:


def NMS(target,compare):
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
    interArea = np.max([(inter_r-inter_l),0])
    interArea = np.max([(inter_t-inter_d),0])
    
    totalArea = (target_r-target_l) * (target_t-target_d) + (compare_r-compare_l) * (compare_t-compare_d) - interArea
    
    return np.max([interArea / (totalArea + 1e-23),0])


# In[ ]:


for i in range(bbox.shape[0]):
    target = i
    for j in range(bbox.shape[0]-i-1):
        compare = j+i+1
        if bbox[target,5] == bbox[compare,5]:              #兩BBox預測種類相同
            if bbox[target,5] != 0:                              #預測種類不為正常豆
                if NMS(bbox[target],bbox[compare]) > 0.5:
                    if bbox[target,4] > bbox[compare,4]:
                        bbox[compare,4] = 0
                    else:
                        bbox[target,4] = 0
            if bbox[target,5] == 0:                             #為相同種類的正常豆
                if NMS(bbox[target],bbox[compare]) > 0:
                    if bbox[target,4] > bbox[compare,4]:
                        bbox[compare,4] = 0
                    else:
                        bbox[target,4] = 0
        if (bbox[target,5] == 0 or bbox[compare,5] == 0):  #其中一 BBOX 預測結果為正常豆，另一個為瑕疵豆
            if NMS(bbox[target],bbox[compare]) > 0:
                if bbox[target,5] == 0:
                    bbox[target,4] = 0
                else:
                    bbox[compare,4] = 0 


# In[ ]:


dim = np.where(bbox[:,4] > 0)
print(dim)
bbox = bbox[dim,:].reshape(-1,6)
print(bbox.shape)


# In[ ]:


label=[]
with open("/home/userr/ncku/Defect_Bean_Batch_GPU0/BeanData/Original/Test_LD/46.txt") as f:
    lines = f.readlines()
    for index,line in enumerate(lines):
#         if index < 10:
        data = line.split(' ')
        label.append(data)
print(label)


# In[ ]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image_path = "/home/userr/ncku/Defect_Bean_Batch_GPU0/BeanData/Original/Test_LD/46.txt"
label=[]
with open(image_path) as f:
    lines = f.readlines()
    for index,line in enumerate(lines):
#         if index < 10:
        data = line.split(' ')
        label.append(data)
print(label)

img = cv2.imread(image_path[0])




for obg in label:
    cv2.circle(img,(int(float(obg[1])*608), int(float(obg[2])*608)), int(float(obg[3])*608), (255,255,255), 3)


for j in range(bbox.shape[0]):
    cv2.circle(img,(int(bbox[j,0]), int(bbox[j,1])), int(bbox[j,2]), (0,0,255), 3)
    cv2.putText(img,str(int(bbox[j,5])), (int(bbox[j,0]), int(bbox[j,1])),cv2.FONT_HERSHEY_TRIPLEX,
                 1, (0, 0, 255), 3, cv2.LINE_AA)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


# fileName = "/home/userr/ncku/Defect_Bean/darknet_function/truedata/cfg/weights/"


# In[ ]:


# os.rename(fileName+"yolov3_200.weights",fileName + "aaaa.weights")


# In[ ]:


import numpy as np
a = np.asarray([3.2,2])
print(np.mod(a,1))


# In[ ]:




