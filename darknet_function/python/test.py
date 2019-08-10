
# coding: utf-8

# In[13]:


from rpidarknet.python import darknet as dn


net = dn.load_net(str.encode("/home/mmdb/Desktop/rpidarknet_(copy)/truedata/cfg/yolov3.cfg"), str.encode("/home/mmdb/Desktop/rpidarknet_(copy)/truedata/cfg/weights/yolov3_10000.weights"), 0)

meta = dn.load_meta(str.encode("/home/mmdb/Desktop/rpidarknet_(copy)/truedata/cfg/obj.data"))
r = dn.detect(net, meta, str.encode("/home/mmdb/Desktop/rpidarknet_(copy)/11.jpg"))
print(r)

