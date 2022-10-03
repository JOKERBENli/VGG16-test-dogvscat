学习时间：2022年9月23日~2022年9月30日

对VGG16网络推荐教程的学习和扩展

使用pytorch框架手把手教你编写利用VGG16网络的猫狗分类程序：https://www.bilibili.com/video/BV1X3411N7aj

## VGG16——网络结构介绍及搭建(PyTorch)

## 一、VGG16的结构层次
### 1、网络结构

VGG16模型很好的适用于分类和定位任务，其名称来自牛津大学几何组（Visual Geometry Group）的缩写。

根据卷积核的大小核卷积层数，VGG共有6种配置，分别为A、A-LRN、B、C、D、E，其中D和E两种是最为常用的VGG16和VGG19。

介绍结构图：

- conv3-64 ：是指第三层卷积后维度变成64，同样地，conv3-128指的是第三层卷积后维度变成128；

- input（224x224 RGB image） ：指的是输入图片大小为224244的彩色图像，通道为3，即224224*3；

- maxpool ：是指最大池化，在vgg16中，pooling采用的是2*2的最大池化方法（如果不懂最大池化，下面有解释）；

- FC-4096 :指的是全连接层中有4096个节点，同样地，FC-1000为该层全连接层有1000个节点；

- padding：指的是对矩阵在外边填充n圈，padding=1即填充1圈，5X5大小的矩阵，填充一圈后变成7X7大小；

- 最后补充，vgg16每层卷积的滑动步长stride=1，padding=1，卷积核大小为3 3 3；

  

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zdGF0aWMub3NjaGluYS5uZXQvdXBsb2Fkcy9zcGFjZS8yMDE4LzAzMTQvMDIzMDQ0X1g0OVJfODc2MzU0LnBuZw?x-oss-process=image/format,png)

如上图VGG16的网络结构为，VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling（最大化池）分开，所有隐层的激活单元都采用ReLU函数。具体信息如下：

- 卷积-卷积-池化-卷积-卷积-池化-卷积-卷积-卷积-池化-卷积-卷积-卷积-池化-卷积-卷积-卷积-池化-全连接-全连接-全连接
-  通道数分别为64，128，512，512，512，4096，4096，1000。卷积层通道数翻倍，直到512时不再增加。通道数的增加，使更多的信息被提取出来。全连接的4096是经验值，当然也可以是别的数，但是不要小于最后的类别。1000表示要分类的类别数。
- 用池化层作为分界，VGG16共有6个块结构，每个块结构中的通道数相同。因为卷积层和全连接层都有权重系数，也被称为权重层，其中卷积层13层，全连接3层，池化层不涉及权重。所以共有13+3=16层。
- 对于VGG16卷积神经网络而言，其13层卷积层和5层池化层负责进行特征的提取，最后的3层全连接层负责完成分类任务。



### 2、VGG16的卷积核

- VGG使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。
- 卷积层全部都是3*3的卷积核，用上图中conv3-xxx表示，xxx表示通道数。其步长为1，用padding=same填充。
- 池化层的池化核为2*2

### **3、卷积计算**



![img](https://img-blog.csdnimg.cn/46be649cabf440c3918f7c48170a7b76.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5qmZ5a2Q5ZCWMjE=,size_20,color_FFFFFF,t_70,g_se,x_16)



具体的过程：

1. 输入图像尺寸为224x224x3，经64个通道为3的3x3的卷积核，步长为1，padding=same填充，卷积两次，再经ReLU激活，输出的尺寸大小为224x224x64
2. 经max pooling（最大化池化），滤波器为2x2，步长为2，图像尺寸减半，池化后的尺寸变为112x112x64
3. 经128个3x3的卷积核，两次卷积，ReLU激活，尺寸变为112x112x128
4. max pooling池化，尺寸变为56x56x128
5. 经256个3x3的卷积核，三次卷积，ReLU激活，尺寸变为56x56x256
6. max pooling池化，尺寸变为28x28x256
7. 经512个3x3的卷积核，三次卷积，ReLU激活，尺寸变为28x28x512
8. max pooling池化，尺寸变为14x14x512
9. 经512个3x3的卷积核，三次卷积，ReLU，尺寸变为14x14x512
10. max pooling池化，尺寸变为7x7x512
11. 然后Flatten()，将数据拉平成向量，变成一维51277=25088。
12. 再经过两层1x1x4096，一层1x1x1000的全连接层（共三层），经ReLU激活
13. 最后通过softmax输出1000个预测结果

从上面的过程可以看出VGG网络结构还是挺简洁的，都是由小卷积核、小池化核、ReLU组合而成。其简化图如下（以VGG16为例）：

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zdGF0aWMub3NjaGluYS5uZXQvdXBsb2Fkcy9zcGFjZS8yMDE4LzAzMTQvMDIzMTExX0dHOWtfODc2MzU0LnBuZw?x-oss-process=image/format,png)

![img](https://img-blog.csdnimg.cn/20210421154043200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hnbnV4Y18xOTkz,size_16,color_FFFFFF,t_70)

### 4、权重参数（不考虑偏置）

1)输入层有0个参数，所需存储容量为224x224x3=150k
2）对于第一层卷积，由于输入图的通道数是3，网络必须要有通道数为3的的卷积核，这样的卷积核有64个，因此总共有（3x3x3）x64 = 1728个参数。
所需存储容量为224x224x64=3.2M
计算量为：输入图像224×224×3，输出224×224×64，卷积核大小3×3。

所以Times=224×224×3x3×3×64=8.7×107

3）池化层有0个参数，所需存储容量为 图像尺寸x图像尺寸x通道数=xxx k
4）全连接层的权重参数数目的计算方法为：前一层节点数×本层的节点数。因此，全连接层的参数分别为：
7x7x512x4096 = 1027,645,444
4096x4096 = 16,781,321
4096x1000 = 4096000
按上述步骤计算的VGG16整个网络总共所占的存储容量为24M*4bytes=96MB/image 。

所有参数为138M
VGG16具有如此之大的参数数目，可以预期它具有很高的拟合能力；

但同时缺点也很明显：
即训练时间过长，调参难度大。
需要的存储容量大，不利于部署



### **5、总结**

- **通过增加深度能有效地提升性能；**
- **VGG16是最佳的模型，从头到尾只有3x3卷积与2x2池化，简洁优美；**
- **卷积可代替全连接，可适应各种尺寸的图片。**

## 二、模型搭建

#### **1、数据加载**

未完成





## 三、vgg16实现MNIST分类

```python

#从keras.model中导入model模块，为函数api搭建网络做准备
from keras.models import Model
from keras.layers import Flatten,Dense,Dropout,MaxPooling2D,Conv2D,BatchNormalization,Input,ZeroPadding2D,Concatenate
from keras.layers.convolutional import AveragePooling2D
from keras import regularizers  #正则化
from keras.optimizers import RMSprop  #优化选择器
from keras.layers import AveragePooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
 
#数据处理
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_test1=X_test
Y_test1=Y_test
X_train=X_train.reshape(-1,28,28,1).astype("float32")/255.0
X_test=X_test.reshape(-1,28,28,1).astype("float32")/255.0
Y_train=np_utils.to_categorical(Y_train,10)
Y_test=np_utils.to_categorical(Y_test,10)
print(X_train.shape)
print(Y_train.shape)
print(X_train.shape)
 
def vgg16():
    x_input = Input((28, 28, 1))  # 输入数据形状28*28*1
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
 
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
 
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
 
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
 
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
 
    #BLOCK 6
    x=Flatten()(x)
    x=Dense(256,activation="relu")(x)
    x=Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    #搭建最后一层，即输出层
    x = Dense(10, activation="softmax")(x)
    # 调用MDOEL函数，定义该网络模型的输入层为X_input,输出层为x.即全连接层
    model = Model(inputs=x_input, outputs=x)
    # 查看网络模型的摘要
    model.summary()
    return model
model=vgg16()
optimizer=RMSprop(lr=1e-4)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
#训练加评估模型
n_epoch=4#-----------------------
batch_size=128#------------------
def run_model(): #训练模型
    training=model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=n_epoch,
    validation_split=0.25,
    verbose=1
    )
    test=model.evaluate(X_train,Y_train,verbose=1)
    return training,test
training,test=run_model()
print("误差：",test[0])
print("准确率：",test[1])
 
def show_train(training_history,train, validation):
    plt.plot(training.history[train],linestyle="-",color="b")
    plt.plot(training.history[validation] ,linestyle="--",color="r")
    plt.title("training history")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["training","validation"],loc="lower right")
    plt.show()
show_train(training,"accuracy","val_accuracy")
 
def show_train1(training_history,train, validation):
    plt.plot(training.history[train],linestyle="-",color="b")
    plt.plot(training.history[validation] ,linestyle="--",color="r")
    plt.title("training history")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["training","validation"],loc="upper right")
    plt.show()
show_train1(training,"loss","val_loss")
 
prediction=model.predict(X_test)
def image_show(image):
    fig=plt.gcf()  #获取当前图像
    fig.set_size_inches(2,2)  #改变图像大小
    plt.imshow(image,cmap="binary")  #显示图像
    plt.show()
def result(i):
    image_show(X_test1[i])
    print("真实值：",Y_test1[i])
    print("预测值：",np.argmax(prediction[i]))
result(0)
result(1)

```

