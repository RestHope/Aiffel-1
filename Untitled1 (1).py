#!/usr/bin/env python
# coding: utf-8

# # EXPLORATION_IC2
# ## 1. 인공지능과 가위바위보 하기
# ### 학습 순서
#     1.데이터 준비
#     2.딥러닝 네트워크 설계
#     3.딥러닝 네트워크 학습
#     4.정확도 확인하기
#     5.네트워크 업그레이드

# #### 평가 기준
#     1. 이미지 분류기 모델이 성공적으로 만들어졌는가? (트레이닝 정상작동)
#     2. 오버피팅을 극복하기 위한 적절한 시도가 있었는가?(데이터셋 다양화/정규화,,)
#     3. 분류모델의 test accuracy가 기준이상인가 (60%이상 도달)

# In[129]:


from PIL import Image
import glob
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt #차트 생성모듈


# In[130]:


def resize_images(img_path):
    images=glob.glob(img_path+"/*.jpg")
    target_size=(28,28) ## 변경후 이미지 사이즈
    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size, Image.ANTIALIAS) #resize ==> from PIL import image:이미지 크기 변경
        new_img.save(img,'JPEG')
    
image_dir_path = os.getenv("HOME")+"/aiffel/EXPLORATION1/rock_scissor_paper/scissor"
image_dir_path_test= os.getenv("HOME") + "/aiffel/EXPLORATION/rock_scissor_paper_test/scissor"

#os.getenv(): 시스템의 환경변수 값을 알 수 있다.
resize_images(image_dir_path)
resize_images(image_dir_path_test)

image_dir_path = os.getenv("HOME")+"/aiffel/EXPLORATION1/rock_scissor_paper/rock" 
image_dir_path_test= os.getenv("HOME") + "/aiffel/EXPLORATION/rock_scissor_paper_test/rock"

resize_images(image_dir_path)
resize_images(image_dir_path_test)

image_dir_path = os.getenv("HOME")+"/aiffel/EXPLORATION1/rock_scissor_paper/paper" 
image_dir_path_test= os.getenv("HOME") + "/aiffel/EXPLORATION/rock_scissor_paper_test/paper"

resize_images(image_dir_path)
resize_images(image_dir_path_test)
print(' 이미지 resize 완료')


# In[131]:


def load_data(img_path, number_of_data=896):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    #,zeros(shape(크기 :N*N=>28*28의 3차원 행렬이 number_of_data만큼 있는 4차원 행렬,dtype,order)
    #.reshape(변경할 배열, 차원)/배열.reshape(차원)
    # 이 크기의 0으로 구성된 행렬을 생성 ==> 기본틀
    labels=np.zeros(number_of_data,dtype=np.int32)
    #print(len(labels))
    #print(labels)
    
    #label은 가위 바위 보 인지를 구별하기 위해 붙이는 거임으로 사진 개수의 행렬을 만들어서 그 사진이 가위인지 보인지 묵인지를 숫자로 표현
    
    idx=0
    for file in glob.iglob(img_path+'/scissor/'+'*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/'+'*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/'+'*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
    #print(labels)   
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

#파일 주소 불러오기
image_dir_path = os.getenv("HOME") + "/aiffel/EXPLORATION/rock_scissor_paper"
image_dir_path_test= os.getenv("HOME") + "/aiffel/EXPLORATION/rock_scissor_paper_test"
print(load_data)
#학습용 데이터와 시험용데이터 나누기 
(x_train, y_train)=load_data(image_dir_path)
(x_test, y_test)=load_data(image_dir_path_test)

x_train_norm = x_train/255.0   # 이미지 픽셀을 구성하는 수를 0-1범위로 만들기 위한 코드 _이미지 정형화
#Sequential APL: 미리 정의된 딥러닝레이어 추가 가능
x_test_norm = x_test/255.0 
print("x_train shape: {}".format(x_train.shape)) #(몇개)
print("y_train shape: {}".format(x_test.shape))


# In[132]:


plt.imshow(x_train[1])
print('라벨: ', y_train[1])


# In[133]:



x_train_reshaped=x_train_norm.reshape( -1, 28, 28, 3)  # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 3)

n_channel_1=10
#n_channel_1=10 /15/30
n_channel_2=32
n_dense=15
#n_dense=10 /15/20/40//11-17
n_train_epoch=15
#epochs=>전체를 학습시킨 횟수
#n_train_epoch=15/30

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(20, activation='softmax'))



model.summary()


# In[134]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history=model.fit(x_train_reshaped, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(x_test_reshaped,y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))


# In[128]:



plt.plot(history.history['accuracy'])
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## 회고 
# ### 1.내가 세운 목표를 달성했는가?
# #### EXPLORATION 목표
# 1) 이미지 분류기 모델이 성공적으로 만들어졌는가? (트레이닝 정상작동)
# 작동은 잘된다!
# 
# 2) 오버피팅을 극복하기 위한 적절한 시도가 있었는가?(데이터셋 다양화/정규화,,)
# 시도는 했다...
# 
# 3) 분류모델의 test accuracy가 기준이상인가 (60%이상 도달)
# 도달하지 못했다. 60%이상의 정확도를 만들기 위해Conv2D레이어의 입력 이미지수를 변경해 봤과 Dense레이어에의 뉴런수를 변경해 보았으며 학습 반복 횟수를 변경해 보았으나 최대가 40% 근처의 값이 출력 되었다. epoch와 accuracy의 상관관계를 파악하기 위해 데이터 그래프를 출력해 보았으나 어떻게 제어해야 accuracy 값이 올라가는지 파악하는 것에 어려움이 있어 활용하지 못했다.
# 
# #### 나의 목표
# 1) 이미지 분류기 모델이 작동하는가? 
# 2) 구조를 이해하였는가?
#  60-80% 이해한것같다. 
# #### 2.나에게 어떤 일들이 있었고 거기서 나의 깨달음은 무엇이었나?
# 발생한 문제점
# 1) 데이터 알집 푸는 문제
# 처음에 리눅스 환경이 아닌 주피터에서 데이터 파일의 .zip를 풀려고 하였는데 unzip 으로는 주피터에서 파일을 풀리지 않았다.그래서 구글링을 통해 shutil 라이브러리를 통해서 풀었는데 test데이터의 zip을 풀때는 풀리지 않았다. 그래서 다시 리룩스로 돌아가서 한번에 원하는 구조로 이동하지 않고 하나씩 이동하여 들어갔다.
# 그런데 파일에 500씩 들어있는 파일들의 .zip가 풀리지 않아서 데이터 수를 절반 정도로 줄이니 압축이 잘 풀렸다.
# 이런 일들을 통해 깨달은 점은 포기하지 않으면 결국은 방법을 찾는 다는 것이었다. 그리고 오타와 여러가지 환경을 항상 점검하면서 코드를 치자는 것이었다. 
# 스스로 생각하기에 코드를 칠때나 파일을 작성할때 소소한 실수들이 많은 것 같다는 것을 느꼈다.
# 
# 2) 구조가 이해 되지 않아 코드를 작성할 수 없었던 점
# 경험이 없는 AI분야의 코드를 읽고 그 구조를 파악하는 것이 어려웠고 사용된 함수나 라이브러리의 기능을 알지 못하는 점이 어려웠다. 또 어떤 프로젝트를 시간 안에 끝 내야 한다는 압박이 때문에 프로젝트를 구현하는 것에 어려움을 느꼈다.
# 이러한 문제들을 해결하기 위해 일단 코드에 사용된 라이브러리나 함수들의 기능을 찾아 정리하였고 그것을 토대로 이 함수가 어떻게 작동하고 어떤 기능을 하는지 이해하기 위해 구조를 그려가면서 이해하려고 했다. 마지막으로 시간안에 끝 내야한다는 압박을 버리기 위해서 목표를 낮췄다. 일단은 이 코드가 구현되고 이 코드가 작동하는 순서등을 이해하는 것을 목표로 잡고 지나온 내용들을 다시 복습했다. 이 문제를 경험하면서 지금 단계가 불가능하다고 느끼면 다른사람의 속도와 달리 다시 이전 단계로 돌아오는법을 배웠고 포기하면 아무 결과가 남지 않지만 목표를 조정하면 나는 한단계 성장하고 언젠간 그 목표에 도달할 수 있는 발판이 된다는 것을 깨닫게 되었다.
# 
# 3) 정확도가 90%가 발생함
# 코드를 다시 보고 빠진코드를 추가해주었다.

# In[ ]:




