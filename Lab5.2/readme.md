# TensorFlow训练石头剪刀布数据集
采用浏览器手动下载
## 解压下载数据集

```bash
import os
import zipfile

local_zip = 'D:/game/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/game/')
zip_ref.close()

local_zip = 'D:/game/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/game/')
zip_ref.close()

```
## 检测数据集的解压结果，打印相关信息。

```bash
rock_dir = os.path.join('D:/game/rps/rock')
paper_dir = os.path.join('D:/game/rps/paper')
scissors_dir = os.path.join('D:/game/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

```bash
total training rock images: 840
total training paper images: 840
total training scissors images: 840
['rock01-000.png', 'rock01-001.png', 'rock01-002.png', 'rock01-003.png', 'rock01-004.png', 'rock01-005.png', 'rock01-006.png', 'rock01-007.png', 'rock01-008.png', 'rock01-009.png']
['paper01-000.png', 'paper01-001.png', 'paper01-002.png', 'paper01-003.png', 'paper01-004.png', 'paper01-005.png', 'paper01-006.png', 'paper01-007.png', 'paper01-008.png', 'paper01-009.png']
['scissors01-000.png', 'scissors01-001.png', 'scissors01-002.png', 'scissors01-003.png', 'scissors01-004.png', 'scissors01-005.png', 'scissors01-006.png', 'scissors01-007.png', 'scissors01-008.png', 'scissors01-009.png']

```

```bash
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

```
图片：
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/1.png)
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/2.png)
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/3.png)
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/4.png)
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/5.png)
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/6.png)
调用TensorFlow的keras进行数据模型的训练和评估。Keras是开源人工神经网络库，TensorFlow集成了keras的调用接口，可以方便的使用。

```bash
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "D:/game/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "D:/game/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

```

```bash
Found 2520 images belonging to 3 classes.
Found 372 images belonging to 3 classes.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6272)              0         
                                                                 
 dropout (Dropout)           (None, 6272)              0         
                                                                 
 dense (Dense)               (None, 512)               3211776   
                                                                 
 dense_1 (Dense)             (None, 3)                 1539      
                                                                 
=================================================================
Total params: 3,473,475
Trainable params: 3,473,475
Non-trainable params: 0
_________________________________________________________________
Epoch 1/25
20/20 [==============================] - 53s 3s/step - loss: 1.4751 - accuracy: 0.3599 - val_loss: 1.1369 - val_accuracy: 0.3333
Epoch 2/25
20/20 [==============================] - 49s 2s/step - loss: 1.1303 - accuracy: 0.3603 - val_loss: 1.0965 - val_accuracy: 0.5108
Epoch 3/25
20/20 [==============================] - 47s 2s/step - loss: 1.0863 - accuracy: 0.4032 - val_loss: 0.9790 - val_accuracy: 0.3978
Epoch 4/25
20/20 [==============================] - 50s 2s/step - loss: 1.0418 - accuracy: 0.5139 - val_loss: 0.8253 - val_accuracy: 0.7258
Epoch 5/25
20/20 [==============================] - 50s 2s/step - loss: 0.8743 - accuracy: 0.6087 - val_loss: 0.4759 - val_accuracy: 0.9651
Epoch 6/25
20/20 [==============================] - 48s 2s/step - loss: 0.8080 - accuracy: 0.6345 - val_loss: 0.6926 - val_accuracy: 0.6183
Epoch 7/25
20/20 [==============================] - 46s 2s/step - loss: 0.6538 - accuracy: 0.7103 - val_loss: 0.2193 - val_accuracy: 0.9785
Epoch 8/25
20/20 [==============================] - 46s 2s/step - loss: 0.5827 - accuracy: 0.7579 - val_loss: 0.2920 - val_accuracy: 0.9731
Epoch 9/25
20/20 [==============================] - 45s 2s/step - loss: 0.4396 - accuracy: 0.8286 - val_loss: 0.0803 - val_accuracy: 1.0000
Epoch 10/25
20/20 [==============================] - 47s 2s/step - loss: 0.3461 - accuracy: 0.8560 - val_loss: 0.3216 - val_accuracy: 0.7634
Epoch 11/25
20/20 [==============================] - 45s 2s/step - loss: 0.3198 - accuracy: 0.8730 - val_loss: 0.0706 - val_accuracy: 0.9651
Epoch 12/25
20/20 [==============================] - 45s 2s/step - loss: 0.2977 - accuracy: 0.8861 - val_loss: 0.0884 - val_accuracy: 0.9651
Epoch 13/25
20/20 [==============================] - 47s 2s/step - loss: 0.2832 - accuracy: 0.8952 - val_loss: 0.0391 - val_accuracy: 0.9839
Epoch 14/25
20/20 [==============================] - 43s 2s/step - loss: 0.1713 - accuracy: 0.9353 - val_loss: 0.0592 - val_accuracy: 0.9758
Epoch 15/25
20/20 [==============================] - 44s 2s/step - loss: 0.2972 - accuracy: 0.8913 - val_loss: 0.1070 - val_accuracy: 0.9839
Epoch 16/25
20/20 [==============================] - 48s 2s/step - loss: 0.1306 - accuracy: 0.9575 - val_loss: 0.0549 - val_accuracy: 0.9785
Epoch 17/25
20/20 [==============================] - 46s 2s/step - loss: 0.1886 - accuracy: 0.9226 - val_loss: 0.0500 - val_accuracy: 0.9866
Epoch 18/25
20/20 [==============================] - 45s 2s/step - loss: 0.1101 - accuracy: 0.9615 - val_loss: 0.0518 - val_accuracy: 0.9651
Epoch 19/25
20/20 [==============================] - 48s 2s/step - loss: 0.1343 - accuracy: 0.9556 - val_loss: 0.0105 - val_accuracy: 1.0000
Epoch 20/25
20/20 [==============================] - 45s 2s/step - loss: 0.1349 - accuracy: 0.9528 - val_loss: 0.2117 - val_accuracy: 0.8952
Epoch 21/25
20/20 [==============================] - 50s 2s/step - loss: 0.0918 - accuracy: 0.9687 - val_loss: 0.0844 - val_accuracy: 0.9651
Epoch 22/25
20/20 [==============================] - 50s 2s/step - loss: 0.2075 - accuracy: 0.9317 - val_loss: 0.0403 - val_accuracy: 0.9839
Epoch 23/25
20/20 [==============================] - 51s 3s/step - loss: 0.0937 - accuracy: 0.9675 - val_loss: 0.7548 - val_accuracy: 0.7070
Epoch 24/25
20/20 [==============================] - 46s 2s/step - loss: 0.0861 - accuracy: 0.9694 - val_loss: 0.1306 - val_accuracy: 0.9435
Epoch 25/25
20/20 [==============================] - 47s 2s/step - loss: 0.1002 - accuracy: 0.9655 - val_loss: 0.0382 - val_accuracy: 0.9839

```
完成模型训练之后，我们绘制训练和验证结果的相关信息。

```bash
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```
![在这里插入图片描述](https://github.com/WananGQ/Android2023/blob/main/Lab5.2/image/7.png)
