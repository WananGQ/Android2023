## 实验5.1TensorFlow Lite Model Maker生成模型

 1. 安装必要的库

```bash
 !pip install tflite-model-maker
```
清华镜像：pip install pywinauto -i https://pypi.tuna.tsinghua.edu.cn/simple/

```bash
Requirement already satisfied: pywinauto in d:\anaconda3\lib\site-packages (0.6.8)
Requirement already satisfied: pywin32 in d:\anaconda3\lib\site-packages (from pywinauto) (305.1)
Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from pywinauto) (1.16.0)
Requirement already satisfied: comtypes in d:\anaconda3\lib\site-packages (from pywinauto) (1.2.0)
```

```bash
!pip install conda-repo-cli==1.0.4
```

```bash
Requirement already satisfied: conda-repo-cli==1.0.4 in d:\anaconda3\lib\site-packages (1.0.4)
Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (5.3.0)
Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2021.3)
Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.8.2)
Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.2.2)
Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.16.0)
Requirement already satisfied: pathlib in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (1.0.1)
Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (6.0)
Requirement already satisfied: requests>=2.9.1 in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (2.26.0)
Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from conda-repo-cli==1.0.4) (61.2.0)
Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.10.0)
Requirement already satisfied: fastjsonschema in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (2.15.1)
Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (5.1.1)
Requirement already satisfied: jsonschema>=2.6 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->conda-repo-cli==1.0.4) (4.4.0)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (0.18.0)
Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->conda-repo-cli==1.0.4) (21.4.0)
Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (3.3)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (1.25.11)
Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests>=2.9.1->conda-repo-cli==1.0.4) (2021.10.8)
Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->conda-repo-cli==1.0.4) (302)

```

```bash
!pip install anaconda-project==0.10.1
```

```bash
Requirement already satisfied: anaconda-project==0.10.1 in d:\anaconda3\lib\site-packages (0.10.1)
Requirement already satisfied: jinja2 in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (2.11.3)
Requirement already satisfied: anaconda-client in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (1.9.0)
Requirement already satisfied: ruamel-yaml in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (0.17.21)
Requirement already satisfied: conda-pack in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (0.6.0)
Requirement already satisfied: requests in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (2.26.0)
Requirement already satisfied: tornado>=4.2 in d:\anaconda3\lib\site-packages (from anaconda-project==0.10.1) (6.1)
Requirement already satisfied: PyYAML>=3.12 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (6.0)
Requirement already satisfied: pytz in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2021.3)
Requirement already satisfied: setuptools in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (61.2.0)
Requirement already satisfied: python-dateutil>=2.6.1 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (2.8.2)
Requirement already satisfied: clyent>=1.2.0 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.2.2)
Requirement already satisfied: nbformat>=4.4.0 in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (5.3.0)
Requirement already satisfied: six in d:\anaconda3\lib\site-packages (from anaconda-client->anaconda-project==0.10.1) (1.16.0)
Requirement already satisfied: fastjsonschema in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (2.15.1)
Requirement already satisfied: jsonschema>=2.6 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (4.4.0)
Requirement already satisfied: jupyter-core in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (4.10.0)
Requirement already satisfied: traitlets>=4.1 in d:\anaconda3\lib\site-packages (from nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (5.1.1)
Requirement already satisfied: attrs>=17.4.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (21.4.0)
Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in d:\anaconda3\lib\site-packages (from jsonschema>=2.6->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (0.18.0)
Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (3.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (2021.10.8)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\anaconda3\lib\site-packages (from requests->anaconda-project==0.10.1) (1.25.11)
Requirement already satisfied: MarkupSafe>=0.23 in d:\anaconda3\lib\site-packages (from jinja2->anaconda-project==0.10.1) (1.1.1)
Requirement already satisfied: pywin32>=1.0 in d:\anaconda3\lib\site-packages (from jupyter-core->nbformat>=4.4.0->anaconda-client->anaconda-project==0.10.1) (302)
Requirement already satisfied: ruamel.yaml.clib>=0.2.6 in d:\anaconda3\lib\site-packages (from ruamel-yaml->anaconda-project==0.10.1) (0.2.6)

```
2.导入相关库

```bash
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```
### 训练模型
	获取数据
	

```bash
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```
加载数据集
```bash
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

```bash
INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.

```
训练Tensorflow模型

```bash
model = image_classifier.create(train_data)
```

```bash
INFO:tensorflow:Retraining the models...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
 rasLayerV1V2)                                                   
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 3,419,429
Trainable params: 6,405
Non-trainable params: 3,413,024
_________________________________________________________________
None
Epoch 1/5


d:\anaconda3\lib\site-packages\keras\optimizer_v2\gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(SGD, self).__init__(name, **kwargs)


103/103 [==============================] - 76s 719ms/step - loss: 0.8647 - accuracy: 0.7782
Epoch 2/5
103/103 [==============================] - 97s 943ms/step - loss: 0.6525 - accuracy: 0.8935
Epoch 3/5
103/103 [==============================] - 92s 896ms/step - loss: 0.6223 - accuracy: 0.9099
Epoch 4/5
103/103 [==============================] - 95s 921ms/step - loss: 0.6021 - accuracy: 0.9226
Epoch 5/5
103/103 [==============================] - 100s 970ms/step - loss: 0.5903 - accuracy: 0.9336

```
评估模型

```bash
loss, accuracy = model.evaluate(test_data)
```
导出TensorFlow模型
```bash
model.export(export_dir='.')
```

```bash
INFO:tensorflow:Assets written to: C:\Users\ll\AppData\Local\Temp\tmpqryprhqv\assets


INFO:tensorflow:Assets written to: C:\Users\ll\AppData\Local\Temp\tmpqryprhqv\assets
d:\anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "


INFO:tensorflow:Label file is inside the TFLite model with metadata.


INFO:tensorflow:Label file is inside the TFLite model with metadata.


INFO:tensorflow:Saving labels in C:\Users\ll\AppData\Local\Temp\tmpjyprgcyp\labels.txt


INFO:tensorflow:Saving labels in C:\Users\ll\AppData\Local\Temp\tmpjyprgcyp\labels.txt


INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite


INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite

```

