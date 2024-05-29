import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pandas import DataFrame
from keras import Model,Layer,Sequential
from keras import models
from keras.layers import (RandomRotation,RandomFlip,RandomContrast,
                                     RandomBrightness,Conv2D,BatchNormalization,
                                    Flatten,Dropout,Dense,MaxPooling2D,Activation,Add,GlobalAveragePooling2D)
from keras.utils import image_dataset_from_directory as DatasetLoader
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import regularizers
from keras.metrics import (AUC,IoU,MeanAbsoluteError,Precision,
                                      Recall,TrueNegatives,TruePositives,
                                     FalseNegatives,FalsePositives)
from keras.callbacks import Callback
from keras.utils import get_custom_objects


train_dic  = './datasets/Training'
test_dic   = './datasets/Testing'
CLASS_NAME = {'glioma':0, 'meningioma':1, 'notumor':2, 'pituitary':3}
inv_class_mappings = {v: k for k, v in CLASS_NAME.items()}
IMG_SIZE   = 256
BATCH_SIZE = 16
N_CLASS    = len(CLASS_NAME)
LR = 0.01



class CustomConv2D(Layer):
  def __init__(self, n_filters, kernel_size, n_strides, padding = 'valid',**kwargs):
    super(CustomConv2D, self).__init__(name = 'custom_conv2d',**kwargs)

    self.conv = Conv2D(
        filters = n_filters,
        kernel_size = kernel_size,
        activation = 'relu',
        strides = n_strides,
        padding = padding)

    self.batch_norm = BatchNormalization()

  def call(self, x, training = True):

    x = self.conv(x)
    x = self.batch_norm(x, training=training)

    return x

  def get_config(self):
    config = super(CustomConv2D, self).get_config()
    config.update({
        "n_filters": self.n_filters,
        "kernel_size": self.kernel_size,
        "n_strides": self.n_strides,
        "padding": self.padding
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class ResidualBlock(Layer):
  def __init__(self, n_channels, n_strides = 1,**kwargs):
    super(ResidualBlock, self).__init__(name = 'res_block',**kwargs)

    self.dotted = (n_strides != 1)

    self.custom_conv_1 = CustomConv2D(n_channels, 3, n_strides, padding = "same")
    self.custom_conv_2 = CustomConv2D(n_channels, 3, 1, padding = "same")

    self.activation = Activation('relu')

    if self.dotted:
      self.custom_conv_3 = CustomConv2D(n_channels, 1, n_strides)

  def call(self, input, training):

    x = self.custom_conv_1(input, training=training)
    x = self.custom_conv_2(x, training=training)

    if self.dotted:
      x_add = self.custom_conv_3(input, training=training)
      x_add = Add()([x, x_add])
    else:
      x_add = Add()([x, input])

    return self.activation(x_add)
  def get_config(self):
    config = super(ResidualBlock, self).get_config()
    config.update({
        "n_channels": self.n_channels,
        "n_strides": self.n_strides
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  

class ResNet34(Model):
  def __init__(self, name='resnet_34', trainable=True, dtype='float32',**kwargs):
    super(ResNet34, self).__init__(name=name, trainable=trainable, dtype=dtype,**kwargs)

    self.conv_1 = CustomConv2D(64, 7, 2, padding = 'same')
    self.max_pool = MaxPooling2D(3,2)

    self.conv_2_1 = ResidualBlock(64)
    self.conv_2_2 = ResidualBlock(64)
    self.conv_2_3 = ResidualBlock(64)

    self.conv_3_1 = ResidualBlock(128, 2)
    self.conv_3_2 = ResidualBlock(128)
    self.conv_3_3 = ResidualBlock(128)
    self.conv_3_4 = ResidualBlock(128)

    self.conv_4_1 = ResidualBlock(256, 2)
    self.conv_4_2 = ResidualBlock(256)
    self.conv_4_3 = ResidualBlock(256)
    self.conv_4_4 = ResidualBlock(256)
    self.conv_4_5 = ResidualBlock(256)
    self.conv_4_6 = ResidualBlock(256)

    self.conv_5_1 = ResidualBlock(512, 2)
    self.conv_5_2 = ResidualBlock(512)
    self.conv_5_3 = ResidualBlock(512)

    self.global_pool = GlobalAveragePooling2D()

    self.fc_3 = Dense(N_CLASS, activation = 'softmax')

  def call(self, x, training = True):
    x = self.conv_1(x)
    x = self.max_pool(x)

    x = self.conv_2_1(x, training=training)
    x = self.conv_2_2(x, training=training)
    x = self.conv_2_3(x, training=training)

    x = self.conv_3_1(x, training=training)
    x = self.conv_3_2(x, training=training)
    x = self.conv_3_3(x, training=training)
    x = self.conv_3_4(x, training=training)

    x = self.conv_4_1(x, training=training)
    x = self.conv_4_2(x, training=training)
    x = self.conv_4_3(x, training=training)
    x = self.conv_4_4(x, training=training)
    x = self.conv_4_5(x, training=training)
    x = self.conv_4_6(x, training=training)

    x = self.conv_5_1(x, training=training)
    x = self.conv_5_2(x, training=training)
    x = self.conv_5_3(x, training=training)

    x = self.global_pool(x)

    return self.fc_3(x)

  def get_config(self):
    config = super(ResNet34, self).get_config()
    config.update({
        'name': self.name,
        'trainable': self.trainable,
        'dtype': self.dtype
    })
    return config


  @classmethod
  def from_config(cls, config):
    return cls(**config)

resnet_34 = ResNet34()
resnet_34(tf.zeros([1,256,256,3]), training = False)

loss_function = CategoricalCrossentropy()

metric = ['accuracy',
          MeanAbsoluteError(name='MeanError'),
          TruePositives(name='tp'),
          FalsePositives(name='fp'), 
          TrueNegatives(name='tn'), 
          FalseNegatives(name='fn'), 
          Precision(name='precision'),
          Recall(name='recall'), 
          AUC(name='AUC')]

resnet_34.compile(
    optimizer = Adam(learning_rate = LR),
    loss = loss_function,
    metrics=metric
)

result_ans = []
class CustomCallback(Callback):
  def on_epoch_end(self,epoch,logs):
    df = {
        epoch+1:[{"Accuracy":logs['accuracy'],
        "Loss":logs['loss'],
        "Precision":logs['precision'],
        "Recall":logs['recall'],
        "AUC":logs['AUC'],
        "True Positives":logs['tp'],
        "True Negatives":logs['tn'],
        "False Positives":logs['fp'],
        "False Negatives":logs['fn']}]
    }
    result_ans.append(df)
    print('\n'*2)
    data = []
    
    index = list(df.keys())[0] 
    metrics = df[index][0]  
    data.append(metrics)  

    df = DataFrame(data)

    print(tabulate(df, headers='keys', tablefmt='psql'))
    

get_custom_objects().update({
    "CustomConv2D": CustomConv2D,
    "ResidualBlock": ResidualBlock,
    "ResNet34": ResNet34
})

resnet_34.save('resnet.keras')

r_model = tf.keras.models.load_model('resnet.keras', custom_objects={
    "CustomConv2D": CustomConv2D,
    "ResidualBlock": ResidualBlock,
    "ResNet34": ResNet34
})

def load_and_preprocess_image(image_path, image_shape=(256, 256)):
    img = image.load_img(image_path, target_size=image_shape, color_mode='rgb') 
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def Classification_Resnet(image_path):
    images = load_and_preprocess_image(image_path)
    prediction = r_model.predict(images)
    predicted_label = inv_class_mappings[np.argmax(prediction)]
    return predicted_label