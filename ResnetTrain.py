import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import seaborn as sns
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
from sklearn.metrics import confusion_matrix

train_dic  = './datasets/Training'
test_dic   = './datasets/Testing'
CLASS_NAME = {'glioma':0, 'meningioma':1, 'notumor':2, 'pituitary':3}
inv_class_mappings = {v: k for k, v in CLASS_NAME.items()}
IMG_SIZE   = 256
BATCH_SIZE = 16
N_CLASS    = len(CLASS_NAME)
LR = 0.01


train_dataset = DatasetLoader(
    train_dic,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAME,
    color_mode='rgb',
    batch_size=None,
    image_size=(IMG_SIZE,IMG_SIZE),
    shuffle=True,
    seed=99,)

test_dataset = DatasetLoader(
    test_dic,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAME,
    color_mode='rgb',
    batch_size=None,
    image_size=(IMG_SIZE,IMG_SIZE),
    shuffle=True,
    seed=99,)

augmantation_layer = Sequential([
    RandomRotation(factor = (0.25,0.2501),),
    RandomFlip(),
    RandomContrast(0.2),
    RandomBrightness(0.2)
])

def augmantation(image,label):
    return augmantation_layer(image,training = True),label

train_data_1 = train_dataset.shuffle(buffer_size = 8 ,reshuffle_each_iteration = True)
train_data_2 = train_dataset.shuffle(buffer_size = 8 ,reshuffle_each_iteration = True)

mix_data = tf.data.Dataset.zip((train_data_1,train_data_2))

def mixup(train_dataset_1,train_dataset_2):

      (img_1,label_1),(img_2,label_2) = train_dataset_1,train_dataset_2


      lamda = tfp.distributions.Beta(0.2,0.2)
      lamda = lamda.sample(1)[0]

      image = lamda*img_1 + (1-lamda)*img_2
      label = lamda*tf.cast(label_1,dtype = tf.float32) + (1-lamda)*tf.cast(label_2,dtype = tf.float32)
      return image,label
    
train_dataset = mix_data.shuffle(buffer_size=8,reshuffle_each_iteration = True).map(mixup).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def normalization(image,lable):
    image = image / 255.0
    
    return image,lable
train_dataset = train_dataset.map(normalization)


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
    
history_1 = resnet_34.fit(
    train_dataset,
    epochs = 50,
    verbose = 1,
    callbacks = [CustomCallback()]
)

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

plt.plot(history_1.history['accuracy'], label='Training Accuracy') 
plt.plot(history_1.history['loss'], label='Training loss')
plt.plot(history_1.history['precision'], label='Training precision')
plt.plot(history_1.history['recall'], label='Training recall')
plt.plot(history_1.history['AUC'], label='Training AUC')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.savefig('resnet.png')
plt.show()

predict = []
label_list = []
for im, labels in test_dataset:
    predict.append(resnet_34(im))
    label_list.append(labels.numpy())
pred = np.concatenate([np.argmax(predict[:-1],axis = -1).flatten(),np.argmax(predict[-1],axis = -1).flatten()])
lab = np.concatenate([np.argmax(label_list[:-1],axis = -1).flatten(),np.argmax(label_list[-1],axis = -1).flatten()])
cm = confusion_matrix(lab, pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAME, yticklabels=CLASS_NAME)
plt.title('Confusion Matrix', color='green')
plt.ylabel('Actual', color='blue')
plt.xlabel('Predict', color='blue')
plt.savefig('resnet_confusionmatrix.png')
plt.show()