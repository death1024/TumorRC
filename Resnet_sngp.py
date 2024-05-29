import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from pandas import DataFrame
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import Independent, MultivariateNormalDiag
from keras import Model,Layer,Sequential
from keras.layers import (RandomRotation,RandomFlip,RandomContrast,
                                     RandomBrightness,Conv2D,BatchNormalization,
                                    Flatten,Dropout,Dense,MaxPooling2D,Activation,Add,GlobalAveragePooling2D)
from keras.utils import image_dataset_from_directory as DatasetLoader
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras import regularizers
from keras.regularizers import l2
from keras.metrics import (AUC,IoU,MeanAbsoluteError,Precision,
                                      Recall,TrueNegatives,TruePositives,
                                     FalseNegatives,FalsePositives)
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
from keras.utils import get_custom_objects

train_dic  = './datasets/Training'
test_dic   = './datasets/Testing'
CLASS_NAME = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE   = 256
BATCH_SIZE = 32
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
    seed=42,)

test_dataset = DatasetLoader(
    test_dic,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAME,
    color_mode='rgb',
    batch_size=None,
    image_size=(IMG_SIZE,IMG_SIZE),
    shuffle=True,
    seed=42,)

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
test_dataset = test_dataset.map(normalization)

class SpectralNormalization(tf.keras.constraints.Constraint):
    def __init__(self, iteration=1):
        self.iteration = iteration

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        for _ in range(self.iteration):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v = v / tf.norm(v)
            u = tf.linalg.matvec(flattened_w, v)
            u = u / tf.norm(u)
        sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {'iteration': self.iteration}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CustomConv2D(Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', activation='relu',l2_reg=0.01,**kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.l2_reg = l2_reg
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=None, 
                           kernel_constraint=SpectralNormalization(),kernel_regularizer=l2(l2_reg))
        self.activation = Activation(activation)
        self.batch_norm = BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        return self.activation(x)

    def get_config(self):
        config = super(CustomConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'l2_reg': self.l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class ResidualBlock(Layer):
    def __init__(self, filters, strides=1, dropout_rate=0.3, l2_reg=0.01, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.conv1 = CustomConv2D(filters, 3, strides, l2_reg=l2_reg, padding='same')
        self.conv2 = CustomConv2D(filters, 3, 1, l2_reg=l2_reg, padding='same')
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.activation = Activation('relu')
        if strides != 1:
            # 使用1x1卷积来调整输入的尺寸和通道数
            self.residual_conv = CustomConv2D(filters, 1, strides, l2_reg=l2_reg, padding='same')
        else:
            self.residual_conv = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.conv2(x, training=training)

        if self.residual_conv:
            residual = self.residual_conv(inputs, training=training)
        else:
            residual = inputs

        x = self.add([x, residual])
        return self.activation(x)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GaussianProcessLayerDiag(Layer):
    def __init__(self, num_classes,**kwargs):
        super(GaussianProcessLayerDiag, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs):
        loc = inputs[..., :self.num_classes]
        scale_diag = tf.nn.softplus(inputs[..., self.num_classes:])
        dist = Independent(MultivariateNormalDiag(loc=loc, scale_diag=scale_diag), reinterpreted_batch_ndims=1)
        logits = loc + tf.random.normal(tf.shape(loc)) * scale_diag
        return logits  # 使用 loc 和 scale_diag 调整后的 logits

    def get_config(self):
        config = super(GaussianProcessLayerDiag,self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResNet34(Model):
    def __init__(self, num_classes, name='resnet_34_sngp', trainable=True, dtype='float32', **kwargs):
        super(ResNet34, self).__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)

        self.conv_1 = CustomConv2D(64, 7, 2, padding='same', activation='relu', l2_reg=0.01)
        self.max_pool = MaxPooling2D(3, 2)

        self.conv_2_1 = ResidualBlock(64, 1, 0.3, 0.01)
        self.conv_2_2 = ResidualBlock(64, 1, 0.3, 0.01)
        self.conv_2_3 = ResidualBlock(64, 1, 0.3, 0.01)

        self.conv_3_1 = ResidualBlock(128, 2, 0.3, 0.01)
        self.conv_3_2 = ResidualBlock(128, 1, 0.3, 0.01)
        self.conv_3_3 = ResidualBlock(128, 1, 0.3, 0.01)
        self.conv_3_4 = ResidualBlock(128, 1, 0.3, 0.01)

        self.conv_4_1 = ResidualBlock(256, 2, 0.3, 0.01)
        self.conv_4_2 = ResidualBlock(256, 1, 0.3, 0.01)
        self.conv_4_3 = ResidualBlock(256, 1, 0.3, 0.01)
        self.conv_4_4 = ResidualBlock(256, 1, 0.3, 0.01)
        self.conv_4_5 = ResidualBlock(256, 1, 0.3, 0.01)
        self.conv_4_6 = ResidualBlock(256, 1, 0.3, 0.01)

        self.conv_5_1 = ResidualBlock(512, 2, 0.3, 0.01)
        self.conv_5_2 = ResidualBlock(512, 1, 0.3, 0.01)
        self.conv_5_3 = ResidualBlock(512, 1, 0.3, 0.01)

        self.global_pool = GlobalAveragePooling2D()

        self.fc_pre_gp = Dense(2 * num_classes, activation='relu') 

        self.gp_layer = GaussianProcessLayerDiag(num_classes)

        self.softmax = Activation('softmax')

    def call(self, inputs, training=True):
        x = self.conv_1(inputs)
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
        x = self.fc_pre_gp(x)
        x = self.gp_layer(x)

        return self.softmax(x)

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
    


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

metric = ['accuracy',
          MeanAbsoluteError(name='MeanError'),
          TruePositives(name='tp'),
          FalsePositives(name='fp'), 
          TrueNegatives(name='tn'), 
          FalseNegatives(name='fn'), 
          Precision(name='precision'),
          Recall(name='recall'), 
          AUC(name='AUC')]

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

early_stopping = EarlyStopping(monitor='accuracy', patience=50, verbose=1, mode='max')
model_checkpoint = ModelCheckpoint('resnet34_sngp.keras', monitor='accuracy', save_best_only=True, mode='max')
callbacks = [CustomCallback(), early_stopping, model_checkpoint]

resnet_34_sngp = ResNet34(num_classes=N_CLASS)
resnet_34_sngp.compile(optimizer=Adam(learning_rate=lr_schedule), loss=CategoricalCrossentropy(), metrics=metric) 

history_1 = resnet_34_sngp.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    verbose=1,
    callbacks=callbacks
)

get_custom_objects().update({
    'CustomConv2D': CustomConv2D,
    'ResidualBlock': ResidualBlock,
    'SpectralNormalization': SpectralNormalization,
    'GaussianProcessLayerDiag': GaussianProcessLayerDiag,
    'ResNet34': ResNet34
})

resnet_34_sngp.save('resnet34_sngp.keras')
loaded_model = tf.keras.models.load_model('resnet34_sngp.keras', custom_objects=get_custom_objects())

plt.plot(history_1.history['accuracy'], label='Training Accuracy') 
plt.plot(history_1.history['loss'], label='Training loss')
plt.plot(history_1.history['precision'], label='Training precision')
plt.plot(history_1.history['recall'], label='Training recall')
plt.plot(history_1.history['AUC'], label='Training AUC')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.savefig('resnet_sngp.png')
plt.show()
