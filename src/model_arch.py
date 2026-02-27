import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

class AttentionGate(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.W_g = layers.Conv2D(self.filters, 1, padding='same', use_bias=True)
        self.W_x = layers.Conv2D(self.filters, 1, padding='same', use_bias=True)
        self.psi = layers.Conv2D(1, 1, padding='same', use_bias=True)
        self.relu = layers.ReLU()
        self.sigmoid = layers.Activation('sigmoid')
        super(AttentionGate, self).build(input_shape)

    def call(self, inputs):
        g, x = inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        return x * psi

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({'filters': self.filters})
        return config

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    focal = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)(y_true, y_pred)
    dice = 1 - dice_coefficient(y_true, y_pred)
    return 0.5 * focal + 0.5 * dice
