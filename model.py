import tensorflow as tf
import tensorflow_hub as hub


class MyModel(tf.keras.models.Model):

    modelUrl = (
        'https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1'
        # 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'
    )

    def __init__(self):
        super(MyModel, self).__init__()

        self.hubLayer = hub.KerasLayer(self.modelUrl)
        self.flatten  = tf.keras.layers.Flatten()
        self.concat   = tf.keras.layers.Concatenate()
        self.dense    = tf.keras.layers.Dense(4)

    def call(self, inputs, training= False):
        y = self.hubLayer(inputs, training= training)
        y = y[1]
        y = [self.flatten(e) for e in y]
        y = self.concat(y)
        y = self.dense(y)

        return y
