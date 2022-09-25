import model
import dataset

import tensorflow as tf

n  = 100

data  = dataset.MyDataset('data').shuffle(1000)
test  = data.take(n)
train = data.skip(n)

model = model.MyModel()
model.compile(
    loss      = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam()
)
model.fit(
    x              = train.batch(600),
    validation_data= test .batch(600),
    epochs         = 200,
    callbacks      = [
        tf.keras.callbacks.BackupAndRestore('training/backup'),
        tf.keras.callbacks.ModelCheckpoint ('training/checkpoint', save_best_only= True)
    ]
)