import tensorflow as tf

num_gpus = len(tf.config.list_physical_devices('GPU'))

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    print('do model things: model.compile for example; model = ... for example')
