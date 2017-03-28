import keras
from keras import backend as K
from keras.engine.topology import Layer
import keras.models as M
import keras.layers as L
import numpy as np

class MeanDiff(Layer):
    def __init__(self, output_dim, **kwargs):
      self.output_dim = output_dim
      super(MeanDiff, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanDiff, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x-K.mean(x,keepdims=True, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

val2 = np.asarray(np.random.rand(32,6), dtype=np.float32)
val1 = np.asarray(np.random.rand(32,1), dtype=np.float32)
out_val_cap = val2-np.mean(val2,axis=1,keepdims=True) + np.repeat(val1, 6,axis=1)

x2 = L.Input(shape=(6,))
y2 = MeanDiff(6)(x2)

x1 = L.Input(shape=(1,))
y1 = L.RepeatVector(6)(x1)
y1 = L.Reshape((6,))(y1)

y = L.add([y1,y2])

model= M.Model([x1,x2],y)
out_val = model.predict_on_batch([val1,val2])

# print(out_val-out_val_cap)
#

class DuelModel():
    def __init__(self, window, input_shape, num_actions):
        self.input = L.Input(shape=tuple([window] + list(input_shape)))
        self.num_actions = num_actions
        self.conv1_op = L.Conv2D(32, 8, strides=4, activation='relu', input_shape=tuple([window] + list(input_shape)),
                                 data_format='channels_first')(self.input)
        self.conv2_op = L.Conv2D(64, 4, strides=2, activation='relu', data_format='channels_first')(self.conv1_op)
        self.conv3_op = L.Conv2D(64, 3, strides=1, activation='relu', data_format='channels_first')(self.conv2_op)
        self.conv_flat = L.Flatten()(self.conv3_op)
        self.stream_sv_fc1 = L.Dense(512, activation='relu')(self.conv_flat)
        self.stream_sv_fc2 = L.Dense(1)(self.stream_sv_fc1)
        self.stream_av_fc1 = L.Dense(512, activation='relu')(self.conv_flat)
        self.stream_av_fc2 = L.Dense(self.num_actions)(self.stream_av_fc1)
        self.stream_av_fc2 = MeanDiff(self.num_actions)(self.stream_av_fc2)
        self.stream_sv_fc2 = L.RepeatVector(self.num_actions)(self.stream_sv_fc2)
        self.stream_sv_fc2 = L.Reshape((self.num_actions,))(self.stream_sv_fc2)
        self.combo_q_val = L.add([self.stream_av_fc2, self.stream_sv_fc2])
        self.model = M.Model([self.input], self.combo_q_val)


    def __call__(self):
        return self.model

tmodel = DuelModel(window=4, input_shape=[84,84], num_actions=6)()

# tinput = np.random.rand(1,4,84,84)
tinput = np.ones(shape=(1,4,84,84))

top = tmodel.predict_on_batch(tinput)

print(top)
