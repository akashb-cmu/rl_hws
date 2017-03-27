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
