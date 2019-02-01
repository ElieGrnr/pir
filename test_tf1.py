#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging, numpy as np, matplotlib.pyplot as plt
import keras, pickle

import pdb

Nechantillion = int(30e3)
Rr = 8.1
om_min, om_max = -1, 1
wr = np.random.uniform(om_min, om_max, Nechantillion)
V = Rr*wr + 0.1*np.random.randn(Nechantillion)


_i = keras.layers.Input((1,), name ="net_i") #wr
_l = keras.layers.Dense(1, activation='linear', kernel_initializer='uniform',
                             input_shape=(1,), use_bias=False, name="plant")
_o = _l(_i)
_ann = keras.models.Model(inputs=_i, outputs=_o)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
_ann.compile(loss='mean_squared_error', optimizer=opt)

_ann_in, _ann_out = wr, V
# Fit the network
#callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
#             keras.callbacks.ModelCheckpoint(filepath='lvel_id_best_model.h5', monitor='val_loss', save_best_only=True)]
history = _ann.fit(_ann_in, _ann_out, epochs=50, batch_size=64,  verbose=1,
                   shuffle=True, validation_split=0.1)#, callbacks=callbacks)

print('weights {}'.format(_l.get_weights()))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#jpu.decorate(plt.gca(), 'loss', xlab='epochs', legend=['training', 'validation'])


plt.figure()
plt.plot(wr, V, '.')
om_test = np.linspace(om_min, om_max, 100) 
V_test = om_test *_l.get_weights()[0][0,0]
plt.plot(om_test, V_test)
plt.show()
