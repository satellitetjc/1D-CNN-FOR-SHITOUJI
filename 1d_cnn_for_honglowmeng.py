from __future__ import print_function, division
import os
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout, Activation
from keras.models import Sequential
from keras.utils import plot_model, normalize
from keras import optimizers

def smc_retrieval(window_size, filter_length, nb_input_lw, nb_output_smc, 
                  nb_filter):
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, 
                      activation='relu', input_shape=(window_size, 
                                                      nb_input_lw)),
        MaxPooling1D(), 
        Convolution1D(60, filter_length=filter_length, 
                      activation='relu'),
        MaxPooling1D(),
        #Dropout(0.05),
        Flatten(),
        Dense(100),
        #BatchNormalization(),
        Activation('relu'),      
        #BatchNormalization(), 
        #Dropout(0.005),        
        Dense(nb_output_smc, activation='relu'),     # For binary classification, change the activation to 'sigmoid'
    ))
    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)    
    #默认lr=0.001
    model.compile(loss='mse', optimizer= adam,  metrics=['mae'])

    return model

def load_save():
    os.chdir("E:\\hongloumen\\workspace\\shuju\\")
    array = np.zeros((),dtype=int)
    array = np.loadtxt(open("quweima2.txt"))
    array2 = array.astype(int)
    #print('所有字',array2)
    #window_size = 50
    x = np.atleast_3d(np.array([array2[start:start + 50] for start in range(0, array.shape[0] - 50)]))
    #x_nol = normalize(x, axis=1, order=2)
    y = array2[50:]
    #print('50个字：',x)
    #print('第51个字：',y)
    #print(x_nol)
    #return x_nol,y
    return x,y
load_save()

def test_save():
     os.chdir("E:\\hongloumen\workspace\\test\\")
     a = np.loadtxt(open("thelast50.txt"))#原文最后50个字
     b = a.astype(int)
     #c = b.reshape(1,50,1)
     shuchu = b.reshape(1,50,1)
     #shuchu = normalize(c, axis=1, order=2)
     #print(shuchu)
     return shuchu
test_save()

def evaluate_train():
    filter_length = 4
    nb_filter = 60
    model = smc_retrieval(window_size=50, filter_length=filter_length,
                          nb_input_lw=1, nb_output_smc=1, nb_filter=nb_filter)
    model.summary()    
    #x_nol,y = load_save()
    x,y = load_save()
    shuchu = test_save()    
    #x_train, x_vol, y_train, y_vol = x_nol[:8000], x_nol[8000:], y[:8000], y[8000:]
    x_train, x_vol, y_train, y_vol = x[:8000], x[8000:], y[:8000], y[8000:]
    model.fit(x_train, y_train, nb_epoch=25, batch_size=90, validation_data=(x_vol, y_vol))
    
    pred = model.predict(shuchu)
    print(pred)
    f = open("shuchu51.txt","w")
    print('shuchu51', sep='\t',file = f)
    #for shuchu51 in pred():
    print(pred, sep='\t',file = f)
       
def main():
 
    evaluate_train()
    
if __name__ == '__main__':
    main()
