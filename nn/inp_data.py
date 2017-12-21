import pandas as pd
import numpy as np
import sys
import random

def read_raw_data():
     
     path = '/home/kb/tf/nn/letterdata.csv'
     d = pd.read_csv(path)     
     d = np.array(d)
     
     data_in = []
     data_out = []

     train_in = []
     test_in = []
     train_out = []
     test_out = []
    
     nrow = d.shape[0]
     ncol = d.shape[1]
    
     for i in range(nrow):
       t = ord(d[i,0].lower()) - 96
       q = np.zeros((26),dtype=np.int)
       q[t-1] = 1
       data_in.append(d[i,1:ncol])  
       data_out.append(q)
     

     data_in = np.array(data_in)
     data_out = np.array(data_out)

     # normalize data     
     for j in range(data_in.shape[1]):
       col_mean = np.mean(data_in[:,j])
       col_std = np.std(data_in[:,j]) 
       data_in[:,j] = (data_in[:,j] - col_mean)/col_std        

     for i in range(nrow):
       r = np.random.rand()
       if (r < 0.8):
         train_in.append(data_in[i,:])
         train_out.append(data_out[i,:])
       else:           
         test_in.append(data_in[i,:])
         test_out.append(data_out[i,:])

     train_in = np.array(train_in)
     test_in = np.array(test_in)
     train_out = np.array(train_out)
     test_out = np.array(test_out)                      
                                                                              
     ntrain = train_in.shape[0]
     ntest = test_in.shape[0]
            
     return ntrain, ntest, train_in, train_out, test_in, test_out 
