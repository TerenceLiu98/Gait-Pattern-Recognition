import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    db_ = pywt.Wavelet('db6')
    # 分解
    coeffs = pywt.wavedec(data, db_)
    # 高频系数置零
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db_)
    return meta

def data_processing(data):
    data_2 = pd.DataFrame()
    for i in data.columns:
        data_2[i] = wavelet_denoising(data[i])
    pca = PCA(n_components=4)
    pca.fit(data_2)
    print('PCA: ', pca.explained_variance_ratio_)
    data_1 = pca.transform(data)
    #data_1 = data_1.reshape(4,-1)[0]
    data_1 = pd.DataFrame(data_1)
    data_1.columns = ['one', 'two','three','four']
    return data_2

def filter(data, w, LorW:bool):
   if LorW == True:
      pass_ = 'lowpass'
   if LorW == False:
      pass_ = 'highpass'
   b, a = signal.butter(1, w, pass_)
   filtedData = signal.filtfilt(b, a, data)
   return filtedData
   #plt.plot(filtedData)
   #plt.show()

def select_data(data, timeframe):
    data_list = []
    for i in range(12):
        tmp = data[data.columns[i]][timeframe * i:(i+1)*timeframe]
        for j in range(len(tmp)):
            data_list.append(tmp[j+i*timeframe])
    return filter(data_list,0.6, True)

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def data_fusion(data, data_frame):
   p = data.filter(regex='p') 
   p = p[['2-1-p', '2-2-p', '2-3-p', '2-4-p',
      '2-5-p', '2-6-p', '2-7-p', '2-8-p', 
      '2-9-p','2-10-p', '2-11-p', '2-12-p']]
   #p = select_data(p, data_frame)
   p = p.T.cumsum().T['2-9-p']
   p = filter(p, 0.4,True)
   x = data.filter(regex='x')
   x = x[['2-1-x', '2-2-x', '2-3-x', '2-4-x',
      '2-5-x', '2-6-x', '2-7-x', '2-8-x', 
      '2-9-x','2-10-x', '2-11-x', '2-12-x']]
   #x = select_data(x, data_frame)
   x = x.T.cumsum().T['2-9-x']
   x = filter(x,  0.04,True)
   y = data.filter(regex='y') 
   y = y[['2-1-y', '2-2-y', '2-3-y', '2-4-y',
      '2-5-y', '2-6-y', '2-7-y', '2-8-y', 
      '2-9-y','2-10-y', '2-11-y', '2-12-y']]
   #y = select_data(y, data_frame)
   y = y.T.cumsum().T['2-9-y']
   y = filter(y, 0.04,True)
   z = data.filter(regex='z') 
   z = z[['2-1-z', '2-2-z', '2-3-z', '2-4-z',
      '2-5-z', '2-6-z', '2-7-z', '2-8-z', 
      '2-9-z','2-10-z', '2-11-z', '2-12-z']]
   #z = select_data(z, data_frame)
   z = z.T.cumsum().T['2-9-z']
   z = filter(z,  0.04,True)
   label = [get_df_name(data)] * len(p)
   return pd.DataFrame({'p':p, 'x':x, 'y':y, 'z':z, 'label':label})

if __name__ == "__main__":
   gao = pd.read_csv('data/GAO_Group_2.csv').drop(['timestamp'], axis=1)
   wang = pd.read_csv('data/WANG_Group_2.csv').drop(['timestamp'], axis=1)
   li = pd.read_csv('data/LI_Group_2.csv')
   yan = pd.read_csv('data/YAN_Group_2.csv')

   gao = gao.fillna(gao.mean())
   wang = wang.fillna(wang.mean())
   li = li.fillna(li.mean())
   yan = yan.fillna(yan.mean()) 

   y1 = data_fusion(yan, int(len(yan)/12))
   w1 = data_fusion(wang, int(len(wang)/12))
   g1 = data_fusion(gao, int(len(gao)/12))
   l1 = data_fusion(li, int(len(li)/12))

   y1.to_csv('data/Y1.csv',index = False)
   w1.to_csv('data/W1.csv',index = False)
   g1.to_csv('data/G1.csv',index = False)
   l1.to_csv('data/L1.csv',index=False)

