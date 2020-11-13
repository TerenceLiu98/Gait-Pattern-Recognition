from statsmodels import api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    db4 = pywt.Wavelet('db3')
    # 分解
    coeffs = pywt.wavedec(data, db4)
    # 高频系数置零
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    return meta

pca = PCA(n_components=1)


YAN = pd.read_excel('data/YAN.xlsx')

pca.fit(YAN)
YAN_1 = pca.transform(YAN)
YAN_1 = YAN_1.reshape(1,-1)[0]

LI = pd.read_excel('data/LI_without_Group3.xlsx')
pca.fit(LI)
LI_1 = pca.transform(LI)
LI_1 = LI_1.reshape(1,-1)[0]

b, a = sm.signal.build_filter(frequency=10,
                              sample_rate=100,
                              filter_type='low',
                              filter_order=4)