import os
import numpy as np
import codecs
#输入相对路径
os.chdir("F:\\Training\\amsr2workspace\\")

array = np.zeros((),dtype=int)
array = np.loadtxt(open("quweima2.txt"))
array2 = array.astype(int)
print('所有字',array2)
window_size = 200
x = np.atleast_3d(np.array([array2[start:start + window_size] for start in range(0, array.shape[0] - window_size)]))
y = array2[window_size:]
print('50个字：',x)
print('第51个字：',y)

a = np.loadtxt(open("thelast50.txt"))#原文最后50个字
b = a.astype(int)
shuchu = b.reshape(1,50,1)
print('原文最后50个字，用作输出：',shuchu)
