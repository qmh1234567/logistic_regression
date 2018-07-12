import numpy as np
from matplotlib import pyplot as plt
import time

a=np.random.rand(1000000)
b=np.random.rand(1000000)

print(a,b)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(c)
print("vectorize:",1000*(toc-tic),"ms")

c=0
tic1=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
print(c)
toc1=time.time()
print("for",1000*(toc1-tic1),"ms")
