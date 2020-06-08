import numpy as np
import pandas as pd
from tensorflow_core.python.data import Dataset

feature1=np.linspace(0,1,num=10000)

feature1=np.random.rand(10000)
print(feature1)

feature2=np.random.randint(low=1, high=100,size=10000)
print(feature2)

feature3=np.random.randint(low=0, high=2,size=10000)
print(feature3)

def getInt(low=0, high=2):
    return np.random.randint(low, high)

def getFloat():
    return str(np.random.rand())

def getInt01():
    return str(getInt())

def getInt20():
    return str(getInt(0,20))


print(getInt())
print(getFloat())

# 0
# 0.05 0.004983 0.05 0 0.021594 0.008 0.15 0.04 0.362 0.166667 0.2 0 0.04
# 2 3 0 0 1 1 0 3 1 0 0 0 0 3 0 0 1 4 1 3 0 0 2 0 1 0
fw=open('/data/criteo/TEA', 'w')
for line in range(100):
    lineList=[]
    lineList.extend([getInt01(),
                     getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),getFloat(),
                     getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),getInt20(),
                     '\n'])
    fw.write(' '.join(lineList))
fw.close()