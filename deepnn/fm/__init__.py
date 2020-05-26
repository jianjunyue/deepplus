import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import np
import pandas as pd

import xlearn as xl

path="C:\\Users\\lejianjun\\git\\deepplus\\data\\";
outpath=path+"out\\"
# Training task
fm_model = xl.create_fm()  # Use factorization machine
fm_model.setTrain(path+"titanic_train.txt")  # Training data
fm_model.setValidate(path+"titanic_train.txt")

param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

fm_model.fit(param, outpath+'modelfm.out')

# Prediction task
fm_model.setTest(path+"titanic_test.txt")  # Set the path of test dataset

# Start to predict
# The output result will be stored in output.txt
fm_model.predict(outpath+"modelfm.out",outpath+ "outputfm.txt")


