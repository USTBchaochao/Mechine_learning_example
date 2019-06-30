import pickle
import numpy
from sklearn import linear_model
import sklearn.metrics as sm
output_model_file = "linear_model.pkl"
with open(output_model_file,'rb') as f:
    model_linear = pickle.load(f)

filename = "data_singlevar.txt"
x = []
y = []
with open(filename,'r') as f:
    for line in f.readlines():
        xt , yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
# print(f)
num = len(x)
x = numpy.array(x[:num]).reshape((num,1))
y = numpy.array(y[:num])
y_test_pred = model_linear.predict(x)
print(round(sm.mean_absolute_error(y_test_pred,y),10))
print(round(sm.r2_score(y_test_pred,y),10))