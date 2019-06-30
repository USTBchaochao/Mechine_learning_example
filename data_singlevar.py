import numpy
from sklearn import linear_model
import sklearn.metrics as sm
import pickle
filename = "data_singlevar.txt"
x = []
y = []
with open(filename,'r') as f:
    for line in f.readlines():
        xt , yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

num_training = int (0.8 * len(x))

x_train = numpy.array(x[:num_training]).reshape((num_training,1))
y_train = numpy.array(y[:num_training])

num_test = len(x) - num_training
x_test = numpy.array(x[num_training:]).reshape((num_test,1))
y_test = numpy.array(y[num_training:])

linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(x_train,y_train)

y_test_pre = linear_regressor.predict(x_test)

mean_absolute_error = round(sm.mean_absolute_error(y_test,y_test_pre),10)#平均绝对误差
mean_square_error = round(sm.mean_squared_error(y_test,y_test_pre),10)#均方差
R2_score = round(sm.r2_score(y_test,y_test_pre),10)#相关系数

print(R2_score)
output_model_file = 'linear_model.pkl'
with open(output_model_file,'wb') as f:
    pickle.dump(linear_regressor,f)

# print(mean_absolute_error,mean_square_error,R2_score)
# plt.figure()
# plt.plot(x_train,y_train_pre,color = 'black',linewidth = 4)
# plt.scatter(x_train,y_train,color = 'green')
# plt.title ('Train data')
# plt.show()
# print("x=",x)
# print("y=",y)
