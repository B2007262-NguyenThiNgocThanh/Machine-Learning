import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor


# Số lần lặp
loop_time = 10

# Đọc file dữ liệu
def getDataset():
    dataset = pd.read_csv("quake.dat", delimiter=',')
    return dataset


# Hiển thị tập train, test
def showData(X_train, X_test, y_train, y_test):

    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:\n", y_train)
    print("y_test:\n", y_test)


# Hồi quy tuyến tính:
def linearRegression(X_train, X_test, y_train, y_test):

    lm = linear_model.LinearRegression()
    bagging_reg = BaggingRegressor(estimator = lm, n_estimators = 10, random_state = 42)
    bagging_reg.fit(X_train, y_train)
    y_pred = bagging_reg.predict(X_test)
    #print("\ny_pred LR = ", y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE of LinearRegression: ", round(mse, 5))
    return mse


def randomForest(X_train, X_test, y_train, y_test):

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    #print("y_pred RF = ", y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE of RandomForest: ", round(mse, 5))
    return mse


def knnRegression(X_train, X_test, y_train, y_test, k):
    #j = i+1
    knn = KNeighborsRegressor(n_neighbors=25 + k, p=1, metric='manhattan')
    #knn = KNeighborsRegressor(n_neighbors=20 + k, p=1, metric='manhattan') #kiểm tra số láng giềng tối ưu
    print('so lang gieng: ',25+k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #print("y_pred KNN = ", y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE of KNeighbors: ", round(mse, 5))
    return mse


# Lặp 'loop_time' lần

j = 0
def execute(attribute, label):

    global mseLR, mseRF, mseKNN,j

    for i in range(0, loop_time):

        # chia các tập train & test với tỉ lệ 2:1 và có xáo trộn
        X_train, X_test, y_train, y_test = train_test_split(attribute, label, test_size=1/3,random_state=42+i)

        print("\n-------------------------------------------------------\nLần ", (i+1))
        # showData(X_train, X_test, y_train, y_test)
        linearRegression(X_train, X_test, y_train, y_test)
        randomForest(X_train, X_test, y_train, y_test)
        knnRegression(X_train, X_test, y_train, y_test, j)
        j = j + 10
        # errKNN.append(knnRegression(X_train, X_test, y_train, y_test,k))

        #print("K= ", k)
errKNN=[]
KNN=[]
k=0
def execute2(attribute, label):

    for i in range(0, loop_time):
        global k
        X_train, X_test, y_train, y_test = train_test_split(attribute, label, test_size=1/3,random_state=42)
        errKNN.append(round(knnRegression(X_train, X_test, y_train, y_test, k),5))
        k = k + 2
        KNN.append(23+k)
    print(errKNN)
    print(KNN)
# vẽ đồ thị với các láng giềng khác nhau thì cho biết k bằng mấy sẽ có tỷ lệ dự đoán sai
    plt.figure( figsize=(10,8) )
    plt.title( "Đồ thị số lượng khác nhau giữa kết quả đự đoán và dataset của thuật toán KNN" )
    plt.plot(KNN,errKNN,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=5)
    plt.xlabel( "Số láng giềng K" )
    plt.ylabel( "Số lượng khác trên tổng số dữ liệu cần test" )
    plt.show()



# lấy dữ kiệu
dt = getDataset()

X = dt.iloc[:,0:3] # lấy từ hàng đầu tiên của cột thứ 1, 2, 3
y = dt.iloc[:,3]   # lấy từ hàng đầu tiên của cột thứ 4

print("Attribute:\n", X)
print("Label:\n", y)

execute(X, y)
