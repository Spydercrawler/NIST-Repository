from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import math
import random

train_values = []
train_values_target = []
num_values = 125001
eqtnvals = 65
for f in range(num_values/3):
    # Sin
    sinlst = []
    sinamplitude = random.random() * 2
    sinperiod = random.random() * 2
    for j in range(eqtnvals):
        sinlst.insert(len(sinlst), sinamplitude * math.sin((math.pi/((eqtnvals-1)/2))*float(j)*sinperiod))
    train_values.insert(f*3, sinlst)
    train_values_target.insert(f*3, "sin")
    # Tan
    tanlst = []
    tanamplitude = random.random() * 2
    tanperiod = random.random() * 2
    for j in range(eqtnvals):
        tanlst.insert(len(tanlst), tanamplitude * math.tan((math.pi/((eqtnvals-1)/2))*float(j)*tanperiod))
    train_values.insert((f*3)+1, tanlst)
    train_values_target.insert((f*3)+1, "tan")
    # 1/cos(e^sin(x))
    speciallst = []
    specialamplitude = random.random() * 2
    specialperiodone = random.random() * 2
    specialperiodtwo = random.random() * 2
    for j in range(eqtnvals):
        speciallst.insert(len(speciallst), specialamplitude* 1.0/(math.cos(specialperiodone*math.pow(math.e, math.sin((math.pi/((eqtnvals-1)/2))*float(j)*specialperiodtwo)))))
    train_values.insert((f * 3) + 2, speciallst)
    train_values_target.insert((f * 3) + 2, "1/(cos(e^sin(x)))")

clf = svm.SVC(gamma=0.001)
train_values_arr = np.array(train_values)
train_values_target_arr = np.array(train_values_target)
print("Training classifier!")
clf.fit(train_values_arr, train_values_target_arr)
joblib.dump(clf, 'SinTangentClassifier.pkl')
print(clf)
