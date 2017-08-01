from sklearn import svm, metrics
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib

train_values = []
train_values_target = []
num_values = 50001
num_test_values = 12501
eqtnvals = 33
matplotlib.rcParams.update({'font.size': 8})
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
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.plot(train_values[i])
    plt.axis('off')
    plt.title("Training: \n%s" % train_values_target[i])

clf = svm.SVC(gamma=0.001)
train_values_arr = np.array(train_values)
train_values_target_arr = np.array(train_values_target)

clf.fit(train_values_arr, train_values_target_arr)

test_values = []
expected = []
for f in range(num_test_values/3):
    sinlst = []
    sinamplitude = random.random() * 2
    sinperiod = random.random() * 2
    for j in range(eqtnvals):
        sinlst.insert(len(sinlst), sinamplitude * math.sin((math.pi/((eqtnvals-1)/2))*float(j)*sinperiod))
    test_values.insert(f*3, sinlst)
    expected.insert(f*3, "sin")
    tanlst = []
    tanamplitude = random.random() * 2
    tanperiod = random.random() * 2
    for j in range(eqtnvals):
        tanlst.insert(len(tanlst), tanamplitude * math.tan((math.pi/((eqtnvals-1)/2))*float(j)*tanperiod))
    test_values.insert((f*3)+1, tanlst)
    expected.insert((f*3)+1, "tan")
    speciallst = []
    specialamplitude = random.random() * 2
    specialperiodone = random.random() * 2
    specialperiodtwo = random.random() * 2
    for j in range(eqtnvals):
        speciallst.insert(len(speciallst), specialamplitude * 1.0 / (math.cos(specialperiodone * math.pow(math.e, math.sin((math.pi / ((eqtnvals - 1) / 2)) * float(j) * specialperiodtwo)))))
    test_values.insert((f * 3) + 2, speciallst)
    expected.insert((f * 3) + 2, "1/(cos(e^sin(x)))")

test_values = np.array(test_values)
expected = np.array(expected)
predicted = clf.predict(test_values)
print("Classification Report For Classifier:")
print(metrics.classification_report(expected, predicted))
print("Confusion Matrix of results:")
print(metrics.confusion_matrix(expected, predicted))
for i in range(6):
    plt.subplot(2, 6, i+7)
    plt.plot(test_values[i])
    plt.axis('off')
    plt.title("Predicted: \n%s" % predicted[i])
plt.show()
