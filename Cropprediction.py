from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras.layers
from keras.models import model_from_json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
main = tkinter.Tk()
main.title("Crop  Prediction using Machine learning")
main.geometry("1000x650")

global train, test, X_train, X_test, y_train, y_test
global filename
global cls


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

        

def traintest(data):     #method to generate test and train data from dataset
    train=data.iloc[:, 0:7].values
    test=data.iloc[: ,8].values
    print(train)
    print(test)
    X_train, X_test, y_train, y_test = train_test_split( 
    train, test, test_size = 0.3, random_state = 0)
    return train, test, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all  features data
    global train, test, X_train, X_test, y_train, y_test
    train1 = pd.read_csv(filename)
    train, test, X_train, X_test, y_train, y_test = traintest(train1)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train1))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(50):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy
    
def runDCT():
    global dct_acc
    global cls
    global train, test, X_train, X_test, y_train, y_test
    #Importing Decision Tree classifier
    from sklearn.tree import DecisionTreeRegressor
    cls=DecisionTreeRegressor()

    #Fitting the classifier into training set
    cls.fit(X_train,y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    dct_acc = cal_accuracy(y_test, prediction_data,'Decision  Tree Accuracy')

def runRF():
    global random_acc
    global cls
    global train, test, X_train, X_test, y_train, y_test
    #Importing Decision Tree classifier
    rf=RandomForestClassifier(n_estimators=50,max_depth=2,random_state=0,class_weight='balanced')

    #Fitting the classifier into training set
    rf.fit(X_train,y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, rf) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest  Accuracy')

def runPAC():
    global pac_acc
    global cls
    global train, test, X_train, X_test, y_train, y_test	
    linear_clf = PassiveAggressiveClassifier()
    linear_clf.fit(X_train,y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, linear_clf) 
    pac_acc = cal_accuracy(y_test, prediction_data,'Random Forest  Accuracy') 

def predicts():
    global clean
    global attack
    global total
    clean = 0;
    attack = 0;
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 0:7]
    total = len(test)
    text.insert(END,filename+" test file loaded\n");
    y_pred = cls.predict(test) 
    #text.insert(END,y_pred+" \n");
    print(y_pred)
    '''for i in range(len(y_pred)):
        text.insert(END,i," \n");'''
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack = attack + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Crop name is rice')+"\n\n")
        elif str(y_pred[i]) == '2.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is wheat')+"\n\n")
        elif str(y_pred[i]) == '3.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is Mung Bean')+"\n\n")         
        elif str(y_pred[i]) == '4.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is Tea')+"\n\n")    
        elif str(y_pred[i]) == '5.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is millet')+"\n\n")
        elif str(y_pred[i]) == '6.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is maize')+"\n\n")
        elif str(y_pred[i]) == '7.0':
            clean = clean + 1
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'crop name is Lentil')+"\n\n")
def graph():
    height = [dct_acc,random_acc,pac_acc]
    bars = ('Decission tree','Random forest','PassiveAggressiveClassifier')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   



font = ('times', 15, 'bold')
title = Label(main, text='Crop  Prediction using Machine Learning', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Agriculture Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=generateModel)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

rnnButton = Button(main, text="Run Decisiontree Algorithm", command=runDCT)
rnnButton.place(x=480,y=100)
rnnButton.config(font=font1)

lstmButton = Button(main, text="Run Randomforest Algorithm", command=runRF)
lstmButton.place(x=700,y=100)
lstmButton.config(font=font1)

ffButton = Button(main, text="Passive Aggressive Algorithm", command=runPAC)
ffButton.place(x=10,y=150)
ffButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=300,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Detect Crop", command=predicts)
predictButton.place(x=550,y=150)
predictButton.config(font=font1)
'''
predictButton = Button(main, text="Predict Disease using Test Data", command=predict)
predictButton.place(x=10,y=200)
predictButton.config(font=font1)

topButton = Button(main, text="Top 6 Crop Yield Graph", command=topGraph)
topButton.place(x=300,y=200)
topButton.config(font=font1)'''

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
