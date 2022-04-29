
from fileinput import filename
from django.shortcuts import render,redirect

from sklearn.ensemble import RandomForestRegressor
# Create your views here.
 
from .models import Student

import joblib
import pandas as pd
from sklearn import naive_bayes, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn import metrics


#DATA PROCESSING
#reading the cleaned data
df=pd.read_csv("../allbranchdata.csv")
#df=pd.read_csv("../classifierdata.csv")

#dividing the column into x and y
x=df[['10th','12th','sem1%','sem2%','sem3%','sem4%']]
y=df[['sem5%','class','sem5']]

'''
Preprocessing data. Scaling input data so
that all the input data lie between 0 to 1
'''
min_max_scaler = preprocessing.MinMaxScaler()
#X_scale = min_max_scaler.fit_transform(x)
X_scale=x

#splitting data into train ,test and validation data
x_train,x_test,y_train,y_test=train_test_split(X_scale,y, train_size=.85,random_state=2)


def get_percentage(sem1,sem2,sem3,sem4):
    if sem1<7:
        sem1=sem1*7.1+12
    else:
        sem1=sem1*7.4+12
    if sem2<7:
        sem2=sem2*7.1+12
    else:
        sem2=sem2*7.4+12
    if sem3<7:
        sem3=sem3*7.1+12
    else:
        sem3=sem3*7.4+12
    if sem4<7:
        sem4=sem4*7.1+12
    else:
        sem4=sem4*7.4+12
    return sem1,sem2,sem3,sem4


def get_pointer(sem5):
    if sem5<63.80:
        sem5=(sem5-12)/7.1
    else:
        sem5=(sem5-12)/7.4
    return sem5
#Decission Tree Regressor Model

def pointer(li):
    # Python code to illustrate
    # map() with lambda()
    # to get double of a list.
    

    final_list = list(map(lambda x: round(get_pointer(x),2), li))
    return final_list

def model2(x_train,y_train,x_test):
    #model= DTR 
    DTRModel = DecisionTreeRegressor(max_depth=7)
    DTRModel.fit(x_train, y_train)
    # save the model to disk
    filename = 'DTR.sav'
    joblib.dump(DTRModel, filename)
    # sem1,sem2,sem3,sem4 = get_percentage(3.85,3.9,4.96,4.3)
    # output=DTRModel.predict([[51.84,49.84,sem1,sem2,sem3,sem4]])
    # print("DTR prediction:",output)
    y_pred = DTRModel.predict(x_test)
    print("Predicted DTR Model")
    return y_pred    
DTRpred=pointer(model2(x_train,y_train.iloc[:,0],x_test))

#Linear Regression Model
def model3(x_train,y_train,x_test):
    #model=LR
    LRModel=LinearRegression()
    LRModel.fit(x_train,y_train)
    y_pred=LRModel.predict(x_test)
    y_pred=[round(item,2) for item in y_pred]
    filename='LR.sav'
    joblib.dump(LRModel,filename)
    
    print("Predicted Linear regression")
    # sem1,sem2,sem3,sem4 = get_percentage(3.85,3.9,4.96,4.3)
    # output=LRModel.predict([[51.84,49.84,sem1,sem2,sem3,sem4]])
    # print("LR prediction:",output)
    return y_pred
LRpred=pointer(model3(x_train,y_train.iloc[:,0],x_test))


#Random Forest Regressor
def model4(x_train,y_train,x_test):
    #random forest
    RFRModel=RandomForestRegressor(n_estimators=40,random_state=0)
    RFRModel.fit(x_train,y_train)
    y_pred=RFRModel.predict(x_test)
    
    filename='RFR.sav'
    joblib.dump(RFRModel,filename)
    print("Predicted Random Forest Regressor")
    return y_pred
RFRpred=pointer(model4(x_train,y_train.iloc[:,0],x_test))

#Naive Bayes Model
def model5(x_train,y_train,x_test):
    #Naive Bayes Classifier
    
    NBModel=naive_bayes.GaussianNB()
    NBModel.fit(x_train,y_train)
    y_pred=NBModel.predict(x_test)

    filename='NB.sav'
    joblib.dump(NBModel,filename)
    print("Predicted Naive Bayes Classifier")
    return y_pred
NBpred=model5(x_train,y_train['class'],x_test)

#Decision tree classifier
def model6(x_train,y_train,x_test):
    #Decision Tree Classifier
    
    DTCModel=DecisionTreeClassifier()
    DTCModel.fit(x_train,y_train)
    y_pred=DTCModel.predict(x_test)
    filename='DTC.sav'
    print("Predicted Decision Tree Classifier")
    joblib.dump(DTCModel,filename)
    return y_pred
DTCpred=model6(x_train,y_train['class'],x_test)


#Home Page view function
def home(request):

    return render(request,"index.html",{})


#Analyses Page view function
def analysis(request):
    from sklearn import metrics
    DTRModel=joblib.load("DTR.sav")
    LRModel=joblib.load("LR.sav")
    RFRModel=joblib.load("RFR.sav")
    context={
        'Predictions':
        [
        {
            "Model":"Decision Tree Regressor",
            "Accuracy":round(DTRModel.score(x_test,y_test.iloc[:,0])*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test.iloc[:,2], DTRpred)*100,2),
            "MeanAbsoluteError":round(metrics.mean_absolute_error(y_test.iloc[:,2], DTRpred)*100,2)
        },
        {
            "Model":'Linear Regression',
            "Accuracy":round(LRModel.score(x_test,y_test.iloc[:,0])*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test.iloc[:,2], LRpred)*100,2),
            "MeanAbsoluteError":round(metrics.mean_absolute_error(y_test.iloc[:,2], LRpred)*100,2)
        },
        {
            "Model":"Random Forest Regressor",
            "Accuracy":round(RFRModel.score(x_test,y_test.iloc[:,0])*100,2),
            "MeanSquareError":round(metrics.mean_squared_error(y_test.iloc[:,2], RFRpred)*100,2),
            "MeanAbsoluteError":round(metrics.mean_absolute_error(y_test.iloc[:,2], RFRpred)*100,2)
        }
        ]
        ,
        "Classification":
        [    
                {
                    "Model":"Decision Tree Classifier",
                    "Accuracy":round(metrics.accuracy_score(y_test['class'],DTCpred)*100,2),
                    "MeanSquaredError":round(metrics.mean_squared_error(y_test.iloc[:,1], DTCpred)*100,2),
                    "RecallScore":round(metrics.recall_score(y_test['class'],DTCpred)*100,2)
                },
                {
                    "Model":"Naive Bayes ",
                    "Accuracy":round(metrics.accuracy_score(y_test['class'],NBpred)*100,2),
                    "MeanSquaredError":round(metrics.mean_squared_error(y_test.iloc[:,1], NBpred)*100,2),
                    "RecallScore":round(metrics.recall_score(y_test['class'],NBpred)*100,2)
                }
        ]
            
    }
    
    return render(request,"analysis.html",context)

def predict(request):
    error=False
    unseenData=[[]]
    if request.method=="POST":
        tenth=float(request.POST["tenth"])
        twelth=float(request.POST['twelth'])
        sem1=float(request.POST['sem1'])
        sem2=float(request.POST['sem2'])
        sem3=float(request.POST['sem3'])
        sem4=float(request.POST['sem4'])
        #sem5=float(request.POST['sem5'])
        # sem6=float(request.POST['sem6'])
        if twelth>100 or twelth<0 and tenth>100 or tenth<0:
            error=True
        if sem1>10 or sem1<0 and sem2>10 or sem2<0 and sem3>10 or sem3<0 and sem4>10 or sem4<0 :#and sem5>10 or sem5<0
            error=True
        if error:
            return render(request,"prediction.html",{error:error})
        else:
            #Transforming input data
            sem1,sem2,sem3,sem4 = get_percentage(sem1,sem2,sem3,sem4)
            unseenData=[[tenth,twelth,sem1,sem2,sem3,sem4]]
            #eval_X=min_max_scaler.fit_transform(unseenData)
            eval_X=unseenData
            prediction=list(analyze_unseen_data(eval_X))
            print("Predicted Values are",prediction)
            return render(request,"prediction.html",{'dtrprediction':float(prediction[0]),'lrprediction':float(prediction[1]),"rfrprediction":float(prediction[2]),
                'nbprediction':prediction[3],'dtcprediction':prediction[4]
                })
    
    return render(request,"prediction.html",{})

def analyze_unseen_data(eval_x):
    
    #loading prediction models
    DTRModel=joblib.load('DTR.sav')
    LRModel=joblib.load("LR.sav")
    RFRModel=joblib.load("RFR.sav")
    DTRresult=round(get_pointer(float(DTRModel.predict(eval_x))),2)
    LRresult=round(get_pointer(float(LRModel.predict(eval_x))),2)
    RFRresult=round(get_pointer(float(RFRModel.predict(eval_x))),2)
    #loading classifier models
    NBModel=joblib.load('NB.sav')
    DTCModel=joblib.load('DTC.sav')
    NBresult=int(NBModel.predict(eval_x))
    DTCresult=int(DTCModel.predict(eval_x))
    if NBresult == 1:
        NBResult = "Pass"
    else:
        NBResult = "Fail"
    if DTCresult == 1:
        DTCResult = "Pass"
    else:
        DTCResult = "Fail"
    
    return (DTRresult,LRresult,RFRresult
    ,NBResult,DTCResult
    )



#table of actual and predicted value
def table_view(request):
    import json
    
    test=pointer(y_test.iloc[:,0].tolist())

    df = pd.DataFrame({'ActualValue':test,"LR":list(LRpred),"RFR":list(RFRpred),"DTR":list(DTRpred)})
    
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'table': data}
    return render(request,"table.html",context)

#graph page
def graph_view(request):

    n=len(y_test)
    x_axis=list(range(1,n+1))
    import json
    #xaxis data
    x_axis=json.dumps(x_axis)
    #test data
    
    test=json.dumps(y_test['sem5'].tolist())
    
    context={'x_axis':x_axis,'test_output':test,'lr_output':list(LRpred),"rfr_output":list(RFRpred)
    
    ,"dtr_output":list(DTRpred)}
    return render(request,"graph.html",context)