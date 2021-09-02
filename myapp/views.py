#Import Libraries
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
import math
from statistics import mean
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Cardio Based Active analysis
def active(request):
    dff=pd.read_csv('C:\\Users\\SAISH\\OneDrive\\Desktop\\MiniProject\\final_cardio.csv')
    
    act_filter = dff.active==1
    no_act_filter = dff.active==0

    act_filter2 = dff.cardio==1
    act_filter3 = dff.cardio==0

    active1=dff.where(act_filter & act_filter2,axis=0).dropna()
    active0=dff.where(act_filter & act_filter3,axis=0).dropna()

    active_yes=active1.groupby(["age"],as_index=False)["cardio"].count()
    active_no= active0.groupby(["age"],as_index=False)["cardio"].count()

    no_active1=dff.where(no_act_filter & act_filter2,axis=0).dropna()
    no_active0=dff.where(no_act_filter & act_filter3,axis=0).dropna()

    no_active_yes=no_active1.groupby(["age"],as_index=False)["cardio"].count()
    no_active_no= no_active0.groupby(["age"],as_index=False)["cardio"].count()

    #variables for plotting
    active_y_age=active_yes.age.tolist()
    active_y_cardio=active_yes.cardio.tolist()

    active_n_age=active_no.age.tolist()
    active_n_cardio=active_no.cardio.tolist()

    n_active_y_age=no_active_yes.age.tolist()
    n_active_y_cardio=no_active_yes.cardio.tolist()

    n_active_n_age=no_active_no.age.tolist()
    n_active_n_cardio=no_active_no.cardio.tolist()

    context={'active_y_age':active_y_age,'active_y_cardio':active_y_cardio,'active_n_age':active_n_age,'active_n_cardio':active_n_cardio,'n_active_y_age':n_active_y_age,'n_active_y_cardio':n_active_y_cardio,'n_active_n_age':n_active_n_age,'n_active_n_cardio':n_active_n_cardio}
    return render(request,'active.html',context)

#Cardio Infection based on cholestrol
def chlo(request):
    
    dff=dff=pd.read_csv('C:\\Users\\SAISH\\OneDrive\\Desktop\\MiniProject\\final_cardio.csv')

    cholesterol = dff.where(dff.active==1,axis=0).dropna()
 

    chl1=cholesterol.where(cholesterol.cholesterol==1,axis=0).dropna()
    chl2=cholesterol.where(cholesterol.cholesterol==2,axis=0).dropna()
    chl3=cholesterol.where(cholesterol.cholesterol==3,axis=0).dropna()
    


    #Active
    chl1_y=chl1.where(chl1.cardio==1,axis=0).dropna()
    chl1_n=chl1.where(chl1.cardio==0,axis=0).dropna()

    chl2_y=chl2.where(chl2.cardio==1,axis=0).dropna()
    chl2_n=chl2.where(chl2.cardio==0,axis=0).dropna()

    chl3_y=chl3.where(chl3.cardio==1,axis=0).dropna()
    chl3_n=chl3.where(chl3.cardio==0,axis=0).dropna()

   


    #cholesterol level 1
    chl1_y=chl1_y.groupby(["age"],as_index=False)["cardio"].count()
    chl1_n=chl1_n.groupby(["age"],as_index=False)["cardio"].count()

    #cholesterol level2
    chl2_y=chl2_y.groupby(["age"],as_index=False)["cardio"].count()
    chl2_n=chl2_n.groupby(["age"],as_index=False)["cardio"].count()

    #cholesterol level 3
    chl3_y=chl3_y.groupby(["age"],as_index=False)["cardio"].count()
    chl3_n=chl3_n.groupby(["age"],as_index=False)["cardio"].count()

    #Active=1
    #cholesterol plotting variables level 1
    chl1_y_age=chl1_y.age.tolist()
    chl1_y_cardio=chl1_y.cardio.tolist()

    chl1_n_age=chl1_n.age.tolist()
    chl1_n_cardio=chl1_n.cardio.tolist()

    #Active=1
    #cholesterol plotting variables level 2
    chl2_y_age=chl2_y.age.tolist()
    chl2_y_cardio=chl2_y.cardio.tolist()

    chl2_n_age=chl2_n.age.tolist()
    chl2_n_cardio=chl2_n.cardio.tolist()

    #Active=1
    #cholesterol plotting variables level 3
    chl3_y_age=chl3_y.age.tolist()
    chl3_y_cardio=chl3_y.cardio.tolist()

    chl3_n_age=chl3_n.age.tolist()
    chl3_n_cardio=chl3_n.cardio.tolist()

    
    context={'chl1_y_age':chl1_y_age,'chl1_y_cardio':chl1_y_cardio,'chl1_n_age':chl1_n_age,'chl1_n_cardio':chl1_n_cardio,'chl2_y_age':chl2_y_age,'chl2_y_cardio':chl2_y_cardio,'chl2_n_age':chl2_n_age,'chl2_n_cardio':chl2_n_cardio,'chl3_y_age':chl3_y_age,'chl3_y_cardio':chl3_y_cardio,'chl3_n_age':chl3_n_age,'chl3_n_cardio':chl3_n_cardio}

    return render(request,'cholest.html',context)

def chlo2(request):
    dff=dff=pd.read_csv('C:\\Users\\SAISH\\OneDrive\\Desktop\\MiniProject\\final_cardio.csv')

    cholesterol_nA=dff.where(dff.active==0,axis=0).dropna()

    chl1_nA=cholesterol_nA.where(cholesterol_nA.cholesterol==1,axis=0).dropna()
    chl2_nA=cholesterol_nA.where(cholesterol_nA.cholesterol==2,axis=0).dropna()
    chl3_nA=cholesterol_nA.where(cholesterol_nA.cholesterol==3,axis=0).dropna()

     #not Active
    Nchl1_y=chl1_nA.where(chl1_nA.cardio==1,axis=0).dropna()
    Nchl1_n=chl1_nA.where(chl1_nA.cardio==0,axis=0).dropna()

    Nchl2_y=chl2_nA.where(chl2_nA.cardio==1,axis=0).dropna()
    Nchl2_n=chl2_nA.where(chl2_nA.cardio==0,axis=0).dropna()

    Nchl3_y=chl3_nA.where(chl3_nA.cardio==1,axis=0).dropna()
    Nchl3_n=chl3_nA.where(chl3_nA.cardio==0,axis=0).dropna()

    #cholesterol level 1
    Nchl1_y=Nchl1_y.groupby(["age"],as_index=False)["cardio"].count()
    Nchl1_n=Nchl1_n.groupby(["age"],as_index=False)["cardio"].count()

    #cholesterol level2
    Nchl2_y=Nchl2_y.groupby(["age"],as_index=False)["cardio"].count()
    Nchl2_n=Nchl2_n.groupby(["age"],as_index=False)["cardio"].count()

    #cholesterol level 3
    Nchl3_y=Nchl3_y.groupby(["age"],as_index=False)["cardio"].count()
    Nchl3_n=Nchl3_n.groupby(["age"],as_index=False)["cardio"].count()


    #Active=0
    #cholesterol plotting variables level 1
    Nchl1_y_age=Nchl1_y.age.tolist()
    Nchl1_y_cardio=Nchl1_y.cardio.tolist()

    Nchl1_n_age=Nchl1_n.age.tolist()
    Nchl1_n_cardio=Nchl1_n.cardio.tolist()

    #Active=0
    #cholesterol plotting variables level 2
    Nchl2_y_age=Nchl2_y.age.tolist()
    Nchl2_y_cardio=Nchl2_y.cardio.tolist()

    Nchl2_n_age=Nchl2_n.age.tolist()
    Nchl2_n_cardio=Nchl2_n.cardio.tolist()

    #Active=0
    #cholesterol plotting variables level 3
    Nchl3_y_age=Nchl3_y.age.tolist()
    Nchl3_y_cardio=Nchl3_y.cardio.tolist()

    Nchl3_n_age=Nchl3_n.age.tolist()
    Nchl3_n_cardio=Nchl3_n.cardio.tolist()


    context={'Nchl1_y_age':Nchl1_y_age,'Nchl1_y_cardio':Nchl1_y_cardio,'Nchl1_n_age':Nchl1_n_age,'Nchl1_n_cardio':Nchl1_n_cardio,'Nchl2_y_age':Nchl2_y_age,'Nchl2_y_cardio':Nchl2_y_cardio,'Nchl2_n_age':Nchl2_n_age,'Nchl2_n_cardio':Nchl2_n_cardio,'Nchl3_y_age':Nchl3_y_age,'Nchl3_y_cardio':Nchl3_y_cardio,'Nchl3_n_age':Nchl3_n_age,'Nchl3_n_cardio':Nchl3_n_cardio}

    return render(request,'cholest2.html',context)



#For home page
def indexPage(request):
    df=pd.read_csv('C:\\Users\\SAISH\\OneDrive\\Desktop\\MiniProject\\final_cardio.csv')
    
    total_data = df[df.columns[-1]].count() 
    df1=df.loc[df.cardio==1]
    df0=df.loc[df.cardio==0] #patients not having cardio

    #Cardio Patient Count
    count=df1[df1.columns[-1]].count()
    cardio_count=df1.groupby(["age"],as_index=False)["cardio"].count()

    age=cardio_count['age']
    age=age.tolist()

    infected=cardio_count.cardio
    infected=infected.tolist()

    #Blood Pressure

    new_df=df1.groupby(["age"],as_index=False)["ap_hi","ap_lo"].mean()
    new_df0=df0.groupby(["age"],as_index=False)["ap_hi","ap_lo"].mean()

    cardio_highB=new_df['ap_hi']
    cardio_highB=cardio_highB.tolist()

    cardio_lowB=new_df['ap_lo']
    cardio_lowB=cardio_lowB.tolist()

    no_cardio_highB=new_df0['ap_hi']
    no_cardio_highB=no_cardio_highB.tolist()

    no_cardio_lowB=new_df0['ap_lo']
    no_cardio_lowB=no_cardio_lowB.tolist()


    age_bp=new_df['age']
    age_bp=age_bp.tolist()
    
    words=df.columns[:11]
    words=words.tolist()

    words[5]="High BP"
    words[6]="Low BP"
    words[8]="Glucose"
    words[10]="Alcoholic"

    cloud=df.corr().cardio
    cloud=cloud.tolist()[:11]

    male_cardio=len(df1.loc[df1.gender==1])
    female_cardio=len(df1.loc[df1.gender==2])

    #number of male not having cardio
    no_male_cardio=len(df0.loc[df0.gender==1])
    no_female_cardio=len(df0.loc[df0.gender==2])

    maleFemale=[male_cardio,no_male_cardio,female_cardio,no_female_cardio]
    mF_label=['male_cardio','no_male_cardio','female_cardio','no_female_cardio']

    male=len(df.loc[df.gender==1])
    female=len(df.loc[df.gender==2])

    total_MF=[male,female]
    print(total_MF)
    total_MF_label=['male','female']
    print(total_MF_label)

    df1 = pd.read_csv("C:\\Users\\SAISH\\OneDrive\\Desktop\\sample1.csv")
    
    
    x=df1[["age","cholesterol","gluc","weight","ap_hi","ap_lo","smoke"]]
    y=df1.cardio
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(criterion = "entropy" , max_depth = 5 ,n_estimators = 40) #Choose and fit model
    model.fit(x_train , y_train)
    score = model.score(x_test,y_test)    #Accuracy
    
    score = score * 100
    score = math.trunc(score)
    context={'count':count,'age':age,'infected':infected ,'age_bp':age_bp,'cardio_highB':cardio_highB,'cardio_lowB':cardio_lowB,'no_cardio_highB':no_cardio_highB,'no_cardio_lowB':no_cardio_lowB ,'words':words,'cloud':cloud,'maleFemale':maleFemale,'mF_label':mF_label,'total_MF':total_MF,'total_MF_label':total_MF_label ,'total_data':total_data,'score':score}
    return render(request,'index.html',context)


def welcome(request):
    return render(request,'homeprice.html')

#For prediction
def predictPrice(request):
    #Passing values to html code
    if(request.method=='POST'):
        age=request.POST.get('age')
    
    if(request.method=='POST'):
        cholesterol=request.POST.get('cholesterol')
    if(request.method=='POST'):
        gluc=request.POST.get('gluc')
    if(request.method=='POST'):
        weight=request.POST.get('weight')

    if(request.method=='POST'):
        ap_hi=request.POST.get('ap_hi')
    if(request.method=='POST'):
        ap_lo=request.POST.get('ap_lo')
    if(request.method=='POST'):
        smoke=request.POST.get('smoke')

    df = pd.read_csv("C:\\Users\\SAISH\\OneDrive\\Desktop\\sample1.csv")
    x=df[["age","cholesterol","gluc","weight","ap_hi","ap_lo","smoke"]]
    y=df.cardio
    from sklearn.model_selection import train_test_split          #Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(criterion = "entropy" , max_depth = 5 ,n_estimators = 40)
    model.fit(x_train , y_train)
    
    pre_age = np.asarray(age, dtype='int')
    pre_cholestrol = np.asarray(cholesterol, dtype='int')
    pre_gluc = np.asarray(gluc, dtype='int')
    pre_weight = np.asarray(weight, dtype='int')
    pre_ap_hi = np.asarray(ap_hi, dtype='int')
    pre_ap_lo = np.asarray(ap_lo, dtype='int')
    pre_smoke = np.asarray(smoke, dtype='int')
    card = model.predict([[pre_age,pre_cholestrol,pre_gluc,pre_weight,pre_ap_hi,pre_ap_lo,pre_smoke]]) #Predict 
    if card == 1:
        print("Patient may be cardio infected")
        ans = "Patient may be cardio infected"
    if card == 0:
        print("Patient may not be cardio infected")
        ans = "Patient may not be cardio infected"
    return render(request,'homeprice.html',{'myname':ans})



