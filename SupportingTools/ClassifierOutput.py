# -*- coding:utf-8 -*-
'''
@author: Yu Qu
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def classifier_output(data_train,label_train,data_test,label_test,grid_sear=False):
#     rf = LogisticRegression()
    rf = RandomForestClassifier(random_state=10)
    
    if(grid_sear==False):
        rf.fit(data_train, label_train)
        predprob=rf.predict(data_test)
        predprob_auc=rf.predict_proba(data_test)[:, 1]
        
        recall=metrics.recall_score(label_test,predprob)
        auc=metrics.roc_auc_score(label_test,predprob_auc)
        precision=metrics.precision_score(label_test,predprob)
        fmeasure=metrics.f1_score(label_test,predprob)
        
        return predprob_auc,predprob,precision,recall,fmeasure,auc
    if(grid_sear==True):
#         parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20), 'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10),'max_features':range(3,11,2)}
        parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
#     rf = LogisticRegression()
        gsearch = GridSearchCV(rf, parameters, scoring='roc_auc', cv=5)
        gsearch.fit(data_train, label_train)
        predprob=gsearch.predict(data_test)
        predprob_auc=gsearch.predict_proba(data_test)[:, 1]
        
        recall=metrics.recall_score(label_test,predprob)
        auc=metrics.roc_auc_score(label_test,predprob_auc)
        precision=metrics.precision_score(label_test,predprob)
        fmeasure=metrics.f1_score(label_test,predprob)

        
        return predprob_auc,predprob,precision,recall,fmeasure,auc