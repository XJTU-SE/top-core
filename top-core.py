# -*- coding:utf-8 -*-
'''
@author: Yu Qu
'''

from SupportingTools.ReadFile import *
from SupportingTools.CDNAnalysis import *
from sklearn.model_selection import RepeatedKFold, train_test_split
from SupportingTools.ClassifierOutput import *
import random
import numpy
import copy
from SMOTUNED.smotuned import SMOTUNED
from SMOTUNED.smote import Smote
from SMOTUNED.wrapper import *
from scipy import stats

####################################################
def get_median(data):
    data.sort()
    half=len(data)//2
    return (data[half] + data[~half]) / 2
####################################################
####################################################
def output_results(kind,twenty_per):
    result_file=open(subject+'/RESULT_'+kind+'.csv','w')
    kind_file=open(subject+"/"+kind+'.csv','r')
    lines=kind_file.readlines()
    count=1
    total_loc=0
    total_bug=0
    result_file.write('0.0,0.0\n')
    for index,each_line in enumerate(lines):
        if(total_loc<twenty_per):
            records=each_line.strip('\n').split(',')
            bug=int(records[1])
            loc=float(records[2])
            total_loc_temp=total_loc+float(loc)
            if(total_loc_temp>twenty_per):    
                total_bug=(float(twenty_per-total_loc)/loc)*bug+total_bug
                total_loc=twenty_per
            else:
                total_bug=total_bug+int(bug)
                total_loc=total_loc_temp
            result_file.write(str(float(total_loc)/All_Loc_num)+','+str(float(total_bug)/All_Bug_num)+'\n')
            count=count+1
        else:
            break
        
    result_file.close()
####################################################
def Output_POPT(ratio):
    result_topcore=open(subject+"/POPT-topcore-"+str(ratio)+".csv","a")

    
    optimal_matrix = numpy.loadtxt(open(subject+"/RESULT_optimal.csv","rb"),delimiter=",",skiprows=0)
    worst_matrix = numpy.loadtxt(open(subject+"/RESULT_worst.csv","rb"),delimiter=",",skiprows=0)
    degree_matrix = numpy.loadtxt(open(subject+"/RESULT_degree.csv","rb"),delimiter=",",skiprows=0)

    betweenness_matrix = numpy.loadtxt(open(subject+"/RESULT_betweenness.csv","rb"),delimiter=",",skiprows=0)
    pagerank_matrix = numpy.loadtxt(open(subject+"/RESULT_pagerank.csv","rb"),delimiter=",",skiprows=0)
    effort_matrix = numpy.loadtxt(open(subject+"/RESULT_effort.csv","rb"),delimiter=",",skiprows=0)
    effortcore_matrix = numpy.loadtxt(open(subject+"/RESULT_effortcore.csv","rb"),delimiter=",",skiprows=0)
    
    optimal=numpy.trapz(optimal_matrix[:,1],x=optimal_matrix[:,0])
    worst=numpy.trapz(worst_matrix[:,1],x=worst_matrix[:,0])
    degree=numpy.trapz(degree_matrix[:,1],x=degree_matrix[:,0])

    betweenness=numpy.trapz(betweenness_matrix[:,1],x=betweenness_matrix[:,0])
    pagerank=numpy.trapz(pagerank_matrix[:,1],x=pagerank_matrix[:,0])
    effort=numpy.trapz(effort_matrix[:,1],x=effort_matrix[:,0])
    effortcore=numpy.trapz(effortcore_matrix[:,1],x=effortcore_matrix[:,0])
    
    P_opt_degree=(degree-worst)/(optimal-worst)
    P_opt_betweenness=(betweenness-worst)/(optimal-worst)
    P_opt_pagerank=(pagerank-worst)/(optimal-worst)
    P_opt_effort=(effort-worst)/(optimal-worst)
    P_opt_effortcore=(effortcore-worst)/(optimal-worst)
    
    P_opt_betweenness_list.append(P_opt_betweenness)
    P_opt_pagerank_list.append(P_opt_pagerank)
    P_opt_degree_list.append(P_opt_degree)
    P_opt_effort_list.append(P_opt_effort)
    P_opt_effortcore_list.append(P_opt_effortcore)
    

    result_topcore.write(str(P_opt_betweenness)+","+str(P_opt_pagerank)+","+str(P_opt_degree)+","+str(P_opt_effort)+","+str(P_opt_effortcore)+"\n")
    result_topcore.flush()
    result_topcore.close()
####################################################    
def label_sum(label_train):
    label_sum=0
    for each in label_train:
        label_sum=label_sum+each
    return label_sum
####################################################
def average_value(list):
    return float(sum(list))/len(list);
####################################################
conf_file=open('Subject.conf')
lines=conf_file.readlines()
for each_line in lines:
    records=each_line.strip('\n').split(",")
    
    subject=records[0]
    print subject
    bug_file_name=records[1]
    bug_file_process=open(subject+'/BugProcessed.csv','w')
    bug_file_process.write('name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc,bug\n')
    
    positive_count=0
    total_count=0
    
    bug_file=open(subject+'/'+bug_file_name,'r')
    lines=bug_file.readlines()
    for index,each_line in enumerate(lines):
        if(index!=0):
            records=each_line.strip('\n').split(',')
            class_name=records[2]
            original_metrics=each_line[each_line.index(class_name)+len(class_name)+1:each_line.rindex(',')]
            total_count=total_count+1
            defect_count=int(each_line[each_line.rindex(',')+1:].strip('\n'))
            if(defect_count>0):
                bug_file_process.write(class_name+","+original_metrics+",1\n")
                positive_count=positive_count+1
            else:
                bug_file_process.write(class_name+","+original_metrics+",0\n")
            bug_file_process.flush()
    bug_file_process.close()
    
    all_data,all_label=read_data(subject+'/BugProcessed.csv')
    
    class_name_list,class_loc_dict,class_defect_dict=defect_data_read(subject+'/'+bug_file_name)
    
    
    G=static_analysis(subject)
    G=G.to_undirected()
    G.remove_edges_from(G.selfloop_edges())
    class_core_dict={}
    j=1
    while(j<100):
        defect_num=0
        count=0
        total_loc=0
        G1=nx.k_core(G,j)
        node_list=G1.nodes()
        if(len(node_list)==0):
            break
        for node in node_list:
            class_core_dict[node]=j
        j=j+1
    
    class_degree_dict={}
    node_list=G.nodes()
    for node in node_list:
        class_degree_dict[node]=G.degree(node)
        
    Betweenness_cen=nx.betweenness_centrality(G)
    Page_rank=nx.pagerank(G)
    
    ratios=[5.0, 3.3, 2.5]
    for ratio in ratios:
        
        P_opt_file=open(subject+"/P-Pot-Average-Value-"+str(ratio)+".csv","w")
        Statistical_file=open(subject+"/STATISTICAL-"+str(ratio)+".csv","w")
        P_opt_betweenness_list=[]
        P_opt_pagerank_list=[]
        P_opt_degree_list=[]
        P_opt_effort_list=[]
        P_opt_effortcore_list=[]
        
        kf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=0)
        for train_index, test_index in kf.split(all_data):
            try:
                data_train=all_data[train_index]
                data_test=all_data[test_index]
                label_train=all_label[train_index]
                label_test=all_label[test_index]
                
                test_class_name=[]
                for each_index in test_index:
                    test_class_name.append(class_name_list[each_index])
                
                if(label_sum(label_train)>(len(label_train)/2)):
                    print "The training data does not need balance."
                    predprob_auc,predprob,precision,recall,fmeasure,auc=classifier_output(data_train,label_train,data_test,label_test,grid_sear=True)  
                else:     
                    train_data_smote, test_data_smote, train_label_smote, test_label_smote=train_test_split(data_train, label_train, test_size=0.5, random_state=10)
                    smotuned=SMOTUNED()
                    k, N, r = smotuned.DE(train_data_smote, train_label_smote, test_data_smote, test_label_smote)
                    opt_para=[k,N,r]
                    data_bin_, label_bin_=smote_wrapper(opt_para, data_train, label_train)
                    predprob_auc,predprob,precision,recall,fmeasure,auc=classifier_output(data_bin_,label_bin_,data_test,label_test,grid_sear=True)#False is only for debugging.
            
                All_Loc_num=0
                All_Bug_num=0    
                class_test_defect_dense={}
                class_test_defect={}
                class_test_loc={}
                for class_name in test_class_name:
                    class_test_defect_dense[class_name]=float(class_defect_dict[class_name])/(int(class_loc_dict[class_name])+0.01)#in case some class's LOC==0.
                    class_test_defect[class_name]=class_defect_dict[class_name]
                    class_test_loc[class_name]=int(class_loc_dict[class_name])
                    All_Loc_num=All_Loc_num+int(class_loc_dict[class_name])
                    All_Bug_num=All_Bug_num+int(class_defect_dict[class_name])
                    
                defect_order=sorted(class_test_defect_dense.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                order_file=open(subject+'/optimal.csv','w')
                for each_turple in defect_order:
                    each_class=each_turple[0]
                    order_file.write(each_class+','+str(class_test_defect[each_class])+','+str(class_loc_dict[each_class])+'\n')
                order_file.close()
                
                reverse_order=sorted(class_test_defect_dense.items(), lambda x, y: cmp(x[1], y[1]), reverse=False)
                reverse_file=open(subject+'/worst.csv','w')
                for each_turple in reverse_order:
                    each_class=each_turple[0]
                    reverse_file.write(each_class+','+str(class_test_defect[each_class])+','+str(class_loc_dict[each_class])+'\n')
                reverse_file.close()
                
                
                class_in_prediction_effortaware={}
                for i in range(len(predprob_auc)):
                    class_name=test_class_name[i]
                    if(float(class_loc_dict[class_name])==0.0):
                        class_in_prediction_effortaware[class_name]=1
                    else:
                        class_in_prediction_effortaware[class_name]=float(predprob_auc[i])/class_loc_dict[class_name]
                effort_order=sorted(class_in_prediction_effortaware.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                effort_file=open(subject+'/effort.csv','w')
                for each_turple in effort_order:
                    class_name=each_turple[0]
                    effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(class_loc_dict[class_name])+','+str(each_turple[1])+'\n')
                effort_file.close()
                ##################################################
                class_in_prediction_effortaware_coreness={}
                for i in range(len(predprob_auc)):
                    class_name=test_class_name[i]
                    if(not class_core_dict.has_key(class_name)):
                        class_in_prediction_effortaware_coreness[class_name]=class_in_prediction_effortaware[class_name]
                    else:
                        class_in_prediction_effortaware_coreness[class_name]=class_in_prediction_effortaware[class_name]*class_core_dict[class_name]
                effort_coreness_order=sorted(class_in_prediction_effortaware_coreness.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                effort_file=open(subject+'/effortcore.csv','w')
                for each_turple in effort_coreness_order:
                    class_name=each_turple[0]
                    effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(class_loc_dict[class_name])+','+str(each_turple[1])+'\n')
                effort_file.close()
                ##################################################
                ##################################################
                class_in_prediction_effortaware_degree={}
                for i in range(len(predprob_auc)):
                    class_name=test_class_name[i]
                    if(not class_degree_dict.has_key(class_name)):
                        class_in_prediction_effortaware_degree[class_name]=class_in_prediction_effortaware[class_name]
                    else:
                        class_in_prediction_effortaware_degree[class_name]=class_in_prediction_effortaware[class_name]*class_degree_dict[class_name]
                effort_degree_order=sorted(class_in_prediction_effortaware_degree.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                effort_file=open(subject+'/degree.csv','w')
                for each_turple in effort_degree_order:
                    class_name=each_turple[0]
                    effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(class_loc_dict[class_name])+','+str(each_turple[1])+'\n')
                effort_file.close()
                ##################################################
                ##################################################
                class_in_prediction_effortaware_Betweenness={}
                for i in range(len(predprob_auc)):
                    class_name=test_class_name[i]
                    if(not Betweenness_cen.has_key(class_name)):
                        class_in_prediction_effortaware_Betweenness[class_name]=class_in_prediction_effortaware[class_name]
                    else:
                        class_in_prediction_effortaware_Betweenness[class_name]=class_in_prediction_effortaware[class_name]*Betweenness_cen[class_name]
                effort_betweenness_order=sorted(class_in_prediction_effortaware_Betweenness.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                effort_file=open(subject+'/betweenness.csv','w')
                for each_turple in effort_betweenness_order:
                    class_name=each_turple[0]
                    effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(class_loc_dict[class_name])+','+str(each_turple[1])+'\n')
                effort_file.close()
                ##################################################
                ##################################################
                class_in_prediction_effortaware_pagerank={}
                for i in range(len(predprob_auc)):
                    class_name=test_class_name[i]
                    if(not Page_rank.has_key(class_name)):
                        class_in_prediction_effortaware_pagerank[class_name]=class_in_prediction_effortaware[class_name]
                    else:
                        class_in_prediction_effortaware_pagerank[class_name]=class_in_prediction_effortaware[class_name]*Page_rank[class_name]
                effort_pagerank_order=sorted(class_in_prediction_effortaware_pagerank.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
                effort_file=open(subject+'/pagerank.csv','w')
                for each_turple in effort_pagerank_order:
                    class_name=each_turple[0]
                    effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(class_loc_dict[class_name])+','+str(each_turple[1])+'\n')
                effort_file.close()
        ####################################################        
                print All_Loc_num
                
                twenty_per=All_Loc_num/ratio
                total_loc=0
                total_bug=0
                output_results('degree',twenty_per)            
                output_results('optimal',twenty_per)
                output_results('worst',twenty_per)
                output_results('betweenness',twenty_per)
                output_results('pagerank',twenty_per)
                output_results('effort',twenty_per)
                output_results('effortcore',twenty_per)
                
                Output_POPT(ratio)
            except Exception, e:
                print str(e)
        
        P_opt_file.write(str(average_value(P_opt_betweenness_list))+','+str(average_value(P_opt_pagerank_list))+','+str(average_value(P_opt_degree_list))+','+str(average_value(P_opt_effort_list))+','+str(average_value(P_opt_effortcore_list))+'\n')
        Statistical_file.write("betweenness&effort: "+str(stats.wilcoxon(P_opt_betweenness_list,P_opt_effort_list))+'\n')
        Statistical_file.write("pagerank&effort: "+str(stats.wilcoxon(P_opt_pagerank_list,P_opt_effort_list))+'\n')
        Statistical_file.write("degree&effort: "+str(stats.wilcoxon(P_opt_degree_list,P_opt_effort_list))+'\n')
        Statistical_file.write("effortcore&effort: "+str(stats.wilcoxon(P_opt_effortcore_list,P_opt_effort_list))+'\n')
        Statistical_file.write("effortcore&degree: "+str(stats.wilcoxon(P_opt_effortcore_list,P_opt_degree_list))+'\n')                                                                                                                                                                                                                                                             