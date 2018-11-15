import pandas as pd
import sys
import numpy as np


def read_data(data_file):
    if 'csv' == data_file.split(".")[-1]:
        train= pd.read_csv(data_file)
        target='bug'
        IDcol='name'
        x_columns = [x for x in train.columns if x not in [target,IDcol]]
        metrics_num=len(x_columns) 
        print "The Number of metrics: "+str(metrics_num)
        all_data = np.loadtxt(data_file, dtype=float, delimiter=',', skiprows=1,usecols=range(1, metrics_num+1))
        all_label = np.loadtxt(data_file, dtype=float, delimiter=',', skiprows=1,usecols=metrics_num+1)

        return all_data,all_label
        
    else:
        print("Unsupported File: " + data_file)
        sys.exit()
    
def defect_data_read(data_file):
    
    class_loc_dict={}
    class_defect_dict={}
    class_name_list=[]
    
    defect_file=open(data_file,'r')
    lines=defect_file.readlines()
    for index,each_line in enumerate(lines):
        if(index!=0):
            records=each_line.strip('\n').split(',')
            class_name=records[2]
            class_loc_dict[class_name]=float(records[13])
            defect_count=int(each_line[each_line.rindex(',')+1:].strip('\n'))
            class_defect_dict[class_name]=defect_count
            class_name_list.append(class_name)
            
    return class_name_list,class_loc_dict,class_defect_dict