# -*- coding:utf-8 -*-
'''
@author: Yu Qu
'''
import networkx as nx

def class_defect_read():
    global Total_defect_num
    global defect_file_num
    Total_defect_num=0
    class_defect_dict={}
    class_defect_file=open(subject+'/'+bug_file_name,'r')
    lines=class_defect_file.readlines()
    for index,each_line in enumerate(lines):
        if(index!=0):
            records=each_line.strip('\n').split(',')
            class_name=records[2]
            defect_count=int(each_line[each_line.rindex(',')+1:].strip('\n'))
            
            Total_defect_num=Total_defect_num+1
            class_defect_dict[class_name]=defect_count
            if(defect_count>0):
                defect_file_num=defect_file_num+1
                class_defect_bool[class_name]=1
            else:
                class_defect_bool[class_name]=0
            
            class_original_metrics[class_name]=each_line[each_line.index(class_name)+len(class_name)+1:each_line.rindex(',')]
            class_loc_dict[class_name]=records[13]


    return class_defect_dict

def static_analysis():
    G = nx.DiGraph()
    findFile = open(subject+'/'+'classgraph.dot','r')
    each_lines= findFile.readlines()
    for each_line in each_lines:
        if each_line.__contains__('>'):
            edge=each_line.split('>');
            edge[0]=edge[0][edge[0].index('\"')+1:edge[0].rindex('\"')]
            edge[1]=edge[1][edge[1].index('\"')+1:edge[1].rindex('\"')]
            if(G.has_edge(edge[0],edge[1])==False):
                G.add_edge(edge[0],edge[1])
        else:
            if each_line.count('\"')==2:
                node=each_line[each_line.index('\"')+1:each_line.rindex('\"')]
                if(G.has_node(node)==False):
                    G.add_node(node)       
    findFile.close()
    return G


conf_file=open('Subject.conf')
lines=conf_file.readlines()
for each_line in lines:
    records=each_line.strip('\n').split(",")
    subject=records[0]
    print subject
    bug_file_name=records[1]
    
    class_original_metrics={}
    class_loc_dict={}
    class_defect_bool={}
    global Total_defect_num
    global defect_file_num
    defect_file_num=0
    class_defect_dict=class_defect_read()
    
    G = static_analysis()
    wcc=nx.weakly_connected_component_subgraphs(G)
    largest_wcc=wcc[0]
    print('Number of Nodes in LWCC:'+str(len(largest_wcc)))
    
    G.remove_edges_from(G.selfloop_edges())
    G=G.to_undirected()
    print('Number of Nodes:'+str(G.number_of_nodes()))
    print('Number of Edges:'+str(G.size()))
    result_file=open(subject+'/'+'RESULT_output.csv','w')
    record_file=open(subject+'/RECORDS_CONSOLE.csv','w')
    record_file.write('Number of Nodes:'+str(G.number_of_nodes())+'\n')
    record_file.write('Number of Edges:'+str(G.size())+'\n')
    
    class_metric_dict={}

    
    j=1
    while(j<100):
        
        defect_num=0
        count=0
        total_loc=0
        defect_bool_num=0
        
        G1=nx.k_core(G,j)
        
        node_list=G1.nodes()
        if(len(node_list)==0):
            break
        for node in node_list:
            G.node[node]['Coreness']=j
            G.node[node]['Layer']=j
            count=count+1
            if(class_defect_dict.has_key(node)):
                defect_num=defect_num+class_defect_dict[node]
                defect_bool_num=defect_bool_num+class_defect_bool[node]
                class_metric_dict[node]=j
                total_loc=total_loc+int(class_loc_dict[node])
                                
        print defect_num
        print count
        if(count==0):
            break
        result_file.write(str(j)+','+str(float(defect_num)/count)+','+str(float(defect_bool_num)/count)+','+str(float(defect_num)/total_loc)+','+str(count)+','+str(total_loc)+','+str(float(total_loc)/count)+'\n')
        j=j+1
    
    nx.write_gexf(G,subject+'/gephi.gexf')
    
    for n in G.nodes_iter():
        if(G.degree(n)==0):
            print n
            
    count_not_in_graph=0        
    all_node_list=G.nodes()
    for each_class in class_defect_dict:
        if not each_class in all_node_list:
            count_not_in_graph=count_not_in_graph+1
            print('*****This class is in defect file but not in the class graph: '+each_class+','+str(class_defect_dict[each_class])+','+str(class_loc_dict[each_class]))
            record_file.write('*****This class is in defect file but not in the class graph: '+each_class+','+str(class_defect_dict[each_class])+','+str(class_loc_dict[each_class])+'\n')
    
    print('The number of class not in class graph: '+str(count_not_in_graph))
    record_file.write('The number of class not in class graph: '+str(count_not_in_graph)+'\n')
    print('The total number of records in defect file is: '+str(Total_defect_num))
    record_file.write('The total number of records in defect file is: '+str(Total_defect_num)+'\n')