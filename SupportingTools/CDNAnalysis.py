# -*- coding:utf-8 -*-
'''
@author: Yu Qu
'''
import networkx as nx

def static_analysis(subject):
    G = nx.DiGraph()
    findFile = open(subject+'/classgraph.dot','r')
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