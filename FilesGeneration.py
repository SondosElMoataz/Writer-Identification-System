import cv2
import numpy as np
import math
import os  
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shutil import copy


parent="C:/Users/dell/Desktop/sem7/pattern/project"     # current directory 
directory ="DataSet"
path = os.path.join(parent, directory)  
os.makedirs(path) 
print (os.path.basename(path)) # current basename of path
print (os.path.basename(os.path.dirname(path))) # 'parent Directory 
print(path)

file_variable = open('forms.txt')
all_lines_variable = file_variable.readlines()
count=0
created=0
numberOfWriters=0
all_lines_variable=all_lines_variable[16:]
for i in range(len(all_lines_variable)):
    line=all_lines_variable[i]
    line=line.split()
    id=line[1]
    setNumber=line[0][0]
    if id == '000':
     dir='0'
    else:
     dir=id.lstrip('0')    
    writerFolder = os.path.join(path,str(dir))
    if not os.path.exists(writerFolder):
     if( setNumber <= "d"):
      os.makedirs(writerFolder)
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formsA-D/"+line[0]+".png", writerFolder)
     if( setNumber > "d" and setNumber <= "h"):
      os.makedirs(writerFolder)
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formsE-H/"+line[0]+".png", writerFolder)
     if( setNumber > "h"):
      os.makedirs(writerFolder)
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formsI-Z/"+line[0]+".png", writerFolder)
    else:
     if( setNumber <= "d"):
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formsA-D/"+line[0]+".png", writerFolder)
     if( setNumber > "d" and setNumber <= "h"):
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formsE-H/"+line[0]+".png", writerFolder)
     if( setNumber > "h"):
      copy("C:/Users/dell/Desktop/sem7/pattern/project/formI-Z/"+line[0]+".png", writerFolder)
     