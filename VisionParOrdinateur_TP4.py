import cv2
import pymld
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def getDescritors(chemins):
    s = cv2.xfeatures2d.SURF_create()
    descriptors = np.array([])
    
    ## Charge and SURF images
    for path in chemins:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in onlyfiles:
            img_in = cv2.imread( str(path)+"/"+str(filename) )

            (kps, dsc) = s.detectAndCompute(img_in,None)   
            descriptors = np.append(descriptors, dsc)      

    desc = np.reshape(descriptors, (len(descriptors)//64, 64))
    desc = np.float32(desc)
    return desc

def vocabulaire(N, desc):

    start = time.time()

    k_means = KMeans(n_clusters=N, n_init=4)
    k_means.fit(desc)
    values = k_means.cluster_centers_
    labels = k_means.labels_
    inertia  = k_means.inertia_
    
    end = time.time()
    
    print(" --\tN : " + str(N) + "\t-- inertia :\t" + str(inertia) + "\t -- time :\t" + str(end - start) + "s")
    return inertia

    # r = lambda: random.randint(0,255)
    # # Now separate the data, Note the flatten()
    # for i in range(N):
    #     tab_classe = desc[labels.ravel()==i]
    #     color = str('#%02X%02X%02X' % (r(),r(),r()) )
    #     plt.scatter(tab_classe[:,0],tab_classe[:,1], c=color, s= 1)

    # plt.scatter(values[:,0],values[:,1],s = 4,c = 'y')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()

    ## Get Centroids to save to file

    # for i in labels:
    #     plt.scatter(np.array(xx)[labels == i] , np.array(yy)[labels == i] , label = i, s = 1)
    # plt.scatter(labels[:,0] , labels[:,1] , s = 40, color = 'k')
    # plt.legend()
    # plt.show()


list_dir = [
    "./101_ObjectCategories/ant",
    "./101_ObjectCategories/camera",
    "./101_ObjectCategories/cougar_face",
    "./101_ObjectCategories/garfield"
]

desc = getDescritors(list_dir)
N_list = []
for N in [2,4,8,16,32,64,128,256,512,1024,2048]:
    N_list.append( vocabulaire(N, desc) )