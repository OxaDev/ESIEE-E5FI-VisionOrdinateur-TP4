import cv2
import pymld
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import glob

def getDescritors(chemins):
    s = cv2.xfeatures2d.SURF_create()
    descriptors = np.array([])
    
    ## Charge and SURF images
    for path in chemins:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in onlyfiles:
            img_in = cv2.imread( join(str(path),str(filename)) )

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

    np.savetxt(join('part1_saves','center_'+str(N)+'.txt'), values, delimiter=',')
    
    end = time.time()
    
    print(" --\tN : " + str(N) + "\t-- inertia :\t" + str(inertia) + "\t -- time :\t" + str(end - start) + "s" + "\t -- Normalized centers :\t" + str(np.linalg.norm(values)))
    return inertia, np.linalg.norm(values)

list_dir = [
    join("101_ObjectCategories","ant"),
    join("101_ObjectCategories","camera"),
    join("101_ObjectCategories","cougar_face"),
    join("101_ObjectCategories","garfield")
]

desc = getDescritors(list_dir)
variance_list = np.array([])
error_max_list = np.array([])
N_list = np.array([2,4,8,16,32,64,128,256,512,1024,2048])

for N in N_list:
    inertia, linalg = vocabulaire(N, desc)
    variance_list = np.append(variance_list, inertia)
    error_max_list = np.append(error_max_list, linalg)

plt.plot(N_list, variance_list )
plt.show()

plt.plot( N_list, error_max_list)
plt.show()