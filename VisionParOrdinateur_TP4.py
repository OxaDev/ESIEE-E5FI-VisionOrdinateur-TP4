import cv2
import pymld
from os import listdir, system, name
from os.path import isfile, join
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

####### Functions #######

def getDescritors(chemins):
    s = cv2.xfeatures2d.SURF_create()
    descriptors = np.array([])
    
    ## Charge and SURF images
    for path in chemins:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for filename in onlyfiles:
            img_in = cv2.imread( join(str(path),str(filename)) )
            img_in = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
            (kps, dsc) = s.detectAndCompute(img_in,None)   
            descriptors = np.append(descriptors, dsc)      

    desc = np.reshape(descriptors, (len(descriptors)//64, 64))
    desc = np.float32(desc)
    return desc


def vocabulaire(N, desc):

    start = time.time()

    k_means = KMeans(n_clusters=N, init="k-means++", n_init=N)
    labels_predict = k_means.fit_predict(desc)
    values = k_means.cluster_centers_
    labels = k_means.labels_
    inertia = k_means.inertia_

    errorMax = 0
    for i in range(len(labels_predict)):
        predicted_class = labels_predict[i]
        current_center = values[predicted_class]
        current_point = desc[i]
        
        temp_max = np.linalg.norm(current_center - current_point)
        if(temp_max > errorMax):
            errorMax = temp_max
    
    np.savetxt(join('part1_saves','center_'+str(N)+'.txt'), values, delimiter=',')
    
    end = time.time()
    
    print(" --\tN : " + str(N) + "\t-- inertia :\t" + str(inertia) + "\t -- time :\t" + str(end - start) + "s" + "\t -- Normalized centers :\t" + str(errorMax))
    return inertia, errorMax

def getVocab(nb_clusters):
    vocab = np.asarray([])
    with open( join('part1_saves','center_'+str(nb_clusters)+'.txt') ) as f:
        lines = f.readlines()
        for elem in lines:
            elem = elem.rstrip()
            tab_values = elem.split(',')
            tab_append = np.asarray([])
            for elem2 in tab_values:
                elem3 = float(elem2)
                tab_append = np.append(tab_append,elem3)
            vocab = np.append(vocab, tab_append)
    return vocab

def vectoriser(pim, kmeans):
    s = cv2.xfeatures2d.SURF_create()
    img_in = cv2.imread( pim )
    img_in = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    (kps, dsc) = s.detectAndCompute(img_in,None) 
    descriptors = np.array([])  
    descriptors = np.append(descriptors, dsc)      

    desc = np.reshape(descriptors, (len(descriptors)//64, 64))
    desc = np.float32(desc)

    labels_predict = kmeans.predict(desc)

    return labels_predict

####### Variables #######
list_dir = [
    join("101_ObjectCategories","ant"),
    join("101_ObjectCategories","camera"),
    join("101_ObjectCategories","cougar_face"),
    join("101_ObjectCategories","garfield")
]
mode = "vectorisation"

######### Execution #########
if(mode== "saveKmeans"):
    N_list = np.array([2,4,8,16,32,64,128,256,512,1024])
    desc = getDescritors(list_dir)

    for N in N_list:    
        print(" -- KMeans for N : " + str(N) + " -- ", end="")
        k_means = KMeans(n_clusters=N, init="k-means++", n_init=N)
        k_means.fit(desc)

        filename = join('part2_saves','kmeans_N_'+str(N)+'.pickle')
        pickle.dump(k_means, open(filename, 'wb'))
        print("saved")


if(mode == "vectorisation"):
    im_vectors = np.asarray([])
    im_filename = []
    Nb_cluster = 32

    filename = join('part2_saves','kmeans_N_'+str(Nb_cluster)+'.pickle')
    k_means = pickle.load(open(filename, 'rb'))

    for path in list_dir:
        print("\nFolder : "+path)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        indic = 1
        for filename in onlyfiles:
            im = join(str(path),str(filename))
            vector_im = vectoriser(im, k_means)

            im_vectors = np.append(im_vectors, vector_im[0])
            im_filename.append(im)
            indic +=1
    
    with open(join('part2_saves','base_im_vectors_N_'+str(Nb_cluster)+'.pickle'), 'wb') as f:
        pickle.dump(im_vectors, f)
    with open(join('part2_saves','base_im_filenames_N_'+str(Nb_cluster)+'.pickle'), 'wb') as f:
        pickle.dump(im_filename, f)

if(mode == "vocabulaire"):
    desc = getDescritors(list_dir)
    variance_list = np.array([])
    error_max_list = np.array([])
    N_list = np.array([2,4,8,16,32,64,128,256,512,1024])

    for N in N_list:
        inertia, linalg = vocabulaire(N, desc)
        variance_list = np.append(variance_list, inertia)
        error_max_list = np.append(error_max_list, linalg)

    plt.plot(N_list, variance_list )
    plt.show()

    plt.plot( N_list, error_max_list)
    plt.show()

