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

def vectoriser(pim, pvoca):
    s = cv2.xfeatures2d.SURF_create()
    img_in = cv2.imread( pim )
    (kps, dsc) = s.detectAndCompute(img_in,None)   
    return_vector = np.asarray([])
    for i in range(len(pvoca)):
        return_vector = np.append(return_vector,0)
    
    indice_dsc = 0
    for elem_desc in dsc:
        #print("Descriptor element "+str(indice_dsc+1)+"/"+str(len(dsc)) + " -- " + str(round((indice_dsc+1)*100/len(dsc), 2 ) )+ "%", end="" )
        min_dist = 9999999
        indice_min = 0
        indice_vocab = 0
        for elem_vocab in pvoca:
            dist = np.linalg.norm(elem_desc - elem_vocab)
            if( dist < min_dist):
                min_dist = dist
                indice_min = indice_vocab
            indice_vocab += 1

        return_vector[indice_min] += 1
        indice_dsc+=1
        #print(" -- Minimal distance : " + str(min_dist) + " -- indice in vocab : "+ str(indice_min))
    return return_vector

####### Variables #######
list_dir = [
    join("101_ObjectCategories","ant"),
    join("101_ObjectCategories","camera"),
    join("101_ObjectCategories","cougar_face"),
    join("101_ObjectCategories","garfield")
]
mode = "vectorisation"

######### Execution #########
if(mode == "vectorisation"):
    nb_clusters = 512
    vocab = getVocab(nb_clusters)
    im_vectors = np.asarray([])
    im_filename = []
    for path in list_dir:
        print("\nFolder : "+path)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        indic = 1
        for filename in onlyfiles:
            print("Filename "+filename+" number \t"+str(indic)+"/"+str(len(onlyfiles)) + " -- " + str(round((indic)*100/len(onlyfiles), 2 ) )+ "%" )
            im = join(str(path),str(filename))
            vector_im = vectoriser(im, vocab)

            im_vectors = np.append(im_vectors, vector_im)
            im_filename.append(im)
            indic +=1
    
    with open(join('part2_saves','base_im_vectors_'+str(nb_clusters)+'.pickle'), 'wb') as f:
        pickle.dump(im_vectors, f)
    with open(join('part2_saves','base_im_filenames_'+str(nb_clusters)+'.pickle'), 'wb') as f:
        pickle.dump(im_filename, f)

if(mode == "vocabulaire"):
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

