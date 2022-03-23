import cv2
from dml import kda
from os import listdir, system, name
from os.path import isfile, join
from sklearn import svm
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

    #k_means = KMeans(n_clusters=N)
    k_means = get_model(N)
    labels_predict = k_means.predict(desc)
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

def get_model(n_cluster):
    filename = join('save_models','kmeans_N_'+str(n_cluster)+'.pickle')
    k_means = pickle.load(open(filename, 'rb'))
    return k_means

def svc_application(classes):
    filename_filenames = join('part2_saves','base_im_filenames_N_'+str(Nb_cluster)+'.pickle')
    filenames = pickle.load(open(filename_filenames, 'rb'))
    filenames_filtered = []
    filename_vectors = join('part2_saves','base_im_vectors_N_'+str(Nb_cluster)+'.pickle')
    vectors = pickle.load(open(filename_vectors, 'rb'))

    mat_X = []
    vect_Y = []
    print(" -- Lecture des données -- ")
    for i in range(len(vectors)):
        current_vector = vectors[i]
        current_filename = filenames[i]

        for i_class in range(len(classes)):
            name_class = classes[i_class]
            if(name_class in current_filename):
                filenames_filtered.append(current_filename)
                vect_Y.append(i_class+1)
                vect_temp = []
                for i_cluster in range(Nb_cluster):
                    vect_temp.append(0)
                
                for elem in current_vector:
                    vect_temp[elem] += 1

                mat_X.append(vect_temp)
                break
    print(" -- Fin de la lecture -- ")

    print(" -- Création du model SVC et Fit des données sur les 2 répertoires -- ")
    svc_model = svm.SVC()
    svc_model.fit(mat_X,vect_Y)

    print(" -- Predict des données avec le SVC -- ")
    predic_svc=  svc_model.predict(mat_X)
    print(" -- Predict du SVC : " + str(predic_svc) + " -- ")

    nb_error = 0
    nb_success = 0
    for i in range(len(predic_svc)):
        current_predict = predic_svc[i]
        current_filename = filenames_filtered[i]
        current_predicted_name = classes[current_predict-1]
        #print("-- For filename : " + str(current_filename)+ " -- predicted : " + str(current_predicted_name))
        if(current_predicted_name in current_filename):
            nb_success +=1
        else:
            nb_error += 1
    print(" -- Nombre de succès de prédictions : " + str(nb_success) + " -- Nombre d'erreur : " + str(nb_error))     

####### Variables #######
list_dir = [
    join("101_ObjectCategories","ant"),
    join("101_ObjectCategories","camera"),
    join("101_ObjectCategories","cougar_face"),
    join("101_ObjectCategories","garfield")
]
mode = "saveKmeans"
mode = "vocabulaire"
mode = "vectorisation"
mode = "Appr_Et_Test_KDA"
mode = "Appr_Et_Test_SVC_2"
#mode = "Appr_Et_Test_SVC_4"

Nb_cluster = 512

######### Execution #########
if(mode== "saveKmeans"):
    N_list = np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096])
    desc = getDescritors(list_dir)

    for N in N_list:    
        print(" -- KMeans for N : " + str(N) + " -- ", end="")
        start = time.time()

        k_means = KMeans(n_clusters=N)
        k_means.fit(desc)

        filename = join('save_models','kmeans_N_'+str(N)+'.pickle')
        pickle.dump(k_means, open(filename, 'wb'))

        end = time.time()
        print("saved -- time : " + str(end - start) + "s")

if(mode == "vocabulaire"):
    desc = getDescritors(list_dir)
    variance_list = np.array([])
    error_max_list = np.array([])
    N_list = np.array([2,4,8,16,32,64,128,256,512,1024,2048,4096])

    for N in N_list:
        inertia, linalg = vocabulaire(N, desc)
        variance_list = np.append(variance_list, inertia)
        error_max_list = np.append(error_max_list, linalg)

    plt.ion()
    plt.plot(N_list, variance_list)
    plt.title("Liste des variances total en fonction de N")
    plt.xlabel("N")
    plt.ylabel("Variance")
    plt.ioff()
    plt.savefig(join("part1_saves","list_variance_totale.png") )

    plt.cla()  

    plt.plot(N_list, error_max_list)
    plt.title("Liste des erreurs max en fonction de N")
    plt.ylabel("Erreur maximum")
    plt.xlabel("N")
    plt.ioff()
    plt.savefig(join("part1_saves","list_error_max.png") )

if(mode == "vectorisation"):
    im_vectors = []
    im_filename = []

    k_means = get_model(Nb_cluster)

    for path in list_dir:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        indic = 1
        for filename in onlyfiles:
            im = join(str(path),str(filename))
            vector_im = vectoriser(im, k_means)

            im_vectors.append(vector_im)
            im_filename.append(im)
            indic +=1

    with open(join('part2_saves','base_im_vectors_N_'+str(Nb_cluster)+'.pickle'), 'wb') as f:
        pickle.dump(im_vectors, f)
    with open(join('part2_saves','base_im_filenames_N_'+str(Nb_cluster)+'.pickle'), 'wb') as f:
        pickle.dump(im_filename, f)
    
    print("-- Vectorisation file saved --")

if(mode == "Appr_Et_Test_KDA"):
    filename_filenames = join('part2_saves','base_im_filenames_N_'+str(Nb_cluster)+'.pickle')
    filenames = pickle.load(open(filename_filenames, 'rb'))

    filename_vectors = join('part2_saves','base_im_vectors_N_'+str(Nb_cluster)+'.pickle')
    vectors = pickle.load(open(filename_vectors, 'rb'))

    mat_X = []
    vect_Y = []
    classes = ['cougar_face','garfield']
    print(" -- Lecture des données -- ")
    for i in range(len(vectors)):
        current_vector = vectors[i]
        current_filename = filenames[i]

        for i_class in range(len(classes)):
            name_class = classes[i_class]
            if(name_class in current_filename):
                vect_Y.append(i_class+1)
                vect_temp = []
                for i_cluster in range(Nb_cluster):
                    vect_temp.append(0)
                
                for elem in current_vector:
                    vect_temp[elem] += 1

                mat_X.append(vect_temp)
                break
    print(" -- Fin de la lecture -- ")

    ## Test KDA ##

    X=np.array([[1,2],[1,3],[2,1],[7,8],[9,8]])
    y=[1,1,1,-1,-1]
    polynomes_deg = [1,2,3,4,5,6,7,8,9,10]
    values_transform=[]

    for poly in polynomes_deg:
        s= kda.KDA(n_components=2, kernel='poly', degree=poly)
        s.fit(X,y)
        temp_value = s.transform(X)
        values_transform.append(temp_value)

        #print(str("\nDegré polynome : " + str(poly) + " -- Valeurs transform : " + str(temp_value)))

    ##############
    """
    Plus le degré du polynome est élevé plus les valeurs seront éloignées entre les deux classes, 
    """

    print(" -- KDA sur la matrice X et le vecteur Y -- ")
    values_transform=[]
    for poly in polynomes_deg:
        s= kda.KDA(n_components=2, kernel='poly', degree=poly)
        s.fit(mat_X,vect_Y)
        temp_value = s.transform(mat_X)
        values_transform.append(temp_value)
        print(str("\nDegré polynome : " + str(poly) + " -- Valeurs transform : " + str(temp_value)))
    print(" -- Fin du KDA -- ")
    '''
    Les projections semblent montrer une bonne séparation des classes lorsque le polynome est élevé ( > à 7 ) 
    car on remarquer que certaines valeurs sont positives et d'autres négatives avec un nombre élevé.
    '''
    
if(mode == "Appr_Et_Test_SVC_2"):
    classes = ['cougar_face','garfield']
    svc_application(classes)
    '''
    Résultats plutôt corrects, 1 seule erreur reportée avec la prédiction du modèle SVC 
    '''

if(mode == "Appr_Et_Test_SVC_4"):

    classes = ['cougar_face','garfield','ant','camera']
    svc_application(classes)
    '''
    Résultats plutôt corrects, 2 erreurs reportées avec la prédiction du modèle SVC 
    '''
        
