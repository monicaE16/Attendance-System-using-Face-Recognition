from commonfunctions import *

IMAGE_DIR = 'data'
import json
DEFAULT_SIZE = [200, 200] 


# def read_images(image_path=IMAGE_DIR, default_size=DEFAULT_SIZE,data="test_set"):
#     images = []
#     images_names = []
#     image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
#     for image_name in image_names:
#         image = Image.open (os.path.join(image_path, image_name))
#         image = image.convert ("L")
#         # resize to given size (if given )
#         if (default_size is not None ):
#             image = image.resize (default_size , Image.ANTIALIAS )
#         images.append(np.asarray (image , dtype =np. uint8 ))
#         #image_name_ = image_name.partition('.')[0]
#         if data=="test_set":
#             image_name_=image_name.split(".")[0]
#         else:
#             image_name_ = image_name[:3]
#         images_names.append(image_name_)
#     #images_names = list(dict.fromkeys(images_names))
#     images = np.array(images)
#     return [images,images_names]



def read_images(image_path=IMAGE_DIR, default_size=DEFAULT_SIZE):
    images = []
    images_names = []
    image_dirs = [image for image in os.listdir(image_path) if not image.startswith('.')]
    for image_dir in image_dirs:
        print(image_dir)
        dir_path = os.path.join(image_path, image_dir)
        image_names = [image for image in os.listdir(dir_path) if not image.startswith('.')]
        for image_name in image_names:
            image = Image.open (os.path.join(dir_path, image_name))
            image = image.convert ("L")
            # resize to given size (if given )
            if (default_size is not None ):
                image = image.resize (default_size , Image.ANTIALIAS )
            images.append(np.asarray (image , dtype =np. uint8 ))
            images_names.append(image_dir)
    return [images,images_names]




def as_row_matrix (X):
    print('inside Reshaping...')
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0].size ), dtype =X [0]. dtype )
    for row in X:
        mat = np.vstack(( mat , np.asarray( row ).reshape(1 , -1))) # 1 x r*c 
    return mat

def get_number_of_components_to_preserve_variance(eigenvalues, variance=.95):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(eigenvalues) / np.sum(eigenvalues)):
        if eigen_value_cumsum > variance:
            return ii
        
def pca (X, y, num_components =0):
    # n : samples , d : dimension of each sample as a row
    print("starting PCA.....")
    [n,d] = X.shape    
    if ( num_components <= 0) or ( num_components >n):
        num_components = n
    mu = X.mean( axis =0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
    else :
        C = np.dot (X,X.T) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors )
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm( eigenvectors [:,i])
            
    # sort eigenvectors descending by their eigenvalue
    
    print('Outside PCA')
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    num_components = get_number_of_components_to_preserve_variance(eigenvalues)
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu] 

def dist_metric(p,q):
    p = np.asarray(p).flatten()
    q = np.asarray (q).flatten()
    return np.sqrt (np.sum (np. power ((p-q) ,2)))

def predict (W, mu , projections, y, X):
    minDist = float("inf")
    minClass = -1
    Q = np.dot (X.reshape (1 , -1) - mu , W)
    print("P",len(projections))
    for i in range (len(projections)):
        dist = dist_metric( projections[i], Q)
        if dist < minDist:
            minDist = dist
            minClass = i
    return minClass

def subplot ( title , images , rows , cols , sptitle ="", sptitles =[] , colormap = plt.cm.gray, filename = None, figsize = (10, 10) ):
    fig = plt.figure(figsize = figsize)
    # main title
    fig.text (.5 , .95 , title , horizontalalignment ="center")
    for i in range ( len ( images )):
        ax0 = fig.add_subplot( rows , cols ,( i +1))
        plt.setp ( ax0.get_xticklabels() , visible = False )
        plt.setp ( ax0.get_yticklabels() , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title("%s #%s" % ( sptitle , str ( sptitles [i ]) )  )
        else:
            plt.title("%s #%d" % ( sptitle , (i +1) )  )
        plt.imshow(np.asarray(images[i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else:
        fig.savefig( filename )
def train_pca():
    [X, y] = read_images()
    X_rows = as_row_matrix(X)
    [eigenvalues, eigenvectors, mu] = pca(X_rows, y, 15)
    print(y)
    # getting features from the given input dataset
    #[X, y] = read_images(image_path='DataBase')
    #X_rows = as_row_matrix(X)
    #projections = np.dot(X_rows - mu, eigenvectors)
    #print(projections.shape)
    # define a video capture object
    #print({'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'mu': mu})
    jsonobject = json.dumps({'eigenvalues': eigenvalues.tolist(), 'eigenvectors': eigenvectors.tolist(), 'mu': mu.tolist()})
    print(jsonobject)
    #jsonFile = open("pca.json", "w")
    with open("pca.json", "w") as file:
        file.write(jsonobject)
        file.close()