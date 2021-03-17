import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

lookupTable = []
n_class = 0
seed = 1337
features = 0

def main(filename):
    global lookupTable
    
    X, y, lookupTable = load_dataset(filename)
    
    offset = len(X) // len(np.unique(y))
    X = X[:n_class*offset]
    y = y[:n_class*offset]
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

    gscore = gaussian(X_train, X_test, y_train, y_test)
    kscore = kNN(X_train, X_test, y_train, y_test)
    """   
    
    gscore = gaussian_cv(X, y, 10)
    kscore = kNN_cv(X, y, 10)
    
    return 1 - gscore, 1 - kscore
        
        
def plot_conf_matrix(y_test, y_pred, method):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Matrice de confusion ' + method)
    plt.colorbar()
    tick_marks = [i for i in range(n_class)]
    plt.xticks(tick_marks, lookupTable[:n_class])
    plt.yticks(tick_marks, lookupTable[:n_class])  
    plt.tight_layout()
    plt.ylabel('Vrai')
    plt.xlabel('Prediction')
    #plt.savefig('images/confmat' + str(features) + '_' + str(n_class) + '_' + method + '.png')
    

def gaussian(X_train, X_test, y_train, y_test):
    start = time.time()
    
    priors = np.array([(y_train == i).sum() / len(y_train) for i in range(len(np.unique(y_train)))])
    g = GaussianBayes(priors=priors, diagonal=True)
    g.fit(X_train, y_train)

    # Score
    y_pred = g.predict(X_test)
    
    end = time.time()
    
    if n_class == 10: plot_conf_matrix(y_test, y_pred, 'gaussian') 
    score = np.sum(y_test == y_pred) / len(X_test)
    
    print_score(score, start, end, n_class, 'gaussien')
    return score


def gaussian_cv(X, y, cv):
    start = time.time()
    
    priors = np.array([(y == i).sum() / len(y) for i in range(len(np.unique(y)))])
    g = GaussianBayes(priors=priors, diagonal=True)
    #score = cross_val_score(g, X, y, cv=cv)
    strat_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    pred = cross_val_predict(g, X, y, cv=strat_k_fold)
    #score = np.mean(score, axis=0)
    score = np.sum(y == pred) / len(X)
             
    end = time.time()
    
    if n_class == 10: plot_conf_matrix(y, pred, 'gaussien')
    print_score(score, start, end, n_class, 'gaussien')
    return score
    
    
def kNN(X_train, X_test, y_train, y_test):
    start = time.time()
    
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(X_train, y_train)
    
    # Score
    y_pred = knn.predict(X_test)   
    end = time.time()
    
    if n_class == 10: plot_conf_matrix(y_test, y_pred, 'kNN')
    score = np.sum(y_test == y_pred) / len(X_test)
    
    print_score(np.mean(score), start, end, n_class, 'kNN') 
    return score


def kNN_cv(X, y, cv):
    start = time.time()
    
    knn = KNeighborsClassifier(n_neighbors = 10)
    #score = cross_val_score(knn, X, y, cv=cv)
    strat_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    pred = cross_val_predict(knn, X, y, cv=strat_k_fold)
    #score = np.mean(score, axis=0)
    score = np.sum(y == pred) / len(X)
    
    end = time.time()
    
    if n_class == 10: plot_conf_matrix(y, pred, 'kNN')
    print_score(score, start, end, n_class, 'kNN')
    return score


def print_score(score, start, end, n_class, method):
    print("precision {:s}: {:.2f}, temps d'execution: {:d}ms, {:d} classes".format(method, score, round((end - start) * 1000), n_class))


if __name__ == "__main__":
    features = 2
    filename = 'data/data' + str(features) + '.csv'
    n_class = 10
    # main(filename)
     
    error = [[],[]]
    vals = [i for i in range(2, 11)]
    
    for i in vals:
        n_class = i
        gerror, kerror = main(filename)
        error[0].append(gerror)
        error[1].append(kerror)
    
    plt.figure()
    plt.title('Evolution de l\'erreur')
    plt.plot(vals, error[0], '--b', label = 'gaussian')
    plt.plot(vals, error[1], '--r', label = 'kNN')
    plt.legend()
    plt.xlabel('classes')
    plt.ylabel('erreur')
    #plt.savefig('images/error' + str(features) + '.png')
        
    
