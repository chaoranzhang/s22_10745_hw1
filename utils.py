import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from scipy.stats import multivariate_normal
import numpy as np

def create_dataset():
    # Create dataset
    np.random.seed(0)
    X=[]
    mu1=[2,2]
    Sigma1=np.array([[3,-2],[-2,3]])
    Gauss1= multivariate_normal(mean=mu1,cov=Sigma1)
    samples1=Gauss1.rvs(20000)

    mu2=[-8,-4]
    Sigma2=np.array([[4,2],[2,4]])
    Gauss2= multivariate_normal(mean=mu2,cov=Sigma2)
    samples2=Gauss2.rvs(40000)

    mu3=[-8,7]
    Sigma3=np.array([[1,0],[0,1]])
    Gauss3= multivariate_normal(mean=mu3,cov=Sigma3)
    samples3=Gauss3.rvs(60000)

    X=np.vstack((samples1,samples2,samples3))
    #np.random.shuffle(X)
    print(X[1:10,:])
    return X

    # Create dataset
    #np.random.seed(0)
    #Xs=[]
    #mu1=[2,2]
    #Sigma1=np.array([[3,-2],[-2,3]])
    #Gauss1= multivariate_normal(mean=mu1,cov=Sigma1)
    #samples1=Gauss1.rvs(200)
    #
    #mu2=[-8,-4]
    #Sigma2=np.array([[4,2],[2,4]])
    #Gauss2= multivariate_normal(mean=mu2,cov=Sigma2)
    #samples2=Gauss2.rvs(400)
    #
    #mu3=[-8,5]
    #Sigma3=np.array([[1,0],[0,1]])
    #Gauss3= multivariate_normal(mean=mu3,cov=Sigma3)
    #samples3=Gauss3.rvs(600)
    #
    #Xs=np.vstack((samples1,samples2,samples3))
    #np.random.shuffle(Xs)
    #print(Xs[1:10,:])

def init_params():
    """ Set the initial mu, covariance and pi values"""

    K=3
    # K=3, d=2
    # This is a Kxd matrix since we assume K Gaussians where each has d dimensions

    #np.random.seed(0)
    #mu = np.random.randint(min(X[:,0]),max(X[:,0]),size=(number_of_sources,len(X[0]))) 

    mu=np.array([[1.0,0.0],[-5.0,2.0],[-2.0,-2.0]])
    print(mu)

    
    # We need a Kxdxd covariance matrix for each Gauss distribution since we have d features 
    #--> We create symmetric covariance matrices with ones on the digonal

    d=2
    cov = np.zeros((K,d,d))
    print(cov.shape)

    for dim in range(len(cov)):
        np.fill_diagonal(cov[dim],2.0)

    print(cov)

    # Set pi to uniform distribution
    pi = np.ones(K)/K 
    print(pi)

    return mu, cov, pi

def plot_results(X, **kwargs):
    """Plot the dataset"""    
    
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(X[:,0],X[:,1])
    ax0.set_title('Results')
    
    try:
        mu = kwargs['mu']
        for m in mu:
            ax0.scatter(m[0],m[1],c='red',zorder=10,s=100)
    except:
        pass
    
    try:
        cov = kwargs ['cov']
        x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
        XY = np.array([x.flatten(),y.flatten()]).T
        reg_cov = 1e-6*np.identity(len(X[0,:]))
        
        for c,m in zip(cov,mu):
            c += reg_cov
            # let us set up the mean and covariance of a multi-dim gaussian
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    
    except:
        pass
    
    plt.show()
    fig.savefig('results.png')
