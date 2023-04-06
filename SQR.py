import numpy as np
from scipy.sparse import spdiags
from scipy import sparse

def SQR(Y,X,C=1,T=10,Q=0.5):

    N,P=np.shape(X)
    b=C*np.log(P)/P
    C1=P+0.25
    power=1/(2-0.5)
    beta=np.zeros((P,1))
    Z=np.ones(P)

    for t in range(0,T):

        L1_error=np.maximum(np.abs(Y-X@beta),0.1)
        T=Y-(1-2*Q)*L1_error

        W=0.25*spdiags(L1_error.squeeze()**-1,0,N,N)

        XTW=sparse.csr_matrix.dot(W,X).T

        XTWX=(XTW*X.T).sum(-1)

        iter=1

        ink1=X[:,1:]@beta[1:,:]
        ink2=np.zeros((N,1))
        beta_previous=np.ones((P,1))
        
        while (np.linalg.norm(beta-beta_previous)>1e-4 and iter<20):

            beta_previous=np.copy(beta)
            iter=iter+1
            
            for j in range(0,P):

                C2=1/b+np.sum(np.abs(beta)**0.5)-np.abs(beta[j])**0.5

                if (j!=0) & (j!=P-1):
                    Z[j]=XTW[j:j+1,:]@(T-ink1-ink2)/XTWX[j]
                elif j==0:
                    Z[0]=XTW[0:1,:]@(T-ink1)/XTWX[j]
                else:
                    Z[P-1]=XTW[P-1:P,:]@(T-ink2)/XTWX[j]

                if np.abs(Z[j])<=2*(C1/(2*C2+2*np.abs(Z[j])**0.5)/XTWX[j])**power:

                    beta[j]=0

                else:

                    beta_old=np.abs(Z[j])
                    beta_new=10
                    k=1

                    while(np.abs(beta_old-beta_new)>1e-4 and k<20 and beta_new>=0):
                        
                        beta_old=np.copy(beta_new)
                        beta_new=np.abs(Z[j])-C1/XTWX[j]/(beta_old+C2*beta_old**0.5)
                        k=k+1
                        
                    if k>=20 or beta_new<0:
                        beta[j]=0 
                    else:
                        beta[j]=beta_new*np.sign(Z[j])

                if (j!=0) & (j!=P-1):
                    ink1=ink1-X[:,j+1:j+2]@beta[j+1:j+2,:]
                    ink2=ink2+X[:,j:j+1]@beta[j:j+1,:]
                elif j==0:
                    ink1=ink1-X[:,1:2]@beta[1:2,:]
                    ink2=X[:,0:1]@beta[0:1,:]
                else:
                    ink1=ink2+X[:,j:j+1]@beta[j:j+1,:]-X[:,0:1]@beta[0:1,:]
                    ink2=np.zeros((N,1))

    return beta