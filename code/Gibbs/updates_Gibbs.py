"""
This file contains the updates for the matrix factorisation models, for Gibbs
sampling.

We make them as general as possible, so they can be used for both single and 
multiple matrix factorisation and tri-factorisation models.

We can provide the parameters for row-wise draws (multivariate Gaussian or 
Truncated Normal), or column-wise draws (individual elements, but each column 
in parallel).

We use the following arguments for the updates for F, S:
    R - a list of main datasets and importances [Rn,Mn,Ftn,Sn,Fun,taun,alphan] for matrix tri-factorisation F(tn) S(n) F(un).T
    C - a list of constraint matrices and importances [Cm,Mm,Ftm,Sm,taum,alpham] for matrix tri-factorisation F(tm) S(m) F(tm).T
    D - a list of feature datasets and importances [Dl,Ml,Ftl,Gl,taul,alphal] for matrix factorisation F(tl) G(l).T
    lambdaF - a list of the hyperparameter or ARD lambda values (length K)
    i, k, l - the row or column number of F or S we are computing the parameters for
    nonnegative - True if we should use the nonnegative updates, False otherwise
    
We always use the rows of the Rn, Cm, Dl - if we want to compute the column 
factors, take the transpose of the matrix and its mask and pass that instead.
    
We use the following arguments for the updates for the lambdas (ARD):
    alpha0, beta0 - hyperparameters
    Fs - a list [Ft,nonneg] of all the factor matrices (F's, G's) that this ARD controls
Note that nonneg is True if we should use the nonnegative updates for Ft, False otherwise.
    
We use the following arguments for the updates for the tau (noise):
    alpha_tau, beta_tau - hyperparameters
    dataset, mask - the Rn, Cm, or Dl matrix
    F, G - factor matrices
    S - if MTF, None if MF
Note that nonneg is True if we should use the nonnegative updates for Ft, False otherwise.
    
Usage for single dataset matrix factorisation:
    U: column_tau_F([],[],[(R,M,U,V,tau,1)]),     column_mu_F(TODO)
    V: column_tau_F([],[],[(R.T,M.T,V,U,tau,1)]), column_mu_F(TODO)
Usage for single dataset matrix tri-factorisation:
    F: column_tau_F([(R,M,F,S,G,tau,1)],[],[]),         column_mu_F(TODO)
    G: column_tau_F([(R.T,M.T,G,S.T,F,tau,1)],[],[]),   column_mu_F(TODO)
"""

import numpy, itertools

###############################################################################
################################### Helpers ###################################
###############################################################################

def triple_dot(M1,M2,M3):
    ''' Do M1*M2*M3. If the matrices have dimensions I,K,L,J, then the complexity
        of M1*(M2*M3) is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
    K,L = M2.shape
    if K < L:
        return numpy.dot(M1,numpy.dot(M2,M3))
    else:
        return numpy.dot(numpy.dot(M1,M2),M3)


###############################################################################
###### Updates for the alpha and beta parameters of the noise parameters ######
###############################################################################

def alpha_tau(alphatau,mask):
    ''' Return the value for alpha for the Gibbs sampler, for the noise tau. '''
    return alphatau + mask.sum() / 2.
          
def beta_tau(betatau,dataset,mask,F,G,S=None):
    ''' Return the value for beta for the Gibbs sampler, for the noise tau. '''
    dataset_pred = numpy.dot(F,G.T) if S is None else triple_dot(F,S,G.T)
    squared_error = (mask*(dataset-dataset_pred)**2).sum()
    return betatau + squared_error / 2.
        
        
###############################################################################
####### Column-wise updates for the tau parameter of the posterior of F #######
###############################################################################

def column_tau_F(R,C,D,lambdaF,k,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for column-wise draws. '''
    tau_F = 0. if nonnegative else lambdaF[k]
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        tau_F += column_tau_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,k)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        tau_F += column_tau_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,k)
        tau_F += column_tau_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,k)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        tau_F += column_tau_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,k)
        
    return tau_F

def column_tau_individual_mf(dataset,mask,F,G,tau,alpha,k):
    ''' Return the component of the tau update for an individual matrix, for matrix factorisation. '''
    return tau * alpha * ( mask * G[:,k]**2 ).sum(axis=1)

def column_tau_individual_mtf(dataset,mask,F,S,G,tau,alpha,k):
    ''' Return the component of the tau update for an individual matrix, for matrix tri-factorisation. '''
    return tau * alpha * ( mask * numpy.dot(S[k,:],G.T)**2 ).sum(axis=1)


###############################################################################
####### Column-wise updates for the mu parameter of the posterior of F ########
###############################################################################

def column_mu_F(R,C,D,lambdaF,tau_Fk,k,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for column-wise draws. '''
    mu_F = -lambdaF[k] if nonnegative else 0.
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        mu_F += column_mu_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,k)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        mu_F += column_mu_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,k)
        mu_F += column_mu_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,k)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        mu_F += column_mu_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,k)
        
    mu_F /= tau_Fk
    return mu_F
    
def column_mu_individual_mf(dataset,mask,F,G,tau,alpha,k):
    ''' Return the component of the mu update for an individual matrix, for matrix factorisation. '''
    return tau * alpha * ( mask * ( ( dataset - numpy.dot(F,G.T) + numpy.outer(F[:,k],G[:,k])) * G[:,k] ) ).sum(axis=1)

def column_mu_individual_mtf(dataset,mask,F,S,G,tau,alpha,k):
    ''' Return the component of the mu update for an individual matrix, for matrix tri-factorisation. '''
    return tau * alpha * ( mask * ( ( dataset - triple_dot(F,S,G.T) + numpy.outer(F[:,k],numpy.dot(S[k,:],G.T)) ) * numpy.dot(S[k,:],G.T) ) ).sum(axis=1)


###############################################################################
##### Row-wise updates for the Precision parameter of the posterior of F ######
############################################################################### 
 
def row_precision_F(R,C,D,lambdaF,i,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for row-wise draws. '''
    precision_F = numpy.zeros((len(lambdaF),len(lambdaF))) if nonnegative else numpy.diag(lambdaF)
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        precision_F += row_precision_F_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,i)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        precision_F += row_precision_F_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,i)
        precision_F += row_precision_F_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,i)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        precision_F += row_precision_F_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,i)
        
    return precision_F

def row_precision_F_individual_mf(dataset,mask,F,G,tau,alpha,i):
    ''' Return the component of the Precision update for an individual matrix, for matrix factorisation. '''
    G_masked = (mask[i] * G.T).T # zero rows when j not in mask[i]
    return tau * alpha * ( numpy.dot(G_masked.T,G_masked) )

def row_precision_F_individual_mtf(dataset,mask,F,S,G,tau,alpha,i):
    ''' Return the component of the Precision update for an individual matrix, for matrix tri-factorisation. '''
    GS_masked = (mask[i] * numpy.dot(G,S.T).T).T # zero rows when j not in mask[i]
    return tau * alpha * ( numpy.dot(GS_masked.T,GS_masked) )

 
###############################################################################
######### Row-wise updates for the mu parameter of the posterior of F #########
############################################################################### 

def row_mu_F(R,C,D,lambdaF,precision_Fi,i,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for row-wise draws. '''
    mu_F = -lambdaF if nonnegative else numpy.zeros(len(lambdaF))
    
    for Rn,Mn,Ftn,Sn,Fun,taun,alphan in R:
        mu_F += row_mu_F_individual_mtf(Rn,Mn,Ftn,Sn,Fun,taun,alphan,i)
    for Cm,Mm,Ftm,Sm,taum,alpham in C:
        mu_F += row_mu_F_individual_mtf(Cm,Mm,Ftm,Sm,Ftm,taum,alpham,i)
        mu_F += row_mu_F_individual_mtf(Cm.T,Mm.T,Ftm,Sm.T,Ftm,taum,alpham,i)
    for Dl,Ml,Ftl,Gl,taul,alphal in D:
        mu_F += row_mu_F_individual_mf(Dl,Ml,Ftl,Gl,taul,alphal,i)
      
    sigma_Fi = numpy.linalg.inv(precision_Fi)
    mu_F = numpy.dot(sigma_Fi,mu_F)
    return mu_F
    
def row_mu_F_individual_mf(dataset,mask,F,G,tau,alpha,i):
    ''' Return the component of the mu update for an individual matrix, for matrix factorisation. '''
    dataset_i_masked = mask[i] * dataset[i]
    return tau * alpha * numpy.dot(dataset_i_masked,G)

def row_mu_F_individual_mtf(dataset,mask,F,S,G,tau,alpha,i):
    ''' Return the component of the mu update for an individual matrix, for matrix tri-factorisation. '''
    dataset_i_masked = mask[i] * dataset[i]
    return tau * alpha * numpy.dot(dataset_i_masked,numpy.dot(G,S.T))
    

###############################################################################
######## Individual updates for the parameters of the posterior of S ##########
###############################################################################

def individual_tau_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for individual draws. '''
    tau_S = 0. if nonnegative else lambdaSkl
    tau_S += tau * alpha * ( mask * numpy.outer(F[:,k]**2,G[:,l]**2) ).sum()
    return tau_S

def individual_mu_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,tau_Skl,nonnegative):
    ''' Return the value for mu for the Gibbs posterior, for individual draws. '''
    mu_S = -lambdaSkl if nonnegative else 0. 
    mu_S += tau * alpha * ( mask * ( ( dataset - triple_dot(F,S,G.T) + S[k,l] * numpy.outer(F[:,k],G[:,l]) ) * numpy.outer(F[:,k],G[:,l]) ) ).sum() 
    mu_S /= tau_Skl      
    return mu_S
        

###############################################################################
######### Row-wise updates for the parameters of the posterior of S ###########
############################################################################### 
 
def row_precision_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,k,nonnegative):
    ''' Return the value for Precision for the Gibbs posterior, for row draws. '''
    I,J = mask.shape
    precision_S = numpy.zeros((len(lambdaSk),len(lambdaSk))) if nonnegative else numpy.diag(lambdaSk)
    
    # Inefficient
    """
    indices_mask = [(i,j) for (i,j) in itertools.product(range(0,I),range(0,J)) if mask[i,j]]
    precision_S += tau * alpha * numpy.array([F[i,k]**2 * numpy.outer(G[j],G[j]) for (i,j) in indices_mask]).sum(axis=0)
    """
    
    # Efficient - we dot F()k**2 with an I x (L x L) matrix that is sum_j in Omegai [ Gj * Gj.T ] 
    G_outer_masked = numpy.array([numpy.dot((mask[i] * G.T),(mask[i] * G.T).T) for i in range(0,I)])
    precision_S += tau * alpha * numpy.tensordot( F[:,k]**2, G_outer_masked, axes=1 )   
    
    return precision_S

def row_mu_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,precision_Sk,k,nonnegative):
    ''' Return the value for tau for the Gibbs posterior, for row draws. '''
    mu_S = -lambdaSk if nonnegative else numpy.zeros(len(lambdaSk))
    mu_S += tau * alpha * numpy.dot( numpy.dot( 
        F[:,k], ( mask * ( dataset - triple_dot(F,S,G.T) + numpy.outer( F[:,k], numpy.dot(S[k,:],G.T) ) ) ) ),
        G )
    
    sigma_Sk = numpy.linalg.inv(precision_Sk)
    mu_S = numpy.dot(sigma_Sk,mu_S)
    return mu_S
    
        
###############################################################################
########## Updates for the parameters of the posterior of lambda_t ############
############################################################################### 
        
def alpha_lambda(alpha0,Fs):
    ''' Return the value for alpha for the Gibbs sampler, for the ARD. '''
    alpha = alpha0
    for F,nonneg in Fs:
        I,_ = F.shape
        alpha += I if nonneg else I / 2.
    return alpha
          
def beta_lambda(beta0,Fs,k):
    ''' Return the value for beta for the Gibbs sampler, for the kth ARD. '''
    beta = beta0
    for F,nonneg in Fs:
        beta += F[:,k].sum() if nonneg else (F[:,k]**2).sum() / 2.
    return beta
        
        
###############################################################################
################# Return both parameters for the variables ####################
###############################################################################
        
def alpha_beta_tau(alphatau,betatau,dataset,mask,F,G,S=None):
    alpha = alpha_tau(
        alphatau=alphatau,mask=mask)
    beta = beta_tau(
        betatau=betatau,dataset=dataset,mask=mask,F=F,G=G,S=S)    
    return (alpha,beta)
    
def alpha_beta_lambda(alpha0,beta0,Fs,k):
    alpha = alpha_lambda(
        alpha0=alpha0,Fs=Fs)
    beta = beta_lambda(
        beta0=beta0,Fs=Fs,k=k)
    return (alpha,beta)

def column_mu_tau_F(R,C,D,lambdaF,k,nonnegative):
    tau_Fk = column_tau_F(
        R=R,C=C,D=D,lambdaF=lambdaF,k=k,nonnegative=nonnegative)
    mu_Fk = column_mu_F(
        R=R,C=C,D=D,lambdaF=lambdaF,k=k,tau_Fk=tau_Fk,nonnegative=nonnegative)
    return (mu_Fk,tau_Fk)
    
def row_mu_precision_F(R,C,D,lambdaF,i,nonnegative):
    precision_Fi = row_precision_F(
        R=R,C=C,D=D,lambdaF=lambdaF,i=i,nonnegative=nonnegative)
    mu_Fi = row_mu_F(
        R=R,C=C,D=D,lambdaF=lambdaF,precision_Fi=precision_Fi,i=i,nonnegative=nonnegative)
    return (mu_Fi,precision_Fi)

def individual_mu_tau_S(dataset,mask,tau,alpha,F,S,G,lambdaSkl,k,l,nonnegative):
    tau_Skl = individual_tau_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSkl=lambdaSkl,k=k,l=l,nonnegative=nonnegative)
    mu_Skl = individual_mu_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSkl=lambdaSkl,k=k,l=l,tau_Skl=tau_Skl,nonnegative=nonnegative)
    return (mu_Skl,tau_Skl)
    
def row_mu_precision_S(dataset,mask,tau,alpha,F,S,G,lambdaSk,k,nonnegative):
    precision_Sk = row_precision_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSk=lambdaSk,k=k,nonnegative=nonnegative)
    mu_Sk = row_mu_S(
        dataset=dataset,mask=mask,tau=tau,alpha=alpha,F=F,S=S,G=G,lambdaSk=lambdaSk,precision_Sk=precision_Sk,k=k,nonnegative=nonnegative)
    return (mu_Sk,precision_Sk)