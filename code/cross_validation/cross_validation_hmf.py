"""
Algorithm for running cross validation on a dataset, for the HMF model. 
We run cross-validation on either an R dataset, or a D dataset.

Arguments:
- folds             - the number of cross-validation folds
- main_dataset      - 'R' if R[index_main] is the cross-validation dataset, 'D' if D[index_main] is
- index_main        - the index of the main dataset in R or D we want to do cross-validation on; so R/D[index_main]. 
- R                 - the datasets, (R,M,entity1,entity2) tuples
- C                 - the kernels, (C,M,entity) tuples
- D                 - the features or datasets, (D,M,entity)
- settings          - the settings for HMF. This should be a dictionary of the form:
                        { 'priorF', 'priorG', priorSn', 'priorG', 'orderF', 'orderSn', 'orderSm', 'orderG', 'ARD' }
- hyperparameters   - the hyperparameters values for HMF. This should be a dictionary of the form:
                        { 'alphatau', 'betatau', 'alpha0', 'beta0', 'lambdaSn', 'lambdaSm', 'lambdaF', 'lambdaG' }
- init              - the initialisation options for HMF. This should be a dictionary of the form:
                        { 'F', 'G', 'Sn', 'Sm', 'lambdat', 'tau' }
- file_performance  - the file in which we store the performances

We start the search using run(iterations,burn_in,thinning).
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")
import HMF.code.generate_mask.mask as mask
from HMF.code.models.hmf_Gibbs import HMF_Gibbs

import numpy

MEASURES = ['R^2','MSE','Rp']
ATTEMPTS_GENERATE_M = 100

class CrossValidation:
    def __init__(self,folds,main_dataset,index_main,R,C,D,K,settings,hyperparameters,init,file_performance):
        self.folds = folds
        self.R = R
        self.C = C
        self.D = D
        self.K = K
        
        self.settings = settings
        self.init = init
        self.hyperparameters = hyperparameters
        
        self.fout = open(file_performance,'w') 
        self.performances = {} # Performances across folds
        
        # Extract the main dataset from R or D
        self.main_dataset = main_dataset
        self.index_main = index_main
        assert self.main_dataset in ['R','D'], "main_dataset has to be 'R' or 'D', not %s!" % self.main_dataset
        if self.main_dataset == 'R':
            (self.main_R,self.main_M,_,_,_) = self.R[self.index_main]
            self.I,self.J = self.main_R.shape
        else:
            (self.main_D,self.main_M,_,_) = self.D[self.index_main]
            self.I,self.J = self.main_D.shape
        
        
    def run(self,iterations,burn_in,thinning):
        ''' Run the cross-validation. '''
        folds_test = mask.compute_folds_attempts(I=self.I,J=self.J,no_folds=self.folds,attempts=ATTEMPTS_GENERATE_M,M=self.main_M)
        folds_training = mask.compute_Ms(folds_test)

        fold_performances = {measure:[] for measure in MEASURES}
        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s." % (i+1)
            
            ''' Modify R or D to only have the training data. '''
            if self.main_dataset == 'R':
                D_search = [(numpy.copy(D),numpy.copy(M),E,alpha) for D,M,E,alpha in self.D]
                R_search = [(numpy.copy(R),numpy.copy(M),E1,E2,alpha) for R,M,E1,E2,alpha in self.R]
                (R,M,E1,E2,alpha) = R_search[self.index_main]
                R_search[self.index_main] = (R,train,E1,E2,alpha)
            else:
                R_search = [(numpy.copy(R),numpy.copy(M),E1,E2,alpha) for R,M,E1,E2,alpha in self.R]
                D_search = [(numpy.copy(D),numpy.copy(M),E,alpha) for D,M,E,alpha in self.D]
                (D,M,E,alpha) = D_search[self.index_main]
                D_search[self.index_main] = (D,train,E,alpha)
                
            ''' Train the model and measure the performance '''
            HMF = HMF_Gibbs(R=R_search,C=self.C,D=D_search,K=self.K,settings=self.settings,hyperparameters=self.hyperparameters)
            HMF.initialise(init=self.init)
            HMF.run(iterations=iterations)
            if self.main_dataset == 'R':
                performance = HMF.predict_Rn(n=self.index_main,M_pred=test,burn_in=burn_in,thinning=thinning)
            else:
                performance = HMF.predict_Dl(l=self.index_main,M_pred=test,burn_in=burn_in,thinning=thinning)
            
            ''' Store the performance for this fold '''
            message = "Performance fold %s: %s.\n" % (i+1,performance)
            print message
            self.fout.write(message)
            self.fout.flush()
            
            for measure in MEASURES:
                fold_performances[measure].append(performance[measure])
        
        ''' Store the final performances and average '''
        average_performance = self.compute_average_performance(fold_performances)
        message = "Average performance: %s." % average_performance
        print message
        self.fout.write(message)        
        self.fout.flush()
        
        
    def compute_average_performance(self,performances):
        ''' Compute the average performance of the given dictionary of performances (MSE, R^2, Rp) '''
        return { measure:(sum(values)/float(len(values))) for measure,values in performances.iteritems() }