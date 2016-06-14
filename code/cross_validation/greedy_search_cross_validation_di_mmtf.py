"""
Algorithm for running cross validation with greedy search on a dataset, for the
DI-MMTF model. We try to improve the values for K one at a time until none of
them improve the metric (AIC, BIC, log likelihood).

Arguments:
- index_main        - the index of the main dataset in R we want to do cross-validation on; so R[index_main], 
- R                 - the datasets, (R,M,entity1,entity2) tuples
- C                 - the kernels, (C,M,entity) tuples
- ranges_K          - a dictionary from the entity names to a list of values for the Kt's
- folds             - number of folds
- prior             - the prior values for DI-MMTF. This should be a dictionary of the form:
                        { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaSn':lambdaSn, 'lambdaSm':lambdaSm }
- initF             - the initialisation of F - 'kmeans', 'exp' or 'random'
- initS             - the initialisation of S - 'exp' or 'random'
- iterations        - number of iterations we run each model
- restarts          - the number of times we try each model when doing model selection
- quality_metric    - the metric we use to measure model quality - MSE, AIC, or BIC
- file_performance  - the file in which we store the performances

We start the search using run(burn_in=<>,thinning=<>).
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#
from DI_MMTF.code.model_selection.greedy_search_2 import GreedySearch2
import DI_MMTF.code.generate_mask.mask as mask
from DI_MMTF.code.models.di_mmtf_gibbs import di_mmtf_gibbs

import numpy

metrics = ['MSE','AIC','BIC'] 
measures = ['R^2','MSE','Rp']
attempts_generate_M = 100

class GreedySearchCrossValidation:
    def __init__(self,index_main,R,C,ranges_K,folds,priors,init_S,init_F,iterations,restarts,quality_metric,file_performance):
        self.R = R
        self.C = C
        self.ranges_K = ranges_K
        self.folds = folds
        self.priors = priors
        self.init_F = init_F
        self.init_S = init_S
        self.iterations = iterations
        self.restarts = restarts
        self.quality_metric = quality_metric
        
        self.fout = open(file_performance,'w')
        assert self.quality_metric in metrics    
        self.performances = {}                  # Performances across folds
        
        # Extract the main dataset from R
        self.index_main = index_main
        (self.main_R,self.main_M,_,_) = self.R[self.index_main]
        self.I,self.J = self.main_R.shape
        
        
    # Run the cross-validation
    def run(self,burn_in,thinning):
        # Generate the masks for the M matrix of the main dataset
        folds_test = mask.compute_folds_attempts(I=self.I,J=self.J,no_folds=self.folds,attempts=attempts_generate_M,M=self.main_M)
        folds_training = mask.compute_Ms(folds_test)

        performances_test = {measure:[] for measure in measures}
        for i,(train,test) in enumerate(zip(folds_training,folds_test)):
            print "Fold %s." % (i+1)
            
            R_search = [(numpy.copy(R),numpy.copy(M),E1,E2) for R,M,E1,E2 in self.R]
            (R,M,E1,E2) = R_search[self.index_main]
            R_search[self.index_main] = (R,train,E1,E2)
            
            # Run the greedy grid search
            greedy_search = GreedySearch2(
                R=R_search,
                C=self.C,
                ranges_K=self.ranges_K,
                priors=self.priors,
                initS=self.init_S,
                initF=self.init_F,
                iterations=self.iterations,
                restarts=self.restarts)
            greedy_search.search(self.quality_metric,burn_in=burn_in,thinning=thinning)
            
            # Store the model fits, and find the best one according to the metric    
            all_performances = greedy_search.all_values(metric=self.quality_metric)
            all_values_K_tried = greedy_search.all_values_K()
            self.fout.write("All model fits for fold %s, metric %s: %s; values K tried: %s.\n" % (i+1,self.quality_metric,all_performances,all_values_K_tried)) 
            self.fout.flush()
            
            best_K = greedy_search.best_value(metric=self.quality_metric)
            self.fout.write("Best values_K for fold %s: %s.\n" % (i+1,best_K))
            self.fout.flush()
            
            # Train a model with this K and measure performance on the test set
            performance = self.run_model(R_train=R_search,M_test=test,values_K=best_K,burn_in=burn_in,thinning=thinning)
            self.fout.write("Performance: %s.\n\n" % performance)
            self.fout.flush()
            
            for measure in measures:
                performances_test[measure].append(performance[measure])
        
        # Store the final performances and average
        average_performance_test = self.compute_average_performance(performances_test)
        message = "Average performance: %s. \nPerformances test: %s." % (average_performance_test,performances_test)
        print message
        self.fout.write(message)        
        self.fout.flush()
        
    # Compute the average performance of the given list of performances (MSE, R^2, Rp)
    def compute_average_performance(self,performances):
        return { measure:(sum(values)/float(len(values))) for measure,values in performances.iteritems() }
        
    # Initialises and runs the model, and returns the performance on the test set
    def run_model(self,R_train,M_test,values_K,burn_in,thinning):
        # We train <restarts> models, and use the one with the best log likelihood to make predictions   
        best_loglikelihood = None
        best_performance = None
        for r in range(0,self.restarts):
            DI_MMTF = di_mmtf_gibbs(
                R=R_train,
                C=self.C,
                K=values_K,
                priors=self.priors
            )
            DI_MMTF.initialise(init_S=self.init_S,init_F=self.init_F)
            DI_MMTF.run(self.iterations)
                
            new_loglikelihood = DI_MMTF.quality(metric='loglikelihood',iterations=self.iterations,burn_in=burn_in,thinning=thinning)
            performance = DI_MMTF.predict_Rn(n=self.index_main,M_pred=M_test,iterations=self.iterations,burn_in=burn_in,thinning=thinning)
                
            if best_loglikelihood is None or new_loglikelihood > best_loglikelihood:
                best_loglikelihood = new_loglikelihood
                best_performance = performance
                
            print "Trained final model, attempt %s. Log likelihood: %s." % (r+1,new_loglikelihood)            
            
        print "Best log likelihood: %s." % best_loglikelihood
        return best_performance