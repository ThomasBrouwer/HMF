'''
DESCRIPTION

Gibbs sampler for Bayesian Hybrid Matrix Factorisation.
We use Gaussian or Exponential priors for the F's and G's, and ARD for model selection.
Updates are done per row or column.

- Ft_ik ~ N(0,lambdat_k^-1)  or ~ Exp(lambdat_k)

- Sn_kl ~ N(0,lambdaSn^-1) or ~ Exp(lambdaSn)
- Sm_kl ~ N(0,lambdaSm^-1) or ~ Exp(lambdaSm)

- lambdat_k ~ Gamma(alpha0,beta0)
- taun, taum, taul ~ Gamma(alphatau,betatau)

We expect the following arguments:
- R, a list of tuples (R^n,M^n,E1,E2,alpha) where:
    R^n    - dataset matrix
    M^n    - mask matrix for R^n
    E1, E2 - row and column entity type, respectively (e.g. a string or integer)
    alpha  - positive real indicating how important this dataset is
- C, a list of tuples (C^m,M^m,E,alpha) where:
    C^m    - constraint matrix for entity type E
    M^m    - mask matrix for C^m
    E      - entity type
    alpha  - positive real indicating how important this dataset is
- D, a list of tuples (D^l,M^l,E,alpha)
    D^l    - feature dataset for entity type E
    M^l    - mask matrix for D^l
    E      - row entity type
    alpha  - positive real indicating how important this dataset is
- K, a dictionary { entity1:K1, ..., entityT:KT } mapping an entity type to the number of clusters
- settings, a dictionary defining the model settings
    { 'priorF', 'priorG', priorS', 'orderFG', 'orderS', 'ARD' }
    priorFG: defines prior over the Ft and Gl; 'exponential' or 'normal'
    priorSn: defines prior over the Sn; 'exponential' or 'normal'
    priorSm: defines prior over the Sm; 'exponential' or 'normal'
    orderFG: draw new values for F and G per column, or per row: 'columns' or 'rows'
    orderS:  draw new values for S per individual element, or per row: 'individual' or 'rows'
    ARD:     True if we use Automatic Relevance Determination over F and G, else False
- hyperparameters, a dictionary defining the priors over the taus, Fs, Ss,
    { 'alphatau', 'betatau', 'alpha0', 'beta0', 'lambdaSn', 'lambdaSm', 'lambdaF', 'lambdaG' }
    alphatau, betatau  - non-negative reals defining prior over noise parameters taun, taum, taul
    alpha0, beta0      - non-negative reals defining ARD prior over entity factors
    lambdaSn, lambdaSm - non-negative reals defining prior over Sn and Sm matrices
    lambdaF            - nonnegative real defining the prior over the Ft (if ARD is False)
    lambdaG            - nonnegative real defining the prior over the Gl (if ARD is False) 
    
We initialise the values of the Ft, Sn, Sm, Gl, lambdat, taun, taum, taul, 
according to the given argument 'init', which is a dictionary:
    { 'F', 'G', 'Sn', 'Sm', 'lambdat', 'tau' }
Options:
- 'random' -> draw initial values randomly from model definition, using given hyperparameters
- 'exp'    -> use the expectation of the model definition, using given hyperparameters
- 'kmeans' -> initialise F and G using Kmeans on the rows of R
- 'least'  -> initialise Sn, Sm using least squares (find S to minimise ||R-FSG.T||^2)
F         -> ['random','exp','kmeans']
G, Sn, Sm -> ['random','exp','least']
lambdat   -> ['random','exp']
tau       -> ['random','exp']

---

USAGE
    HMF = hmf_gibbs(R,C,D,K,settings,priors)
    HMF.initisalise(init)
    HMF.run(iterations)
Or:
    HMF = hmf_gibbs(R,C,D,K,settings,priors)
    HMF.train(init,iterations)
    
The expectation can be computed by specifying a burn-in and thinning rate, and using:
    HMF.approx_expectation_Ft(E,burn_in,thinning)
    HMF.approx_expectation_Gl(l,burn_in,thinning)
    HMF.approx_expectation_Sn(n,burn_in,thinning)
    HMF.approx_expectation_Sm(m,burn_in,thinning)
    HMF.approx_expectation_taun(n,burn_in,thinning)
    HMF.approx_expectation_taum(m,burn_in,thinning)
    HMF.approx_expectation_taul(l,burn_in,thinning)

We can test the performance of our model on a test dataset, specifying our test set with a mask M. 
    performance = HMF.predict_Rn(n,M_pred,burn_in,thinning)
    performance = HMF.predict_Cm(m,M_pred,burn_in,thinning)
    performance = HMF.predict_Dl(l,M_pred,burn_in,thinning)
This gives a dictionary of performances,
    performance = { 'MSE', 'R^2', 'Rp' }
    
We also store information about each iteration, namely:
- iterations_all_performances - dictionary from 'MSE', 'R^2', or 'Rp' to a list of performances
- iterations_all_times        - list of timesteps at which the iteration was completed
- iterations_all_Ft           - dictionary from entity type name to list of numpy arrays of Ft at each iteration
- iterations_all_Sn           - list of length N of lists of numpy arrays of Sn at each iteration (so N x It x size Sn)
- iterations_all_Sm           - list of length M of lists of numpy arrays of Sn at each iteration (so M x It x size Sm)
- iterations_all_Gl           - list of length L of lists of numpy arrays of Gl at each iteration (so L x It x size Gl)
- iterations_all_lambdat      - dictionary from entity type name to list of numpy vectors of lambdat at each iterations
- iterations_all_taun         - list of length N of lists of taun values at each iteration
- iterations_all_taum         - list of length M of lists of taum values at each iteration

Finally, we can return the goodness of fit of the data using the 
quality(metric,thinning,burn_in) function:
- metric = 'loglikelihood' -> return p(D|theta)
         = 'BIC'           -> return Bayesian Information Criterion
         = 'AIC'           -> return Afaike Information Criterion
         = 'MSE'           -> return Mean Square Error 
'''

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")

from HMF.code.Gibbs.draws_Gibbs import draw_tau, draw_lambda, draw_F, draw_S
from HMF.code.Gibbs.init_Gibbs import init_lambdak, init_FG, init_S, init_V, init_tau

import numpy, math, time


''' Default settings '''
IMPORTANCE_DATASET = 1.
DEFAULT_SETTINGS = {
    'priorF'  : 'exponential',
    'priorG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'columns',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}
OPTIONS_PRIOR_F = ['exponential','normal']
OPTIONS_PRIOR_S = ['exponential','normal']
OPTIONS_PRIOR_G = ['exponential','normal']
OPTIONS_ORDER_F = ['rows','columns']
OPTIONS_ORDER_S = ['rows','individual']
OPTIONS_ORDER_G = ['rows','columns']

DEFAULT_PRIORS = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 1.,
    'lambdaG'  : 1.,
    'lambdaSn' : 1.,
    'lambdaSm' : 1.
}

DEFAULT_INIT = {
    'F'       : 'kmeans',
    'G'       : 'least',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}
OPTIONS_INIT_F = ['kmeans','random','exp']
OPTIONS_INIT_G = ['least','random','exp']
OPTIONS_INIT_S = ['least','random','exp']
OPTIONS_INIT_LAMBDAT = ['random','exp']
OPTIONS_INIT_TAU = ['random','exp']

METRICS_PERFORMANCE = ['MSE','R^2','Rp'] 
METRICS_QUALITY = ['loglikelihood','BIC','AIC']


''' Class definition '''
class HMF_Gibbs:
    
    def __init__(self,R,C,D,K,settings={},hyperparameters={}):
        ''' Lists of the R, C, D datasets '''
        self.all_E = [] # list of all entity type names
        
        self.all_Rn = []      # list of all datasets Rn
        self.all_Mn = []      # list of all masks Mn for Rn
        self.E_per_Rn = []    # gives (E1,E2) tuples per entry in all_Rn
        self.size_Omegan = [] # list of the no. of observed datapoints in Rn
        self.all_alphan = []  # list of alphan values for each Rn
        
        self.all_Cm = []      # list of all constraint datasets Cm  
        self.all_Mm = []      # list of all masks Mm for Cm
        self.E_per_Cm = []    # gives E per entry in all_Mm
        self.size_Omegam = [] # list of the no. of observed datapoints in each Cm
        self.all_alpham = []  # list of alpham values for each Cm
        
        self.all_Dl = []      # list of all feature datasets Dl
        self.all_Ml = []      # list of all masks Ml for Dl
        self.E_per_Dl = []    # gives E per entry in all_Dl
        self.size_Omegal = [] # list of the no. of observed datapoints in each Dl
        self.all_alphal = []  # list of alphal values for each Dl
        
        self.all_Ft = {}      # dictionary from entity type name to Ft matrix 
        self.all_Sn = []      # list of Sn matrices, for Rn
        self.all_Sm = []      # list of Sm matrices, for Cm
        self.all_Gl = []      # list of Gl matrices, for Dl
        self.all_taun = []    # list of taun values, for Rn
        self.all_taum = []    # list of taum values, for Rm
        self.all_taul = []    # list of taul values, for Rl
        self.all_lambdat = {} # dictionary from entity type name to lambdat vector
        
        self.K = K            # dictionary from entity name to Kt
        self.I = {}           # dictionary from entity name to It
        self.J = []           # list of J values, for Dl

        ''' Dictionary mapping entity name to list of indexes of datasets '''
        self.U1t = {}        # indices n of Rn where t = tn (first entity type for Rn)
        self.U2t = {}        # indices n of Rn where t = un (second entity type for Rn)
        self.Vt = {}         # indices m of Cm where t = tm (self-relational entity type for Cm)
        self.Wt = {}         # indices l of Dl where t = tl (entity type for Dl)
        
        ''' Extract the info from R '''
        for n,(Rn,Mn,E1,E2,alpha) in enumerate(R):
            assert E1 != E2, "Gave same entity type for R%s: %s." % (n,E1)
            assert len(Rn.shape) == 2, "R%s is not 2-dimensional, but instead %s-dimensional." % (n,len(Rn.shape))
            assert len(Mn.shape) == 2, "M%s is not 2-dimensional, but instead %s-dimensional." % (n,len(Mn.shape))
            assert Rn.shape == Mn.shape, "Different shapes for R%s and M%s: %s and %s." % (n,n,Rn.shape,Mn.shape)
                
            (I,J) = Rn.shape
            if E1 in self.I.keys(): assert self.I[E1] == I, \
                "Different number of rows (%s) in R%s for entity type %s than before (%s)!" % (I,n,E1,self.I[E1])
            if E2 in self.I.keys(): assert self.I[E2] == J, \
                "Different number of columns (%s) in R%s for entity type %s than before (%s)!" % (J,n,E2,self.I[E2])
            self.I[E1], self.I[E2] = I, J
            
            if E1 not in self.all_E: self.all_E.append(E1)
            if E2 not in self.all_E: self.all_E.append(E2)
            self.E_per_Rn.append((E1,E2))
            
            self.all_Rn.append(numpy.array(Rn))
            self.all_Mn.append(numpy.array(Mn))
            self.size_Omegan.append(Mn.sum())
            self.all_alphan.append(alpha)
            
            self.U1t.setdefault(E1,[]).append(n) # if U1t is undefined set it to []; append n either way 
            self.U2t.setdefault(E2,[]).append(n)
            
        ''' Extract the info from C '''
        for m,(Cm,Mm,E,alpha) in enumerate(C):            
            assert len(Cm.shape) == 2, "C%s is not 2-dimensional, but instead %s-dimensional." % (m,len(Cm.shape))
            assert len(Mm.shape) == 2, "M%s is not 2-dimensional, but instead %s-dimensional." % (m,len(Mm.shape))
            assert Cm.shape == Mm.shape, "Different shapes for C%s and M%s: %s and %s." % (m,m,Cm.shape,Mm.shape)
            
            (I,J) = Cm.shape
            assert I == J, "C%s is not a square matrix: %s." % (m,Cm.shape)
            if E in self.I.keys(): assert self.I[E] == I, \
                "Different number of rows (%s) in C%s for entity type %s than before (%s)!" % (I,m,E,self.I[E])
            self.I[E] = I
            
            if E not in self.all_E: self.all_E.append(E)
            self.E_per_Cm.append(E)
            
            # Set the diagonal entries as unobserved
            for i in range(0,I):
                Mm[i,i] = 0.
            
            self.all_Cm.append(numpy.array(Cm))
            self.all_Mm.append(numpy.array(Mm))
            self.size_Omegam.append(Mm.sum())
            self.all_alpham.append(alpha)
            
            self.Vt.setdefault(E,[]).append(m)
        
        ''' Extract the info from D '''
        for l,(Dl,Ml,E,alpha) in enumerate(D):            
            assert len(Dl.shape) == 2, "D%s is not 2-dimensional, but instead %s-dimensional." % (l,len(Dl.shape))
            assert len(Ml.shape) == 2, "M%s is not 2-dimensional, but instead %s-dimensional." % (l,len(Ml.shape))
            assert Dl.shape == Ml.shape, "Different shapes for D%s and M%s: %s and %s." % (l,l,Dl.shape,Ml.shape)
            
            (I,J) = Dl.shape
            if E in self.I.keys(): assert self.I[E] == I, \
                "Different number of rows (%s) in D%s for entity type %s than before (%s)!" % (I,l,E,self.I[E])
            self.I[E] = I
            self.J.append(J)
            
            if E not in self.all_E: self.all_E.append(E)
            self.E_per_Dl.append(E)
            
            self.all_Dl.append(numpy.array(Dl))
            self.all_Ml.append(numpy.array(Ml))
            self.size_Omegal.append(Ml.sum())
            self.all_alphal.append(alpha)
            
            self.Wt.setdefault(E,[]).append(l)
        
        ''' Compute the number of datasets and entity types '''
        self.N = len(self.all_Rn) 
        self.M = len(self.all_Cm)
        self.L = len(self.all_Dl)
        self.T = len(self.all_E)       
        
        ''' Make sure all entities have at least the empty list in U1t, U2t, Vt, Wl '''
        for E in self.all_E:
            self.U1t.setdefault(E,[])
            self.U2t.setdefault(E,[])
            self.Vt.setdefault(E,[])
            self.Wt.setdefault(E,[])
            self.all_Ft.setdefault(E,[])
            assert E in self.K.keys(), "Did not get an entry for entity %s in K = %s." % (E,self.K)
            
        ''' Check whether the Rs and Cs have the right number of rows, columns, etc '''
        for E in self.all_E:
            self.check_same_size(E)
        
        ''' Check whether each entity instance has at least one observed entry across all datasets '''
        for E in self.all_E:
            self.check_observed_entry(E)      
      
        ''' Extract the hyperparameters and model settings '''
        self.alphatau = hyperparameters.get('alphatau',DEFAULT_PRIORS['alphatau'])
        self.betatau =  hyperparameters.get('betatau', DEFAULT_PRIORS['betatau'])
        self.alpha0 =   hyperparameters.get('alpha0',  DEFAULT_PRIORS['alpha0'])
        self.beta0 =    hyperparameters.get('beta0',   DEFAULT_PRIORS['beta0'])
        self.lambdaF =  hyperparameters.get('lambdaF', DEFAULT_PRIORS['lambdaF'])
        self.lambdaSn = hyperparameters.get('lambdaSn', DEFAULT_PRIORS['lambdaSn'])
        self.lambdaSm = hyperparameters.get('lambdaSm', DEFAULT_PRIORS['lambdaSm'])
        self.lambdaG =  hyperparameters.get('lambdaG', DEFAULT_PRIORS['lambdaG'])
        
        self.prior_F =  settings.get('priorF', DEFAULT_SETTINGS['priorF'])
        self.prior_G =  settings.get('priorG', DEFAULT_SETTINGS['priorG'])
        self.prior_Sn = settings.get('priorSn', DEFAULT_SETTINGS['priorSn'])
        self.prior_Sm = settings.get('priorSm', DEFAULT_SETTINGS['priorSm'])
        
        assert self.prior_F  in OPTIONS_PRIOR_F, "Unexpected prior for F: %s."  % self.prior_F
        assert self.prior_G  in OPTIONS_PRIOR_G, "Unexpected prior for G: %s."  % self.prior_G
        assert self.prior_Sn in OPTIONS_PRIOR_S, "Unexpected prior for Sn: %s." % self.prior_Sn
        assert self.prior_Sm in OPTIONS_PRIOR_S, "Unexpected prior for Sm: %s." % self.prior_Sm
        
        self.order_F =  settings.get('orderF',  DEFAULT_SETTINGS['orderF'])
        self.order_G =  settings.get('orderG',  DEFAULT_SETTINGS['orderG'])
        self.order_Sn = settings.get('orderSn', DEFAULT_SETTINGS['orderSn'])
        self.order_Sm = settings.get('orderSm', DEFAULT_SETTINGS['orderSm'])
        
        assert self.order_F in OPTIONS_ORDER_F,  "Unexpected order for F: %s." % self.prior_F
        assert self.order_G in OPTIONS_ORDER_G,  "Unexpected order for G: %s." % self.order_G
        assert self.order_Sn in OPTIONS_ORDER_S, "Unexpected order for Sn: %s." % self.order_Sn
        assert self.order_Sm in OPTIONS_ORDER_S, "Unexpected order for Sm: %s." % self.order_Sm
        
        self.ARD =      settings.get('ARD',     DEFAULT_SETTINGS['ARD'])
        
        self.rows_F =         True if self.order_F  == 'rows' else False
        self.rows_G =         True if self.order_G  == 'rows' else False
        self.rows_Sn =        True if self.order_Sn == 'rows' else False
        self.rows_Sm =        True if self.order_Sm == 'rows' else False
        self.nonnegative_F  = True if self.prior_F  == 'exponential' else False
        self.nonnegative_G  = True if self.prior_G  == 'exponential' else False
        self.nonnegative_Sn = True if self.prior_Sn == 'exponential' else False
        self.nonnegative_Sm = True if self.prior_Sm == 'exponential' else False
        
        print "Instantiated HMF model with: F ~ %s, G ~ %s, Sn ~ %s, Sm ~ %s, ARD = %s, and update order %s for F, %s for G, %s for Sn, %s for Sm." % \
            (self.prior_F,self.prior_G,self.prior_Sn,self.prior_Sm,self.ARD,self.order_F,self.order_G,self.order_Sn,self.order_Sm)
        
      
      
    """ Initialise the Fs, Ss, Gs, and taus. """
    def initialise(self,init={}):
        self.init_F =       init.get('F',      DEFAULT_INIT['F'])
        self.init_G =       init.get('G',      DEFAULT_INIT['G'])
        self.init_Sn =      init.get('Sn',     DEFAULT_INIT['Sn'])
        self.init_Sm =      init.get('Sm',     DEFAULT_INIT['Sm'])
        self.init_lambdat = init.get('lambdat',DEFAULT_INIT['lambdat'])
        self.init_tau =     init.get('tau',    DEFAULT_INIT['tau'])
        
        assert self.init_F in OPTIONS_INIT_F, "Unexpected init for F: %s." % self.init_F
        assert self.init_G in OPTIONS_INIT_G, "Unexpected init for G: %s." % self.init_G
        assert self.init_Sn in OPTIONS_INIT_S, "Unexpected init for Sn: %s." % self.init_Sn
        assert self.init_Sm in OPTIONS_INIT_S, "Unexpected init for Sm: %s." % self.init_Sm
        assert self.init_lambdat in OPTIONS_INIT_LAMBDAT, "Unexpected init for lambdat: %s." % self.init_lambdat
        assert self.init_tau in OPTIONS_INIT_TAU, "Unexpected init for tau: %s." % self.init_tau
        
        ''' Initialise the ARD: lambdat '''
        if self.ARD:
            self.all_lambdat = {
                E : init_lambdak(
                    init=self.init_lambdat,
                    K=self.K[E],
                    alpha0=self.alpha0,
                    beta0=self.beta0)
                for E in self.all_E            
            }      
        
        ''' Initialise the Ft - for KMeans use first dataset with 1+ observed 
            entry for each entity instance. '''
        for E in self.all_E:
            lambdaFt = self.all_lambdat[E] if self.ARD else self.lambdaF*numpy.ones(self.K[E])
            (R, M) = self.find_dataset(E) if self.init_F == 'kmeans' else (None, None)
            self.all_Ft[E] = init_FG(
                prior=self.prior_F,
                init=self.init_F,
                I=self.I[E],
                K=self.K[E],
                lambdak=lambdaFt,
                R=R,
                M=M) 
        
        ''' Initialise the Gl '''
        for l in range(0,self.L):
            E = self.E_per_Dl[l]
            
            lambdaGl = self.all_lambdat[E] if self.ARD else self.lambdaG*numpy.ones(self.K[E])
            Gl = init_V(
                prior=self.prior_G,
                init=self.init_G,
                I=self.J[l],
                K=self.K[E],
                lambdak=lambdaGl,
                R=self.all_Dl[l],
                M=self.all_Ml[l],
                U=self.all_Ft[E])
            self.all_Gl.append(Gl)
            
            taul = init_tau(
                init=self.init_tau,
                alphatau=self.alphatau,
                betatau=self.betatau,
                R=self.all_Dl[l],
                M=self.all_Ml[l],
                F=self.all_Ft[E],
                G=self.all_Gl[l])
            self.all_taul.append(taul)
            
        ''' Initialise the Sn and taun '''
        for n in range(0,self.N):
            E1,E2 = self.E_per_Rn[n]
            
            lambdaS = self.lambdaSn * numpy.ones((self.K[E1],self.K[E2]))
            Sn = init_S(
                prior=self.prior_Sn,
                init=self.init_Sn,
                K=self.K[E1],
                L=self.K[E2],
                lambdaS=lambdaS,
                R=self.all_Rn[n],
                M=self.all_Mn[n],
                F=self.all_Ft[E1],
                G=self.all_Ft[E2])
            self.all_Sn.append(Sn)
            
            taun = init_tau(
                init=self.init_tau,
                alphatau=self.alphatau,
                betatau=self.betatau,
                R=self.all_Rn[n],
                M=self.all_Mn[n],
                F=self.all_Ft[E1],
                G=self.all_Ft[E2],
                S=self.all_Sn[n])
            self.all_taun.append(taun)
            
        ''' Initialise the Sm and taum '''
        for m in range(0,self.M):
            E = self.E_per_Cm[m]
            
            lambdaS = self.lambdaSm * numpy.ones((self.K[E],self.K[E]))
            Sm = init_S(
                prior=self.prior_Sm,
                init=self.init_Sm,
                K=self.K[E],
                L=self.K[E],
                lambdaS=lambdaS,
                R=self.all_Cm[m],
                M=self.all_Mm[m],
                F=self.all_Ft[E],
                G=self.all_Ft[E])
            self.all_Sm.append(Sm)
                
            taum = init_tau(
                init=self.init_tau,
                alphatau=self.alphatau,
                betatau=self.betatau,
                R=self.all_Cm[m],
                M=self.all_Mm[m],
                F=self.all_Ft[E],
                G=self.all_Ft[E],
                S=self.all_Sm[m])
            self.all_taum.append(taum)
      
        print "Finished initialising with settings: F: %s. G: %s. Sn: %s. Sm: %s. lambdat: %s. tau: %s." % \
            (self.init_F,self.init_G,self.init_Sn,self.init_Sm,self.init_lambdat,self.init_tau)
      
      
    def run(self,iterations):
        """ Run the Gibbs sampler """
        self.iterations = iterations
        self.iterations_all_Ft =      {E:[] for E in self.all_E}
        self.iterations_all_lambdat = {E:[] for E in self.all_E}
        self.iterations_all_Sn =      [[] for n in range(0,self.N)]
        self.iterations_all_Sm =      [[] for m in range(0,self.M)]
        self.iterations_all_Gl =      [[] for l in range(0,self.L)]
        self.iterations_all_taun =    [[] for n in range(0,self.N)]
        self.iterations_all_taum =    [[] for m in range(0,self.M)]
        self.iterations_all_taul =    [[] for l in range(0,self.L)]
        self.all_times = [] 
        
        metrics = ['MSE','R^2','Rp']
        self.all_performances_Rn = {}
        self.all_performances_Cm = {}
        self.all_performances_Dl = {}
        for metric in metrics:
            self.all_performances_Rn[metric] = [[] for n in range(0,self.N)]
            self.all_performances_Cm[metric] = [[] for m in range(0,self.M)]
            self.all_performances_Dl[metric] = [[] for l in range(0,self.L)]
        
        ''' Print the initial statistics. '''
        print "Initial statistics. Performances (MSE, R^2, Rp):"
        for n in range(0,self.N):  
            perf = self.predict_while_running_Rn(n)
            print "R%s %s. %s. %s. %s." % (n,self.E_per_Rn[n],perf['MSE'],perf['R^2'],perf['Rp'])
            for metric in metrics:
                self.all_performances_Rn[metric][n].append(perf[metric])
        for m in range(0,self.M):
            perf = self.predict_while_running_Cm(m)
            print "C%s (%s). %s. %s. %s." % (m,self.E_per_Cm[m],perf['MSE'],perf['R^2'],perf['Rp'])
            for metric in metrics:
                self.all_performances_Cm[metric][m].append(perf[metric])
        for l in range(0,self.L):
            perf = self.predict_while_running_Dl(l)
            print "D%s (%s). %s. %s. %s." % (l,self.E_per_Dl[l],perf['MSE'],perf['R^2'],perf['Rp'])
            for metric in metrics:
                self.all_performances_Dl[metric][l].append(perf[metric])
        
        ''' Perform the specified number of Gibbs iterations. '''
        time_start = time.time()
        for it in range(0,self.iterations):  
            ''' Draw new values for the lambdat '''
            if self.ARD:
                for E in self.all_E:
                    Fs = [(self.all_Ft[E],self.nonnegative_F)]
                    for l in self.Wt[E]:
                        Fs.append((self.all_Gl[l],self.nonnegative_G))
                    self.all_lambdat[E] = draw_lambda(
                        alpha0=self.alpha0,
                        beta0=self.beta0,
                        Fs=Fs,
                        K=self.K[E])
                        
            ''' Draw new values for the Ft '''
            for E in self.all_E:
                R, C, D = self.construct_RCD(E)
                lambdaFt = self.all_lambdat[E] if self.ARD else self.lambdaF*numpy.ones(self.K[E])
                self.all_Ft[E] = draw_F(
                    R=R,C=C,D=D,
                    lambdaF=lambdaFt,
                    nonnegative=self.nonnegative_F,
                    rows=self.rows_F) 
                
            ''' Draw new values for the Gl '''
            for l in range(0,self.L):
                E = self.E_per_Dl[l]
                D = [(self.all_Dl[l].T,self.all_Ml[l].T,self.all_Gl[l],
                     self.all_Ft[E],self.all_taul[l],self.all_alphal[l])]
                lambdaGl = self.all_lambdat[E] if self.ARD else self.lambdaG*numpy.ones(self.K[E])
                self.all_Gl[l] = draw_F(
                    R=[],C=[],D=D,
                    lambdaF=lambdaGl,
                    nonnegative=self.nonnegative_G,
                    rows=self.rows_G) 
                
            ''' Draw new values for the Sn '''
            for n in range(0,self.N):
                E1,E2 = self.E_per_Rn[n]
                lambdaSn = self.lambdaSn * numpy.ones((self.K[E1],self.K[E2]))
                self.all_Sn[n] = draw_S(
                    dataset=self.all_Rn[n],
                    mask=self.all_Mn[n],
                    tau=self.all_taun[n],
                    alpha=self.all_alphan[n],
                    F=self.all_Ft[E1],
                    S=self.all_Sn[n],
                    G=self.all_Ft[E2],
                    lambdaS=lambdaSn,
                    nonnegative=self.nonnegative_Sn,
                    rows=self.rows_Sn) 
                
            ''' Draw new values for the Sm '''
            for m in range(0,self.M):
                E = self.E_per_Cm[m]
                lambdaSm = self.lambdaSm * numpy.ones((self.K[E],self.K[E]))
                self.all_Sm[m] = draw_S(
                    dataset=self.all_Cm[m],
                    mask=self.all_Mm[m],
                    tau=self.all_taum[m],
                    alpha=self.all_alpham[m],
                    F=self.all_Ft[E],
                    S=self.all_Sm[m],
                    G=self.all_Ft[E],
                    lambdaS=lambdaSm,
                    nonnegative=self.nonnegative_Sm,
                    rows=self.rows_Sm) 
            
            ''' Draw new values for the taun, taum, taul '''
            for n in range(0,self.N):
                E1,E2 = self.E_per_Rn[n]
                self.all_taun[n] = draw_tau(
                    alphatau=self.alphatau,
                    betatau=self.betatau,
                    dataset=self.all_Rn[n],
                    mask=self.all_Mn[n],
                    F=self.all_Ft[E1],
                    G=self.all_Ft[E2],
                    S=self.all_Sn[n])
                    
            for m in range(0,self.M):
                E = self.E_per_Cm[m]
                self.all_taum[m] = draw_tau(
                    alphatau=self.alphatau,
                    betatau=self.betatau,
                    dataset=self.all_Cm[m],
                    mask=self.all_Mm[m],
                    F=self.all_Ft[E],
                    G=self.all_Ft[E],
                    S=self.all_Sm[m])
                    
            for l in range(0,self.L):
                E = self.E_per_Dl[l]
                self.all_taul[l] = draw_tau(
                    alphatau=self.alphatau,
                    betatau=self.betatau,
                    dataset=self.all_Dl[l],
                    mask=self.all_Ml[l],
                    F=self.all_Ft[E],
                    G=self.all_Gl[l])
                
            ''' Store the draws - have to make a deep copy for the dicts and lists '''
            for E in self.all_E:
                self.iterations_all_lambdat[E].append(numpy.copy(self.all_lambdat[E]))
                self.iterations_all_Ft[E].append(numpy.copy(self.all_Ft[E]))
            for n in range(0,self.N):
                self.iterations_all_Sn[n].append(numpy.copy(self.all_Sn[n]))
                self.iterations_all_taun[n].append(self.all_taun[n])
            for m in range(0,self.M):
                self.iterations_all_Sm[m].append(numpy.copy(self.all_Sm[m]))
                self.iterations_all_taum[m].append(self.all_taum[m])
            for l in range(0,self.L):
                self.iterations_all_Gl[l].append(numpy.copy(self.all_Gl[l]))
                self.iterations_all_taul[l].append(self.all_taul[l])
            
            ''' Print the performancs and store them '''
            print "Iteration %s. Performances (MSE, R^2, Rp):" % (it+1)
            for n in range(0,self.N):  
                perf = self.predict_while_running_Rn(n)
                print "R%s %s. %s. %s. %s." % (n,self.E_per_Rn[n],perf['MSE'],perf['R^2'],perf['Rp'])
                for metric in metrics:
                    self.all_performances_Rn[metric][n].append(perf[metric])
            for m in range(0,self.M):
                perf = self.predict_while_running_Cm(m)
                print "C%s (%s). %s. %s. %s." % (m,self.E_per_Cm[m],perf['MSE'],perf['R^2'],perf['Rp'])
                for metric in metrics:
                    self.all_performances_Cm[metric][m].append(perf[metric])
            for l in range(0,self.L):
                perf = self.predict_while_running_Dl(l)
                print "D%s (%s). %s. %s. %s." % (l,self.E_per_Dl[l],perf['MSE'],perf['R^2'],perf['Rp'])
                for metric in metrics:
                    self.all_performances_Dl[metric][l].append(perf[metric])
        
            time_iteration = time.time()
            self.all_times.append(time_iteration-time_start)       
      

    """ Return the average value for the Fs, Ss,, Gs taus - i.e. our approximation to the expectations. """
    def approx_expectation_Ft(self,E,burn_in,thinning):
        ''' Expectation of F belonging to entity type E '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_Ft[E][i] for i in indices]).sum(axis=0) / float(len(indices))   
        
    def approx_expectation_lambdat(self,E,burn_in,thinning):
        ''' Expectation of lambdat belonging to entity type E '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_lambdat[E][i] for i in indices]).sum(axis=0) / float(len(indices))   
        
    def approx_expectation_Gl(self,l,burn_in,thinning):
        ''' Expectation of Gl '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_Gl[l][i] for i in indices]).sum(axis=0) / float(len(indices))   
        
    def approx_expectation_Sn(self,n,burn_in,thinning):
        ''' Expectation of Sn '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_Sn[n][i] for i in indices]).sum(axis=0) / float(len(indices))   
      
    def approx_expectation_Sm(self,m,burn_in,thinning):
        ''' Expectation of Sm '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_Sm[m][i] for i in indices]).sum(axis=0) / float(len(indices))   
      
    def approx_expectation_taun(self,n,burn_in,thinning):
        ''' Expectation of taun '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_taun[n][i] for i in indices]).sum(axis=0) / float(len(indices))   
        
    def approx_expectation_taum(self,m,burn_in,thinning):
        ''' Expectation of taum '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_taum[m][i] for i in indices]).sum(axis=0) / float(len(indices))   
        
    def approx_expectation_taul(self,l,burn_in,thinning):
        ''' Expectation of taul '''
        assert burn_in < self.iterations, "burn_in=%s should not be greater than the number of iterations=%s." % (burn_in,self.iterations)
        indices = range(burn_in,self.iterations,thinning)  
        return numpy.array([self.iterations_all_taul[l][i] for i in indices]).sum(axis=0) / float(len(indices))   
        

    """ Compute the expectation of F, S, and G, and use it to predict missing values """
    def predict_Rn(self,n,M_pred,burn_in,thinning):
        E1,E2 = self.E_per_Rn[n]
        exp_F1 = self.approx_expectation_Ft(E1,burn_in,thinning)
        exp_F2 = self.approx_expectation_Ft(E2,burn_in,thinning)
        exp_Sn = self.approx_expectation_Sn(n,burn_in,thinning)
        R_pred = self.triple_dot(exp_F1,exp_Sn,exp_F2.T)
        return self.compute_statistics(M_pred,self.all_Rn[n],R_pred)
        
    def predict_Cm(self,m,M_pred,burn_in,thinning):
        E = self.E_per_Cm[m]
        exp_F = self.approx_expectation_Ft(E,burn_in,thinning)
        exp_Sm = self.approx_expectation_Sm(m,burn_in,thinning)
        R_pred = self.triple_dot(exp_F,exp_Sm,exp_F.T)
        return self.compute_statistics(M_pred,self.all_Cm[m],R_pred)
        
    def predict_Dl(self,l,M_pred,burn_in,thinning):
        E = self.E_per_Dl[l]
        exp_F = self.approx_expectation_Ft(E,burn_in,thinning)
        exp_G = self.approx_expectation_Gl(l,burn_in,thinning)
        R_pred = numpy.dot(exp_F,exp_G.T)
        return self.compute_statistics(M_pred,self.all_Dl[l],R_pred)
        
    def predict_while_running_Rn(self,n):
        E1,E2 = self.E_per_Rn[n]
        R_pred = self.triple_dot(self.all_Ft[E1],self.all_Sn[n],self.all_Ft[E2].T)
        return self.compute_statistics(self.all_Mn[n],self.all_Rn[n],R_pred)    
        
    def predict_while_running_Cm(self,m):
        E = self.E_per_Cm[m]
        C_pred = self.triple_dot(self.all_Ft[E],self.all_Sm[m],self.all_Ft[E].T)
        return self.compute_statistics(self.all_Mm[m],self.all_Cm[m],C_pred)  
        
    def predict_while_running_Dl(self,l):
        E = self.E_per_Dl[l]
        D_pred = numpy.dot(self.all_Ft[E],self.all_Gl[l].T)
        return self.compute_statistics(self.all_Ml[l],self.all_Dl[l],D_pred)      
        
    
    """ Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) """
    def compute_statistics(self,M,R,R_pred): 
        MSE = self.compute_MSE(M,R,R_pred)
        R2 = self.compute_R2(M,R,R_pred)    
        Rp = self.compute_Rp(M,R,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}
        
    def compute_MSE(self,M,R,R_pred):
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else numpy.inf
        
    def compute_Rp(self,M,R,R_pred):
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))


    """ Functions for model selection, measuring the goodness of fit vs model complexity """          
    def quality(self,metric,burn_in,thinning): 
        assert metric in METRICS_QUALITY, 'Unrecognised metric for model quality: %s.' % metric

        log_likelihood = self.log_likelihood(burn_in,thinning)   
        if metric == 'loglikelihood':
            return log_likelihood
        elif metric == 'BIC':
            # -2*loglikelihood + (no. free parameters * log(no data points))
            return - 2 * log_likelihood + self.no_parameters() * math.log(self.no_datapoints())
        elif metric == 'AIC':
            # -2*loglikelihood + 2*no. free parameters
            return - 2 * log_likelihood + 2 * self.no_parameters()
      
    def log_likelihood(self,burn_in,thinning):
        ''' Compute the log likelihood of the model '''
        log_likelihood = 0.
        
        for n in range(0,self.N):
            E1,E2 = self.E_per_Rn[n]
            expF1 = self.approx_expectation_Ft(E1,burn_in,thinning)
            expF2 = self.approx_expectation_Ft(E2,burn_in,thinning)
            expS = self.approx_expectation_Sn(n,burn_in,thinning)
            exptau = self.approx_expectation_taun(n,burn_in,thinning)
            log_likelihood += self.log_likelihood_mtf(self.all_Rn[n],self.all_Mn[n],expF1,expS,expF2,exptau)
            
        for m in range(0,self.M):
            E1 = self.E_per_Cm[m]
            expF1 = self.approx_expectation_Ft(E1,burn_in,thinning)
            expS = self.approx_expectation_Sm(m,burn_in,thinning)
            exptau = self.approx_expectation_taum(m,burn_in,thinning)
            log_likelihood += self.log_likelihood_mtf(self.all_Cm[m],self.all_Mm[m],expF1,expS,expF1,exptau)
            
        for l in range(0,self.L):
            E = self.E_per_Dl[l]
            expF = self.approx_expectation_Ft(E,burn_in,thinning)
            expG = self.approx_expectation_Gl(l,burn_in,thinning)
            exptau = self.approx_expectation_taul(l,burn_in,thinning)
            log_likelihood += self.log_likelihood_mf(self.all_Dl[l],self.all_Ml[l],expF,expG,exptau)
            
        return log_likelihood            
            
      
    """ Helper methods """
    def check_same_size(self,E):
        '''Method for checking whether the datasets have the same number of 
            rows/columns for the same entity type. '''
        U1,U2,V,W = self.U1t[E],self.U2t[E],self.Vt[E],self.Wt[E]
        sizes = [self.all_Rn[n].shape[0] for n in U1] + [self.all_Rn[n].shape[1] for n in U2] + \
                [self.all_Cm[m].shape[0] for m in V]  + [self.all_Dl[l].shape[0] for l in W]
        assert len(set(sizes)) == 1, "Different dataset sizes for entity type %s across datasets R%s, C%s, M%s: %s, respectively." % (E,U1+U2,V,W,sizes)

    def check_observed_entry(self,E):
        ''' Method for checking whether each entity has at least one observed 
            datapoint in one of the datasets. '''
        U1,U2,V,W = self.U1t[E],self.U2t[E],self.Vt[E],self.Wt[E]
        sums_per_entity = numpy.zeros(self.I[E])
        for M in [self.all_Mn[n] for n in U1] + [self.all_Mn[n].T for n in U2]:
            ''' Sum each row, and add that to sums_per_entity '''
            sums_rows = M.sum(axis=1)
            sums_per_entity = numpy.add(sums_per_entity,sums_rows)
        for M in [self.all_Mm[m] for m in V]:
            ''' Sum each row and column, and add that to sums_per_entity '''
            sums_rows = M.sum(axis=1)
            sums_columns = M.sum(axis=0)
            sums_per_entity = numpy.add(sums_per_entity,numpy.add(sums_rows,sums_columns))
        for M in [self.all_Ml[l] for l in W]:
            ''' Sum each row, and add that to sums_per_entity '''
            sums_rows = M.sum(axis=1)
            sums_per_entity = numpy.add(sums_per_entity,sums_rows)
            
        for i,s in enumerate(sums_per_entity):
            assert s > 0, "No observed datapoints in any dataset for entity %s of type %s." % (i,E) 
            
    def find_dataset(self,E):
        ''' For the given entity type, find a dataset R, C, or D, for which 
            every entity instance has at least one observed entry. 
            If there is none, raise an AssertError.'''
        datasets = [(self.all_Rn[n],  self.all_Mn[n])   for n in self.U1t[E]] + \
                   [(self.all_Rn[n].T,self.all_Mn[n].T) for n in self.U2t[E]] + \
                   [(self.all_Cm[m],  self.all_Mm[m])   for m in self.Vt[E]] + \
                   [(self.all_Cm[m].T,self.all_Mm[m].T) for m in self.Vt[E]] + \
                   [(self.all_Dl[l],  self.all_Ml[l].T) for l in self.Wt[E]]
        for (R,M) in datasets:
            if all([True if row.sum() > 0 else False for row in M]):   
                return R,M
        assert False, "Could not initialise F for entity type %s with K-means as no dataset R or C has at least one datapoint for each entity." % E
         
    def construct_RCD(self,E):
        ''' Construct the R, C, D lists needed for the updates of Ft. '''
        R, C, D = [], [], []
        
        for n in self.U1t[E]:
            _,E2 = self.E_per_Rn[n]
            R.append((self.all_Rn[n],self.all_Mn[n],self.all_Ft[E],self.all_Sn[n],
                      self.all_Ft[E2],self.all_taun[n],self.all_alphan[n]))
                      
        for n in self.U2t[E]:
            ''' In this case we take the transpose of R, M, S so that our F is the first matrix. '''
            E1,_ = self.E_per_Rn[n]
            R.append((self.all_Rn[n].T,self.all_Mn[n].T,self.all_Ft[E],self.all_Sn[n].T,
                      self.all_Ft[E1],self.all_taun[n],self.all_alphan[n]))
                      
        for m in self.Vt[E]:
            C.append((self.all_Cm[m],self.all_Mm[m],self.all_Ft[E],self.all_Sm[m],
                      self.all_taum[m],self.all_alpham[m]))

        for l in self.Wt[E]:
            D.append((self.all_Dl[l],self.all_Ml[l],self.all_Ft[E],self.all_Gl[l],
                      self.all_taul[l],self.all_alphal[l]))
                      
        return (R,C,D)
         
    def triple_dot(self,M1,M2,M3):
        ''' Triple matrix multiplication: M1*M2*M3. 
            If the matrices have dimensions I,K,L,J, then the complexity of M1*(M2*M3) 
            is ~IJK, and (M1*M2)*M3 is ~IJL. So if K < L, we use the former. '''
        K,L = M2.shape
        if K < L:
            return numpy.dot(M1,numpy.dot(M2,M3))
        else:
            return numpy.dot(numpy.dot(M1,M2),M3)
        
    def log_likelihood_mf(self,R,M,expF,expG,exptau):
        ''' Return the matrix tri-factorisation likelihood of the data given 
            the trained model's parameters '''
        explogtau = math.log(exptau)
        return M.sum() / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exptau / 2. * (M*( R - numpy.dot(expF,expG.T) )**2).sum()  
        
    def log_likelihood_mtf(self,R,M,expF1,expS,expF2,exptau):
        ''' Return the matrix tri-factorisation likelihood of the data given 
            the trained model's parameters '''
        explogtau = math.log(exptau)
        return M.sum() / 2. * ( explogtau - math.log(2*math.pi) ) \
             - exptau / 2. * (M*( R - self.triple_dot(expF1,expS,expF2.T) )**2).sum()         
          
    def no_datapoints(self):
        ''' Return the total number of observed datapoints '''
        return sum(self.size_Omegan) + sum(self.size_Omegam) + sum(self.size_Omegal)     
        
    def no_parameters(self):
        ''' Return the number of model parameters '''
        count = 0.
        # Ft and lambdat
        for E1 in self.all_E:
            count += self.I[E1]*self.K[E1] + self.K[E1]
        # Sn
        for n in range(0,self.N):
            E1,E2 = self.E_per_Rn[n]
            count += self.K[E1]*self.K[E2]
        # Sm
        for m in range(0,self.M):
            E1 = self.E_per_Cm[m]
            count += self.K[E1]**2
        # Gl
        for l in range(0,self.L):
            E1 = self.E_per_Dl[l]
            count += self.J[l]*self.K[E1]
        # taun, taum, taul
        count += self.N + self.M + self.L
        return count
        