"""
Generate a toy dataset for the matrix factorisation case, and store it.
F,G ~ Exp
S ~ N
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from bmf_models.code.distributions.exponential import exponential_draw
from bmf_models.code.distributions.normal import normal_draw
from bmf_models.code.generate_mask.mask import try_generate_M

import numpy, itertools, matplotlib.pyplot as plt

def generate_dataset(I,J,K,L,lambdaF,lambdaS,lambdaG,tau):
    # Generate U, V
    F = numpy.zeros((I,K))
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        F[i,k] = exponential_draw(lambdaF)
    S = numpy.zeros((K,L))
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        S[k,l] = normal_draw(mu=0,tau=lambdaS)
    G = numpy.zeros((J,L))
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        G[j,l] = exponential_draw(lambdaG)
        
    # Generate R
    true_R = numpy.dot(F,numpy.dot(S,G.T))
    R = add_noise(true_R,tau)    
    
    return (F,S,G,tau,true_R,R)
    
def add_noise(true_R,tau):
    if numpy.isinf(tau):
        return numpy.copy(true_R)
    (I,J) = true_R.shape
    R = numpy.zeros((I,J))
    for i,j in itertools.product(xrange(0,I),xrange(0,J)):
        R[i,j] = normal_draw(true_R[i,j],tau)
    return R
    
##########

if __name__ == "__main__":
    output_folder = project_location+"bmf_models/data/toy/bsnmtf/"

    I,J,K,L = 100, 80, 10, 8
    fraction_unknown = 0.1
    alpha, beta = 1., 1.
    lambdaF = 1.
    lambdaS = 1.
    lambdaG = 1.
    tau = alpha / beta
    
    (F,S,G,tau,true_R,R) = generate_dataset(I,J,K,L,lambdaF,lambdaS,lambdaG,tau)
    
    # Try to generate M
    M = try_generate_M(I,J,fraction_unknown,attempts=1000)
    
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"F.txt",'w'),F)
    numpy.savetxt(open(output_folder+"S.txt",'w'),S)
    numpy.savetxt(open(output_folder+"G.txt",'w'),G)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    numpy.savetxt(open(output_folder+"M.txt",'w'),M)
    
    print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())
    fig = plt.figure()
    plt.hist(R.flatten(),bins=range(int(R.min())-1,int(R.max())+1))
    plt.show()