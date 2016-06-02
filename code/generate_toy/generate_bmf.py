"""
Generate a toy dataset for the matrix factorisation case, and store it.
U,V ~ N
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)
from bmf_models.code.distributions.normal import normal_draw
from bmf_models.code.generate_mask.mask import try_generate_M

import numpy, itertools, matplotlib.pyplot as plt

def generate_dataset(I,J,K,lambdaU,lambdaV,tau):
    # Generate U, V
    U = numpy.zeros((I,K))
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        U[i,k] = normal_draw(mu=0,tau=lambdaU)
    V = numpy.zeros((J,K))
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        V[j,k] = normal_draw(mu=0,tau=lambdaV)
    
    # Generate R
    true_R = numpy.dot(U,V.T)
    R = add_noise(true_R,tau)    
    
    return (U,V,tau,true_R,R)
    
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
    output_folder = project_location+"bmf_models/data/toy/bmf/"

    I,J,K = 100, 80, 10 #20, 10, 5 #
    fraction_unknown = 0.1
    alpha, beta = 1., 1.
    lambdaU = 1.
    lambdaV = 1.
    tau = alpha / beta
    
    (U,V,tau,true_R,R) = generate_dataset(I,J,K,lambdaU,lambdaV,tau)
    
    # Try to generate M
    M = try_generate_M(I,J,fraction_unknown,attempts=1000)
    
    # Store all matrices in text files
    numpy.savetxt(open(output_folder+"U.txt",'w'),U)
    numpy.savetxt(open(output_folder+"V.txt",'w'),V)
    numpy.savetxt(open(output_folder+"R_true.txt",'w'),true_R)
    numpy.savetxt(open(output_folder+"R.txt",'w'),R)
    numpy.savetxt(open(output_folder+"M.txt",'w'),M)
    
    print "Mean R: %s. Variance R: %s. Min R: %s. Max R: %s." % (numpy.mean(R),numpy.var(R),R.min(),R.max())
    fig = plt.figure()
    plt.hist(R.flatten(),bins=range(int(R.min())-1,int(R.max())+1))
    plt.show()