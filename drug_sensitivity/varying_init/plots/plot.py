'''
Plot the convergence of both methods on the four datasets.
'''

import matplotlib.pyplot as plt



''' Load in the performances. '''
folder_performances = './../results/'
mf_ccle_ec = eval(open(folder_performances+'hmf_data_mf_ccle_ec.txt','r').readline())
mf_ccle_ic = eval(open(folder_performances+'hmf_data_mf_ccle_ic.txt','r').readline())
mf_ctrp_ec = eval(open(folder_performances+'hmf_data_mf_ctrp_ec.txt','r').readline())
mf_gdsc_ic = eval(open(folder_performances+'hmf_data_mf_gdsc_ic.txt','r').readline())
mtf_ccle_ec = eval(open(folder_performances+'hmf_data_mtf_ccle_ec.txt','r').readline())
mtf_ccle_ic = eval(open(folder_performances+'hmf_data_mtf_ccle_ic.txt','r').readline())
mtf_ctrp_ec = eval(open(folder_performances+'hmf_data_mtf_ctrp_ec.txt','r').readline())
mtf_gdsc_ic = eval(open(folder_performances+'hmf_data_mtf_gdsc_ic.txt','r').readline())


''' Plot settings. '''
iterations = range(len(mf_ccle_ec[0])+1)
iterations_mf, iterations_mtf = 50, 50
colours = ['#0000ff','#00ff00','#ff0000','#00ffff','#ff00ff','#ffff00','#00007f']#,'#007f00']
markersize = 1


''' Method for plotting. '''
def plot(name, MSE_min, MSE_max, all_performances, no_iterations):
    fig = plt.figure(figsize=(4.5,3.0))
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.12, top=0.97)
    plt.xlabel("Iterations", fontsize=12, labelpad=1) #fontsize=8
    plt.ylabel("MSE", fontsize=12, labelpad=2) #fontsize=8, labelpad=-1
    plt.yticks(fontsize=8) #fontsize=6
    plt.xticks(fontsize=8) #fontsize=6
    
    plt.plot(iterations[:no_iterations], all_performances[0][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='Exp', c=colours[0], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[1][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='Random', c=colours[1], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[2][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='K-means, exp', c=colours[2], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[3][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='K-means, random', c=colours[3], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[4][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='Exp, least squares', c=colours[4], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[5][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='Random, least squares', c=colours[5], markersize=markersize)
    plt.plot(iterations[:no_iterations], all_performances[6][:no_iterations], 
             linestyle='-', linewidth=1.2, marker='o', label='K-means, least squares', c=colours[6], markersize=markersize)
    
    plt.xlim(0,no_iterations+1)
    plt.ylim(MSE_min,MSE_max)
    
    plt.legend(loc='upper right',fontsize=8)
    plt.savefig(folder_plots+"%s.png" % name, dpi=600)
    
    
''' Make the plots. '''
folder_plots = './'
plot('ccle_ec_mf',  0.055, 0.13,  mf_ccle_ec,  iterations_mf)
plot('ccle_ic_mf',  0.015, 0.07,  mf_ccle_ic,  iterations_mf)
plot('ctrp_ec_mf',  0.07,  0.085, mf_ctrp_ec,  iterations_mf)
plot('gdsc_ic_mf',  0.055, 0.07,  mf_gdsc_ic,  iterations_mf)
plot('ccle_ec_mtf', 0.055, 0.17,  mtf_ccle_ec, iterations_mtf)
plot('ccle_ic_mtf', 0.025, 0.07,  mtf_ccle_ic, iterations_mtf)
plot('ctrp_ec_mtf', 0.076, 0.12,  mtf_ctrp_ec, iterations_mtf)
plot('gdsc_ic_mtf', 0.06,  0.12,  mtf_gdsc_ic, iterations_mtf)