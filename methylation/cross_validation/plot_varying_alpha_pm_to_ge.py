'''
Plot the performance of PM -> GE predictions for HMF D-MF and HMF D-MTF, as we
vary the values of alpha (but keeping alpha1+alpha2=2).
'''

import matplotlib.pyplot as plt, numpy

''' Data '''
values_alpha_PM = [1., 0.67, 0.5, 0.33, 0.2, 0.1]
MF_MSE  = [1.14667879891, 1.02299310513,  0.846795028167, 0.843159103115, 0.852543339052, 0.883781922686]
MTF_MSE = [0.92424600746, 0.876466937064, 0.874113635057, 0.908049050549, 0.943227235142, 0.967237667979]

''' Plot '''
fig = plt.figure(figsize=(3.8,3.0))
fig.subplots_adjust(left=0.12, right=0.965, bottom=0.14, top=0.97)
plt.xlabel("Importance GE", fontsize=10, labelpad=1) #fontsize=8
plt.ylabel('MSE', fontsize=10, labelpad=0) #fontsize=8, labelpad=-1
plt.yticks(fontsize=8) #fontsize=6
plt.xticks(fontsize=8) #fontsize=6

plt.plot(values_alpha_PM,MF_MSE,linestyle='-', linewidth=1.2, marker='o', label='HMF D-MF', c='r', markersize=5) #linewidth=0.5, markersize=2)
plt.plot(values_alpha_PM,MTF_MSE,linestyle='-', linewidth=1.2, marker='o', label='HMF D-MTF', c='b', markersize=5) #linewidth=0.5, markersize=2)
 
plt.xlim(0.,1.05)
plt.ylim(0.8,1.2)

plt.legend(loc='upper left',fontsize=10,ncol=2)
    
plt.savefig("./varying_alpha_pm_to_ge.png", dpi=600)