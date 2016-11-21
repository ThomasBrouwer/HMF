'''
Plot the performances of HMF D-MF, D-MTF, and S-MF, with varying values for Kt, 
on the PM+GM to GE datasets.
'''

import matplotlib.pyplot as plt

MSE_max, MSE_min = 1.5, 0.8
fraction_min, fraction_max = 0.25, 0.95

values_K = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]

#perf_hmf_mf_pm_gm_to_ge_ARD = 
#perf_hmf_mf_pm_gm_to_ge_no_ARD = 

perf_hmf_mtf_pm_gm_to_ge_ARD = [1.0008179293234307, 1.0027935398482632, 0.9725686537406224, 0.89406943742551093, 0.90911482664537524, 0.93221899674787956, 0.96545060035367736, 1.0105451043621445, 0.96346641937002442, 1.0115351385666709, 0.97922436610607677, 0.94731013699836597, 0.98923686335869621, 1.0222804345913752]
perf_hmf_mtf_pm_gm_to_ge_no_ARD = [1.0011176009123834, 1.001208239789396, 0.99740877082053725, 0.91132346942128706, 0.94069263452160179, 0.95233362349919037, 1.1101377682300149, 1.0986209628136276, 1.2216930864968425, 1.2224873115331814, 1.2334890299622294, 1.2569353083033512, 1.3493863257094616, 1.362851123555044]

#perf_hmf_smf_pm_gm_to_ge_ARD = 
#perf_hmf_smf_pm_gm_to_ge_no_ARD = 


''' Plot '''
fig = plt.figure(figsize=(3.8,3.0))
fig.subplots_adjust(left=0.14, right=0.965, bottom=0.12, top=0.97)
plt.xlabel("Kt", fontsize=12, labelpad=1) #fontsize=8
plt.ylabel("MSE", fontsize=12, labelpad=2) #fontsize=8, labelpad=-1
plt.yticks(fontsize=8) #fontsize=6
plt.xticks(fontsize=8) #fontsize=6

#plt.plot(values_K, perf_hmf_mf_pm_gm_to_ge_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MF (ARD)', c='red', markersize=5)
#plt.plot(values_K, perf_hmf_mf_pm_gm_to_ge_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MF (no ARD)', c='red', markersize=5)

plt.plot(values_K, perf_hmf_mtf_pm_gm_to_ge_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MTF (ARD)', c='blue', markersize=5)
plt.plot(values_K, perf_hmf_mtf_pm_gm_to_ge_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MTF (no ARD)', c='blue', markersize=5)
 
#plt.plot(values_K, perf_hmf_smf_pm_gm_to_ge_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF S-MF (ARD)', c='blue', markersize=5)
#plt.plot(values_K, perf_hmf_smf_pm_gm_to_ge_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF S-MF (no ARD)', c='blue', markersize=5)
 
plt.xlim(0,values_K[-1])
plt.ylim(MSE_min,MSE_max)

plt.legend(loc='upper left',fontsize=10)
    
plt.savefig("./varying_K_pm_gm_to_ge.png", dpi=600)