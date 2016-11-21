'''
Plot the performances of HMF D-MF, D-MTF, and S-MF, with varying values for Kt, 
on the GE+GM to PM datasets.
'''

import matplotlib.pyplot as plt

MSE_max, MSE_min = 1.1, 0.7
fraction_min, fraction_max = 0.25, 0.95

values_K = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]

#perf_hmf_mf_ge_gm_to_pm_ARD = 
#perf_hmf_mf_ge_gm_to_pm_no_ARD = 

perf_hmf_mtf_ge_gm_to_pm_ARD = [0.96687919064173988, 0.95642859255925061, 0.94183674422775865, 0.91058768715878879, 0.90272335932951753, 0.84169156169043335, 0.80773291957782178, 0.78792060879368808, 0.77616162161044078, 0.76364626947916814, 0.74255960811908162, 0.73981982592197293, 0.74183804024271838, 0.73888837609339864]
perf_hmf_mtf_ge_gm_to_pm_no_ARD = [0.96899210395395552, 0.95584451613916599, 0.94668711345253498, 0.91234460070836376, 0.91957780803072209, 0.86065341792206862, 0.83832077958264151, 0.84694769579052465, 0.83790157315686342, 0.8797200549187455, 0.88414963700170401, 0.86713496498571341, 0.8517108828305473, 0.87947057376576621]

#perf_hmf_smf_ge_gm_to_pm_ARD = 
#perf_hmf_smf_ge_gm_to_pm_no_ARD = 


''' Plot '''
fig = plt.figure(figsize=(3.8,3.0))
fig.subplots_adjust(left=0.14, right=0.965, bottom=0.12, top=0.97)
plt.xlabel("Kt", fontsize=12, labelpad=1) #fontsize=8
plt.ylabel("MSE", fontsize=12, labelpad=2) #fontsize=8, labelpad=-1
plt.yticks(fontsize=8) #fontsize=6
plt.xticks(fontsize=8) #fontsize=6

#plt.plot(values_K, perf_hmf_mf_ge_gm_to_pm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MF (ARD)', c='red', markersize=5)
#plt.plot(values_K, perf_hmf_mf_ge_gm_to_pm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MF (no ARD)', c='red', markersize=5)

plt.plot(values_K, perf_hmf_mtf_ge_gm_to_pm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MTF (ARD)', c='blue', markersize=5)
plt.plot(values_K, perf_hmf_mtf_ge_gm_to_pm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MTF (no ARD)', c='blue', markersize=5)
 
#plt.plot(values_K, perf_hmf_smf_ge_gm_to_pm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF S-MF (ARD)', c='blue', markersize=5)
#plt.plot(values_K, perf_hmf_smf_ge_gm_to_pm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF S-MF (no ARD)', c='blue', markersize=5)
 
plt.xlim(0,values_K[-1])
plt.ylim(MSE_min,MSE_max)

plt.legend(loc='upper left',fontsize=10)
    
plt.savefig("./varying_K_ge_gm_to_pm.png", dpi=600)