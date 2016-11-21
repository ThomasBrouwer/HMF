'''
Plot the performances of HMF D-MF, D-MTF, and S-MF, with varying values for Kt, 
on the GE+PM to GM datasets.
'''

import matplotlib.pyplot as plt

MSE_max, MSE_min = 1.2, 0.5
fraction_min, fraction_max = 0.25, 0.95

values_K = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]

#perf_hmf_mf_ge_pm_to_gm_ARD = 
#perf_hmf_mf_ge_pm_to_gm_no_ARD = 

perf_hmf_mtf_ge_pm_to_gm_ARD = [0.99383572803203502, 0.95391284959823019, 0.84776759328788653, 0.84094448811766553, 0.77566413993160921, 0.76304090476894482, 0.70646398236194119, 0.6586612666037126, 0.65482822893424797, 0.65882198539051362, 0.66379046061158986, 0.6597802640846866, 0.64481772162981643, 0.64435415246853678]
perf_hmf_mtf_ge_pm_to_gm_no_ARD = [1.0014626014434833, 0.9499471433684803, 0.87065457180402195, 0.85962734627382653, 0.81246637433189106, 0.79595938498334906, 0.73773501948029907, 0.70490524467933513, 0.70545667776187559, 0.71824200581717512, 0.73066188989774172, 0.7231452669785956, 0.75500732023075456, 0.78374059194073575]

#perf_hmf_smf_ge_pm_to_gm_ARD = 
#perf_hmf_smf_ge_pm_to_gm_no_ARD = 


''' Plot '''
fig = plt.figure(figsize=(3.8,3.0))
fig.subplots_adjust(left=0.14, right=0.965, bottom=0.12, top=0.97)
plt.xlabel("Kt", fontsize=12, labelpad=1) #fontsize=8
plt.ylabel("MSE", fontsize=12, labelpad=2) #fontsize=8, labelpad=-1
plt.yticks(fontsize=8) #fontsize=6
plt.xticks(fontsize=8) #fontsize=6

#plt.plot(values_K, perf_hmf_mf_ge_pm_to_gm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MF (ARD)', c='red', markersize=5)
#plt.plot(values_K, perf_hmf_mf_ge_pm_to_gm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MF (no ARD)', c='red', markersize=5)

plt.plot(values_K, perf_hmf_mtf_ge_pm_to_gm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF D-MTF (ARD)', c='blue', markersize=5)
plt.plot(values_K, perf_hmf_mtf_ge_pm_to_gm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF D-MTF (no ARD)', c='blue', markersize=5)
 
#plt.plot(values_K, perf_hmf_smf_ge_pm_to_gm_ARD,    linestyle='-', linewidth=1.2, marker='o', label='HMF S-MF (ARD)', c='blue', markersize=5)
#plt.plot(values_K, perf_hmf_smf_ge_pm_to_gm_no_ARD, linestyle='-', linewidth=1.2, marker='x', label='HMF S-MF (no ARD)', c='blue', markersize=5)
 
plt.xlim(0,values_K[-1])
plt.ylim(MSE_min,MSE_max)

plt.legend(loc='upper left',fontsize=10)
    
plt.savefig("./varying_K_ge_pm_to_gm.png", dpi=600)