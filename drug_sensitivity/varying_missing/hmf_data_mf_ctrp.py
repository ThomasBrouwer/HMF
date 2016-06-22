"""
Test the performance of HMF for recovering the CTRP dataset, where we vary the 
fraction of entries that are missing.
We repeat this 10 times per fraction and average that.

GDSC has 0.7356934001670844 observed entries
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from HMF.code.models.hmf_Gibbs import HMF_Gibbs
from HMF.code.generate_mask.mask import try_generate_M_from_M
from HMF.drug_sensitivity.load_dataset import load_data_without_empty,load_data_filter

import numpy, random


''' Settings '''
metrics = ['MSE', 'R^2', 'Rp']

fractions_unknown = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
repeats = 10

iterations, burn_in, thinning = 200, 180, 2
settings = {
    'priorF'  : 'exponential',
    'orderG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'columns',
    'orderG'  : 'rows',
    'orderSn' : 'rows',
    'orderSm' : 'rows',
    'ARD'     : True
}
hyperparameters = {
    'alphatau' : 1.,
    'betatau'  : 1.,
    'alpha0'   : 0.001,
    'beta0'    : 0.001,
    'lambdaF'  : 0.1,
    'lambdaG'  : 0.1,
    'lambdaSn' : 0.1,
    'lambdaSm' : 0.1,
}
init = {
    'F'       : 'kmeans',
    'Sn'      : 'least',
    'Sm'      : 'least',
    'G'       : 'least',
    'lambdat' : 'exp',
    'tau'     : 'exp'
}

K = {'Cell_lines':10, 'Drugs':10}
alpha_l = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Load data '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data = location+"data_row_01/"

R_ctrp,     M_ctrp,   cell_lines, drugs   = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)


''' Seed all of the methods the same '''
numpy.random.seed(0)
random.seed(0)

''' Generate matrices M - one list of (M_train,M_test)'s for each fraction '''
M_attempts = 1000
all_Ms_train_test = [ 
    [try_generate_M_from_M(M=M_ctrp,fraction=fraction,attempts=M_attempts) for r in range(0,repeats)]
    for fraction in fractions_unknown
]

''' Make sure each M has no empty rows or columns '''
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for Ms_train_test,fraction in zip(all_Ms_train_test,fractions_unknown):
    for (M_train,M_test) in Ms_train_test:
        check_empty_rows_columns(M_train,fraction)

''' Run the method on each of the M's for each fraction '''
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (fraction,Ms_train_test) in zip(fractions_unknown,all_Ms_train_test):
    print "Trying fraction %s." % fraction
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for repeat,(M_train,M_test) in zip(range(0,repeats),Ms_train_test):
        print "Repeat %s of fraction %s." % (repeat+1, fraction)
     
        D = [(R_ctrp,    M_train,   'Cell_lines', alpha_l[0]), 
             (R_gdsc,    M_gdsc,    'Cell_lines', alpha_l[1]), 
             (R_ccle_ic, M_ccle_ic, 'Cell_lines', alpha_l[2]),
             (R_ccle_ec, M_ccle_ec, 'Cell_lines', alpha_l[3])]
        R, C = [], []

        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        
        # Measure the performances
        performances = HMF.predict_Dl(l=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this fraction
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
    
 
print "repeats=%s \nfractions_unknown = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,fractions_unknown,all_performances,average_performances)

'''
repeats=10 
fractions_unknown = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] 
all_performances = {'R^2': [[0.4613831479115331, 0.37514804479062, 0.47277655260949125, 0.45332797028919014, 0.4372671612907343, 0.4193870381172182, 0.44609155895509667, 0.3823508072973548, 0.444203681457947, 0.5486045088199711], [0.4245091834740865, 0.3958501774227826, 0.4183254570133078, 0.43113486873257234, 0.4393636867565629, 0.46031885618844737, 0.45869270089765457, 0.4346289623243622, 0.4582830956901225, 0.49090242865332867], [0.4413941377812871, 0.41828905478831524, 0.38969740002215913, 0.4144907369374128, 0.4539304672119835, 0.45640437092190766, 0.4225378726240694, 0.4106577908377478, 0.42084249312929456, 0.4462900130516938], [0.4243613145322912, 0.3914676019577493, 0.4087788070646874, 0.4341639675170741, 0.405573919550281, 0.39886694637094344, 0.40070092656415535, 0.43512763652693387, 0.4265537152440565, 0.39262304305033113], [0.4071111319573322, 0.41637619819634963, 0.39476010977496934, 0.43017492211498054, 0.41466345021553197, 0.4137339953667746, 0.4197459288845281, 0.40470476910398734, 0.4332210685981984, 0.4068941385093142], [0.3807434383525008, 0.4143876342623045, 0.39194997008356613, 0.40101540333271757, 0.4305280719146497, 0.41892282810608583, 0.41411545999146027, 0.41302336995985456, 0.4197985363195973, 0.41642672452046514], [0.3866643006062891, 0.37655747297531506, 0.4212267948006233, 0.3820242772963367, 0.40531371628843627, 0.3883744643470829, 0.37424214838445624, 0.4217386117160654, 0.3876680431181707, 0.38798945620299086], [0.3973360484247642, 0.4104608821054071, 0.3824818760064337, 0.3706209054901214, 0.3858755190451535, 0.3990325961921939, 0.404102637009063, 0.39618507621000265, 0.3819755135348385, 0.3744605063854298], [0.3926620581853516, 0.3749291385495406, 0.38280351398412105, 0.3746396772936912, 0.4021498609523, 0.3763769657427194, 0.3795295048627879, 0.3844479656550318, 0.40153134031508575, 0.39494051176462586], [0.38133866702560326, 0.386799761947326, 0.3590280858803543, 0.3734739708301321, 0.37522174084996385, 0.3659652701694135, 0.3767548212249957, 0.3938572219985459, 0.3827498770893596, 0.3860578930652726], [0.37385924125177117, 0.3469614086680223, 0.36637358907190964, 0.3786417106844915, 0.3766024997484301, 0.36467814065427673, 0.35862725978300125, 0.35431156929204866, 0.3583970053539983, 0.38693003579668217], [0.3466214734399369, 0.3572829454275762, 0.35204686926472795, 0.36026944075322165, 0.3478913362381395, 0.3588760774710782, 0.36540755943839287, 0.3519141134033076, 0.35491400709011445, 0.3561067340993419], [0.33596825466074, 0.34005906691558363, 0.34059593079913275, 0.35359651706243767, 0.31440182998954636, 0.3395194747955691, 0.33206074693898013, 0.3449462199097827, 0.3258955516299282, 0.32286970886322863], [0.3032673968803209, 0.30481956460541104, 0.3243692755718748, 0.3301917321468404, 0.3085800517841273, 0.3261291749173827, 0.3168399845888411, 0.31674927451749446, 0.2973547423332914, 0.31800678323872145], [0.3006880657960189, 0.29007628965669274, 0.3030881154538856, 0.2663305810076495, 0.31249392775488594, 0.3211984065670308, 0.31161917424420327, 0.30424307299898634, 0.30051656524610504, 0.30360887463119746], [0.2839954633582106, 0.28771207052544645, 0.26773995330221567, 0.29690751126023207, 0.3016038178128073, 0.29211541009878117, 0.2708779432838655, 0.2987209557558451, 0.29601147698052266, 0.3140807514099534]], 'MSE': [[0.089345130486071789, 0.089485481628775496, 0.082590175007456398, 0.089818364369539813, 0.096954835207966425, 0.095773440379089719, 0.087356079699563466, 0.1050119885045426, 0.089176159336870897, 0.069945115198864066], [0.089304358246954887, 0.096201527969059175, 0.09255599001087314, 0.0933272302301765, 0.090755497164209775, 0.086645471660286008, 0.082189775396350528, 0.091749924002450148, 0.089477286736105319, 0.081983491158273761], [0.090346233831696099, 0.094140285542759147, 0.096892784516284164, 0.091877479886421989, 0.085968333134592553, 0.085609398652795932, 0.093642969429736822, 0.091397842729869436, 0.091558009189706399, 0.089330524327367378], [0.09309976098828654, 0.094416568579833979, 0.097237354276792823, 0.089248588853071145, 0.095480135027383006, 0.096230346271494585, 0.09613233395180415, 0.091739744007507423, 0.092052950369098455, 0.097417958976761324], [0.092527586043420776, 0.091920803502911819, 0.095248451450179475, 0.090542677720429718, 0.094882620131292117, 0.092341274977450663, 0.091175985792617875, 0.094194355763436419, 0.091838530493747725, 0.092511211089207401], [0.098138824781584047, 0.093477477907411954, 0.096211493918092514, 0.095605226563563808, 0.090024549217481251, 0.091828360914744986, 0.094891544312022555, 0.092517927001351169, 0.093183278660585053, 0.092876304160790193], [0.097580514289489412, 0.098427272948462249, 0.091786244229805303, 0.098134926074477491, 0.095010575534995104, 0.096804574168926169, 0.1003042372618395, 0.092087416388007451, 0.095502983718695061, 0.097886353595234843], [0.096438446165120664, 0.093309244687748621, 0.098604911323037336, 0.098380083413842176, 0.097160862711316812, 0.095682156923924178, 0.094898602983694638, 0.095582697969766109, 0.097756498760307203, 0.099174320686700332], [0.096288124816306206, 0.098097001157533864, 0.098870033249358932, 0.099543451843196365, 0.094671551553243194, 0.099349527271032467, 0.099136841478860038, 0.09735776883450338, 0.096777590890502821, 0.096220096745032013], [0.098836004250587639, 0.098089493237958431, 0.10113185225826411, 0.099738083305848338, 0.10025678313116029, 0.10009360689807073, 0.098420738079116554, 0.096948218809805781, 0.09817583761577002, 0.097922757386939827], [0.099943873093766733, 0.10423371962706442, 0.1005773627096177, 0.098774485437605156, 0.09897754887416059, 0.10123301985295695, 0.10135091355721165, 0.10350342263399662, 0.10218614831052948, 0.09829047364334513], [0.10361240428015739, 0.10118071336491585, 0.10325668986877204, 0.1009039001007818, 0.10456182421513739, 0.10124373546825055, 0.10107218768974088, 0.10393032762060637, 0.10270676254431772, 0.10177808376866464], [0.10608829508305469, 0.10447000167445573, 0.10435009255700128, 0.10280451208624515, 0.10847453624182533, 0.10415001644775423, 0.1075863494881077, 0.10355627926598014, 0.10700357854141333, 0.10736565140993402], [0.1114765805150407, 0.11079109927694332, 0.10785997276820886, 0.10665024645728501, 0.11058373558644768, 0.10756336119235893, 0.1085284234848183, 0.10918234112698268, 0.11144312155429413, 0.10884971773300146], [0.11061028756648089, 0.11238034529259668, 0.11031047828256967, 0.11718303385964744, 0.10931152490312908, 0.10779387854094394, 0.10940022778162427, 0.11040414911375244, 0.11190874924599291, 0.11028053251306479], [0.11386162505832415, 0.1136421391441932, 0.11623888079424216, 0.11188427469815365, 0.11085208829500731, 0.11247729604933127, 0.11623771653825132, 0.11164086190644601, 0.1119160721491408, 0.10882470469531512]], 'Rp': [[0.68180100525841902, 0.61957682771001132, 0.68885275785191158, 0.67466472975794578, 0.66871772655147987, 0.65607522033871501, 0.66918015256433994, 0.62881200039059526, 0.66893961443696304, 0.74090320488767736], [0.65414958613128282, 0.63366481130888896, 0.64955646869247596, 0.65835892738810053, 0.66477961778215977, 0.67981294324894048, 0.67987432528827385, 0.66094798697651169, 0.67741154648893032, 0.70108234027379068], [0.66550780965800904, 0.64931723102600847, 0.62965126459685195, 0.64641986963096332, 0.67467233074675925, 0.67690295237945719, 0.65423525495886437, 0.64562055787016415, 0.65247116706442265, 0.6699915166142143], [0.65340790856227449, 0.63224146592953745, 0.64478710776874737, 0.66157573660739488, 0.64196642336366483, 0.63934993436946042, 0.63758037593883443, 0.66250823802606984, 0.65511748192666119, 0.63295703695107441], [0.64416772320135929, 0.64989973641915899, 0.63274733709737829, 0.65802394720973834, 0.6461092582494794, 0.64890090307148296, 0.64999552215545686, 0.64017978048333779, 0.66012120917523054, 0.64254171684253203], [0.62401793442444176, 0.64665305587331046, 0.63212046907233355, 0.6380268342893064, 0.65822851411605787, 0.64862865595033647, 0.64719299498295801, 0.64617216074010608, 0.65047287957602584, 0.64899684370906452], [0.63013295680648773, 0.62195351588708969, 0.65150218438751695, 0.62394017425792714, 0.64014924648072169, 0.62907990373545619, 0.61904847049185907, 0.65173263440728557, 0.6274793281829244, 0.62781067156902071], [0.63535496603138286, 0.64350204830780444, 0.62362527802739942, 0.61927224485532328, 0.62807198638555639, 0.63644314538787716, 0.6379188312289179, 0.63330997452408555, 0.62485968591540908, 0.61992738125985281], [0.63193234726591851, 0.61974047260947562, 0.62329153222290712, 0.62128624485701078, 0.63740177121983166, 0.61943463310715041, 0.62311646402969401, 0.62673929643465776, 0.63586041617147859, 0.63183129313434094], [0.62323262618518649, 0.62536349343497533, 0.60906355741926754, 0.61671817167107501, 0.61986240400722747, 0.61237751391236117, 0.61990786805523612, 0.63106616969917351, 0.62320240966267493, 0.62595151572485574], [0.61697244416472052, 0.59835564118612727, 0.61294981976814411, 0.61926102920544057, 0.61869250243094509, 0.60933565651117116, 0.60782085369991845, 0.60198456956749613, 0.60789955744132185, 0.62620240514618941], [0.59790095480070049, 0.60461758283736533, 0.60180281694910376, 0.60755368335887672, 0.59881815764278457, 0.61018393215404676, 0.61134997680982539, 0.60203633258128741, 0.60500246856706241, 0.60348818475894839], [0.59107818312016946, 0.5946658154277934, 0.59575422798748046, 0.59997700301336976, 0.57808888030936778, 0.59602272432343106, 0.58870111125896196, 0.5998421428734938, 0.58804782387308696, 0.58704305719629613], [0.57158503263975702, 0.57150166435997385, 0.5824398023164763, 0.5844724716862727, 0.57588476769885333, 0.58785893298246372, 0.57793472598276807, 0.5770517930018253, 0.56879387005067605, 0.57745518341241731], [0.57118411666069069, 0.56283519494777812, 0.56895679519630604, 0.55021731830698228, 0.57444942795192622, 0.57923633534821828, 0.57217247111641656, 0.56871436092724492, 0.56389640440964039, 0.57064221684680672], [0.56040920699808772, 0.56074449702300599, 0.55032425831865406, 0.56376145812756118, 0.5694679464989354, 0.5614129368043328, 0.54706436805627912, 0.56348631731749099, 0.56706120585820974, 0.56907985696246155]]} 
average_performances = {'R^2': [0.44405404715391567, 0.44120094171532276, 0.42745343373058703, 0.4118217878378503, 0.41413857127219666, 0.41009114368432015, 0.39317992857357664, 0.3902531560403408, 0.38640105373052547, 0.3781247310080967, 0.36653824603046314, 0.3551330556625837, 0.33499133015649285, 0.3146307980584305, 0.30138630733566557, 0.290976535378788], 'MSE': [0.089545676981874073, 0.089419055257473928, 0.091076386124122985, 0.094305574130203323, 0.092718349696469388, 0.093875498743762747, 0.096352509820993235, 0.096698782562545813, 0.097631198783956943, 0.098961337497352156, 0.10090709677402546, 0.10242466289213448, 0.10558493127957716, 0.10929285996953812, 0.1109583207099802, 0.11275756593284048], 'Rp': [0.66975232397480577, 0.66596385535793545, 0.65647899545457145, 0.64614917094437196, 0.64726871339051539, 0.64405103427339405, 0.63228290862062908, 0.6302285541923609, 0.62706344710524653, 0.6206745729772033, 0.61194744791214739, 0.60427540904600019, 0.59192209693834508, 0.57749782441314834, 0.56823046417120104, 0.56128120519650182]}
'''