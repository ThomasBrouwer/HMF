"""
Test the performance of HMF for recovering the CTRP dataset, where we vary the 
fraction of entries that are missing.
We repeat this 10 times per fraction and average that.
This is the real-valued D-MTF version with column-wise posterior draws.
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
    'priorF'  : 'normal',
    'orderG'  : 'normal',
    'priorSn' : 'normal',
    'priorSm' : 'normal',
    'orderF'  : 'rows',
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
alpha_n = [1., 1., 1., 1.] # GDSC, CTRP, CCLE IC, CCLE EC


''' Load data '''
location = project_location+"HMF/drug_sensitivity/data/overlap/"
location_data = location+"data_row_01/"

R_ctrp,     M_ctrp,   cell_lines, drugs   = load_data_without_empty(location_data+"ctrp_ec50_row_01.txt")
R_ccle_ec,  M_ccle_ec                     = load_data_filter(location_data+"ccle_ec50_row_01.txt",cell_lines,drugs)
R_gdsc,     M_gdsc                        = load_data_filter(location_data+"gdsc_ic50_row_01.txt",cell_lines,drugs)
R_ccle_ic,  M_ccle_ic                     = load_data_filter(location_data+"ccle_ic50_row_01.txt",cell_lines,drugs)


#''' Seed all of the methods the same '''
#numpy.random.seed(0)
#random.seed(0)

''' Generate matrices M - one list of (M_train,M_test)'s for each fraction '''
M_attempts = 10000
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
     
        R = [(R_ctrp,    M_train,   'Cell_lines', 'Drugs', alpha_n[1]),
             (R_gdsc,    M_gdsc,    'Cell_lines', 'Drugs', alpha_n[0]),  
             (R_ccle_ic, M_ccle_ic, 'Cell_lines', 'Drugs', alpha_n[2]),
             (R_ccle_ec, M_ccle_ec, 'Cell_lines', 'Drugs', alpha_n[3])]
        C, D = [], []

        HMF = HMF_Gibbs(R,C,D,K,settings,hyperparameters)
        HMF.initialise(init)
        HMF.run(iterations)
        
        # Measure the performances
        performances = HMF.predict_Rn(n=0,M_pred=M_test,burn_in=burn_in,thinning=thinning)
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
all_performances = {'R^2': [[0.43123999002768065, 0.4775732422869503, 0.556513566560351, 0.556030148667487, 0.41608940339893896, 0.5199051332533658, 0.5258089404345638, 0.4715184695845248, 0.3375968386692494, 0.6029913843442687], [0.410533004838871, 0.4714662712313683, 0.4986651197591996, 0.48801994973836305, 0.4410728870915943, 0.47231348481467417, 0.385937242577271, 0.4593995128359377, 0.46245707112250667, 0.39913841577524745], [0.42891596120292974, 0.4394041997950173, 0.48469722258574854, 0.43937854955444877, 0.4504668604772152, 0.45296173424590336, 0.4433353118466049, 0.45566718881059054, 0.4399539145099589, 0.4340645919308841], [0.4264999661862894, 0.439540801405231, 0.4536079328340904, 0.4297086051566008, 0.4390551298555937, 0.43872954145262066, 0.4182264550990724, 0.44339275256261823, 0.4312610173458865, 0.4365097388498128], [0.4510662682903377, 0.4609080731315156, 0.43567498212524114, 0.3999733075399763, 0.44581647457828266, 0.45135919345515496, 0.4408188559932976, 0.43487459636966375, 0.44317990949770913, 0.43682738662042475], [0.4280353815823901, 0.43249178684219103, 0.436865164928506, 0.43311031861406313, 0.4473469067901401, 0.4201932958097577, 0.4346648173270411, 0.42363284073337504, 0.40394014290436653, 0.4383080050166598], [0.4046568401422983, 0.41385244612921646, 0.422102521007942, 0.42314468543634165, 0.42312258211099396, 0.4285621228191391, 0.4189289736269166, 0.4263420659269691, 0.4236273077463777, 0.4207699747870325], [0.42827767302454756, 0.43879871232541723, 0.4094872074007485, 0.4152771575770928, 0.41629119125411485, 0.42413095588386307, 0.42872015338833125, 0.42250434778933144, 0.39514835182484565, 0.4209731611573768], [0.4162070406461865, 0.4084457058960699, 0.4220235460193449, 0.42108744523222597, 0.42270666867737583, 0.40275656253943737, 0.39501159452071954, 0.41237211749055536, 0.4068422560622674, 0.4166233987931418], [0.41542857225815766, 0.3988997899242678, 0.40419611185138327, 0.41644433779619094, 0.40580215235852435, 0.40796925151854524, 0.4072739532524874, 0.41713251444848376, 0.40407467003619035, 0.40780225354526634], [0.40062816526339906, 0.4051118526470766, 0.3969238582391401, 0.4150339256999931, 0.3930307680220072, 0.401362410837516, 0.41602123621615794, 0.40558050536996837, 0.4130761952397026, 0.39391621060512516], [0.39360666959052326, 0.3741029464150102, 0.3995605349558562, 0.39205770198043877, 0.396542426227229, 0.38816696805095996, 0.39488986426148986, 0.3772702301257529, 0.4035578769359939, 0.39325435185529245], [0.37286704988335984, 0.3825228385166146, 0.37015543827442066, 0.3847823479927467, 0.3772468484555458, 0.3957786242583009, 0.3600089802583477, 0.36910014395756063, 0.3960221644931471, 0.36914074134506847], [0.36687171214460235, 0.365550707447831, 0.3871885117947337, 0.3679146280773844, 0.34557048942828483, 0.37143404345539066, 0.35083708916081546, 0.34284292503938296, 0.3582657084702986, 0.38404457839097195], [0.34557729085050326, 0.3599231420877296, 0.35650430116848564, 0.35603676329226097, 0.3274105448452551, 0.3566379320026366, 0.3448916969765685, 0.3502325702348761, 0.3012214677114555, 0.3549269426442391], [0.2864674120784567, 0.3206444141861412, 0.3399618608232122, 0.3404870776782628, 0.3342507606328584, 0.30069851829254235, 0.32019276778799066, 0.2789411867549145, 0.3197187347526338, 0.29567344498038406]], 'MSE': [[0.097662621061420152, 0.079203286869448483, 0.073738258954148761, 0.071608989543144222, 0.094997459949246271, 0.071113333556058034, 0.074958880325313074, 0.089155619602073799, 0.10023445960737108, 0.065101310485405586], [0.095878781717518624, 0.082490921759598379, 0.08051050939270267, 0.079752121365435358, 0.085096358310581391, 0.079880028327786221, 0.098490847939055867, 0.085732065296322935, 0.087460115701123636, 0.095621608757425786], [0.087079748290594172, 0.088266178395515774, 0.079907638531463779, 0.090031239204315799, 0.08950946921401845, 0.084567699357312481, 0.088409674712404254, 0.085904162172028314, 0.087265148699135042, 0.088189637367316179], [0.090546360484799623, 0.090852879825476893, 0.085212511087581225, 0.091033665181136716, 0.088442056213235795, 0.088533373678191876, 0.091985103672522897, 0.087379304462582089, 0.091272398232766769, 0.087487706105317323], [0.086619608789935507, 0.086958089202386596, 0.090999114218097948, 0.096519375200950289, 0.087253719715298286, 0.085946676404652228, 0.090471364337354185, 0.0904917179366415, 0.090834997703201206, 0.090170982314959583], [0.089636155771276801, 0.089872101656236839, 0.08939098648733948, 0.09053305310986326, 0.087406794503322699, 0.091932710729608846, 0.090498203568630309, 0.090857242854865794, 0.094117660935174777, 0.088981840690496083], [0.095496526887722977, 0.094749149598750151, 0.091235074480100001, 0.090473717572426129, 0.091539869820475836, 0.091982587775428865, 0.092050898370913667, 0.091469675357603183, 0.089761640295238754, 0.093204845872631295], [0.090765785614909264, 0.089688360002823458, 0.093768865667092235, 0.093916427198401334, 0.092562336744484094, 0.092480443506491816, 0.09141314855564546, 0.091857627000334285, 0.096467622167694334, 0.090991471381169739], [0.092973553038017243, 0.09392592100805211, 0.091934423387968797, 0.091628084674067453, 0.091778293675554137, 0.094533243142375448, 0.094890512208122293, 0.092601783232006588, 0.093760987566311044, 0.092573245955369005], [0.09406913927701499, 0.095561582430588293, 0.09524044600283757, 0.092885053932272907, 0.094501571004248025, 0.093544699024210698, 0.093533651530530698, 0.093186264571913022, 0.095396976844356593, 0.09361696284218074], [0.09643011354097672, 0.095231331701185792, 0.094819226719751196, 0.093621707183964356, 0.096175892031892016, 0.094568340652968208, 0.093159241283755009, 0.094020213738235325, 0.09282110178944758, 0.095019647954630526], [0.096298468957230599, 0.098930122714277327, 0.095052092467512367, 0.096703110357061639, 0.096171225679998154, 0.098268979341349635, 0.096412348515952609, 0.098538564254432984, 0.094709799871219261, 0.096839000355609897], [0.099968539829210945, 0.098382718677793454, 0.10011383572763091, 0.09778793131644975, 0.098542122453348785, 0.095966128817267987, 0.10138433205573834, 0.099702471758078254, 0.096634047580891044, 0.10007946346213976], [0.10063085509035959, 0.10134181828583377, 0.097523987429226064, 0.10038784321887416, 0.10336662381022187, 0.10009370023968754, 0.10373287885925649, 0.10450704194555102, 0.10230874097467958, 0.097947368563280318], [0.10425964390141108, 0.1018786579462185, 0.10275269555398873, 0.10247248025125802, 0.10677847530671347, 0.10237382602426331, 0.10441820068141443, 0.10336891226062349, 0.11155182230979467, 0.10268076119554248], [0.113850895392042, 0.10808926622311184, 0.1047575839747087, 0.10507775346488969, 0.10591583694620706, 0.11092743295261608, 0.10794302482139273, 0.11461267497357054, 0.10839684322674328, 0.11244478321764766]], 'Rp': [[0.65967428920187965, 0.69156632327871514, 0.74650644714363168, 0.74568713737649339, 0.6472285109278072, 0.72198349327128408, 0.72722903901844849, 0.68768831445913825, 0.58813311762333931, 0.77783734489499456], [0.64272209381423318, 0.68734092106103939, 0.70750935317187602, 0.69868546941325405, 0.66645189049746212, 0.6878140594387665, 0.62384914112344114, 0.68098996992954952, 0.68033865003939276, 0.63320716966652812], [0.65777536651079194, 0.6650293641442494, 0.69631566484298046, 0.66435727165655933, 0.67152926195455653, 0.67399300939996154, 0.66660281389229259, 0.67568036460026293, 0.66592802659659167, 0.66174021503302538], [0.65530498225679323, 0.66404267655946447, 0.67480405255305631, 0.6575011575325882, 0.66394585246500204, 0.66438445244616451, 0.64757986687937719, 0.66725600506801852, 0.65741359509868258, 0.66265907620250175], [0.67185411075800261, 0.67938230172704328, 0.66282351344906287, 0.63573708589433531, 0.66885038338962977, 0.67222384078600017, 0.66498440271232684, 0.66022132952129298, 0.66634811142655637, 0.66173582854293933], [0.65637201749526741, 0.65979462230596175, 0.66180152553741389, 0.65938150709907084, 0.6690446643710678, 0.64960111944298171, 0.66015877024210412, 0.65238138766002485, 0.63851651827138367, 0.66242569117169803], [0.63976210956880653, 0.64537540542394112, 0.65110074667908713, 0.65237902138700676, 0.6523635829552541, 0.65601799014799822, 0.64840489810448432, 0.65403826111400876, 0.651896595446816, 0.65110644791061356], [0.65517685418063221, 0.66331782121456218, 0.64168754010507378, 0.64666411074775032, 0.6485871206085001, 0.65271031060789675, 0.65547497390322551, 0.65205215151056373, 0.63227893520756306, 0.65006601601484371], [0.64643985631638601, 0.64150261180072066, 0.65102974800545943, 0.64967202716209038, 0.65138649493358003, 0.63737973871970688, 0.63242647316694589, 0.64400075372683152, 0.64099236907339863, 0.64672717010064118], [0.64702326350684169, 0.63501707779927796, 0.63870580136222943, 0.6466328744933092, 0.63943803947276023, 0.64129022130553759, 0.63975073344354272, 0.64703213011805061, 0.63858990384580883, 0.64173148464260577], [0.63431811156353546, 0.63736902033030918, 0.63333300526655656, 0.64524192763173815, 0.63031313957268353, 0.63681717418312256, 0.64613844477533355, 0.63929649184047033, 0.64386685818737577, 0.63204218712042892], [0.63053094096148776, 0.61669411162335508, 0.63536688641232042, 0.63015030558055862, 0.63099019714677163, 0.62584488080971601, 0.6318692851947636, 0.62111923424611293, 0.63686420656157272, 0.63042298458113011], [0.61474571401837641, 0.62202302933116749, 0.61381798012303301, 0.62446594002552669, 0.61961878975989371, 0.63097348363483297, 0.60580466295398194, 0.61497946515448287, 0.63153672216225765, 0.61490076544459027], [0.61230145332922192, 0.60889146733762367, 0.62361526362515507, 0.61128889558511668, 0.59764482335756941, 0.61260237440565124, 0.59921235582381349, 0.59512128736896552, 0.6035185659360659, 0.62135530829894259], [0.5946273104723504, 0.60245897138464433, 0.60169029621886394, 0.60148221680670677, 0.58499952220092533, 0.60106453731056253, 0.59451142301678117, 0.59623763437242339, 0.57382927999470723, 0.60115938106211464], [0.5676997642043432, 0.57872945524372565, 0.58813063652062714, 0.58944762083932523, 0.58721425394199278, 0.57244463614773966, 0.57873323854796377, 0.55631954013130736, 0.57846645711631473, 0.56349627783446032]]} 
average_performances = {'R^2': [0.48952671172273804, 0.4489002959785033, 0.44688455349593015, 0.4356531940747816, 0.44004990476016037, 0.429858866054849, 0.4205109519733227, 0.41996089116256685, 0.41240763358773247, 0.40850236069894963, 0.4040685128140086, 0.3913009570398547, 0.3777625177435112, 0.3640520393409696, 0.34533626518140104, 0.31370361779673966], 'MSE': [0.081777421995362948, 0.087091335856755075, 0.086913059594410425, 0.089274535894361129, 0.089626564582347718, 0.090322675030681476, 0.092196398603129087, 0.092391208783904583, 0.093060004788784415, 0.094153634746015347, 0.094586681659680677, 0.096792371251464449, 0.098856159167854923, 0.10118408584169705, 0.10425354754312281, 0.10920160951929296], 'Rp': [0.69935340171957328, 0.67089087181555418, 0.6698951358631271, 0.66148917170616484, 0.66441609082071884, 0.65694778235969742, 0.65024450587380167, 0.649801583410061, 0.64415572430057599, 0.64152115299899637, 0.63787363604715541, 0.62898530331177871, 0.61928665526081428, 0.60855517950681259, 0.59520605728400799, 0.57606818805278004]}
'''