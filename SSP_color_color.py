import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

##### BEGIN: User input #####

data_file="Galaxy_NSCs_1Z.dat"
data=pd.DataFrame({'f606w' : [23.09],\
                   'f606w_err' : [0.03],\
                   'f814w' : [22.2],\
                   'f814w_err' : [0.04],\
                   'f160w' : [20.8],\
                   'f160w_err' : [0.05]})
                      
parser = argparse.ArgumentParser()
parser.add_argument("--m606",type=float)
parser.add_argument("--m814",type=float)
parser.add_argument("--m160",type=float)
input=parser.parse_args()
if input.m606 :
    data['f606w']=input.m606
    print(input.m606)
if input.m814 :
    data['f814w']=input.m814
    print(input.m814)
if input.m160 :
    data['f160w']=input.m160
    print(input.m160)

ssp_model_file_Z="bc03/cb07_12/cb07_hr_stelib_m62_kroup_ssp_colnm.dat"
ssp_model_file_0p005Z="bc03/cb07_12/cb07_hr_stelib_m22_kroup_ssp_colnm.dat"
ssp_model_file_0p04Z="bc03/cb07_12/cb07_hr_stelib_m32_kroup_ssp_colnm.dat"
ssp_model_file_0p02Z="bc03/cb07_12/cb07_hr_stelib_m42_kroup_ssp_colnm.dat"
ssp_model_file_0p2Z="bc03/cb07_12/cb07_hr_stelib_m52_kroup_ssp_colnm.dat"
ssp_model_file_2p5Z="bc03/cb07_12/cb07_hr_stelib_m72_kroup_ssp_colnm.dat"

lg_age_limit=9.3; print("SSP for Age > {:.2f}".format(1e-9*10**(lg_age_limit)))
DM=34.9              # Distance modulus
M_sun_f606w=4.66    # Abs magnitude of the Sun
ssp_Z_lum_scale=1e6   # Scale the SSP model to 1e6 L_sun
##### END: User input #####

ssp_model_Z = pd.read_table(ssp_model_file_Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])
ssp_model_0p005Z = pd.read_table(ssp_model_file_0p005Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])
ssp_model_0p02Z = pd.read_table(ssp_model_file_0p02Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])
ssp_model_0p04Z = pd.read_table(ssp_model_file_0p04Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])
ssp_model_0p2Z = pd.read_table(ssp_model_file_0p2Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])
ssp_model_2p5Z = pd.read_table(ssp_model_file_2p5Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,95,109,114])

select=ssp_model_Z['log_age_yr']>=lg_age_limit; select=ssp_model_0p005Z['log_age_yr']>=lg_age_limit; select=ssp_model_2p5Z['log_age_yr']>=lg_age_limit; 

ssp_model_Z['mag_Z']=ssp_model_Z['Vmag'][select]-ssp_model_Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_Z['M_star_tot_to_Lv'][select])) +DM
ssp_model_Z['606m814']=ssp_model_Z['V_m_F814w_uvis'][select]-ssp_model_Z['V_m_F606w_uvis'][select]
ssp_model_Z['606m160']=ssp_model_Z['V_m_F160w_wfc3'][select]-ssp_model_Z['V_m_F606w_uvis'][select]
mag_0p005Z=ssp_model_0p005Z['Vmag'][select]-ssp_model_0p005Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p005Z['M_star_tot_to_Lv'][select]))+DM
ssp_model_0p005Z['606m814']=ssp_model_0p005Z['V_m_F814w_uvis'][select]-ssp_model_0p005Z['V_m_F606w_uvis'][select]
ssp_model_0p005Z['606m160']=ssp_model_0p005Z['V_m_F160w_wfc3'][select]-ssp_model_0p005Z['V_m_F606w_uvis'][select]
mag_0p02Z=ssp_model_0p02Z['Vmag'][select]-ssp_model_0p02Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p02Z['M_star_tot_to_Lv'][select]))+DM
ssp_model_0p02Z['606m814']=ssp_model_0p02Z['V_m_F814w_uvis'][select]-ssp_model_0p02Z['V_m_F606w_uvis'][select]
ssp_model_0p02Z['606m160']=ssp_model_0p02Z['V_m_F160w_wfc3'][select]-ssp_model_0p02Z['V_m_F606w_uvis'][select]
mag_0p04Z=ssp_model_0p04Z['Vmag'][select]-ssp_model_0p04Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p04Z['M_star_tot_to_Lv'][select]))+DM
ssp_model_0p04Z['606m814']=ssp_model_0p04Z['V_m_F814w_uvis'][select]-ssp_model_0p04Z['V_m_F606w_uvis'][select]
ssp_model_0p04Z['606m160']=ssp_model_0p04Z['V_m_F160w_wfc3'][select]-ssp_model_0p04Z['V_m_F606w_uvis'][select]
mag_0p2Z=ssp_model_0p2Z['Vmag'][select]-ssp_model_0p2Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p2Z['M_star_tot_to_Lv'][select]))+DM
ssp_model_0p2Z['606m814']=ssp_model_0p2Z['V_m_F814w_uvis'][select]-ssp_model_0p2Z['V_m_F606w_uvis'][select]
ssp_model_0p2Z['606m160']=ssp_model_0p2Z['V_m_F160w_wfc3'][select]-ssp_model_0p2Z['V_m_F606w_uvis'][select]
mag_2p5Z=ssp_model_2p5Z['Vmag'][select]-ssp_model_2p5Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_2p5Z['M_star_tot_to_Lv'][select]))+DM
ssp_model_2p5Z['606m814']=ssp_model_2p5Z['V_m_F814w_uvis'][select]-ssp_model_2p5Z['V_m_F606w_uvis'][select]
ssp_model_2p5Z['606m160']=ssp_model_2p5Z['V_m_F160w_wfc3'][select]-ssp_model_2p5Z['V_m_F606w_uvis'][select]

# Figure formatting entire session
plt.figure(facecolor='white', figsize=plt.figaspect(0.65)); plt.rcParams["axes.formatter.useoffset"]=False
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.size"] = 11
plt.rcParams["xtick.top"] = True; plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 5; plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 2; plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.direction"] = "in"; plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True; plt.rcParams["ytick.minor.visible"] = True

mxtics = AutoMinorLocator(4)
mytics = AutoMinorLocator(4)
plt.axes().xaxis.set_minor_locator(mxtics)
plt.axes().yaxis.set_minor_locator(mytics)

plt.xlabel("F606W-F160W [mag]")
plt.ylabel("F606W-F814W [mag]")
#plt.ylim(0,1.8) # plt.gca().invert_yaxis()
#plt.xlim(0,2.8)
       
plt.plot(ssp_model_Z['606m160'],ssp_model_Z['606m814'], color='red', linestyle='-', label='1Z')
plt.plot(ssp_model_0p005Z['606m160'],ssp_model_0p005Z['606m814'], color='blue', linestyle='-', label='0.005Z')
plt.plot(ssp_model_0p02Z['606m160'],ssp_model_0p02Z['606m814'], color='darkblue', linestyle='-', label='0.02Z')
#plt.plot(color2_0p04Z,color1_0p04Z, color='royalblue', linestyle='-')
plt.plot(ssp_model_0p2Z['606m160'],ssp_model_0p2Z['606m814'], color='darkorange', linestyle='-', label='0.2Z')
plt.plot(ssp_model_2p5Z['606m160'],ssp_model_2p5Z['606m814'], color='brown', linestyle='-', label='2.5Z')
plt.errorbar(data.f606w-data.f160w,data.f606w-data.f814w,\
             xerr=np.sqrt(data.f606w_err**2+data.f160w_err**2),\
             yerr=np.sqrt(data.f606w_err**2+data.f814w_err**2),marker='.')

temp = pd.DataFrame(columns=['ml_col1'])
for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
                                       (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
                                       (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
    for mod_col1,mod_col2,mod_col3 in zip(ssp_model_0p2Z['606m160'],ssp_model_0p2Z['606m814'],ssp_model_0p2Z['V_m_F160w_wfc3']-ssp_model_0p2Z['V_m_F814w_uvis'],) :
        a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
        b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
        c=np.log(a*b)
        temp = temp.append({'ml_col1': c}, ignore_index=True)

ssp_model_0p2Z['ml_col1']=temp
age=pd.to_numeric(1e-9*10**(ssp_model_0p2Z['log_age_yr'][(ssp_model_0p2Z['ml_col1']==np.max(temp['ml_col1']))]))
m_to_l=pd.to_numeric(ssp_model_0p2Z['M_star_tot_to_Lv'][(ssp_model_0p2Z['ml_col1']==np.max(temp['ml_col1']))])

#for a,m in zip(age,m_to_l):
#    print(a,m); k='$M/L_V$ = {:.3g}'.format(m) + "; Age = {:.3g} Gyr".format(a)+''
#    plt.annotate(k,xy=(1.6,1))
print(data)

plt.legend(loc=4,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

plt.tight_layout()

plt.savefig("test.pdf")

plt.show()