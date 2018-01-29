import argparse
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

##### BEGIN: User input #####

data_file="Galaxy_NSCs_1Z.dat"
data=pd.DataFrame({'f606w' : [25.09],\
                   'f606w_err' : [0.03],\
                   'f814w' : [24.2],\
                   'f814w_err' : [0.04],\
                   'f160w' : [22.8],\
                   'f160w_err' : [0.05]})
                      
ssp_model_file_0p005Z="bc03/cb07_12/cb07_hr_stelib_m22_kroup_ssp_colnm.dat"
ssp_model_file_0p02Z="bc03/cb07_12/cb07_hr_stelib_m32_kroup_ssp_colnm.dat"
ssp_model_file_0p04Z="bc03/cb07_12/cb07_hr_stelib_m42_kroup_ssp_colnm.dat"
ssp_model_file_0p2Z="bc03/cb07_12/cb07_hr_stelib_m52_kroup_ssp_colnm.dat"
ssp_model_file_Z="bc03/cb07_12/cb07_hr_stelib_m62_kroup_ssp_colnm.dat"
ssp_model_file_2p5Z="bc03/cb07_12/cb07_hr_stelib_m72_kroup_ssp_colnm.dat"

lg_age_limit=9.3
DM=34.9              # Distance modulus
M_sun_f606w=4.66    # Abs magnitude of the Sun
ssp_lum_scale=1e6   # Scale the SSP model to 1e6 L_sun

parser = argparse.ArgumentParser()
parser.add_argument("--m606",type=float)
parser.add_argument("--m814",type=float)
parser.add_argument("--m160",type=float)
parser.add_argument("--lg_age_lim",type=float)
parser.add_argument("--mM",type=float)
parser.add_argument("--M_sun_f606w",type=float)
parser.add_argument("--ssp_lum_scale",type=float)
input=parser.parse_args()
if input.m606 :
    data['f606w']=input.m606
if input.m814 :
    data['f814w']=input.m814
if input.m160 :
    data['f160w']=input.m160
if input.lg_age_lim :
    lg_age_limit=input.lg_age_lim
if input.mM :
    DM=input.mM
if input.M_sun_f606w :
    M_sun_f606w=input.M_sun_f606w
if input.m160 :
    ssp_lum_scale=input.ssp_lum_scale

##### END: User input #####

print("SSP models for Age > {:.2f}".format(1e-9*10**(lg_age_limit)))
print(data)

ssp_model_0p005Z = pd.read_table(ssp_model_file_0p005Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])
ssp_model_0p02Z = pd.read_table(ssp_model_file_0p02Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])
ssp_model_0p04Z = pd.read_table(ssp_model_file_0p04Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])
ssp_model_0p2Z = pd.read_table(ssp_model_file_0p2Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])
ssp_model_Z = pd.read_table(ssp_model_file_Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])
ssp_model_2p5Z = pd.read_table(ssp_model_file_2p5Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv', 'M_star_liv_to_Lv','V_m_F160w_wfc3', 'V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,61,95,109,114])

select_0p005Z=ssp_model_0p005Z['log_age_yr']>=lg_age_limit
select_0p02Z=ssp_model_0p02Z['log_age_yr']>=lg_age_limit
select_0p04Z=ssp_model_0p04Z['log_age_yr']>=lg_age_limit
select_0p2Z=ssp_model_0p2Z['log_age_yr']>=lg_age_limit
select_Z=ssp_model_Z['log_age_yr']>=lg_age_limit
select_2p5Z=ssp_model_2p5Z['log_age_yr']>=lg_age_limit

for ssp_model,select_age in zip([ssp_model_0p005Z,ssp_model_0p02Z,ssp_model_0p04Z,ssp_model_0p2Z,ssp_model_Z,ssp_model_2p5Z],\
                                [select_0p005Z,select_0p02Z,select_0p04Z,select_0p2Z,select_Z,select_2p5Z]):
    ssp_model['606m814']=ssp_model['V_m_F814w_uvis'][select_age]-ssp_model['V_m_F606w_uvis'][select_age]
    ssp_model['606m160']=ssp_model['V_m_F160w_wfc3'][select_age]-ssp_model['V_m_F606w_uvis'][select_age]
    ssp_model['814m160']=ssp_model['V_m_F160w_wfc3'][select_age]-ssp_model['V_m_F814w_uvis'][select_age]
    # Remove inf and nans
    all_inf_or_nan = ssp_model.isin([np.inf, -np.inf, np.nan]).all(axis='columns')
    ssp_model=ssp_model[~all_inf_or_nan]

# Old
#mag_0p005Z=ssp_model_0p005Z['Vmag'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_0p005Z['M_star_tot_to_Lv'][select_0p005Z]))+DM
#ssp_model_0p005Z['606m814']=ssp_model_0p005Z['V_m_F814w_uvis'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z]
#ssp_model_0p005Z['606m160']=ssp_model_0p005Z['V_m_F160w_wfc3'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z]


# Figure formatting entire session
plt.rcParams["axes.formatter.useoffset"]=False
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.size"] = 11
plt.rcParams["xtick.top"] = True; plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 5; plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 2; plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.direction"] = "in"; plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True; plt.rcParams["ytick.minor.visible"] = True

#plt.figure(1,facecolor='white', figsize=plt.figaspect(.65))
fig = plt.figure(facecolor='white', figsize=(12,6))
ax1 = fig.add_axes([0.045,0.07,.5,.92])
mxtics = AutoMinorLocator(4)
mytics = AutoMinorLocator(4)
ax1.xaxis.set_minor_locator(mxtics)
ax1.yaxis.set_minor_locator(mytics)

ax1.set_xlabel("F606W-F160W [mag]")
ax1.set_ylabel("F606W-F814W [mag]")

# SSP constant age lines
#age_low=np.log10(1.99e9); age_up=np.log10(2.01e9)

#m606m160_2Gyr=pd.DataFrame(); m606m814_2Gyr=pd.DataFrame()
#m606m160_5Gyr=pd.DataFrame(); m606m814_5Gyr=pd.DataFrame()
#m606m160_14Gyr=pd.DataFrame(); m606m814_14Gyr=pd.DataFrame()
m606m160_2Gyr=[]; m606m814_2Gyr=[]
m606m160_5Gyr=[]; m606m814_5Gyr=[]
m606m160_14Gyr=[]; m606m814_14Gyr=[]
for age_low,age_up,ssp_const_age_col1,ssp_const_age_col2 in zip([np.log10(1.99e9),np.log10(4.99e9),np.log10(13.99e9)],\
                                                                [np.log10(2.01e9),np.log10(5.01e9),np.log10(14.01e9)],\
                                                                [m606m160_2Gyr,m606m160_5Gyr,m606m160_14Gyr],\
                                                                [m606m814_2Gyr,m606m814_5Gyr,m606m814_14Gyr]):
    ssp_const_age_col1 += [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m160'].values[0],\
                          ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m160'].values[0],\
                          ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m160'].values[0],\
                          ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m160'].values[0],\
                          ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m160'].values[0]]
    ssp_const_age_col2 += [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m814'].values[0],\
                          ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m814'].values[0],\
                          ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m814'].values[0],\
                          ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m814'].values[0],\
                          ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m814'].values[0]]

# OLD
#m606m160_2Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m160'].values[0],\
#ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m160'].values[0],\
#ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m160'].values[0],\
#ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m160'].values[0],\
#ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m160'].values[0]]
#m606m814_2Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m814'].values[0],\
#ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m814'].values[0],\
#ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m814'].values[0],\
#ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m814'].values[0],\
#ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m814'].values[0]]


ax1.plot(m606m160_2Gyr,m606m814_2Gyr,linestyle='--',color='gray',zorder=0)
ax1.plot(m606m160_5Gyr,m606m814_5Gyr,linestyle='--',color='gray',zorder=0)
ax1.plot(m606m160_14Gyr,m606m814_14Gyr,linestyle='--',color='gray',zorder=0)

ax1.plot(ssp_model_0p005Z['606m160'],ssp_model_0p005Z['606m814'], color='darkblue', linestyle='-', label='0.005Z$_\odot$',zorder=0)
#ax1.plot(ssp_model_0p02Z['606m160'],ssp_model_0p02Z['606m814'], color='darkblue', linestyle='-', label='0.02Z$_\odot$',zorder=0)
ax1.plot(ssp_model_0p04Z['606m160'],ssp_model_0p04Z['606m814'], color='blue', linestyle='-', label='0.04Z$_\odot$',zorder=0)
ax1.plot(ssp_model_0p2Z['606m160'],ssp_model_0p2Z['606m814'], color='darkorange', linestyle='-', label='0.2Z$_\odot$',zorder=0)
ax1.plot(ssp_model_Z['606m160'],ssp_model_Z['606m814'], color='red', linestyle='-', label='1Z')
ax1.plot(ssp_model_2p5Z['606m160'],ssp_model_2p5Z['606m814'], color='brown', linestyle='-', label='2.5Z$_\odot$',zorder=0)
ax1.errorbar(data.f606w-data.f160w,data.f606w-data.f814w,\
             xerr=np.sqrt(data.f606w_err**2+data.f160w_err**2),\
             yerr=np.sqrt(data.f606w_err**2+data.f814w_err**2),marker='.',color='gray',zorder=1)
ax1.plot(data.f606w-data.f160w,data.f606w-data.f814w,marker='o',color='black',zorder=2)

# Calculate the maximum likelihood value for each model
for ssp_model,likelihood_column_name,max_ml_name,ml_age_name,le_ml_age_name,ue_ml_age_name,\
    m_to_l_name,le_m_to_l_name,ue_m_to_l_name,mass_name,le_mass_name,ue_mass_name \
    in zip([ssp_model_0p005Z,ssp_model_0p02Z,ssp_model_0p04Z,ssp_model_0p2Z,ssp_model_Z,ssp_model_2p5Z],\
           ['ml_mod0p005Z','ml_mod0p02Z','ml_mod0p04Z','ml_mod0p2Z','ml_modZ','ml_mod2p5Z'],\
           ['max_ml_0p005Z','max_ml_0p02Z','max_ml_0p04Z','max_ml_0p2Z','max_ml_Z','max_ml_2p5Z'],\
           ['age_0p005Z','age_0p02Z','age_0p04Z','age_0p2Z','age_Z','age_2p5Z'],\
           ['le_age_0p005Z','le_age_0p02Z','le_age_0p04Z','le_age_0p2Z','le_age_Z','age_2p5Z'],\
           ['ue_age_0p005Z','ue_age_0p02Z','ue_age_0p04Z','ue_age_0p2Z','ue_age_Z','ue_age_2p5Z'],\
           ['m_to_l_0p005Z','m_to_l_0p02Z','m_to_l_0p04Z','m_to_l_0p2Z','m_to_l_Z','m_to_l_2p5Z'],\
           ['le_m_to_l_0p005Z','le_m_to_l_0p02Z','le_m_to_l_0p04Z','le_m_to_l_0p2Z','le_m_to_l_Z','m_to_l_2p5Z'],\
           ['ue_m_to_l_0p005Z','ue_m_to_l_0p02Z','ue_m_to_l_0p04Z','ue_m_to_l_0p2Z','ue_m_to_l_Z','ue_m_to_l_2p5Z'],\
           ['mass_0p005Z','mass_0p02Z','mass_0p04Z','mass_0p2Z','mass_Z','mass_2p5Z'],\
           ['le_mass_0p005Z','le_mass_0p02Z','le_mass_0p04Z','le_mass_0p2Z','le_mass_Z','mass_2p5Z'],\
           ['ue_mass_0p005Z','ue_mass_0p02Z','ue_mass_0p04Z','ue_mass_0p2Z','ue_mass_Z','ue_mass_2p5Z']):
        temp = pd.DataFrame(columns=[likelihood_column_name])
        for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
                                                             (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
                                                             (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
            for mod_col1,mod_col2,mod_col3 in zip(ssp_model['606m160'],ssp_model['606m814'],ssp_model['V_m_F160w_wfc3']-ssp_model['V_m_F814w_uvis']) :
                a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
                b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
                c=np.log(a*b)
                temp = temp.append({likelihood_column_name: c}, ignore_index=True)
        ssp_model[likelihood_column_name]=temp; del temp
        all_inf_or_nan = ssp_model.isin([np.inf, -np.inf, np.nan]).all(axis='columns')
        ssp_model=ssp_model[~all_inf_or_nan]

        max_ml=ssp_model[likelihood_column_name].max() ; data[max_ml_name]=data['f606w']*0. + max_ml
        ml_age=((ssp_model['log_age_yr'][ssp_model[likelihood_column_name]==max_ml])).max(); data[ml_age_name]=data['f606w']*0 + ml_age
        q1=ssp_model['log_age_yr'][(ssp_model['log_age_yr']>=ml_age-ssp_model['log_age_yr'].std()*.5)\
                         & (ssp_model['log_age_yr']<=ml_age+ssp_model['log_age_yr'].std()*.5)].quantile(.3255)
        q3=ssp_model['log_age_yr'][(ssp_model['log_age_yr']>=ml_age-ssp_model['log_age_yr'].std()*.5) \
                         & (ssp_model['log_age_yr']<=ml_age+ssp_model['log_age_yr'].std()*.5)].quantile(.675)

        min_age=(1e-9*10**(ssp_model['log_age_yr'][(ssp_model['log_age_yr']>=q1) & (ssp_model['log_age_yr']<=q3)])).min() # [(ssp_model[likelihood_column_name]>=max_ml-q1) & (ssp_model[likelihood_column_name]<=max_ml+q3)])).quantile(.3255)
        max_age=(1e-9*10**(ssp_model['log_age_yr'][(ssp_model['log_age_yr']>=q1) & (ssp_model['log_age_yr']<=q3)])).max() # [(ssp_model[likelihood_column_name]>=max_ml-q1) & (ssp_model[likelihood_column_name]<=max_ml+q3)])).quantile(.6745)
        ml_age=1e-9*10**ml_age
        if ml_age-min_age<0:
            min_age=1e-9*10**(lg_age_limit)

        le_ml_age=ml_age-min_age ; data[le_ml_age_name]=data['f606w']*0 + le_ml_age
        ue_ml_age=max_age-ml_age ; data[ue_ml_age_name]=data['f606w']*0 + ue_ml_age

        ml_m_to_l=((ssp_model['M_star_tot_to_Lv'][ssp_model[likelihood_column_name]==max_ml])).max() ; data[m_to_l_name]=data['f606w']*0 + ml_m_to_l
        q1=ssp_model['M_star_tot_to_Lv'][(ssp_model['M_star_tot_to_Lv']>=ml_m_to_l-ssp_model['M_star_tot_to_Lv'].std()*.5)\
                         & (ssp_model['M_star_tot_to_Lv']<=ml_m_to_l+ssp_model['M_star_tot_to_Lv'].std()*.5)].quantile(.3255)
        q3=ssp_model['M_star_tot_to_Lv'][(ssp_model['M_star_tot_to_Lv']>=ml_m_to_l-ssp_model['M_star_tot_to_Lv'].std()*.5) \
                         & (ssp_model['M_star_tot_to_Lv']<=ml_m_to_l+ssp_model['M_star_tot_to_Lv'].std()*.5)].quantile(.675)
        
        min_m_to_l=(ssp_model['M_star_tot_to_Lv'][(ssp_model['M_star_tot_to_Lv']>=q1) & (ssp_model['M_star_tot_to_Lv']<=q3)]).min()
        max_m_to_l=(ssp_model['M_star_tot_to_Lv'][(ssp_model['M_star_tot_to_Lv']>=q1) & (ssp_model['M_star_tot_to_Lv']<=q3)]).max()

        le_ml_m_to_l=ml_m_to_l-min_m_to_l ; data[le_m_to_l_name]=data['f606w']*0 + le_ml_m_to_l
        ue_ml_m_to_l=max_m_to_l-ml_m_to_l ; data[ue_m_to_l_name]=data['f606w']*0 + ue_ml_m_to_l
        
        mass=ml_m_to_l*( 10**( -0.4*(data.f606w-DM-M_sun_f606w) ) ) * 1e-5 ; data[mass_name]=ml_m_to_l*( 10**( -0.4*(data.f606w-DM-M_sun_f606w) ) ) * 1e-5
        le_mass=.5*np.sqrt( ( 10**( -0.4*(data.f606w-DM-M_sun_f606w)) )*\
                           (le_ml_m_to_l**2)*( 10**( -0.4*(data.f606w-DM-M_sun_f606w)) ) + \
                           (((.5*data.f606w_err)**2 * ml_m_to_l)**2)*np.log(10.)) * 1e-5 ; data[le_mass_name]=le_mass
        ue_mass=.5*np.sqrt( ( 10**( -0.4*(data.f606w-DM-M_sun_f606w)) )*\
                           (ue_ml_m_to_l**2)*( 10**( -0.4*(data.f606w-DM-M_sun_f606w)) ) + \
                           (((.5*data.f606w_err)**2 * ml_m_to_l)**2)*np.log(10.)) * 1e-5 ; data[ue_mass_name]=ue_mass
        le=le_mass.max() ; ue=ue_mass.max()

ml_sel=np.exp(data['max_ml_0p005Z'])>=0
ml_age=( data['age_0p005Z'][ml_sel]*np.exp(data['max_ml_0p005Z']) + data['age_0p02Z'][ml_sel]*np.exp(data['max_ml_0p02Z'][ml_sel]) +\
        data['age_0p04Z'][ml_sel]*np.exp(data['max_ml_0p04Z'][ml_sel]) + data['age_0p2Z'][ml_sel]*np.exp(data['max_ml_0p2Z'][ml_sel]) +\
        data['age_Z'][ml_sel]*np.exp(data['max_ml_Z'][ml_sel]) + data['age_2p5Z'][ml_sel]*np.exp(data['max_ml_2p5Z'][ml_sel]))/\
        (np.exp(data['max_ml_0p005Z'][ml_sel]) + np.exp(data['max_ml_0p02Z'][ml_sel]) + np.exp(data['max_ml_0p04Z'][ml_sel]) +\
         np.exp(data['max_ml_0p2Z'][ml_sel]) + np.exp(data['max_ml_Z'][ml_sel]) + np.exp(data['max_ml_2p5Z'][ml_sel]))

print(data['age_0p005Z'][ml_sel],np.exp(data['max_ml_0p005Z']),data['age_0p02Z'][ml_sel],np.exp(data['max_ml_0p02Z'][ml_sel]),\
        data['age_0p04Z'][ml_sel],np.exp(data['max_ml_0p04Z'][ml_sel]),data['age_0p2Z'][ml_sel],np.exp(data['max_ml_0p2Z'][ml_sel]),\
        data['age_Z'][ml_sel],np.exp(data['max_ml_Z'][ml_sel]),data['age_2p5Z'][ml_sel],np.exp(data['max_ml_2p5Z'][ml_sel]))

#test_label='Age = ${:.2f}'.format(ml_age)+'^{ +'+'{:.2f}'.format(ue_ml_age)+'}_{ -'+'{:.2f}'.format(le_ml_age)+'}$ Gyr; '+\
#           'M/L$_V$ = ${:.2f}'.format(ml_m_to_l)+'^{ +'+'{:.2f}'.format(ue_ml_m_to_l)+'}_{ -'+'{:.2f}'.format(le_ml_m_to_l)+'}$'+\
#           'Z = 0.2Z$_\odot; M = {:.2f}'.format(mass[0]) + '^{+'+'{:.2f}'.format(ue)+'}_{-'+'{:.2f}'.format(le)+'}\ x\ 10^5 M_\odot$'

#print('Age = {:.3f}'.format((1e-9*10**(ssp_model['log_age_yr'])*(ssp_model['ml_mod0p2Z'])).sum() / ((ssp_model_0p2Z['ml_mod0p2Z'])).sum()))
#print('M/Lv = {:.3f}'.format(((ssp_model_0p2Z['M_star_tot_to_Lv'])*ssp_model_0p2Z['ml_mod0p2Z']).sum() / ssp_model_0p2Z['ml_mod0p2Z'].sum()))

### Annotate the most likely SSP parameters
age_0p2Z=(1e-9*10**(ssp_model_0p2Z['log_age_yr'][(ssp_model_0p2Z['ml_mod0p2Z']==np.max(ssp_model_0p2Z['ml_mod0p2Z']))]))
m_to_l_0p2Z=(ssp_model_0p2Z['M_star_tot_to_Lv'][(ssp_model_0p2Z['ml_mod0p2Z']==np.max(ssp_model_0p2Z['ml_mod0p2Z']))])
age_0p2Z_val=age_0p2Z.values[0]; m_to_lv_0p2Z=m_to_l_0p2Z.values[0]

k=r'Z = 0.2Z$_\odot; M = {:.2f}'.format(mass[0]) + '^{+'+'{:.2f}'.format(ue)+'}_{-'+'{:.2f}'.format(le)+'}\ x\ 10^5 M_\odot$'
ax1.annotate(k,xy=(1.56,1))
#ax1.annotate(test_label,xy=(1.56,1.05))

ax1.legend(loc=4,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

ax22 = fig.add_axes([.905,.39,.09,.27])
#sbn.kdeplot(np.exp(ssp_model_0p2Z['ml_mod0p2Z']),bw=((.1*np.exp(ssp_model_0p2Z['ml_mod0p2Z'])).max()/2.),color='darkorange',label='KDE', vertical=True)
ax22.plot(ssp_model_0p2Z['M_star_tot_to_Lv'],np.exp(ssp_model_0p2Z['ml_mod0p2Z']),linestyle='--',color='darkorange')
ax22.yaxis.set_visible(False)
ax22.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax2 = fig.add_axes([.6,.39,.305,.27])
ax2.set_ylabel("Likelihood value")
ax2.plot(1e-9*10**(ssp_model_0p2Z['log_age_yr']),np.exp(ssp_model_0p2Z['ml_mod0p2Z']),color='darkorange',linestyle='-',label='0.2Z$_\odot$')
ax2.vlines(min_age,np.exp(ssp_model_0p2Z['ml_mod0p2Z']).min(),np.exp(ssp_model_0p2Z['ml_mod0p2Z'][(ssp_model_0p2Z['log_age_yr']>=q1) & (ssp_model_0p2Z['log_age_yr']<=q3)].min()),linestyle='--', colors='gray')
ax2.vlines(max_age,np.exp(ssp_model_0p2Z['ml_mod0p2Z']).min(),np.exp(ssp_model_0p2Z['ml_mod0p2Z'][(ssp_model_0p2Z['log_age_yr']>=q1) & (ssp_model_0p2Z['log_age_yr']<=q3)].max()),linestyle='--', colors='gray')
ax2.vlines(ml_age,(np.exp(ssp_model_0p2Z['ml_mod0p2Z'])).min(),np.exp(max_ml),linestyle='-', colors='gray')
ax2.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

### OLD ###
#model=ssp_model_Z
#temp = pd.DataFrame(columns=['ml_modZ'])
#for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
#                                       (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
#                                       (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
#    for mod_col1,mod_col2,mod_col3 in zip(model['606m160'],model['606m814'],model['V_m_F160w_wfc3']-model['V_m_F814w_uvis']) :
#        a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
#        b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
#        c=np.log(a*b)
#        temp = temp.append({'ml_modZ': c}, ignore_index=True)#
#
#all_inf_or_nan = temp.isin([np.inf, -np.inf, np.nan]).all(axis='columns')
#model['ml_modZ']=temp
#model=model[~all_inf_or_nan]
#age_Z=pd.to_numeric(1e-9*10**(model['log_age_yr'][(model['ml_modZ']==np.max(temp['ml_modZ']))]))
#m_to_l_Z=pd.to_numeric(model['M_star_tot_to_Lv'][(model['ml_modZ']==np.max(temp['ml_modZ']))])
#m_to_lv_Z=m_to_l_Z.values[0]

ax33 = fig.add_axes([.905,.71,.09,.27])
sbn.kdeplot(np.e**(ssp_model_Z['ml_modZ']),bw=(.1*np.e**(ssp_model_Z['ml_modZ']).max()/2.),color='red',label='KDE', vertical=True)
ax33.yaxis.set_visible(False)
ax33.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax3 = fig.add_axes([.6,.71,.305,.27])
ax3.set_ylabel("Likelihood value")
ax3.plot(1e-9*10**(ssp_model_Z['log_age_yr']),np.e**(ssp_model_Z['ml_modZ']),color='red',linestyle='-',label='Z$_\odot$')
ax3.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)


ax44 = fig.add_axes([.905,.07,.09,.27])
ax44.set_xlabel(r"$\rho_{Likelihood}$")
sbn.kdeplot(np.e**(ssp_model_0p04Z['ml_mod0p04Z']),bw=(.1*np.e**(ssp_model_0p04Z['ml_mod0p04Z']).max()/2.),color='blue',label='KDE', vertical=True)
ax44.yaxis.set_visible(False)
ax44.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax4 = fig.add_axes([.6,.07,.305,.27])
ax4.set_xlabel("Age [Gyr]") ; plt.ylabel("Likelihood value")
ax4.plot(1e-9*10**(ssp_model_0p04Z['log_age_yr']),np.e**(ssp_model_0p04Z['ml_mod0p04Z']),color='blue',linestyle='-',label='0.04Z$_\odot$')
ax4.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

plt.tight_layout()

plt.savefig("test.pdf")

plt.show()