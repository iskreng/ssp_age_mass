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

print("SSP for Age > {:.2f}".format(1e-9*10**(lg_age_limit)))
select_0p005Z=ssp_model_0p005Z['log_age_yr']>=lg_age_limit
select_0p02Z=ssp_model_0p02Z['log_age_yr']>=lg_age_limit
select_0p04Z=ssp_model_0p04Z['log_age_yr']>=lg_age_limit
select_0p2Z=ssp_model_0p2Z['log_age_yr']>=lg_age_limit
select_Z=ssp_model_Z['log_age_yr']>=lg_age_limit
select_2p5Z=ssp_model_2p5Z['log_age_yr']>=lg_age_limit

mag_0p005Z=ssp_model_0p005Z['Vmag'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_0p005Z['M_star_tot_to_Lv'][select_0p005Z]))+DM
ssp_model_0p005Z['606m814']=ssp_model_0p005Z['V_m_F814w_uvis'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z]
ssp_model_0p005Z['606m160']=ssp_model_0p005Z['V_m_F160w_wfc3'][select_0p005Z]-ssp_model_0p005Z['V_m_F606w_uvis'][select_0p005Z]
mag_0p02Z=ssp_model_0p02Z['Vmag'][select_0p02Z]-ssp_model_0p02Z['V_m_F606w_uvis'][select_0p02Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_0p02Z['M_star_tot_to_Lv'][select_0p02Z]))+DM
ssp_model_0p02Z['606m814']=ssp_model_0p02Z['V_m_F814w_uvis'][select_0p02Z]-ssp_model_0p02Z['V_m_F606w_uvis'][select_0p02Z]
ssp_model_0p02Z['606m160']=ssp_model_0p02Z['V_m_F160w_wfc3'][select_0p02Z]-ssp_model_0p02Z['V_m_F606w_uvis'][select_0p02Z]
mag_0p04Z=ssp_model_0p04Z['Vmag'][select_0p04Z]-ssp_model_0p04Z['V_m_F606w_uvis'][select_0p04Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_0p04Z['M_star_tot_to_Lv'][select_0p04Z]))+DM
ssp_model_0p04Z['606m814']=ssp_model_0p04Z['V_m_F814w_uvis'][select_0p04Z]-ssp_model_0p04Z['V_m_F606w_uvis'][select_0p04Z]
ssp_model_0p04Z['606m160']=ssp_model_0p04Z['V_m_F160w_wfc3'][select_0p04Z]-ssp_model_0p04Z['V_m_F606w_uvis'][select_0p04Z]
mag_0p2Z=ssp_model_0p2Z['Vmag'][select_0p2Z]-ssp_model_0p2Z['V_m_F606w_uvis'][select_0p2Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_0p2Z['M_star_tot_to_Lv'][select_0p2Z]))+DM
ssp_model_0p2Z['606m814']=ssp_model_0p2Z['V_m_F814w_uvis'][select_0p2Z]-ssp_model_0p2Z['V_m_F606w_uvis'][select_0p2Z]
ssp_model_0p2Z['606m160']=ssp_model_0p2Z['V_m_F160w_wfc3'][select_0p2Z]-ssp_model_0p2Z['V_m_F606w_uvis'][select_0p2Z]
ssp_model_Z['mag_Z']=ssp_model_Z['Vmag'][select_Z]-ssp_model_Z['V_m_F606w_uvis'][select_Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_Z['M_star_tot_to_Lv'][select_Z])) +DM
ssp_model_Z['606m814']=ssp_model_Z['V_m_F814w_uvis'][select_Z]-ssp_model_Z['V_m_F606w_uvis'][select_Z]
ssp_model_Z['606m160']=ssp_model_Z['V_m_F160w_wfc3'][select_Z]-ssp_model_Z['V_m_F606w_uvis'][select_Z]
mag_2p5Z=ssp_model_2p5Z['Vmag'][select_2p5Z]-ssp_model_2p5Z['V_m_F606w_uvis'][select_2p5Z] - 2.5*np.log10(8.55e1*ssp_lum_scale/(ssp_model_2p5Z['M_star_tot_to_Lv'][select_2p5Z]))+DM
ssp_model_2p5Z['606m814']=ssp_model_2p5Z['V_m_F814w_uvis'][select_2p5Z]-ssp_model_2p5Z['V_m_F606w_uvis'][select_2p5Z]
ssp_model_2p5Z['606m160']=ssp_model_2p5Z['V_m_F160w_wfc3'][select_2p5Z]-ssp_model_2p5Z['V_m_F606w_uvis'][select_2p5Z]

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

# SSP constant age
age_low=np.log10(2.99e9); age_up=np.log10(3.01e9)
m606m160_2Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m160'].values[0]]
m606m814_2Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m814'].values[0]]

age_low=np.log10(4.99e9); age_up=np.log10(5.01e9)
m606m160_5Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m160'].values[0]]
m606m814_5Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m814'].values[0]]

age_low=np.log10(13.99e9); age_up=np.log10(14.01e9)
m606m160_14Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m160'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m160'].values[0]]
m606m814_14Gyr = [ssp_model_0p005Z.loc[((ssp_model_0p005Z['log_age_yr']>=age_low) & (ssp_model_0p005Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p02Z.loc[((ssp_model_0p02Z['log_age_yr']>=age_low) & (ssp_model_0p02Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p04Z.loc[((ssp_model_0p04Z['log_age_yr']>=age_low) & (ssp_model_0p04Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_0p2Z.loc[((ssp_model_0p2Z['log_age_yr']>=age_low) & (ssp_model_0p2Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_Z.loc[((ssp_model_Z['log_age_yr']>=age_low) & (ssp_model_Z['log_age_yr']<=age_up)), '606m814'].values[0],\
ssp_model_2p5Z.loc[((ssp_model_2p5Z['log_age_yr']>=age_low) & (ssp_model_2p5Z['log_age_yr']<=age_up)), '606m814'].values[0]]

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

# Estimate the maximum likelihood model value for age and M/Lv
model=ssp_model_0p2Z
temp = pd.DataFrame(columns=['ml_mod0p2Z'])
for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
                                       (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
                                       (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
    for mod_col1,mod_col2,mod_col3 in zip(model['606m160'],model['606m814'],model['V_m_F160w_wfc3']-model['V_m_F814w_uvis']) :
        a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
        b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
        c=np.log(a*b)
        temp = temp.append({'ml_mod0p2Z': c}, ignore_index=True)

model['ml_mod0p2Z']=temp
age_0p2Z=(1e-9*10**(model['log_age_yr'][(model['ml_mod0p2Z']==np.max(temp['ml_mod0p2Z']))]))
m_to_l_0p2Z=(model['M_star_tot_to_Lv'][(model['ml_mod0p2Z']==np.max(temp['ml_mod0p2Z']))])
age_0p2Z_val=age_0p2Z.values[0]; m_to_lv_0p2Z=m_to_l_0p2Z.values[0]

print(data)

ax1.legend(loc=4,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

ax22 = fig.add_axes([.905,.39,.09,.27])
sbn.kdeplot(np.e**(model['ml_mod0p2Z']),bw=(.1*np.e**(model['ml_mod0p2Z']).max()/2.),color='darkorange',label='KDE', vertical=True)
ax22.yaxis.set_visible(False)
ax22.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax2 = fig.add_axes([.6,.39,.305,.27])
ax2.set_ylabel("Likelihood value")
ax2.plot(1e-9*10**(model['log_age_yr']),np.e**(model['ml_mod0p2Z']),color='darkorange',linestyle='-',label='0.2Z$_\odot$')
ax2.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

#############
del model,temp
model=ssp_model_Z
temp = pd.DataFrame(columns=['ml_modZ'])
for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
                                       (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
                                       (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
    for mod_col1,mod_col2,mod_col3 in zip(model['606m160'],model['606m814'],model['V_m_F160w_wfc3']-model['V_m_F814w_uvis']) :
        a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
        b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
        c=np.log(a*b)
        temp = temp.append({'ml_modZ': c}, ignore_index=True)

model['ml_modZ']=temp
age_Z=pd.to_numeric(1e-9*10**(model['log_age_yr'][(model['ml_modZ']==np.max(temp['ml_modZ']))]))
m_to_l_Z=pd.to_numeric(model['M_star_tot_to_Lv'][(model['ml_modZ']==np.max(temp['ml_modZ']))])
m_to_lv_Z=m_to_l_Z.values[0]

ax33 = fig.add_axes([.905,.71,.09,.27])
sbn.kdeplot(np.e**(model['ml_modZ']),bw=(.1*np.e**(model['ml_modZ']).max()/2.),color='red',label='KDE', vertical=True)
ax33.yaxis.set_visible(False)
ax33.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax3 = fig.add_axes([.6,.71,.305,.27])
ax3.set_ylabel("Likelihood value")
ax3.plot(1e-9*10**(model['log_age_yr']),np.e**(model['ml_modZ']),color='red',linestyle='-',label='Z$_\odot$')
ax3.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

#############
del model,temp
model=ssp_model_0p04Z
temp = pd.DataFrame(columns=['ml_mod0p04Z'])
for col1,col1_err,col2,col2_err,col3,col3_err in zip((data.f606w-data.f160w),np.sqrt(data.f606w_err**2+data.f160w_err**2),\
                                       (data.f606w-data.f814w),np.sqrt(data.f606w_err**2+data.f814w_err**2),\
                                       (data.f814w-data.f160w),np.sqrt(data.f814w_err**2+data.f160w_err**2)):
    for mod_col1,mod_col2,mod_col3 in zip(model['606m160'],model['606m814'],model['V_m_F160w_wfc3']-model['V_m_F814w_uvis']) :
        a=(1./(col1_err*np.sqrt(2.*np.pi)))*(1./(col2_err*np.sqrt(2.*np.pi)))*(1./(col3_err*np.sqrt(2.*np.pi)))
        b=np.exp(-(col1-mod_col1)**2 / (2*col1_err**2))*np.exp(-(col2-mod_col2)**2 / (2*col2_err**2))*np.exp(-(col3-mod_col3)**2 / (2*col3_err**2))
        c=np.log(a*b)
        temp = temp.append({'ml_mod0p04Z': c}, ignore_index=True)

model['ml_mod0p04Z']=temp
age_0p04Z=pd.to_numeric(1e-9*10**(model['log_age_yr'][(model['ml_mod0p04Z']==np.max(temp['ml_mod0p04Z']))]))
m_to_l_0p04Z=pd.to_numeric(model['M_star_tot_to_Lv'][(model['ml_mod0p04Z']==np.max(temp['ml_mod0p04Z']))])
m_to_lv_0p04Z=m_to_l_0p04Z.values

ax44 = fig.add_axes([.905,.07,.09,.27])
ax44.set_xlabel(r"$\rho_{Likelihood}$")
sbn.kdeplot(np.e**(model['ml_mod0p04Z']),bw=(.1*np.e**(model['ml_mod0p04Z']).max()/2.),color='blue',label='KDE', vertical=True)
ax44.yaxis.set_visible(False)
ax44.legend(fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)
ax4 = fig.add_axes([.6,.07,.305,.27])
ax4.set_xlabel("Age [Gyr]") ; plt.ylabel("Likelihood value")
ax4.plot(1e-9*10**(model['log_age_yr']),np.e**(model['ml_mod0p04Z']),color='blue',linestyle='-',label='0.04Z$_\odot$')
ax4.legend(loc=2,fontsize=10,ncol=2,columnspacing=.5,markerscale=0.28,framealpha=0)

### Annotate the most likely SSP parameters
mass=m_to_lv_0p2Z*( 10**( -0.4*(data.f606w-DM-M_sun_f606w) ) ) * 1e-6
k='Z = 0.2Z$_\odot$; Age = {:.3g} Gyr'.format(age_0p2Z_val)+'; $M/L_V$ = {:.3g}'.format(m_to_lv_0p2Z)+'\n$M$ = {:.2f}'.format(mass[0]) + 'x$10^6 M_\odot$'
ax1.annotate(k,xy=(1.56,1))


plt.tight_layout()

plt.savefig("test.pdf")

plt.show()