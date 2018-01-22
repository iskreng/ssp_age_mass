import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

##### BEGIN: User input #####

#data_file="Galaxy_NSCs_1Z.dat"
data=pd.DataFrame({'f606w' : [23.],\
                   'f606w_err' : [0.03],\
                   'f814w' : [22.2],\
                   'f814w_err' : [0.04]})
                      
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
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])
ssp_model_0p005Z = pd.read_table(ssp_model_file_0p005Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])
ssp_model_0p02Z = pd.read_table(ssp_model_file_0p02Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])
ssp_model_0p04Z = pd.read_table(ssp_model_file_0p04Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])
ssp_model_0p2Z = pd.read_table(ssp_model_file_0p2Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])
ssp_model_2p5Z = pd.read_table(ssp_model_file_2p5Z, delim_whitespace=True, engine='c', na_values='INDEF',
                              header=None, comment='#', names=['log_age_yr','Vmag','M_star_tot_to_Lv','V_m_F606w_uvis', 'V_m_F814w_uvis'], usecols=[0,13,58,109,114])

select=ssp_model_Z['log_age_yr']>=lg_age_limit; select=ssp_model_0p005Z['log_age_yr']>=lg_age_limit; select=ssp_model_2p5Z['log_age_yr']>=lg_age_limit; 

mag_Z=ssp_model_Z['Vmag'][select]-ssp_model_Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_Z['M_star_tot_to_Lv'][select])) +DM
color_Z=ssp_model_Z['V_m_F814w_uvis'][select]-ssp_model_Z['V_m_F606w_uvis'][select]
mag_0p005Z=ssp_model_0p005Z['Vmag'][select]-ssp_model_0p005Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p005Z['M_star_tot_to_Lv'][select]))+DM
color_0p005Z=ssp_model_0p005Z['V_m_F814w_uvis'][select]-ssp_model_0p005Z['V_m_F606w_uvis'][select]
mag_0p02Z=ssp_model_0p02Z['Vmag'][select]-ssp_model_0p02Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p02Z['M_star_tot_to_Lv'][select]))+DM
color_0p02Z=ssp_model_0p02Z['V_m_F814w_uvis'][select]-ssp_model_0p02Z['V_m_F606w_uvis'][select]
mag_0p04Z=ssp_model_0p04Z['Vmag'][select]-ssp_model_0p04Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p04Z['M_star_tot_to_Lv'][select]))+DM
color_0p04Z=ssp_model_0p04Z['V_m_F814w_uvis'][select]-ssp_model_0p04Z['V_m_F606w_uvis'][select]
mag_0p2Z=ssp_model_0p2Z['Vmag'][select]-ssp_model_0p2Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_0p2Z['M_star_tot_to_Lv'][select]))+DM
color_0p2Z=ssp_model_0p2Z['V_m_F814w_uvis'][select]-ssp_model_0p2Z['V_m_F606w_uvis'][select]
mag_2p5Z=ssp_model_2p5Z['Vmag'][select]-ssp_model_2p5Z['V_m_F606w_uvis'][select] - 2.5*np.log10(8.55e1*ssp_Z_lum_scale/(ssp_model_2p5Z['M_star_tot_to_Lv'][select]))+DM
color_2p5Z=ssp_model_2p5Z['V_m_F814w_uvis'][select]-ssp_model_2p5Z['V_m_F606w_uvis'][select]

#print(1e-6*10 ** (-0.4 * (np.array(mag_Z)-DM-M_sun_f606w)))

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

plt.xlabel("F606W-F814W [mag]")
plt.ylabel("F606W [mag]")
plt.ylim(27.5,18.5) # plt.gca().invert_yaxis()
plt.xlim(0,1.8)
       
plt.axhline(25.66, color='k', linestyle='dashed', linewidth=1)
plt.annotate("90% compl.",xy=(1.5,25.5), fontsize=9)
plt.plot(color_Z,mag_Z, color='red', linestyle='-')
plt.plot(color_0p005Z,mag_0p005Z, color='blue', linestyle='-')
plt.plot(color_0p02Z,mag_0p02Z, color='darkblue', linestyle='-')
plt.plot(color_0p02Z,mag_0p2Z, color='darkorange', linestyle='-')
plt.plot(color_2p5Z,mag_2p5Z, color='brown', linestyle='-')
plt.errorbar(data.f606w-data.f814w,data.f606w,xerr=np.sqrt(data.f606w_err**2+data.f814w_err**2),yerr=data.f606w_err,marker='.')

test = pd.DataFrame(columns=['ml'])

for mag1c,mag1c_err in zip(data.f606w,data.f606w_err):
    for mu1c in mag_0p005Z :
        a=(1./(mag1c_err*np.sqrt(2.*np.pi)))
        b=np.log(a*np.exp(-(mag1c-mu1c)**2 / (2*mag1c_err**2)))
#        print("{:.5f}".format( a*b ))
        test = test.append({'ml': b}, ignore_index=True)
#    return 

print(np.max(test))

ylim = plt.ylim()

# Add anoter plot whose yaxis and tiscks will appear on the x2 axis
ax=plt.twinx()
plt.ylabel("$L\ [L_\odot$]", rotation=0)
ax.ticklabel_format(useOffset=False)

ax.set_yscale("log")
plt.ylim(10 ** (-0.4 * (np.array(ylim)-DM-M_sun_f606w)))

#plt.plot(color_Z,(10 ** (-0.4 * (np.array(mag_Z)-DM-4.7))), color='gray', linestyle='-')

plt.tight_layout()

plt.savefig("test.pdf")

plt.show()