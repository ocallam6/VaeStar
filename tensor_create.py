import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir("Data")
cluster_df=pd.read_csv('cluster_members_final.csv')
os.chdir("../")


probability='p3'
f, axes = plt.subplots(1, 1)
f.set_figheight(5)
f.set_figwidth(5)

cluster_df=cluster_df[cluster_df['phot_bp_mean_flux_error']<1000].reset_index(drop=True)

scatter=cluster_df[cluster_df[probability]>0.9]





scatter=scatter[scatter['rgeo']>=1.8].reset_index(drop=True)
scatter=scatter[scatter['rgeo']<=2.5].reset_index(drop=True)
#scatter=scatter[scatter['bp_rp']<=1.6].reset_index(drop=True)

orig_scatter=scatter

orig_scatter['rgeo']=scatter['rgeo'].median() #still assuming everything at the same distance
#set the main distance.
orig_scatter['rgeo']=2.0

cluster_df=scatter

cluster_df=cluster_df[['ra','dec','parallax','phot_g_mean_mag','phot_g_mean_flux','phot_rp_mean_flux','phot_bp_mean_flux','phot_bp_mean_mag','phot_rp_mean_mag','bp_rp','mean_absolute_mag_g_band','ra_error','dec_error','parallax_error','phot_g_mean_flux_error','phot_bp_mean_flux_error','phot_rp_mean_flux_error']].reset_index(drop=True)
cluster_df

cluster_df['phot_g_mean_mag_error']=np.sqrt(((2.5/np.log(10))*cluster_df['phot_g_mean_flux_error']/cluster_df['phot_g_mean_flux'])**2+0.0027553202**2)
cluster_df['bp_error']=np.sqrt(((2.5/np.log(10))*cluster_df['phot_bp_mean_flux_error']/cluster_df['phot_bp_mean_flux'])**2+0.0037793818**2)
cluster_df['rp_error']=np.sqrt(((2.5/np.log(10))*cluster_df['phot_rp_mean_flux_error']/cluster_df['phot_rp_mean_flux'])**2+0.0027901700**2)
cluster_df['bp_rp_error']=np.sqrt(cluster_df['rp_error']**2+cluster_df['bp_error']**2)

mean_x=cluster_df[['ra','dec','parallax','phot_g_mean_mag','bp_rp']].to_numpy()
var_x=cluster_df[['ra_error','dec_error','parallax_error','phot_g_mean_mag_error','bp_rp_error']].to_numpy()**2


x_params=[]
x_var=[]
draw_size=32
for i in range (len(mean_x)):
    mu=mean_x[i]
    sig=np.diag(var_x[i])
    draws = np.random.multivariate_normal(mu, sig, size=draw_size)

    x_params.append(draws)
    x_var.append(np.tile(sig,(draw_size,1)))

x_params=np.array(x_params)
x_params=x_params.reshape((x_params.shape[0]*x_params.shape[1],x_params.shape[2]))

x_var=np.array(x_var)
x_var=x_var.reshape((x_var.shape[0]*x_var.shape[1],x_var.shape[2]))


resampled_x=pd.DataFrame(data=x_params,columns=[['ra','dec','parallax','phot_g_mean_mag','bp_rp']])
resampled_err=pd.DataFrame(data=x_params,columns=[['ra_error','dec_error','parallax_error','phot_g_mean_mag_error','bp_rp_error']])
resampled_x['ra']=resampled_x['ra']%360
resampled_x['dec']=resampled_x['dec']%360


def mean_basolute_mag_g_band(selection_gaia):

    return selection_gaia['phot_g_mean_mag'].values+5*np.log10(selection_gaia['parallax'].values/1000)+5
    

resampled_x['G']=mean_basolute_mag_g_band(resampled_x)
cluster_df['G']=mean_basolute_mag_g_band(cluster_df)


x_resample=pd.concat([resampled_x,resampled_err],axis=1)

x_resample=x_resample.dropna().reset_index(drop=True)

import minimint

filters = ["Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3']
ii = minimint.Interpolator(filters)

# Compute the isochrones
massgrid = 10*np.linspace(np.log10(0.1),np.log10(10), 1000)
logagegrid = [8.5,8.0,7.0]
fehgrid = [0.17,-1,-0.5]
dfs=[]
for feh in fehgrid:
    for lage in logagegrid:
        iso = pd.DataFrame(ii(massgrid, lage, feh))
        dfs.append(iso[iso['phase']==0])


# Compute the isochrones
massgrid = 10*np.linspace(np.log10(0.1),np.log10(10), 1000)
logagegrid = [8.0,8.1,8.3,8.5]
fehgrid = [0.17]
dfs=[]
for feh in fehgrid:
    for lage in logagegrid:
        iso = pd.DataFrame(ii(massgrid, lage, feh))
        dfs.append(iso[iso['phase']==0])


massgrid = 10*np.linspace(np.log10(0.1),np.log10(10), 1000)
logagegrid = np.linspace(5,10.3,105)
fehgrid = np.linspace(-4,0.5,90)
dfs=[]
for feh in fehgrid:
    for lage in logagegrid:
        iso = pd.DataFrame(ii(massgrid, lage, feh))
        dfs.append(iso[iso['phase']==0].reset_index(drop=True))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def isochrone_selector(feh,age):
    if(feh<-4 or feh>0.5):
        raise NotImplementedError
    if(age<5 or age>10.3):
        raise NotImplementedError
    else:
        logagegrid = np.linspace(5,10.3,105)
        fehgrid = np.linspace(-4,0.5,90)
        feh,feh_idx=find_nearest(fehgrid,feh)
        age,age_idx=find_nearest(logagegrid,age)

        return feh_idx*len(logagegrid)+age_idx
    

    
dfs[isochrone_selector(0,7)]
print('start')
for isochrone in dfs:
    p_slopes=[]
    isochrone['BPRP']=isochrone['Gaia_BP_EDR3']-isochrone['Gaia_RP_EDR3']
    x=isochrone['BPRP']
    y=isochrone['Gaia_G_EDR3']
    
    for i in range(len(isochrone)-1):    
        dy=y[i+1]-y[i]
        dx=x[i+1]-x[i]
        p_slopes.append(-1*dx/dy)
    p_slopes.append(0)

    isochrone['p_slopes']=p_slopes
    isochrone['slopes']=-1/isochrone['p_slopes']

    high_c=[]
    low_c=[]
    for i in range(len(isochrone)-1):
        high_c.append(isochrone['Gaia_G_EDR3'][i+1] - isochrone['p_slopes'][i]*isochrone['BPRP'][i+1])
        low_c.append(isochrone['Gaia_G_EDR3'][i] - isochrone['p_slopes'][i]*isochrone['BPRP'][i])
    high_c.append(0)
    low_c.append(0)
    isochrone['low_c']=high_c #high c is low c oops
    isochrone['high_c']=low_c
    isochrone=isochrone[:-2]

import torch

collection_tensors=[]
for isochrone in dfs:
    rows=cluster_df.values
    rows_ten=torch.tensor(rows)
    rows_ten=rows_ten.reshape((rows_ten.shape[0],rows_ten.shape[1],1))
    rows_ten=rows_ten.repeat(1,1,len(isochrone))

    isoc=torch.tensor(isochrone.values).reshape((1,isochrone.shape[1],isochrone.shape[0])).repeat(len(rows_ten),1,1)

    final=torch.cat((rows_ten,isoc),1)

    collection_tensors.append(final)

print(len(collection_tensors))