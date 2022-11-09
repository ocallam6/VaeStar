import os
from sys import implementation
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


class StarSample:
    def __init__(self,input_dataframe, param_columns,error_columns):
        if(type(input_dataframe)==pd.DataFrame):
            self.data=input_dataframe
            self.data_columns=param_columns
            self.error_columns=error_columns
            self.resampled=None
            self.dropnan=False
            self.stacked=None
            try:
                self.data['phot_g_mean_mag_error']=np.sqrt(((2.5/np.log(10))*self.data['phot_g_mean_flux_error']/self.data['phot_g_mean_flux'])**2+0.0027553202**2)
                self.data['bp_error']=np.sqrt(((2.5/np.log(10))*self.data['phot_bp_mean_flux_error']/self.data['phot_bp_mean_flux'])**2+0.0037793818**2)
                self.data['rp_error']=np.sqrt(((2.5/np.log(10))*self.data['phot_rp_mean_flux_error']/self.data['phot_rp_mean_flux'])**2+0.0027901700**2)
                self.data['bp_rp_error']=np.sqrt(self.data['rp_error']**2+self.data['bp_error']**2)
                self.error_columns+=['phot_g_mean_mag_error','bp_error','rp_error','bp_rp_error']
            except:
                print('Some Gaia parameters not available')    

        else:
            raise NotImplemented

    def dropna(self):
        self.data=self.data.dropna().reset_index(drop=True)
        self.dropnan=True

    def cut_on_condition(self,condition,column,val):
        x=self.data
        if(condition=='ge'):
            x=x[x[column]>=val].reset_index(drop=True)
        elif(condition=='le'):
            x=x[x[column]<=val].reset_index(drop=True)



    def to_tensor(self,expand=True,expandsize=1):
        x_values=self.data.values
        x_values=torch.tensor(x_values)          
        if(expand==True):     
            x_values=x_values.reshape((x_values.shape[0],1,x_values.shape[1]))
            pad_size=np.max(expandsize)
            x_values=x_values.repeat(1,pad_size,1)
        return x_values


    def mean_absolute_mag_g_band(self,abs_column_name,magnitude_column,parallax_column,method='parallax'):
        if(method=='parallax'):
            self.data[abs_column_name] = self.data[magnitude_column].values+5*np.log10(self.data[parallax_column].values/1000)+5
        else:
            raise NotImplemented
   
    def resample(self,draw_size,input_columns,error_columns,store=False):
        mean_x=self.data[input_columns].to_numpy()
        var_x=self.error[error_columns].to_numpy()**2
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

        resampled_x=pd.DataFrame(data=x_params,columns=[input_columns])
        resampled_err=pd.DataFrame(data=x_params,columns=[error_columns])
        if(store==True):
            self.resampled=pd.concat([resampled_x,resampled_err],axis=1)
        return pd.concat([resampled_x,resampled_err],axis=1)

    def save_dataframe(self,full=True):
        if(full==True):
            self.data.to_csv('star_sample_full')
        else:
            raise NotImplemented

    def hr_plot(self,title,mag_column,colour_column,hue_column,height=10,width=10,**kwargs):
        f, axes = plt.subplots(1, 1)
        f.set_figheight(height)
        f.set_figwidth(width)
        sn.scatterplot(x=self.data[colour_column],y=self.data[mag_column],hue=self.data[hue_column])#,kwargs=kwargs)
        axes.invert_yaxis()
        plt.legend()
        axes.set_title(title)
        plt.xlim(0,3)
        plt.show()

    
import minimint

class Isochrones:
    def __init__(self,filters,logagegrid,massgrid,fehgrid,phase=0,override=None):
        ii = minimint.Interpolator(filters)
        self.isochrones_list=[]
        self.pad_size=None
        self.logagegrid=logagegrid
        self.fehgrid=fehgrid
        for feh in fehgrid:
            for lage in logagegrid:
                iso = pd.DataFrame(ii(massgrid, lage, feh))
                self.isochrones_list.append(iso[iso['phase']==phase].reset_index(drop=True))
        
        self.slopes_creator(override=override)
        
        

    def plot_isochrones(self, index=None):
        fig = plt.figure(figsize=(6, 6), dpi=120)
        if(index==None):
            for isochrone in self.isochrones_list:
                plt.scatter(isochrone['Gaia_BP_EDR3']-isochrone['Gaia_RP_EDR3'], isochrone['Gaia_G_EDR3'],
                                )
        else:
            isochrone=self.isochrones_list[index]
            plt.scatter(isochrone['Gaia_BP_EDR3']-isochrone['Gaia_RP_EDR3'], isochrone['Gaia_G_EDR3'],
                                )

            
        plt.title('Plot of isochrones')
        plt.ylim(20, -15)

    
    def slopes_creator(self,override=None):
        for isochrone in self.isochrones_list:
            p_slopes=[]
            distance=[]
            isochrone['BPRP']=isochrone['Gaia_BP_EDR3']-isochrone['Gaia_RP_EDR3']
            x=isochrone['BPRP']
            y=isochrone['Gaia_G_EDR3']
            
            for i in range(len(isochrone)-1):    
                dy=y[i+1]-y[i]
                dx=x[i+1]-x[i]

                distance.append(np.sqrt(dx**2+dy**2))

            distance.append(0)

            isochrone['distance']=distance

            
            isochrone['distance_flag']=(isochrone['distance']<=0.3*isochrone['distance'].mean())
            isochrone.drop(isochrone.loc[isochrone['distance_flag']==True].index, inplace=True)
            isochrone.reset_index(drop=True,inplace=True)

            isochrone.drop('distance_flag',axis=1,inplace=True)
            isochrone.drop('distance',axis=1,inplace=True)

            x=isochrone['BPRP']
            y=isochrone['Gaia_G_EDR3']
            for i in range(len(isochrone)-1):    
                dy=y[i+1]-y[i]
                dx=x[i+1]-x[i]
                p_slopes.append(-1*dx/dy)

            p_slopes.append(0)


            isochrone['p_slopes']=p_slopes
            isochrone['slopes']=-1/isochrone['p_slopes']
            if(type(override)==int):
                isochrone['p_slopes']=override
            high_c=[]
            low_c=[]
            for i in range(len(isochrone)-1):
                high_c.append(isochrone['Gaia_G_EDR3'][i+1] - isochrone['p_slopes'][i]*isochrone['BPRP'][i+1])
                low_c.append(isochrone['Gaia_G_EDR3'][i] - isochrone['p_slopes'][i]*isochrone['BPRP'][i])
            high_c.append(0)
            low_c.append(0)
            isochrone['low_c']=high_c #high c is low c oops because its inverted on HR diagram and the ploot in reverse order?
            isochrone['high_c']=low_c
            isochrone.drop(isochrone.tail(3).index, inplace = True) #this is to try get off the turn off
    
    def stack_isochrones(self):
        tens=[]
        lengths=[]
        
        for isochrone in self.isochrones_list:
            isoc=torch.tensor(isochrone.values).reshape((1,isochrone.shape[0],isochrone.shape[1]))
            tens.append(isoc)
            lengths.append(isoc.shape[-2])
        stack_tens=[]
        import torch.nn.functional as F
        self.pad_size=np.max(lengths)
        for i in range(len(tens)):
            
            t=F.pad(input=tens[i], pad=(0,0,0,self.pad_size-tens[i].shape[-2]), value=torch.nan)
            
            stack_tens.append(t.reshape(t.shape[1:]))
        self.stacked = torch.stack(stack_tens)
        return self.stacked

    def find_nearest(self,array, value):
        #array = np.asarray(array)
        idx = (torch.abs(array - value)).argmin()
        return array[idx],idx

    def isochrone_selector(self,feh,age):

        logagegrid = torch.tensor(self.logagegrid)
        fehgrid = torch.tensor(self.fehgrid)
        feh,feh_idx=self.find_nearest(fehgrid,feh)
        age,age_idx=self.find_nearest(logagegrid,age)

        return feh_idx*len(logagegrid)+age_idx

    def stack_isochrones_subsample(self,feh_list,age_list):
        substack=[]
        for feh in feh_list:
            for age in age_list:
                substack.append(self.stacked[self.isochrone_selector(feh,age)])
        return torch.stack(substack)
    