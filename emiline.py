import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
from scipy.signal import savgol_filter
#from scipy.optimize import curve_fit
from scipy.signal import find_peaks,peak_prominences
from scipy.integrate import simps

class emission_line():
	def __init__(self):
		self.name="halfa"
		self.lam0=6562.83
		self.k1=31
		self.folder="Doppler_VIS2"
		self.phase_file="phasev.txt"
		self.gamma=1
		self.wl=101  #window_length for smoothing func. must be odd number.
		self.prom=0.6
		self.polyorder=3
		self.grid=2000
		self.vel_inf_lim=-2000
		self.vel_sup_lim=2000
		self.peak1=0 #positive peak of coadded spectrum
		self.peak2=0 #negative peak of coadded spectrum
		

	def sep_peaks(self,vel_sp,spec,calc_peak_prom=False):
		cl=2.997e5
		smooth=savgol_filter(spec,window_length=self.wl,polyorder=self.polyorder)
		peaks,_=find_peaks(smooth,prominence=self.prom)
			
		if len(peaks) < 2:
			raise Exception("Could not find two peaks, try modifying prominence with option self.prom")		
		
		#if len(peaks)>2:			
		pos_peak=peaks[1]
		neg_peak=peaks[0]
		
		if calc_peak_prom:
			mypeaks=[smooth[peaks[0]],smooth[peaks[1]]]   #To get the minimum of the peaks, this will give prominence
			minpeak=mypeaks.index(min(mypeaks))
			prominences = peak_prominences(smooth,[peaks[minpeak]])[0]
			self.prom=prominences[0] 
			
		sep_vel=vel_sp[pos_peak]-vel_sp[neg_peak]
		sep_ang=(sep_vel*self.lam0/cl)
		
		self.peak2=vel_sp[pos_peak]
		self.peak1=vel_sp[neg_peak]						
		return sep_vel,sep_ang		
		

	def coadd_spec(self,vel_lims=[-2000,2000]):
		"""Coadds the spectra contained in folder"""

		k1=self.k1
		df=pd.read_csv(self.folder+"/"+self.phase_file,sep="\s+",names=["files","phases"])
		files=list(df.files)
		phases=list(df.phases)
		spec0=np.loadtxt(self.folder+"/"+files[0],unpack=True)
		wave_length=spec0[0]
		vel_sp=np.linspace(vel_lims[0],vel_lims[1],self.grid)
		avg_spec=np.zeros(vel_sp.size)
		cl=2.997e5
		spec_list=[]
		area_list=[]
		peaks_list=[]
		for i,txtfile in enumerate(files):
			print(txtfile)
			v_r=self.gamma+self.k1*np.sin(2*np.pi*phases[i]) #radial velocity of the White Dwarf
			delta_lam=self.lam0*v_r/cl
			lam0_wd=self.lam0+delta_lam      #lambda 0 in the WD's reference frame
			vell=(np.array(wave_length)-lam0_wd)*cl/lam0_wd #convert wave_length sp. to velocity sp. via Dop. shift	
			spectrum=np.loadtxt(self.folder+"/"+txtfile,unpack=True)[1]	
			spec_vel=np.interp(vel_sp,vell,spectrum)
			
			smooth=savgol_filter(spec_vel,window_length=self.wl,polyorder=self.polyorder)
			popt_neg,popt_pos=sep_gauss(vel_sp,smooth)
			peaks_list.append((popt_neg[0],popt_pos[0]))
			
			area=self.integ_flux(vel_sp,spec_vel)
			area_list.append(area)
			spec_list.append(spec_vel)


			avg_spec=np.add(avg_spec,spec_vel)
		
		spec_array=np.array(spec_list)
		avg_spec=avg_spec/len(files)
		return vel_sp,avg_spec,area_list,spec_array,phases,peaks_list
		
	def integ_flux(self,vel_sp,spec,line_lims=[-1000,1000]):  #To sum the flux
		a=line_lims[0]
		b=line_lims[1]			
		area= simps(spec[np.logical_and(vel_sp>=a,vel_sp<=b)], dx=1)
		#sum_flux=np.sum(spec[np.logical_and(vel_sp>=a,vel_sp<=b)])
		return area
	
	def plot_line(self,vel_sp,spec,smooth=True,xlims=[-1500,1500],ylims=[-.1,5],color_sm="red",color="blue",label="emission line"):
	#	plt.figure(num="Emission line"+self.name)
		width=0.9
		plt.hlines(0,xlims[0],xlims[1],linewidth=1)
		plt.vlines(0,ylims[0],ylims[1],linestyle="dashed",linewidth=1)
		plt.vlines(self.peak1,ylims[0],ylims[1],linestyle="dashed",linewidth=1,color=color_sm)
		plt.vlines(self.peak2,ylims[0],ylims[1],linestyle="dashed",linewidth=1,color=color_sm)
		plt.plot(vel_sp,spec,color=color,label=label,linewidth=0.7,alpha=0.8)
		if smooth:
			smooth=savgol_filter(spec,window_length=self.wl,polyorder=self.polyorder)
			plt.plot(vel_sp,smooth
			,color=color_sm,linewidth=0.7,alpha=0.8)
			
		plt.xlim(xlims[0],xlims[1])
		plt.ylim(ylims[0],ylims[1])
		#plt.show()

		
def sep_gauss(vel,spec):
	
	def gaussian(x,mu,A,sigma):
		return A*np.exp(-(x-mu)**2/2/sigma**2)
	#spec_neg=np.copy(spec)
	#spec_pos=np.copy(spec)
	#spec_neg[vel>0]=0
	#spec_pos[vel<0]=0	
	neg_max=np.argmax(spec[vel<0])
	pos_max=np.argmax(spec[vel>0])+len(spec[vel<0])  #To account of the negative part
	
	negpeak=spec[0:neg_max+1] 
	pospeak=spec[pos_max:]
	
	negpeak2= np.flipud(negpeak)
	pospeak2= np.flipud(pospeak)
	
	spec_neg=np.concatenate((negpeak,negpeak2))
	spec_pos=np.concatenate((pospeak2,pospeak))
	
	vel_neg=vel[0:len(spec_neg)]
	vel_pos=vel[-len(spec_pos):]

	popt_neg, _ = optimize.curve_fit(gaussian, vel_neg, spec_neg)
	popt_pos, _ = optimize.curve_fit(gaussian, vel_pos, spec_pos)

	return popt_neg,popt_pos			

		
#def plot_line_gauss(vel,spec):
		
	







