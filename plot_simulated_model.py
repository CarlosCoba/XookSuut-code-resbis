import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from scipy import special
from scipy.special import erf, gamma

def Polyex(r, M_I):
	#r_PE = r_PE / r_opt
	#r = r/r_opt
	V0,r_PE,alpha  = params(M_I)

	#r = r/40
	#r_PE = r_PE*40
	V = V0*(1-np.exp(-r/r_PE))*(1+alpha*r/r_PE)
	return V


def xi_sq_pdf(x,k=3):
	xi = (1/2.)**(k/2.)*(np.exp(-x/2.))*x**(k/2.-1)/gamma(k/2.)
	xi = xi/np.nanmax(xi)
	return xi

# Bertola 1991
def bertola(r_sky,v_max,c0,p):
	v = v_max*r_sky/(r_sky**2 + c0**2)**(p/2.)
	return v


def meseta(x,a,n,A):
	y = np.exp(-2*(x/a)**n)
	return y*A




def gamma_pdf(x,t,k):
	A = np.exp(-x/t)*x**(k-1)/gamma(k)*t**k
	return A/np.nanmax(A)



def plot_true(name,vmode,axis):

	galaxy = "%s.fits.gz"%name
	#hdu = fits.open("/home/carlos/simulation/manga/sim/%s"%galaxy)
	hdu = fits.open("/home/carlos/simulation_paper/%s"%galaxy)
	#hdu = fits.open("/home/carlos/simulation/%s"%galaxy)
	hdr = hdu[0].header
	#PA, INC,XC,YC,VSYS,PHI_BAR,AMP_VT2,AMP_VR2,R_OPT,R_BAR,CDELT = hdr["PA"], hdr["INC"],hdr["XC"],hdr["YC"],hdr["VSYS"],hdr["PHI_BAR"],hdr["AMP_VT2"],hdr["AMP_VR2"],hdr["R_OPT"],hdr["R_bar"],abs(hdr["CDELT1"]) 
	

	PA, INC,XC,YC,VSYS,PHI_BAR,AMP_VT2,AMP_VR2,R_OPT,CDELT = hdr["PA"], hdr["INC"],hdr["XC"],hdr["YC"],hdr["VSYS"],hdr["PHI_BAR"],hdr["AMP_VT2"],hdr["AMP_VR2"],hdr["R_OPT"],abs(hdr["CDELT1"]) 
	
	pix_scale = CDELT*3600
	#R_arc = np.linspace(0,R_OPT,100)*pix_scale
	R_arc = np.linspace(0,R_OPT,100)
	#Vrot = bertola(R_arc,180,0.5*pix_scale,0.9)
	Vrot = bertola(R_arc, v_max = 170, c0 = 2.68, p = 1.12 )

	#R_BAR, n = R_BAR*pix_scale, 4
	#Vr2 = meseta(R_arc,R_BAR,n,AMP_VR2)
	#Vt2 = meseta(R_arc,R_BAR,n,AMP_VT2)

	Amp_vt2, Amp_vr2 = 30, 20
	Vt2 = gamma_pdf(R_arc,t=2,k=3.5)*Amp_vt2
	Vr2 = gamma_pdf(R_arc,t=2,k=3)*Amp_vr2



	if vmode == "bisymmetric": 

		axis.plot(R_arc,Vt2,color = "#2fa7ce",linestyle='--', alpha = 1, linewidth=1)
		axis.plot(R_arc,Vr2,color = "#c73412",linestyle='--', alpha = 1, linewidth=1)
		axis.plot(R_arc,Vrot,color = "k",linestyle='--', alpha = 1, linewidth=1)

	if vmode == "radial": 

		axis.plot(R_arc,Vt2,color = "#2fa7ce",linestyle='--', alpha = 1, linewidth=1)
		axis.plot(R_arc,Vrot,color = "k",linestyle='--', alpha = 1, linewidth=1)

	if vmode == "circular":
		axis.plot(R_arc,Vrot,color = "k",linestyle='--', alpha = 1, linewidth=1)

