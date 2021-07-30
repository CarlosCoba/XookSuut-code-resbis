import numpy as np
from astropy.io import fits
from phi_bar_sky import error_pa_bar_sky


def save_model(galaxy,vmode,R,Vrot,e_Vrot,Vrad,e_Vrad,Vtan,e_Vtan,PA,INC,XC,YC,VSYS,THETA,PA_BAR_MAJOR,PA_BAR_MINOR,errors_fit,save = 1):
	#m = len(MODELS)
	n = len(Vrot)


	eVrot = e_Vrot
	eVrad = e_Vrad
	eVtan = e_Vtan


	if vmode == "circular":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			data = np.zeros((3,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = eVrot

	if vmode == "radial":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			#Vrad,eVrad = MODELS[1],eMODELS[1]
			data = np.zeros((5,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = eVrot
			data[4][:] = eVrad



	if vmode == "bisymmetric" or vmode == "resbis"  or vmode == "twostep":
			#Vrot,eVrot = MODELS[0],eMODELS[0]
			#Vrad,eVrad = MODELS[1],eMODELS[1]
			#Vtan,eVtan = MODELS[2],eMODELS[2]
			data = np.zeros((7,n))
			data[0][:] = R
			data[1][:] = Vrot
			data[2][:] = Vrad
			data[3][:] = Vtan
			data[4][:] = eVrot
			data[5][:] = eVrad
			data[6][:] = eVtan


	if save == 1:

		e_Vrot,e_Vrad,e_PA,e_INC,XC_e,YC_e,e_Vsys,e_theta,e_Vtan = errors_fit

		hdu = fits.PrimaryHDU(data)

		if vmode == "circular":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'error circular velocity (km/s)'
		if vmode == "radial":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'radial velocity (km/s)'
			hdu.header['NAME3'] = 'error circular velocity (km/s)'
			hdu.header['NAME4'] = 'error radial velocity (km/s)'
		if vmode == "bisymmetric" or vmode == "resbis"  or vmode == "twostep":
			hdu.header['NAME0'] = 'deprojected distance (arcsec)'
			hdu.header['NAME1'] = 'circular velocity (km/s)'
			hdu.header['NAME2'] = 'radial velocity (km/s)'
			hdu.header['NAME3'] = 'tangencial velocity (km/s)'
			hdu.header['NAME4'] = 'error circular velocity (km/s)'
			hdu.header['NAME5'] = 'error radial velocity (km/s)'
			hdu.header['NAME6'] = 'error tangencial velocity (km/s)'


		hdu.header['PA'] = PA
		hdu.header['e_PA'] = e_PA
		hdu.header['INC'] = INC
		hdu.header['e_INC'] = e_INC
		hdu.header['VSYS'] = VSYS
		hdu.header['e_VSYS'] = e_Vsys
		hdu.header['XC'] = XC
		hdu.header['e_XC'] = XC_e
		hdu.header['YC'] = YC
		hdu.header['e_YC'] = YC_e

		if vmode == "bisymmetric" or vmode == "resbis" or vmode == "twostep":
			hdu.header['HIERARCH PHI_BAR'] = THETA
			hdu.header['HIERARCH e_PHI_BAR'] = e_theta
			hdu.header['HIERARCH PA_BAR_MAJOR'] = PA_BAR_MAJOR
			hdu.header['HIERARCH e_PA_BAR_MAJOR'] = error_pa_bar_sky(PA,INC,THETA,e_PA,e_INC,e_theta)
			hdu.header['HIERARCH PA_BAR_MINOR'] = PA_BAR_MINOR
			hdu.header['HIERARCH e_PA_BAR_MINOR'] = error_pa_bar_sky(PA,INC,THETA-90,e_PA,e_INC,e_theta)
		
		hdu.writeto("./models/%s.%s.1D_model.fits"%(galaxy,vmode),overwrite=True)



