#!/home/paul/anaconda2/bin/python
import scipy.integrate as integrate
import numpy as np


def harm_pot(xarr, fcon, zero):
    pot = (fcon/2.0)*(xarr-zero)**2
    return pot


def riemanint(xarray, yarray):
    binwidth = (xarray[-1]-xarray[0])/len(xarray)
    result = 0
    for height in yarray[:-1]:
        result = result+height*binwidth

    return result


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def calc_G_I(xydata_np_arary, siteintervall, bulk):
    xarr = xydata_np_arary.T[0]
    yarr = xydata_np_arary.T[1]
    kbT = 0.5961
    index_bulk = find_nearest(xarr, bulk)
    id_s = [find_nearest(xarr, siteintervall[0]),
            find_nearest(xarr, siteintervall[1])]
    yarr_site = yarr[id_s[0]:id_s[1]]
    xarr_site = xarr[id_s[0]:id_s[1]]

    Ipart1 = integrate.simps(np.exp(-(yarr_site)/kbT), xarr_site)
    Ipart2 = np.exp((yarr[index_bulk])/kbT)
    I = Ipart1*Ipart2
    result = -kbT*np.log(I)
    return result


# my dumb much to late try...
def calc_G_I_alt(xydata_np_arary, siteintervall, bulk):
    xarr = xydata_np_arary.T[0]
    yarr = xydata_np_arary.T[1]
    kbT = 0.5961
    id_s = [find_nearest(xarr, siteintervall[0]),
            find_nearest(xarr, siteintervall[1])]
    yarr_site = yarr[id_s[0]:id_s[1]]
    xarr_site = xarr[id_s[0]:id_s[1]]
    xarr_bulk = xarr[find_nearest(xarr, bulk):find_nearest(xarr, bulk+1)]
    yarr_bulk = yarr[find_nearest(xarr, bulk):find_nearest(xarr, bulk+1)]

    Ipart1 = integrate.simps(np.exp(-(yarr_site)/kbT),
                             xarr_site)
    Ipart2 = integrate.simps(np.exp(-yarr_bulk)/kbT, xarr_bulk)
    I = Ipart1/Ipart2
    result = -kbT*np.log(I)
    return result


def calc_G_a_bulk(fcon_alpha, zero_alpha,
                  fcon_beta, zero_beta,):
    ''' alpha, beta and gamma are the orientational threepoin angle and
    dihedral angels. meaning alpha goes from 0 to pi and the other to 0
    to 2pi'''
    kbT = 0.5961

    alpha = np.linspace(0, np.pi, 1000)
    f_alpha = np.sin(alpha)*np.exp(-harm_pot(alpha, fcon_alpha, zero_alpha)/kbT)
    result_alpha = integrate.simps(f_alpha, alpha)

    beta = np.linspace(0, 2*np.pi, 1000)
    f_beta = np.exp(-harm_pot(beta, fcon_beta, zero_beta)/kbT)
    result_beta = integrate.simps(f_beta, beta)

    result = -kbT*np.log((result_beta*result_alpha)/(4*np.pi))
    return result


# my dumb much to late try...ends here


def calc_G_S(xydata_np_arary, bulk,
             theta_fcon, theta_zero,
             phi_fcon, phi_zero):
    xarr = xydata_np_arary.T[0]
    index_bulk = find_nearest(xarr, bulk)
    kbT = 0.5961
    theta = np.arange(0, np.pi, 0.01)
    theta_pot = harm_pot(theta, theta_fcon, theta_zero)
    U_theta = np.sin(theta)*np.exp(-(theta_pot)/kbT)
    Spart_theta = integrate.simps(U_theta, theta)

    phi = np.arange(0, 2*np.pi, 0.01)
    phi_pot = harm_pot(phi, phi_fcon, phi_zero)
    U_phi = np.exp(-(phi_pot)/kbT)
    Spart_phi = integrate.simps(np.exp(-(U_phi)/kbT), phi)
    S = (xarr[index_bulk]**2)*Spart_phi*Spart_theta
    result = -kbT*np.log(S)
    return result


def calc_std_C():
    return -0.59*np.log(1.0/1661.0)


def calc_G_rmsd(xydata_np_arary, fcon, zero):
    xarr = xydata_np_arary.T[0]
    yarr = xydata_np_arary.T[1]
    kbT = 0.5961
    norm = integrate.simps(np.exp(-(yarr)/kbT), xarr)
    potential = harm_pot(xarr, fcon, zero)
    energy = integrate.simps(np.exp(-(yarr+potential)/kbT), xarr)
    result = -kbT*np.log(energy/norm)
    return result


def calc_G_FEP_harmpot(sampled_data, fcon, zero):
    kbT = 0.5961
    average = np.sum(np.exp(-(1/kbT)*(fcon/2)*(sampled_data-zero)**2))
    average = average/len(sampled_data)
    result = -kbT*np.log(average)
    return result


def calc_G_FEP_2harmpots(sampled_data1, fcon1, zero1,
                         sampled_data2, fcon2, zero2):
    kbT = 0.5961
    average = np.sum(np.exp(-(1/kbT)*(((fcon1/2)*(sampled_data1-zero1)**2) +
                            (fcon2/2)*(sampled_data2-zero2)**2)))
    average = average/len(sampled_data2)
    result = -kbT*np.log(average)
    return result


def clac_G_FEP(samples, pots):
    '''takes a list of numpy arrays with sample data and one with potential
    functions to apply onto this data, potential functions and dta are connected
    by their positions in the lists'''
    kbT = 0.5961
    toaverage = 1
    for sample, pot in zip(samples, pots):
        toaverage = toaverage*np.exp(-(1.0/kbT)*pot(sample))
    average = sum(toaverage)/len(toaverage)
    result = -kbT*np.log(average)
    return result


def calc_G_o_bulk(fcon_alpha, zero_alpha,
                  fcon_beta, zero_beta,
                  fcon_gamma, zero_gamma):
    ''' alpha, beta and gamma are the orientational threepoin angle and
    dihedral angels. meaning alpha goes from 0 to pi and the other to 0
    to 2pi'''
    kbT = 0.5961

    alpha = np.linspace(0, np.pi, 1000)
    f_alpha = np.sin(alpha)*np.exp(-harm_pot(alpha, fcon_alpha, zero_alpha)/kbT)
    result_alpha = integrate.simps(f_alpha, alpha)

    beta = np.linspace(0, 2*np.pi, 1000)
    f_beta = np.exp(-harm_pot(beta, fcon_beta, zero_beta)/kbT)
    result_beta = integrate.simps(f_beta, beta)

    gamma = np.linspace(0, 2*np.pi, 1000)
    f_gamma = np.exp(-harm_pot(gamma, fcon_gamma, zero_gamma)/kbT)
    result_gamma = integrate.simps(f_gamma, gamma)

    result = -kbT*np.log((result_gamma*result_beta*result_alpha)/(8*np.pi**2))
    return result
