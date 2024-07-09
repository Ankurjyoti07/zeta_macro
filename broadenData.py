import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import erf
from lmfit import Parameters, minimize
import warnings
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


class LineListReader:
    @staticmethod
    def read_line_list(filename):
        line_centers = []
        line_widths = []

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()  # removing whitespaces
                if not line:
                    continue  # skipping empty lines
                parts = line.split()
                center = float(parts[0])
                if len(parts) > 1:
                    width = float(parts[1])
                else:
                    width = 10.0  # default
                line_centers.append(center)
                line_widths.append(width)

        return line_centers, line_widths

class GaussianModel:
    @staticmethod
    def gauss(x, a, center, R, gamma):
        sigma = 4471 / (2.0 * R * np.sqrt(2.0 * np.log(2)))
        return a * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + gamma

class DataGenerator:
    @staticmethod
    def generate_data(wave, flux, line_centers, line_widths, wavelength_slices):
        interp_func = interp1d(wave, flux, kind='linear')
        wave_slices = []
        flux_slices = []
        for center, width in zip(line_centers, line_widths):
            new_wave = np.linspace(center - width, center + width, wavelength_slices)
            new_flux = interp_func(new_wave)
            wave_slices.append(new_wave)
            flux_slices.append(new_flux)
        return np.concatenate(wave_slices), np.concatenate(flux_slices)


class Model_broad:
    def __init__(self, wave, flux):
        self.x = wave
        self.y = flux

class ModelBroadener:
    @staticmethod
    def broaden(model, vsini, epsilon=0.5, linear=False, findcont=False):
        # Implementation of Broaden method as provided# Remove NaN values from the flux array and corresponding wavelength values
        #non_nan_idx = ~np.isnan(model.y)
        
        wvl = model.x
        flx = model.y
        
        dwl = wvl[1] - wvl[0]
        binnu = int(np.floor((((vsini/10)/ 299792.458) * max(wvl)) / dwl)) + 1 #adding extra bins for error handling
        #validIndices = np.arange(len(flx)) + binnu => this was used in rotbroad as a user cond ==> this is always on here
        front_fl = np.ones(binnu) * flx[0]
        end_fl = np.ones(binnu) * flx[-1]
        flux = np.concatenate((front_fl, flx, end_fl))

        front_wv = (wvl[0] - (np.arange(binnu) + 1) * dwl)[::-1]
        end_wv = wvl[-1] + (np.arange(binnu) + 1) * dwl
        wave = np.concatenate((front_wv, wvl, end_wv))

        if not linear:
            x = np.logspace(np.log10(wave[0]), np.log10(wave[-1]), len(wave))
        else:
            x = wave
            
        if findcont:
            # Find the continuum
            model.cont = np.ones_like(flux)  # Placeholder for continuum finding
            
        # Make the broadening kernel
        dx = np.log(x[1] / x[0])
        c = 299792458  # Speed of light in m/s
        lim = vsini / c
        if lim < dx:
            warnings.warn("vsini too small ({}). Not broadening!".format(vsini))
            return Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
        
        d_logx = np.arange(0.0, lim, dx)
        d_logx = np.concatenate((-d_logx[::-1][:-1], d_logx))
        alpha = 1.0 - (d_logx / lim) ** 2
        B = (1.0 - epsilon) * np.sqrt(alpha) + epsilon * np.pi * alpha / 4.0  # Broadening kernel
        B /= np.sum(B)  # Normalize

        # Do the convolution
        broadened = Model_broad(wave.copy(), flux.copy())  # Create a copy of the Model object
        broadened.y = fftconvolve(flux, B, mode='same')
        
        return broadened



class MacroBroadener:
    @staticmethod
    def macro_broaden(xdata, ydata, vmacro):
        c = 299792458 #~constants.c.cgs.value * units.cm.to(units.km)
        sq_pi = np.sqrt(np.pi)
        lambda0 = np.median(xdata)
        xspacing = xdata[1] - xdata[0]
        mr = vmacro * lambda0 / c
        ccr = 2 / (sq_pi * mr)

        px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
        pxmr = abs(px) / mr
        profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

        before = ydata[int(-profile.size / 2 + 1):]
        after = ydata[:int(profile.size / 2 +1)] #add one to fix size mismatch
        extended = np.r_[before, ydata, after]

        first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
        last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
        
        x2 = np.linspace(first, last, extended.size)  #newdata x array ==> handles edge effects

        conv_mode = "valid"

        newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

        return newydata
        # Implementation of macro_broaden method as provided

def generate_broaden(params, line_centers, line_widths, wavelength_slices):
    model_slices = []
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        wave = np.linspace(center - width, center + width, wavelength_slices)

        instrum = GaussianModel.gauss(wave, params[f'a{i}'], params[f'center{i}'], 20000, params[f'gamma{i}'])
        broad_rot = ModelBroadener.broaden(Model_broad(wave, instrum), params['vsini'])

        broad_macro = MacroBroadener.macro_broaden(broad_rot.x, broad_rot.y, params[f'vmacro{i}'])
        interp = interp1d(broad_rot.x, broad_macro, kind='linear')
        broad_flux = interp(wave)
        model_slices.append(broad_flux)

    return np.concatenate(model_slices)


def objective(params, wave, flux, line_centers, line_widths, wavelength_slices):
    wave_data, flux_data = DataGenerator.generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    model = generate_broaden(params, line_centers, line_widths, wavelength_slices)
    return flux_data - model

def fit_lines(wave, flux, line_centers, line_widths, wavelength_slices):
    params = Parameters()
    wave_data, flux_data = DataGenerator.generate_data(wave, flux, line_centers, line_widths, wavelength_slices)
    for i, (center, width) in enumerate(zip(line_centers, line_widths)):
        params.add(f'a{i}', value=-1)   # Initial guess for amplitude
        params.add(f'center{i}', value=center)  # Initial guess for center
        params.add(f'gamma{i}', value=1)
        params.add(f'vmacro{i}', value=150000, min=0, max=500000)
    params.add('vsini', value=150000, min=0, max=500000)

    result = minimize(objective, params=params, args=(wave_data, flux_data, line_centers, line_widths, wavelength_slices))
    return result

class BroadenData:
    def __init__(self, vsini, vmacro, resolution, line_list_file, wave, flux, wavelength_slices):
        self.vsini = vsini
        self.vmacro = vmacro
        self.resolution = resolution
        self.line_list_file = line_list_file
        self.wave = wave
        self.flux = flux
        self.wavelength_slices = wavelength_slices

    def fit_lines(self):
        # Read line list
        line_centers, line_widths = LineListReader.read_line_list(self.line_list_file)

        # Generate data for fitting
        wave_data, flux_data = DataGenerator.generate_data(self.wave, self.flux, line_centers, line_widths, self.wavelength_slices)

        # Create parameters for fitting
        params = Parameters()
        for i, (center, width) in enumerate(zip(line_centers, line_widths)):
            params.add(f'a{i}', value=-1)   # Initial guess for amplitude
            params.add(f'center{i}', value=center)  # Initial guess for center
            params.add(f'gamma{i}', value=1)
            params.add(f'vmacro{i}', value=self.vmacro, min=0, max=500000)
        params.add('vsini', value=self.vsini, min=0, max=500000)

        # Perform fitting
        result = minimize(objective, params=params, args=(wave_data, flux_data, line_centers, line_widths, self.wavelength_slices))
        return result
