import bilby
import matplotlib.pyplot as plt
from scipy.special import erfinv
import numpy as np
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.stats import norm 
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline
from bilby.core.utils import check_directory_exists_and_if_not_mkdir


class power_gaussian(bilby.core.prior.Prior):
    
    #input all parameters
    def __init__(self, mu, mu2, sigma, sigma2, alpha, beta, gamma, minimum, maximum, name = None, unit = None, boundary = None, latex_label = r'm_1 (M$_\odot)$'):
        # define what these parameters are
        super (power_gaussian, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, unit = None, boundary = None
        )
        self.alpha = alpha # this is the power-law factor
        self.beta = beta # this is the exponential factor
        self.gamma = gamma # this is just a constant needed to get the correct shape
        self.mu = mu  # this is the mean of the first Gaussian peak
        self.mu2 = mu2 # this is the mean of the second Gaussian peak
        self.sigma = sigma # this is the standard deviation of the first Gaussian peak
        self.sigma2 = sigma2 # this is the standard deviation of the second Gaussian peak
        
        
    
    # Creating the function 
    
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) * np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (40 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) + (200 * (np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2))
      
      
    
# Inputting values for the parameters  

prior_power_gaussian = power_gaussian(name="name", mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(100), alpha = 0.31, beta = -0.06, gamma = 0.1, minimum = 0.00001, maximum = 120)

# this is range of x values (mass) going from just above 0 to 120 solar masses

x, dx = np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000, retstep=True)

# y_i represents the normalised probability values

y_i = prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)) / np.sum(dx * prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)))

# producing the cdf of the function numerically

cdf = np.cumsum(y_i * dx)

  
# finding the inverse cdf through interpolation

from scipy.interpolate import InterpolatedUnivariateSpline

x_int = cdf
y_int = x


f = InterpolatedUnivariateSpline(x_int, y_int, k=1)

from scipy.interpolate import interp1d
f = interp1d(x_int, y_int, kind = 'linear')

# Creating a dense array of x values

x_dense = x_int

y_dense = f(x_dense)

# storing the values of the inverse cdf in an array

y_array = np.array([])

for i in np.arange(prior_power_gaussian.minimum, 1,0.001):
    y_new = np.interp(i, x_dense, y_dense)
    y_array = np.append(y_array, y_new)

    
# now want to define a class with the prior probabilities and the inverse cdf values for the rescale


class power_gaussian2(bilby.core.prior.Prior):
    
    #input all parameters
    def __init__(self, mu, mu2, sigma, sigma2, alpha, beta, gamma, xx, yy, minimum, maximum, name = None, unit = None, boundary = None, latex_label = r'm_1 (M$_\odot)$'):
        # define what these parameters are
        super (power_gaussian2, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, unit = None, boundary = None
        )
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        self.xx = xx
        self.yy = yy
        self.inverse_cumulative_distribution = None

    # would like the rescale to be the array of inverse cdf values calculated numerically
    # rescale requires two positional arguments
    
    def _initialize_attributes(self):
        self.inverse_cumulative_distribution = interp1d(self.yy, self.xx, kind = 'linear')
    
    def rescale(self,val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled
        
    
    # Same probability function as power_gaussian
    
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return (np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) * np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (40 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) + (200 * (np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2)))
        


# Inputting values for the parameters 


prior_power_gaussian2 = power_gaussian2(name="name", mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(100), alpha = 0.31, beta = -0.06, gamma = 0.1, xx = x, yy = cdf,  minimum = 0.00001, maximum = 120)


# Injecting parameters

duration = 4
sampling_frequency =  2048
outdir = "visualising2"
label = "example"

injection_parameters = dict(
	mass_1 = 36.0,
	mass_2 = 29.0,
	a_1 = 0.4,
	a_2 = 0.3,
	tilt_1 = 0.5,
	tilt_2 = 1.0,
	phi_12 = 1.7,
	phi_jl = 0.3,
	luminosity_distance = 1000.0,
	theta_jn = 0.4,
	phase = 1.3,
	ra = 1.375,
	dec = -1.2108,
	geocent_time = 1126259542.413,
	psi = 2.659,
)


# Defining the waveform arguments

waveform_arguments = dict(
	waveform_approximant = "IMRPhenomXP",
	reference_frequency = 50.0,
)

# Defining the waveform generator

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
	sampling_frequency = sampling_frequency,
	duration = duration,
	frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
	parameters = injection_parameters,
	waveform_arguments = waveform_arguments,
)


# Setting up interferometers

ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
	duration=duration,
	sampling_frequency = sampling_frequency,
	start_time = injection_parameters["geocent_time"] - 2,
)

_ = ifos.inject_signal(
	waveform_generator = waveform_generator, parameters = injection_parameters
)

# Define priors as a delta function at a particular value,
# then manually change the mass and luminosity distance prior


priors = bilby.gw.prior.BBHPriorDict(injection_parameters.copy())

# Using the new prior for masses


priors["mass_1"] = prior_power_gaussian2
priors["mass_2"] = prior_power_gaussian2
priors["luminosity_distance"] = bilby.core.prior.Uniform(400, 2000, "luminosity_distance")


# Calculating Likelihood

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
	interferometers = ifos, waveform_generator = waveform_generator)
	
# Obtaining results

result = bilby.core.sampler.run_sampler(
	likelihood = likelihood,
	priors = priors,
	sampler = "dynesty",
	npoints = 100,
	injection_parameters = injection_parameters,
	outdir = outdir,
	label = label,
	walks = 5,
	nact = 2,
)
result.plot_corner()




