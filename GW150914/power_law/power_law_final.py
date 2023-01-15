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
from scipy.interpolate import interp1d


outdir_bp = "power_law_test"
check_directory_exists_and_if_not_mkdir(outdir_bp)



class exp_PowerLaw(bilby.core.prior.Prior):
    
    def __init__(self, alpha, beta, mu, sigma, minimum, maximum, name = None, latex_label = r'M$_\odot$'):
        super (exp_PowerLaw, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum, maximum=maximum
        )
        self.alpha = alpha 
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        
        
    def rescale(self, val):
        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val * 
                (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))



    
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        if self.alpha == -1:
            return ((self.beta *
            np.exp(self.beta * val)
            / (np.exp(self.beta * self.maximum) - np.exp(self.beta * self.minimum))
            * in_prior)  + np.nan_to_num(1 / val / np.log(self.maximum/self.minimum)) * self.is_in_prior_range(val)) 
        else: 
            return np.nan_to_num(self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.alpha / self.beta) * np.log(self.maximum/self.minimum)) / (self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (80 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) 
        

        
prior = exp_PowerLaw(name="name", alpha = 0.31, beta = -0.06, mu = 35, sigma = np.sqrt(5), minimum = 0.0001, maximum = 120) 



x1 = np.linspace(prior.minimum, prior.maximum, 1000)
x2 = np.linspace(80, prior.maximum, 1000)
x3 = [*x1, *x2]

xi = np.asarray(x3)

#ma_xi = np.ma.masked_inside(xi, 20, 30)


plt.figure(figsize=(12,5))
plt.hist(prior.sample(100000), bins=100, histtype = "step", density = True)
if not isinstance(prior, bilby.core.prior.Gaussian):
        plt.plot(
            x1,
    prior.prob(x1),
) 

else:
    plt.plot(np.linspace(-5, 5, 1000), prior.prob(np.linspace(-5, 5, 1000)))

plt.xlabel("{}".format(prior.latex_label))
plt.show()
plt.savefig('power_law_test/bpl_base.jpg')



      
c = -3
l = 1
p = (c*(x1-50)**(l)) + np.interp(50,x1,prior.prob(x1))

slope, intercept = np.polyfit(p, x1, 1)


    

plt.plot(x1,p)
plt.savefig("power_law_test/comp_powerlaw.png")
    
class broken_PowerLaw(bilby.core.prior.Prior):
    
    def __init__(self, alpha, beta, phi, mu, sigma, epsilon, minimum, maximum, name = None, latex_label = r'M$_\odot$'):
        super (broken_PowerLaw, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum, maximum=maximum
        )
        self.alpha = alpha 
        self.beta = beta
        self.phi = phi
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        

    def prob_2(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        if self.minimum <= val < 50:
            return np.nan_to_num(self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.alpha / self.beta) * np.log(self.maximum/self.minimum)) / (self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (80 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) 
        
        else: 
            return np.nan_to_num(-500*(val-50) ** self.phi * (1 + self.phi) /
                                 ((self.maximum-50) ** (1 + self.phi) - 
                                  (self.minimum-50) ** (1 + self.phi))) * self.is_in_prior_range(val) + np.interp(50,x1,prior.prob(x1))
    
        
print(np.interp(50,x1,prior.prob(x1)))
print(intercept)
print(slope)

    
prior_2 = broken_PowerLaw(name="name", alpha = 0.31, beta = -0.06, phi=1,  mu = 35, sigma = np.sqrt(5), epsilon = -1, minimum = 0.0001, maximum = 120) 

f_val = np.array([prior_2.prob_2(val) for val in x1]) 

f_val_pos = f_val[f_val>=0]

#print(f_val_pos)

plt.figure(figsize=(12,5))
plt.plot(np.linspace(prior.minimum, prior.maximum, len(f_val_pos)), f_val_pos, label = r"factor $=$ 500, phi $=$ 1")
plt.xlabel("{}".format(prior_2.latex_label))
plt.legend(fontsize=10)
plt.show()
plt.savefig("power_law_test/pl_test.png")


plt.figure(figsize=(12,5))
#plt.hist(prior_2.sample(100000), bins=100, histtype = "step", density = True)
if not isinstance(prior_2, bilby.core.prior.Gaussian):
        plt.plot(
            x1,
             f_val, label = r"factor $=$ 500, phi $=$ 1"
        )
else:
    plt.plot(np.linspace(-5, 5, 1000), prior_2.prob_2(np.linspace(-5, 5, 1000)))

plt.xlabel("{}".format(prior_2.latex_label))
plt.ylim(0,150)
plt.legend(fontsize=10)
plt.show()
plt.savefig("power_law_test/v17_ebpg.png")



plt.figure(figsize=(12,5))
#plt.hist(prior_2.sample(100000), bins=100, histtype = "step", density = True)
if not isinstance(prior_2, bilby.core.prior.Gaussian):
        plt.plot(
            np.linspace(prior_2.minimum, prior_2.maximum, len(f_val_pos)),
             f_val_pos, label = r"factor $=$ 500, phi $=$ 1"
        )
else:
    plt.plot(np.linspace(-5, 5, 1000), prior_2.prob_2(np.linspace(-5, 5, 1000)))

plt.xlabel("{}".format(prior_2.latex_label))
plt.ylim(0,150)
plt.legend(fontsize=10)
plt.show()
plt.savefig("power_law_test/pos_graph.png")

# to get truncated change val - 50 to val - intercept
#plt.savefig("scripts_bp/truncated.png")

x, dx = np.linspace(prior_2.minimum, prior_2.maximum, len(f_val_pos), retstep=True)
y_i = f_val_pos / np.sum(dx * f_val_pos)

plt.figure(figsize=(12,5))
plt.plot(np.linspace(prior_2.minimum, prior_2.maximum, len(f_val_pos)), y_i, label = r"factor $=$ 500, phi $=$ 1")
plt.xlabel("{}".format(prior_2.latex_label))
plt.legend(fontsize=10)
plt.show()
plt.savefig("power_law_test/normalised.png")

# Only keep the positive y data
# Next step is to normalise the y data and then use this as well as x data for interpolation

class Interped(bilby.core.prior.Prior):

    def __init__(self, xx, yy, minimum=np.nan, maximum=np.nan, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Creates an interpolated prior function from arrays of xx and yy=p(xx)

        Parameters
        ==========
        xx: array_like
            x values for the to be interpolated prior function
        yy: array_like
            p(xx) values for the to be interpolated prior function
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass

        Attributes
        ==========
        probability_density: scipy.interpolate.interp1d
            Interpolated prior probability distribution
        cumulative_distribution: scipy.interpolate.interp1d
            Interpolated cumulative prior probability distribution
        inverse_cumulative_distribution: scipy.interpolate.interp1d
            Inverted cumulative prior probability distribution
        YY: array_like
            Cumulative prior probability distribution

        """
        self.xx = xx
        self.min_limit = min(xx)
        self.max_limit = max(xx)
        self._yy = yy
        self.YY = None
        self.probability_density = None
        self.cumulative_distribution = None
        self.inverse_cumulative_distribution = None
        self.__all_interpolated = interp1d(x=xx, y=yy, bounds_error=False, fill_value=0)
        minimum = float(np.nanmax(np.array((min(xx), minimum))))
        maximum = float(np.nanmin(np.array((max(xx), maximum))))
        super(Interped, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                       minimum=minimum, maximum=maximum, boundary=boundary)
        self._update_instance()


    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if np.array_equal(self.xx, other.xx) and np.array_equal(self.yy, other.yy):
            return True
        return False

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return self.probability_density(val)


    def cdf(self, val):
        return self.cumulative_distribution(val)


    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return rescaled


    @property
    def minimum(self):
        """Return minimum of the prior distribution.

        Updates the prior distribution if minimum is set to a different value.

        Yields an error if value is set below instantiated x-array minimum.

        Returns
        =======
        float: Minimum of the prior distribution

        """
        return self._minimum

    @minimum.setter
    def minimum(self, minimum):
        if minimum < self.min_limit:
            raise ValueError('Minimum cannot be set below {}.'.format(round(self.min_limit, 2)))
        self._minimum = minimum
        if '_maximum' in self.__dict__ and self._maximum < np.inf:
            self._update_instance()

    @property
    def maximum(self):
        """Return maximum of the prior distribution.

        Updates the prior distribution if maximum is set to a different value.

        Yields an error if value is set above instantiated x-array maximum.

        Returns
        =======
        float: Maximum of the prior distribution

        """
        return self._maximum

    @maximum.setter
    def maximum(self, maximum):
        if maximum > self.max_limit:
            raise ValueError('Maximum cannot be set above {}.'.format(round(self.max_limit, 2)))
        self._maximum = maximum
        if '_minimum' in self.__dict__ and self._minimum < np.inf:
            self._update_instance()

    @property
    def yy(self):
        """Return p(xx) values of the interpolated prior function.

        Updates the prior distribution if it is changed

        Returns
        =======
        array_like: p(xx) values

        """
        return self._yy

    @yy.setter
    def yy(self, yy):
        self._yy = yy
        self.__all_interpolated = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self._update_instance()

    def _update_instance(self):
        self.xx = np.linspace(self.minimum, self.maximum, len(self.xx))
        self._yy = self.__all_interpolated(self.xx)
        self._initialize_attributes()

    def _initialize_attributes(self):
        from scipy.integrate import cumtrapz
        #if np.trapz(self._yy, self.xx) != 1:
            #logger.debug('Supplied PDF for {} is not normalised, normalising.'.format(self.name))
        self._yy /= np.trapz(self._yy, self.xx)
        self.YY = cumtrapz(self._yy, self.xx, initial=0)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        #self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=False)

# inputting values for interpolated function (xx and yy already contain the values for alpha, beta etc.)        
prior_int = Interped(xx = x,  yy = y_i, minimum = 0.0001, maximum = 120)


plt.figure(figsize=(12,5))
plt.hist(prior_int.sample(100000), bins=100, histtype="step", density=True)
plt.plot(x,  prior_int.prob(np.linspace(prior_int.minimum, prior_int.maximum, len(f_val_pos))))
plt.xlabel("{}".format(prior_2.latex_label)) 
plt.show()
plt.savefig('power_law_test/prob.jpg')

plt.figure(figsize=(12,5))
plt.hist(prior_int.sample(100000), bins=100, histtype="step", density=True)
plt.plot(np.linspace(0,1,len(f_val_pos)),  prior_int.rescale(np.linspace(0, 1, len(f_val_pos))))
plt.xlabel("{}".format(prior_2.latex_label)) 
plt.show()
plt.savefig('power_law_test/res.jpg')

# Injecting parameters

duration = 4
sampling_frequency =  2048
outdir = "power_law_test"
label = "power_law"

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


priors["mass_1"] = prior_int
priors["mass_2"] = prior_int
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
