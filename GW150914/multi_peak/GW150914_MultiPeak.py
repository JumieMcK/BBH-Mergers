import bilby
from gwpy.timeseries import TimeSeries
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

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW150914"

# Note you can get trigger times using the gwosc package, e.g.:
# > from gwosc import datasets
# > datasets.event_gps("GW150914")
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

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
        
    
    # Creating the function - factors of 40 and 200 required for the height of the Gaussian peaks
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        try:
            return np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) * np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (40 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) + (200 * (np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2))
        except ZeroDivisionError:
            return np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) * np.log(120/0.00001)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (40 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)) + (200 * (np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2))
            
      

    
# Inputting values for the parameters - setting minimum to zero alters shape despite ZeroDivisionError, so it is just above 0    
prior_power_gaussian = power_gaussian(name="name", mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(100), alpha = 0.31, beta = -0.06, gamma = 0.1, minimum = 0.00001, maximum = 120)

# this is range of x values (mass) going from just above 0 to 120 solar masses
x, dx = np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000, retstep=True)

# y_i represents the normalised probability values
y_i = prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)) / np.sum(dx * prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)))

# numerically solving for the cumulative distribution function
cdf = np.cumsum(y_i * dx)


# Interpolating the prior function using the mass values (x=xx) and the normalised probability values (y_i = yy)

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
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=False)

# inputting values for interpolated function (xx and yy already contain the values for alpha, beta etc.)        




# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict("multipeak.prior")
priors["mass_1"] = Interped(name="mass_1",xx = x,  yy = y_i, minimum = 10, maximum = 120)
priors["mass_2"] = Interped(name="mass_2",xx = x,  yy = y_i, minimum = 10, maximum = 120)

# Add the geocent time prior
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)



# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomXP",
        "reference_frequency": 50,
    },
)

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
# Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    phase_marginalization=False,
    distance_marginalization=True,
)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample='rwalk',
    walks= 100,
    nact= 50,
    outdir=outdir,
    label=label,
    nlive=1000,
    check_point_delta_t=600,
    check_point_plot=True,
    npool=1,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
result.plot_corner()

