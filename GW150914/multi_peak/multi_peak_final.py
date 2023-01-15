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

# Creating out directory for plots

outdir = "multi_peak_prior"
check_directory_exists_and_if_not_mkdir(outdir)

# Creating the multi-peak function. This has been solved analytically

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
prior_int = Interped(xx = x,  yy = y_i, minimum = 0.00001, maximum = 120)


# In the examples the dictionary of priors is given the variable name 'priors' and the mass priors are named "mass_1" and "mass_2"
# In order to replace the standard mass priors with this new one use: priors["mass_1"] = prior_int 
# Similarly use: priors["mass_2"] = prior_int 
 



