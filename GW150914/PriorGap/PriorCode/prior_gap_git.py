import bilby
import matplotlib.pyplot as plt
from scipy.special import erfinv
import numpy as np
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.stats import norm 
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from scipy.interpolate import interp1d

outdir_bp = "prior_peak_plus_gap"
check_directory_exists_and_if_not_mkdir(outdir_bp)



class power_gaussian(bilby.core.prior.Prior):
    
    #input all parameters
    def __init__(self, mu, mu2, sigma, sigma2, alpha, beta, gamma, minimum, maximum, name = None, unit = None, boundary = None, latex_label = r'm_1 (M$_\odot)$'):
        # define what these parameters are
        super (power_gaussian, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, unit = None, boundary = None
        )
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        
    
    # Added the pieces from the power law and gaussian examples - not sure this is right.
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.abs( np.where(x <=  50, np.nan_to_num(self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) *  np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (2 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)),
                         np.where(x >=  90, np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) *  np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (20 * (np.exp(-(self.mu2 - val) ** 2 / (1 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2)),np.nan)))
    

# Inputting values for the parameters    
prior_power_gaussian = power_gaussian(name="name", mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(150), alpha = 0.31, beta = -0.06, gamma = 0.5, minimum = 2, maximum = 120)


x, dx = np.linspace(2, 120, 1000, retstep=True)



y_i = prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)) / np.nansum(dx * prior_power_gaussian.prob(np.linspace(prior_power_gaussian.minimum, prior_power_gaussian.maximum, 1000)))
cdf = np.nancumsum(y_i * dx)


class power_gaussian2(bilby.core.prior.Prior):
    
    #input all parameters
    def __init__(self, mu, mu2, sigma, sigma2, alpha, beta, gamma, minimum, maximum, name = None, unit = None, boundary = None, latex_label = r'm_1 (M$_\odot)$'):
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

    def rescale(self,val):
        return y_array
        
    
    # Added the pieces from the power law and gaussian examples - not sure this is right.
    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        return np.abs( np.where(x <=  50, np.nan_to_num(self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) *  np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (2 * (np.exp(-(self.mu - val) ** 2 / (2 * self.sigma **2)) / (2 * np.pi) ** 0.5 / self.sigma)),
                         np.where(x >=  90, np.nan_to_num( self.beta * val ** self.alpha * np.exp(self.beta * val) * (1 - (self.gamma * self.alpha / self.beta) *  np.log(self.maximum/self.minimum)) / (self.gamma * self.maximum ** self.alpha * np.exp(self.beta * self.maximum) - (self.minimum ** self.alpha * np.exp(self.beta * self.minimum))) * in_prior)  * self.is_in_prior_range(val) + (20 * (np.exp(-(self.mu2 - val) ** 2 / (2 * self.sigma2 **2)) / (2 * np.pi) ** 0.5 / self.sigma2)),np.nan)))
        
prior_power_gaussian2 = power_gaussian2(name="name", mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(150), alpha = 0.31, beta = -0.06, gamma = 0.5, minimum = 2, maximum = 120)


class Interped(bilby.core.prior.Prior):

    def __init__(self, xx, yy, mu, mu2, dx, sigma, sigma2, alpha, beta, gamma, minimum=np.nan, maximum=np.nan, name=None,
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
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        self.dx = dx
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
        return np.nan_to_num(self.cumulative_distribution(val))
    
    # print this out to see how it looks and compare to numerical
    

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the prior.

        This maps to the inverse CDF. This is done using interpolation.
        """
        
        rescaled = self.inverse_cumulative_distribution(val)
        if rescaled.shape == ():
            rescaled = float(rescaled)
        return np.nan_to_num(rescaled, nan=1)
        
       

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
        self._yy /= np.nan_to_num(np.trapz(self._yy, self.xx), nan=1)
        self.YY = np.nancumsum(self._yy * self.dx)
        # Need last element of cumulative distribution to be exactly one.
        self.YY[-1] = 1
        self.probability_density = interp1d(x=self.xx, y=self._yy, bounds_error=False, fill_value=0)
        self.cumulative_distribution = interp1d(x=self.xx, y=self.YY, bounds_error=False, fill_value=(0, 1))
        self.inverse_cumulative_distribution = interp1d(x=self.YY, y=self.xx, bounds_error=False)
        
prior_int_gap = Interped(  xx = x,  yy = y_i, dx = dx, mu = 35, mu2 = 70, sigma = np.sqrt(10), sigma2 = np.sqrt(150), alpha = 0.31, beta = -0.06, gamma = 0.5, minimum = 2, maximum = 120)


# Use this plot to check if the prior makes sense (sampling histogram should follow probability curve)

plt.figure(figsize=(12,5))
plt.hist(prior_int_gap.sample(100000), bins=100, histtype="step", density=True)
plt.plot(x,  prior_int_gap.prob(np.linspace(prior_int_gap.minimum, prior_int_gap.maximum, 1000)))
plt.xlabel("{}".format(prior_int_gap.latex_label)) 
plt.show()
plt.savefig('prior_peak_plus_gap/prob.jpg')




