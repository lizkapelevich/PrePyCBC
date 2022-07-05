import numpy as np
import math
from scipy import interpolate

def get_noise(t_0, t_end, del_T):
    
    """
    This function will take as input the boundaries of a time stamp
    from some noise and the interval between each value.
    
    INPUT:
    ------
    t_0 : start time of noise
    t_end : end time of noise
    del_T : interval of time between each value
    
    RETURNS:
    --------
    The time series for random noise.
    
    """

    duration_data = t_end - t_0
    number_of_values = duration_data / del_T                        # calculating number of time stamps
    del_T_prime = duration_data / math.ceil(number_of_values)       # new interval of the noise data to allow  
                                                                    # array to reach end point
    
    time_series = np.arange(t_0, t_end + del_T_prime, del_T_prime)  # array from beginning to end of noise time stamp
                                                                    # with increments of the standard deviation
    noise = -1 + 2 * np.random.random(len(time_series))             # random values with same length as time_series
    
    return (time_series, noise, del_T_prime)


def get_signal(A, t_0, t_end, del_T, f, sigma):
    
    """
    This function will take as input the amplitude, boundaries of a time stamp
    from a signal, the frequency, and the standard deviation.
    
    INPUT:
    ------
    f : frequency
    A : amplitude
    sigma : standard deviation
    t_0 : start time of a signal
    t_end : end time of a signal
    del_T : interval of time between each value
    
    RETURNS:
    --------
    The time series for a signal.
    
    """

    duration_data = t_end - t_0
    number_of_values = duration_data / del_T                        # calculating number of time stamps
    del_T_prime = duration_data / math.ceil(number_of_values)       # new interval of the noise data to allow  
                                                                    # array to reach end point
    t = np.arange(t_0, t_end + del_T_prime, del_T_prime)
    
    t_mean = (t_0 + t_end) / 2                                      # finding center of signal
    S = A*np.sin(2*np.pi*f*t)*np.exp((-(t - t_mean)**2)/(2*sigma))  # calculating sine gaussian

    return (t, S, del_T_prime)


def final_data(a, t_signal_start, t_signal_end, t_noise_start, 
               t_noise_end, del_T, f, sigma):
    """
    This function will take as input the amplitude, time boundaries
    of the signal and noise, interval between their values, the
    frequency, and standard deviation.
    
    INPUT:
    ------
    A : amplitude
    f : frequency
    sigma : standard deviation
    t_noise_end : end time of noise
    t_signal_end : end time of signal
    t_noise_start : start time of noise
    t_signal_start : start time of signal
    del_T : interval of time between each value
    
    RETURNS:
    --------
    The calculation for embedding a signal within random noise.
    
    """
    time_series_signal, signal_values, del_T_signal = get_signal(a, t_signal_start, t_signal_end,
                                     del_T, f, sigma)
    time_series_noise, noise_values, del_T_noise = get_noise(t_noise_start, t_noise_end, del_T)
    
    s = interpolate.interp1d(time_series_signal, signal_values)
    
    index1 = time_series_noise > time_series_signal[0]            # boolean = False before signal in noise time-series
    index2 = time_series_noise < time_series_signal[-1]           # boolean = False after signal in noise time_series
    time_series_intersect = index1*index2                         # multiplying arrays into one array of booleans
    
    zeroes = np.zeros_like(noise_values)                          # create an array of zeroes same length as noise values

    signal_time_stamps = time_series_noise[time_series_intersect]       # keeping only boolean = True
    signal = s(signal_time_stamps)

    data_in_signal = noise_values[time_series_intersect] + signal       # embedding signal in noise
    data_before_signal = noise_values[index2*(~time_series_intersect)]  # prepping to graph noise before signal
    data_after_signal = noise_values[index1*(~time_series_intersect)]   # prepping to graph noise after signal

    alldata = np.hstack((data_before_signal, data_in_signal, data_after_signal)) # combining the three arrays
    
    return (time_series_noise, alldata)


def template(a, f, sigma, t_0, t_duration, data_time_stamps, ad_hoc_grid = 10000):
    
    """
    This function will take as input the frequency, standard
    deviation, initial time, duration time, and interval
    of time between each value.
    
    INPUT:
    ------
    A : amplitude
    f : array of frequencies
    sigma : array of standard deviations
    t_0 : start time of a signal
    t_duration : duration of time for the signal
    data_time_stamps : time stamps from the data
    
    RETURNS:
    --------
    A template that will match the given signal with 
    the proper frequency and standard deviation using
    zero-padding.

    """
    
    del_T_data = np.diff(data_time_stamps)[0]                                # finding interval between values
    t_end = t_duration + t_0
    time_stamps = np.linspace(t_0, t_end + del_T_data, ad_hoc_grid)          # creating array of values with random step-size
    
    t_mean = np.mean(time_stamps)
    t_stamps, frequency = np.meshgrid(time_stamps, f)                        # setting up a matrix for interpolation

    S = a*np.sin(2*np.pi*frequency*t_stamps)*np.exp((-(t_stamps - t_mean)**2)/(2*sigma)) # evaluation in ad-hoc grid

    all_templates = []                                                       # empty list that will store Template values
    
    for i in range(len(f)):
        s = S[i]
        interpolant = interpolate.interp1d(time_stamps, s)           # prepping interp. in time-stamps of data-grid
        Template = interpolant(data_time_stamps[(data_time_stamps >= t_0)*(data_time_stamps <= t_end)]) # interpolating

        template_series = np.zeros_like(data_time_stamps)            # template time-stamps same length as data time-stamps

        index_prefix = template_series[data_time_stamps < t_0]       # zero-padding template on the left side
        index_suffix = template_series[data_time_stamps > t_end]     # zero-padding template on the right side

        closest_to_t0 = data_time_stamps[np.argmin(np.abs(t_0 - data_time_stamps))]     # approximating min value near start
        closest_to_tend = data_time_stamps[np.argmin(np.abs(t_end - data_time_stamps))] # approximating min value near end
        
        result = np.hstack((index_prefix, Template, index_suffix))       # horizontally stacking zero-padding with template
        all_templates.append(result)                                     # appending all values into Template
        
    return all_templates


def integrator(data_time_series, a, f, sigma, t_0, t_duration, del_T):

    """
    This function will take as input the amplitude, an array 
    of frequencies, standard deviation, initial time, duration 
    time, interval of time between each value, and the array 
    of data plus the time stamps. 

    Note that the data_time_series parameter may be a different
    size array from data_time_stamps.

    INPUT:
    ------
    a : amplitude
    f : array of frequencies
    sigma : standard deviation
    t_0 : start time of a signal
    del_T : interval of time between each value
    t_duration : duration of time for the signal
    data_time_series : data plus the time stamps

    RETURNS:
    --------
    An array of integration results between the time series 
    of the template and the data.
    
    """
    
    temp = template(a, f, sigma, t_0, t_duration, data_time_series[0]) # an array of zero-padded templates

    result = []
    for i in temp:
        result.append(np.sum(data_time_series[1] * i) * del_T)       # integrating over all templates in increments of del_T
    return result


def cross_correlation(del_T_0, t_start, t_max, data_time_series, a, f, sigma, t_duration, del_T):

    """
    This function will take as input the interval of time between
    each value, the first value of the signal start time, the last 
    value of the signal time, the array of times for the data, frequency, 
    standard deviation, and the duration of time for the template.
    
    INPUT:
    ------
    a : amplitude
    f : array of frequencies
    sigma : standard deviation
    t_max : last value of the time array
    t_start : initial value of the time array    
    del_T_0 : interval of time for the template
    del_T : interval of time between each value
    t_duration : duration of time for the signal
    data_time_series : array of times for the data
    
    RETURNS:
    --------
    An array of integration results for all values in the range.
    
    """

    C = []                                         # initializing the value of the variable
    time_stamps = []                               # creating empty list to save values for plotting
    
    while t_start <= t_max:
        time_stamps.append(t_start)
        
        integ = integrator(data_time_series, a, f, sigma, t_start, t_duration, del_T)
        C.append(integ)                            # computing integral over all 'sections'
        
        t_start += del_T_0                         # moving intial time of template over increments of del_T
        
    return time_stamps, C                          # returning list of times and cross-correlation values`


def fs_search(del_T_0, t_start, t_max, data_time_series, a, t_duration, del_T, f_start, f_end, df, s_start, s_end, ds):
    
    """
    This function will take as input the interval of time,
    the starting and ending times, the array of data times, 
    amplitude, frequency, standard deviation, and the
    duration of time.
    
    INPUT:
    ------
    a : amplitude
    f : frequency    
    t_max : final time of data
    sigma : standard deviation
    t_duration : duration of time
    t_start : initial time of data
    data_time_series : full time array of data
    del_T_0 : interval of time between each value
    
    RETURNS:
    --------
    One value of frequency and one value of standard deviation
    that correlate with the highest value of cross-correlation.
    
    """
    
    frequency_list = []                                                          # creating empty array to store other arrays
    sigma_list = []
    time_list = []
    crosscorr_list = []
    max_crosscorr = 0
    
    frequency_values = np.arange(f_start, f_end, df)                             # setting up ranges for which to run loops
    sigma_values = np.arange(s_start, s_end, ds)
    
    for frequency in frequency_values:
        for s in sigma_values:
            times, C = cross_correlation(del_T_0, t_start, t_max,                # running calculation
                            data_time_series, a, frequency, s, t_duration, del_T)
            if np.amax(C) > max_crosscorr:
                max_crosscorr = np.amax(C)
                frequency_correct = frequency
                sigma_correct = s
            
            time_list.append(times)                                              # appending arrays into empty lists
            crosscorr_list.append(C)
            frequency_list.append(frequency)
            sigma_list.append(s)
    
    return frequency_correct, sigma_correct                                      # returning two values