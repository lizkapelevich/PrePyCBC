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
    
    index1 = time_series_noise > time_series_signal[0]   # boolean = False before signal in noise time-series
    index2 = time_series_noise < time_series_signal[-1]  # boolean = False after signal in noise time_series
    time_series_intersect = index1*index2                # multiplying arrays into one array of booleans
    
    zeroes = np.zeros_like(noise_values)                          # create an array of zeroes same length 
                                                                  # as noise_values
    signal_time_stamps = time_series_noise[time_series_intersect] # keeping only boolean = True
    signal = s(signal_time_stamps)

    data_in_signal = noise_values[time_series_intersect] + signal       # embedding signal in noise
    data_before_signal = noise_values[index2*(~time_series_intersect)]  # prepping to graph noise before signal
    data_after_signal = noise_values[index1*(~time_series_intersect)]   # prepping to graph noise after signal

    alldata = np.hstack((data_before_signal, data_in_signal, data_after_signal)) # combining the three arrays
    
    return (time_series_noise, alldata)


def template(A, f, sigma, t_0, t_duration, data_time_stamps, ad_hoc_grid = 10000):

    """
    This function will take as input the frequency, standard
    deviation, initial time, duration time, and interval
    of time between each value.
    
    INPUT:
    ------
    A : amplitude
    f : frequency
    sigma : standard deviation
    t_0 : start time of a signal
    t_duration : duration of time for the signal
    data_time_stamps : time stamps from the data
    
    RETURNS:
    --------
    A template that will match the given signal using
    zero-padding if necessary.
    """
    
    t_end = t_duration + t_0
    time_stamps = np.linspace(t_0, t_end, ad_hoc_grid)         # template time stamps with random step-size
    
    t_mean = np.mean(time_stamps)

    S = A*np.sin(2*np.pi*f*time_stamps)*np.exp((-(time_stamps - t_mean)**2)/(2*sigma)) # evaluation in ad-hoc grid
    
    del_T_data = np.diff(data_time_stamps)[0]
    
    s = interpolate.interp1d(time_stamps, S)                   # interpolating signal in time-stamps of data-grid
    template_time_stamps = np.arange(t_0, t_end, del_T_data)
    Template = s(template_time_stamps)                         # evaluating sine-gaussian at data-grid points
    
    template_series = np.zeros_like(data_time_stamps)          # template time-stamps same length as data time-stamps
    
    index_prefix = template_series[data_time_stamps < t_0]     # zero-padding template on the left side
    index_suffix = template_series[data_time_stamps > t_end]   # zero-padding template on the right side
    
    return np.hstack((index_prefix, Template, index_suffix))


def integrator(data_time_series, a, f, sigma, t_0, t_duration, del_T):

    """
    This function will take as input the amplitude, frequency, standard
    deviation, initial time, duration time, interval of time between 
    each value, and the array of data plus the time stamps. Note that
    the data_time_series parameter may be a differens size array from
    data_time_stamps.
    
    INPUT:
    ------
    A : amplitude
    f : frequency
    sigma : standard deviation
    t_0 : start time of a signal
    t_duration : duration of time for the signal
    data_time_series : data plus the time stamps
    
    RETURNS:
    --------
    A result of integration between the time series of
    the template and the data.
    
    """
#    print("Len data_time_stamps: " + str(len(data_time_stamps)))
#    print(str(a) + "," + str(f) + "," + str(sigma) + "," + str(t_0) + "," + str(t_duration) + "," + str(data_time_series[0]))
    temp = template(a, f, sigma, t_0, t_duration, data_time_series[0])
    
    result = 0                                          # initializing the value of the variable

    for i in range(0, len(data_time_series[1])):        # data_time_series[1] is y-values bc it's a tuple
        result += data_time_series[1][i] * temp[i]
        
    return result*del_T


def cross_correlation(del_T_0, t_start, t_max, data_time_series, a, f, sigma, t_duration, del_T):

    """
    This function will take as input the interval of time between
    each value, the first value of the signal start time, the last 
    value of the signal time, the array of times for the data, frequency, 
    standard deviation, and the duration of time for the template.
    
    INPUT:
    ------
    del_T_0 : interval of time for the template
    del_T : interval of time between each value
    t_start : initial value of the time array
    t_max : last value of the time array
    data_time_series : array of times for the data
    A : amplitude
    f : frequency
    sigma : standard deviation
    t_duration : duration of time for the signal
    
    RETURNS:
    --------
    A list of integration results for all values in the range.
    
    """

    C = []                                         # initializing the value of the variable
    time_stamps = []                               # creating empty list to save values for plotting
    
    while t_start <= t_max:
        time_stamps.append(t_start)
        
        integ = integrator(data_time_series, a, f, sigma, t_start, t_duration, del_T)
        C.append(integ)                            # computing integral over all 'sections'
        
        t_start += del_T_0                         # moving intial time of template over increments of del_T
        
    return time_stamps, C                          # returning the list of integration results


def fs_search(del_T_0, t_start, t_max, data_time_series, a, t_duration, del_T):
    
    """
    This function will take as input the interval of time,
    the starting and ending times, the array of data times, 
    amplitude, frequency, standard deviation, and the
    duration of time.
    
    INPUT:
    ------
    del_T_0 : interval of time between each value
    t_start : initial time of data
    t_max : final time of data
    data_time_series : full time array of data
    a : amplitude
    f : frequency
    sigma : standard deviation
    t_duration : duration of time
    
    RETURNS:
    --------
    One value of frequency and one value of standard deviation
    that correlate with the highest value of cross-correlation.
    
    """
    
    frequency_list = []                                                           # creating empty array to store other arrays
    sigma_list = []
    time_list = []
    crosscorr_list = []
    
    frequency_values = np.arange(0.01, 20, 1)                                     # setting up ranges for which to run loops
    sigma_values = np.arange(0.01, 10, 1)
    
    for frequency in frequency_values:
        for s in sigma_values:
            times, C = cross_correlation(del_T_0, t_start, t_max,             # running calculation
                            data_time_series, a, frequency, s, t_duration, del_T)
            time_list.append(times)                                               # appending arrays into empty lists
            crosscorr_list.append(C)
            frequency_list.append(frequency)
            sigma_list.append(s)
            
            print(f'\r{frequency, s}')
            
    frequency_list = frequency_list.flatten()                                     # flattening 2-D arrays into 1-D
    sigma_list = sigma_list.flatten()
    time_list = time_list.flatten()
    crosscorr_list = crosscorr_list.flatten()
    
    largest_value_index = np.argmax(crosscorr_list)                               # finding index of largest value in array
    frequency_correct = frequency_list[largest_value_index]                       # finding corresponding value using index
    sigma_correct = sigma_list[largest_value_index]
    
    return frequency_correct, sigma_correct                                       # returning two values