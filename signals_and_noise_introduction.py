def get_noise(t_0, t_end, del_T):
    
    """
    This function will take as input the boundaries of a time stamp
    from some noise and the interval between each value.
    
    INPUT:
    ------
      t_0 : start time of a data
    t_end : end time of a data
      del_T : interval of time between each value
    
    RETURNS:
    --------
    The time series for random noise.
    
    """

    duration_data = t_end - t_0               
    x = duration_data / del_T
    del_T_prime = duration_data / math.ceil(x)
    
    time_series = np.arange(t_0, t_end + del_T_prime, del_T_prime)
    noise = np.random.random(len(time_series))

    return (time_series, noise, del_T_prime)


def get_signal(A, t_0, t_end, del_T, f, sigma):

    """
    This function will take as input the amplitude, boundaries of a time stamp
    from a signal, the frequency, and the standard deviation.

    INPUT:
    ------
        f : frequency
        A : amplitude
      t_0 : start time of a signal
    t_end : end time of a signal
    del_T : interval of time between each value
    sigma : standard deviation

    RETURNS:
    --------
    The time series for a signal.

    """

    duration_signal = t_end - t_0

    x = duration_signal / del_T

    del_T_prime = duration_signal / math.ceil(x)

    t = np.arange(t_0, t_end + del_T_prime, del_T_prime)
    t_mean = (t_0 + t_end) / 2

    S = A*np.sin(2*np.pi*f*t)*np.exp((-(t - t_mean)**2)/(2*sigma))

    return (t, S, del_T_prime)



def final_data(a, t_signal_start, t_signal_end, t_noise_start, t_noise_end,
               del_T, f, sigma):
    """
    This function will take as input the amplitude, time boundaries
    of the signal and noise, interval between their values, the
    frequency, and standard deviation.

    INPUT:
    ------
             A : amplitude
             f : frequency
         sigma : standard deviation
         del_T : interval of time between each value
   t_noise_end : end time of noise
  t_signal_end : end time of signal
 t_noise_start : start time of noise
t_signal_start : start time of signal

    RETURNS:
    --------
    The calculation for cross-correlating a signal embedded
    within random noise with zero-padding if necessary.

    """
    t1, x, del_T_signal = get_signal(a, t_signal_start, t_signal_end,
                                     del_T, f, sigma)
    t2, noise, del_T_noise = get_noise(t_noise_start, t_noise_end, del_T)

    s = interpolate.interp1d(t1, x)

    index1 = t2 > t1[0]
    index2 = t2 < t1[-1]
    index = index1*index2

    zeroes = np.zeros_like(noise)

    signal_time_stamps = t2[index]
    signal = s(signal_time_stamps)

    data_in_signal = noise[index] + signal
    data_before_signal = noise[index2*(~index)]
    data_after_signal = noise[index1*(~index)]

    alldata = np.hstack((data_before_signal, data_in_signal, data_after_signal))

    return (t2, alldata)
