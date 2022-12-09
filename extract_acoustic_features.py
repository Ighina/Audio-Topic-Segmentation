# -*- coding: utf-8 -*-
"""
All code imported from https://librosa.org/doc/main/_modules/librosa/core/pitch.html#yin

the main change is the fact that the yin function is modified to return also the voicing intensity
as defined in https://ieeexplore.ieee.org/abstract/document/8268981

The final modified yin function is then combined with the librosa melspectrogram function to
obtain all the acoustic features described in https://ieeexplore.ieee.org/abstract/document/8268981,
including the pauses, defined as the portions of the audio in which voicing intensity < 0.5
"""
import numpy as np
import librosa
import warnings
warnings.filterwarnings("default")
# import librosa.util as util
# from librosa.util.exceptions import ParameterError

def get_pause_durations(voicing_intensities, delta = 0.5):
    """
    Get the pause durations and the voiced segments (i.e. voicing intensities where there is no pause detection)
    
    Parameters
    ----------
    voicing_intensities : np.ndarray [shape=(..., n_frames)]
    delta : int [define the threshold under which a frame is considered as a pause]
    Returns
    -------
    pause_durations : np.ndarray [shape=(n_pauses)]
    voiced_segments : np.ndarray [shape=(n_voiced_segments)]
    """
    pauses = []
    voiced_segments = []
    pause = 0
    add = False
    for sample in voicing_intensities:
        if sample<delta:
            pause+=1
            add = True
        else:
            if add:
                pauses.append(pause)
                pause = 0
                add = False
            voiced_segments.append(sample)
    
    if not pauses:
        if pause>0:
            pauses.append(pause)
            voiced_segments.append(0)
        else:
            pauses.append(0)
            voiced_segments = voicing_intensities
    return np.array(pauses), np.array(voiced_segments) 
    

# def _cumulative_mean_normalized_difference(
#     y_frames, frame_length, win_length, min_period, max_period
# ):
#     """Cumulative mean normalized difference function (equation 8 in [#]_)

#     .. [#] De Cheveigné, Alain, and Hideki Kawahara.
#         "YIN, a fundamental frequency estimator for speech and music."
#         The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

#     Parameters
#     ----------
#     y_frames : np.ndarray [shape=(frame_length, n_frames)]
#         framed audio time series.

#     frame_length : int > 0 [scalar]
#          length of the frames in samples.

#     win_length : int > 0 [scalar]
#         length of the window for calculating autocorrelation in samples.

#     min_period : int > 0 [scalar]
#         minimum period.

#     max_period : int > 0 [scalar]
#         maximum period.

#     Returns
#     -------
#     yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
#         Cumulative mean normalized difference function for each frame.
#     """
#     # Autocorrelation.
#     a = np.fft.rfft(y_frames, frame_length, axis=0)
#     b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)
#     acf_frames = np.fft.irfft(a * b, frame_length, axis=0)[win_length:]
#     acf_frames[np.abs(acf_frames) < 1e-6] = 0

#     # Energy terms.
#     energy_frames = np.cumsum(y_frames ** 2, axis=0)
#     energy_frames = energy_frames[win_length:, :] - energy_frames[:-win_length, :]
#     energy_frames[np.abs(energy_frames) < 1e-6] = 0

#     # Difference function.
#     yin_frames = energy_frames[0, :] + energy_frames - 2 * acf_frames

#     # Cumulative mean normalized difference function.
#     yin_numerator = yin_frames[min_period : max_period + 1, :]
#     tau_range = np.arange(1, max_period + 1)[:, None]
#     cumulative_mean = np.cumsum(yin_frames[1 : max_period + 1, :], axis=0) / tau_range
#     yin_denominator = cumulative_mean[min_period - 1 : max_period, :]
#     yin_frames = yin_numerator / (yin_denominator + util.tiny(yin_denominator))
#     return yin_frames

# def _parabolic_interpolation(y_frames):
#     """Piecewise parabolic interpolation for yin and pyin.

#     Parameters
#     ----------
#     y_frames : np.ndarray [shape=(frame_length, n_frames)]
#         framed audio time series.

#     Returns
#     -------
#     parabolic_shifts : np.ndarray [shape=(frame_length, n_frames)]
#         position of the parabola optima
#     """

#     parabolic_shifts = np.zeros_like(y_frames)
#     parabola_a = (
#         y_frames[..., :-2, :] + y_frames[..., 2:, :] - 2 * y_frames[..., 1:-1, :]
#     ) / 2
#     parabola_b = (y_frames[..., 2:, :] - y_frames[..., :-2, :]) / 2
#     parabolic_shifts[..., 1:-1, :] = -parabola_b / (
#         2 * parabola_a + util.tiny(parabola_a)
#     )
#     parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
#     return parabolic_shifts

# def yin(
#     y,
#     *,
#     fmin,
#     fmax,
#     sr=22050,
#     frame_length=2048,
#     win_length=280,
#     hop_length=None,
#     trough_threshold=0.1,
#     center=True,
#     pad_mode="constant",
# ):
#     """Fundamental frequency (F0) estimation using the YIN algorithm.

#     YIN is an autocorrelation based method for fundamental frequency estimation [#]_.
#     First, a normalized difference function is computed over short (overlapping) frames of audio.
#     Next, the first minimum in the difference function below ``trough_threshold`` is selected as
#     an estimate of the signal's period.
#     Finally, the estimated period is refined using parabolic interpolation before converting
#     into the corresponding frequency.

#     .. [#] De Cheveigné, Alain, and Hideki Kawahara.
#         "YIN, a fundamental frequency estimator for speech and music."
#         The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

#     Parameters
#     ----------
#     y : np.ndarray [shape=(..., n)]
#         audio time series. Multi-channel is supported..
#     fmin : number > 0 [scalar]
#         minimum frequency in Hertz.
#         The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
#         though lower values may be feasible.
#     fmax : number > 0 [scalar]
#         maximum frequency in Hertz.
#         The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
#         though higher values may be feasible.
#     sr : number > 0 [scalar]
#         sampling rate of ``y`` in Hertz.
#     frame_length : int > 0 [scalar]
#         length of the frames in samples.
#         By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
#         a sampling rate of 22050 Hz.
#     win_length : None or int > 0 [scalar]
#         length of the window for calculating autocorrelation in samples.
#         If ``None``, defaults to ``frame_length // 2``
#     hop_length : None or int > 0 [scalar]
#         number of audio samples between adjacent YIN predictions.
#         If ``None``, defaults to ``frame_length // 4``.
#     trough_threshold : number > 0 [scalar]
#         absolute threshold for peak estimation.
#     center : boolean
#         If ``True``, the signal `y` is padded so that frame
#         ``D[:, t]`` is centered at `y[t * hop_length]`.
#         If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
#         Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
#         time grid by means of ``librosa.core.frames_to_samples``.
#     pad_mode : string or function
#         If ``center=True``, this argument is passed to ``np.pad`` for padding
#         the edges of the signal ``y``. By default (``pad_mode="constant"``),
#         ``y`` is padded on both sides with zeros.
#         If ``center=False``,  this argument is ignored.
#         .. see also:: `np.pad`

#     Returns
#     -------
#     f0: np.ndarray [shape=(..., n_frames)]
#         time series of fundamental frequencies in Hertz.

#         If multi-channel input is provided, f0 curves are estimated separately for each channel.

#     See Also
#     --------
#     librosa.pyin :
#         Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

#     Examples
#     --------
#     Computing a fundamental frequency (F0) curve from an audio input

#     >>> y = librosa.chirp(fmin=440, fmax=880, duration=5.0)
#     >>> librosa.yin(y, fmin=440, fmax=880)
#     array([442.66354675, 441.95299983, 441.58010963, ...,
#         871.161732  , 873.99001454, 877.04297681])
#     """

#     if fmin is None or fmax is None:
#         raise ParameterError('both "fmin" and "fmax" must be provided')

#     # Set the default window length if it is not already specified.
#     if win_length is None:
#         win_length = frame_length // 2

#     if win_length >= frame_length:
#         raise ParameterError(
#             "win_length={} cannot exceed given frame_length={}".format(
#                 win_length, frame_length
#             )
#         )

#     # Set the default hop if it is not already specified.
#     if hop_length is None:
#         hop_length = frame_length // 4

#     # Check that audio is valid.
#     util.valid_audio(y, mono=False)

#     # Pad the time series so that frames are centered
#     if center:
#         padding = [(0, 0) for _ in y.shape]
#         padding[-1] = (frame_length // 2, frame_length // 2)
#         y = np.pad(y, padding, mode=pad_mode)

#     # Frame audio.
#     y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

#     # Calculate minimum and maximum periods
#     min_period = max(int(np.floor(sr / fmax)), 1)
#     max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

#     # Calculate cumulative mean normalized difference function.
#     yin_frames = _cumulative_mean_normalized_difference(
#         y_frames, frame_length, win_length, min_period, max_period
#     )

#     # Parabolic interpolation.
#     parabolic_shifts = _parabolic_interpolation(yin_frames)

#     # Find local minima.
#     is_trough = util.localmin(yin_frames, axis=-2)
#     is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

#     # Find minima below peak threshold.
#     is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

#     # Absolute threshold.
#     # "The solution we propose is to set an absolute threshold and choose the
#     # smallest value of tau that gives a minimum of d' deeper than
#     # this threshold. If none is found, the global minimum is chosen instead."
#     target_shape = list(yin_frames.shape)
#     target_shape[-2] = 1

#     global_min = np.argmin(yin_frames, axis=-2)
#     yin_period = np.argmax(is_threshold_trough, axis=-2)

#     global_min = global_min.reshape(target_shape)
#     yin_period = yin_period.reshape(target_shape)

#     no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
#     yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

#     # Refine peak by parabolic interpolation.

#     yin_period = (
#         min_period
#         + yin_period
#         + np.take_along_axis(parabolic_shifts, yin_period, axis=-2)
#     )[..., 0, :]

#     # Convert period to fundamental frequency.
#     f0 = sr / yin_period
#     return f0, np.max(yin_frames, axis = 0)


def get_acoustic_features(y, sr, previous_f0s = None, mfcc = False):
    
    stat_fn = [np.nanmean, np.nanstd]
    
    statistics = []
    
    if mfcc:
        
        x = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
        
        delta_x = librosa.feature.delta(x)
        
        for fn in stat_fn:
            statistics.extend(fn(x, axis=1).tolist())
            statistics.extend(fn(delta_x, axis=1).tolist())
        
    else:
        f0, _, voicing_intensity = librosa.pyin(y, fmin=70, fmax=500, sr = sr)
        
        if sum(np.isnan(f0))==len(f0):
            f0[np.isnan(f0)] = 0
        
        # f0[np.isnan(f0)]=0
        
        pauses, voiced_segments = get_pause_durations(voicing_intensity)
        
        mel_filter = librosa.feature.melspectrogram(y=y, n_mels=40, sr = sr)
        
        delta_mel = librosa.feature.delta(mel_filter)
        
        feats = [f0, pauses, voiced_segments, mel_filter, delta_mel]
        
        for feat in feats:
            for fn in stat_fn:
                try:
                    statistics.extend(fn(feat, axis=1).tolist())
                except:
                    statistics.append(fn(feat, axis = 0))
        # statistics = [fn(feat, axis = 0) for feat in feats for fn in stat_fn]
        
        if previous_f0s is None:
            pitch_jump = 0
        else:
            pitch_jump = np.nanmean(f0[:len(f0)//5]/np.nanmean(f0)) - np.nanmean(previous_f0s[-len(previous_f0s)//5:]/np.nanmean(previous_f0s))
            if np.isnan(pitch_jump):
                print("could not compute pitch jump!")
                pitch_jump = 0
        
        statistics.append(pitch_jump)
    
    statistics = np.array(statistics)
    
    if sum(np.isnan(statistics))>0:
        print(statistics)
        print(f0)
        raise ValueError
    
    return statistics


    
    
    