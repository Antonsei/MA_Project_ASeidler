import numpy as np
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.dates as dt
import math
import copy
from scipy import stats
from scipy.interpolate import griddata
import os
import cartopy.crs as ccrs
import cartopy.feature as ft


if os.name == 'nt':  # only Windows computers
    os.environ[
        'PROJ_LIB'] = r'C:\Users\seidl\Miniconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'


def colors():
    return 10000 * ['orange', 'g', 'r', 'b', 'm']     # default colors for plots

def markers():
    return 10000 * ['^', 'd', '*']                   # default markers for plots


class EarthquakeSubset:
    """
    create an object of class EarthquakeSubset

    Parameters
    ---------
    tlocmag:    4-tuple of numpy arrays (time, location, magnitude, depth)
    minmag:     minimum magnitude of events to be kept
    maxmag:     maximum magnitude of events to be kept
    min_depth:  minimum depth of events to be kept
    max_depth:  maximum depth of events to be kept

    Methods
    --------
    dt(self):               Return non-normalized inter-event times
    get_dtau(self):         Return normalised inter-event times
    rate(self):             Return n0 events per time
    pdf(self):              Return pdf of Exp(X): e-function e^(dtau)
    cut_by_depth(self, min_z=-1000, max_z=1e4):
        removes events outside depth range [min_z, max_z] in km
        default min_z, max_z = -1000, 1e4
        Return self
    cut_by_mag(self, minmag=None, maxmag=None):
        removes events with magnitudes below threshold minmag and/or above threshold maxmag
        default minmag, maxmag = None
        Return copy of self

    cut_by_location(self, minlat, minlat, maxlon, maxlat)
        remove elements exceeding minimum/ maximum latitude/longitude
        Return copy of self

    cut_by_time_frame(self, mintime, maxtime)
        removes events outside given time span [mintime, maxtime]
        parameters mintime, maxtime: obspy
    """

    def __init__(self, tlocmag=None, minmag=None, maxmag=None, min_depth=-1,
                 max_depth=1e4):  # requires tuple (time,location,mag,depth)
        if tlocmag is not None:

            self.time = tlocmag[0]
            self.location = tlocmag[1]
            self.magnitudes = tlocmag[2]
            self.depth = tlocmag[3]

            if minmag is not None or maxmag is not None:
                new_obj = self.cut_by_mag(minmag=minmag, maxmag=maxmag)
                self.__dict__.update(new_obj.__dict__)

            self.cut_by_depth(min_depth, max_depth)

        else:
            self.time = None
            self.location = None
            self.magnitudes = None
            self.depth = None

    def __len__(self):
        if self.magnitudes is not None:
            return len(self.magnitudes)
        elif self.time is not None:
            return len(self.time)
        else:
            return 0

    def dt(self):                           # total inter-event times
        return np.diff(np.sort(self.time))

    def get_dtau(self):                     # normalised inter-event times
        return np.sort(self.rate() * self.dt())

    def rate(self):                         # rate parameter: number of events / time interval
        return len(self.time) / (np.max(self.time) - np.min(self.time))

    def pdf(self):                          # pdf of exponential distribution (e^(dtau))
        return math.e ** (-self.get_dtau())

    def cut_by_depth(self, min_z=-1000, max_z=1e4):     # remove events with depths outside interval [min_z max_z]
        if self.depth is None:
            return self
        l_depth = np.logical_and(self.depth > min_z, self.depth < max_z)
        if np.sum(l_depth) == 0:
            print('No earthquakes are located at that depth')
            return
        elif np.sum(l_depth) == 1:
            print('Just one event at that depth, therefore no subset generated')
            return
        else:
            self.time = self.time[l_depth]
            self.location = self.location[l_depth, :]
            self.magnitudes = self.magnitudes[l_depth]
            self.depth = self.depth[l_depth]
        return self

    def cut_by_mag(self, minmag=None, maxmag=None):  # yields empty list if no events within that region,
        ret = copy.deepcopy(self)
        if minmag is None and maxmag is None:
            return ret
        if minmag is None:
            minmag = np.min(ret.magnitudes)
        if maxmag is None:
            maxmag = np.max(ret.magnitudes)
        lmag = np.logical_and(ret.magnitudes >= minmag, ret.magnitudes <= maxmag)
        if np.sum(lmag) == 0:
            print('No earthquakes within that magnitude range')
            return None
        elif np.sum(lmag) == 1:
            print('Just one event at that depth, therefore no subset generated')
            return None
        else:
            ret.time = ret.time[lmag] if ret.time is not None else None
            ret.location = ret.location[lmag] if ret.location is not None else None # lat, lon
            ret.magnitudes = ret.magnitudes[lmag]
            ret.depth = ret.depth[lmag] if ret.depth is not None else None
            return ret

        return ret

    def cut_by_location(self, minlon=-180, minlat=-90, maxlon=360, maxlat=180):
        ret = copy.deepcopy(self)
        l_lon = np.logical_and(ret.location[:, 0] > minlat, ret.location[:, 0] < maxlat)
        l_lat = np.logical_and(ret.location[:, 1] > minlon, ret.location[:, 1] < maxlon)
        l_loc = np.logical_and(l_lon, l_lat)
        if np.sum(l_loc) == 0:
            print('No earthquakes are located in the requested area')
            return
        elif np.sum(l_loc) == 1:
            print('Just one event in subregion, therefore no subset generated')
            return
        else:
            ret.time = ret.time[l_loc]
            ret.location = ret.location[l_loc, :]
            ret.depth = ret.depth[l_loc] if ret.depth is not None else None
            ret.magnitudes = ret.magnitudes[l_loc]
        return ret

    def random_times(self, n=1, timeframe=None):
        """
        :param n: number of synthetic catalogs produced
               timeframe : 2-tuple with mintime, maxtime of synthetic catalog
        :return:
        list of EarthquakeSubset objects with randomized uniformely distributed times, other properties same
        """
        if timeframe is None:
            timeframe = (np.min(self.time), np.max(self.time))
        span = timeframe[1] - timeframe[0]
        nn = len(self)
        synthquakes = []
        for i in range(n):
            synth_quake = copy.deepcopy(self)
            times_sec = np.sort(span * np.random.rand(nn, ))
            synth_time = np.full((nn,), timeframe[0]) + times_sec
            synth_quake.time = synth_time
            synthquakes.append(synth_quake)
        return synthquakes

    def cut_by_time_frame(self, mintime=obspy.UTCDateTime(2006, 1, 1), maxtime=obspy.UTCDateTime(2020, 12, 30)):
        """
        input: mintime, maxtime of in obspy UTCDateTime format
        returns copy of subset containing events between interval mintime and maxtime
        """
        ret = copy.deepcopy(self)
        l_time = np.logical_and(ret.time > mintime, ret.time < maxtime)
        if np.sum(l_time) == 0:
            print('No earthquakes are located in the requested time frame')
            return
        else:
            ret.time = ret.time[l_time]
            ret.location = ret.location[l_time, :]
            ret.depth = ret.depth[l_time]
            ret.magnitudes = ret.magnitudes[l_time]
        return ret

    def minmag(self):
        return np.min(self.magnitudes)

    def gr_plot(self):
        fig = gutenberg_richter_plot(self.magnitudes, show=True)
        return fig

    def plot_seismicity_rate(self, show=False, title=None):
        plot_seismicity_rate(self.time, show, title)


def load_catalogue_ipoc(catalog='data/ipoc/Sippl2018_data.txt'):
    # input: catalog ('Sippl2018_data.txt')
    # output: Time, location (latitude, longitude, depth) and Magnitude as numpy arrays update: using negative
    df = pd.read_csv(catalog, header=None, sep="\s+")
    time = convert2UTC(df.iloc[:, 0:6].values)
    location = df.iloc[:, 6:8].values
    depth = df.iloc[:, 8].values
    mag = df.iloc[:, -2].values
    return time, location, mag, depth


def load_catalogue_ipoc_modified(catalog='data/ipoc/chile_cat_wo_as.dat'):
    """
    input: ipoc catalog without Topocilla 2007 and Iquique 2014 fore- and aftershock series
    columns: t [sec] after 01/01/2007, lon, lat, depth, magnitude

    output: t in UTC format, location [lat, lon, depth], t in secs after 01/01/2007
    """
    df = pd.read_csv(catalog, header=None, sep="\s+")
    t = df.iloc[:, 0].values
    loc = df.iloc[:, 1:4].values
    location = loc[:, (1, 0)]       # transformation into lat,lon
    depth = loc[:, 2]  # depth in km (positive values)
    mag = df.iloc[:, -1].values
    t_utc = []
    for j in range(t.shape[0]):
        t_utc.append(obspy.UTCDateTime(t[j] + (obspy.UTCDateTime(2007, 1, 1) - obspy.UTCDateTime(0))))
    return np.array(t_utc), location, mag, depth, t


def load_catalogue_engdahl():
    filename = 'data/engdahl/centennial_altered.txt'
    test_file = open(filename)
    t, lon, lat, depth, mag, depth = [], [], [], [], [], []
    n_errs = 0
    for line in test_file:
        year = int(line[12:16])
        month = int(line[17:20])
        day = int(line[20:23])
        hour = int(line[24:27])
        minute = int(line[27:30])
        second = int(float(line[31:36]))
        try:
            t.append(obspy.UTCDateTime(year=year, month=month, day=day, hour=hour, minute=minute, second=second))
        except:
            n_errs += 1
            continue
        mag.append(float(line[67:70]))
        depth.append(float(line[53:59]))  # positive depth
        lon.append(float(line[44:53]))
        lat.append(float(line[37:44]))
    test_file.close()
    print(n_errs, 'errors occurred loading engdahl catalog')

    return np.array(t), np.array([lat, lon]).T, np.array(mag), np.array(depth)


def convert2UTC(time):
    
    """
    converts list or array of times into obspy.UTCDateTime
    :param time in other format
    :return: array of obspy.UTCDateTime
    """
    timeint = time.astype(int)  # required for conversion to UTC
    t_utc = []
    i_errs = 0
    for j in range(timeint.shape[0]):
        try:
            t_utc.append(obspy.UTCDateTime(*timeint[j, :]))
        except:
            i_errs += 1
            print('problem occurred converting time!')

    return np.array(t_utc)


def gutenberg_richter_plot(magnitudes, show=False, xlims=None, ylims=None, ax=None, markersize=4, bins=20, norm=False):
    """
    compute normalised magnitude-frequency distribution for given set of magnitudes

    :param magnitudes:      array or list of magnitudes
    :param show:            Boolean, call plt.show() if true
    :param xlims:           tuple or list defining limits (minimum, maximum) of x-axis
    :param ylims:           tuple or list defining limits (minimum, maximum) of y-axis
    :param ax:              plot on given AxesSubplot object
    :param markersize:      size of markers
    :param bins:            bins (number of bins or given array of bin centres, default 20)
    :param norm:            normalised vs. frequency
    :return:                figure object
    """
    weights, bin_edges = np.histogram(magnitudes, bins=bins)  # bin_edges identical to bins!
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    y = np.cumsum(weights[::-1])[::-1] / len(magnitudes) if norm else np.cumsum(weights[::-1])[::-1]
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
    ax.plot(bin_centres, y, '.', markersize=markersize)
    ax.set_yscale('log')
    ax.set_xlabel('M', fontsize=7)
    ax.set_ylabel(r'$N/N_{tot}$', fontsize=7)
    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])
    if show:
        plt.show()
    if ax is None:
        return fig


def plot_seismicity_rate(times, show=False, bins=100, logscale=True, title=None, legend=None, bar=False, ax=None,
                     plt_kwargs=None):
    """

    :param times:       times in UTCdatetime
    :param show:        if true call plt.show()
    :param bins:        bins of time
    :param logscale:    if true x-values are plotted on a logarithmic scale
    :param title:       title of figure
    :param legend:      legend object
    :param bar:         plot as bar-plot
    :param ax:          plot on subplot.axes object
    :param plt_kwargs:  dictionary of arguments of plt.plot() functions
    :return:            None
    """

    if plt_kwargs is None:
        plt_kwargs = {}
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    if type(times) == list:
        weights = []

        for i, ti in enumerate(times):
            plt_kwargs_old = plt_kwargs
            if type(plt_kwargs) == list:

                plt_kwargs = plt_kwargs[i]
            t = [time.matplotlib_date for time in ti]
            w, bin_edges = np.histogram(t, bins=bins)
            # if i==0:
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            dates = dt.num2date(np.array(bin_centres))
            weights.append(w)
            print('type dates', type(dates), type(w))
            print('shape dates = ', len(dates), 'shape weights', len(w))
            if bar:
                ax.bar(dates, w)
            else:
                ax.plot(dates, w, '.-', lw=0.9, **plt_kwargs)
            plt.tick_params(labelsize=6)
            plt_kwargs = plt_kwargs_old

    else:
        t = [time.matplotlib_date for time in times]

        weights, bin_edges = np.histogram(t, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        dates = dt.num2date(np.array(bin_centres))
        if bar:
            ax.bar(dates, weights)
        else:
            if 'marker' not in plt_kwargs:
                plt_kwargs['marker'] = '.'
            if 'linestyle' not in plt_kwargs or 'ls' not in plt_kwargs:
                plt_kwargs['marker']= '.'
            ax.plot(dates, weights, **plt_kwargs)
    if legend is not None:
        ax.legend(legend, fontsize=6)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('occurrences', fontsize=8)
    if logscale:
        ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=6)
    if show:
        plt.show()


def compute_gamma(dtau, opt='norm'):
    """
    computes pdf of gamma distribution based on normalised inter-event times
    standard form as function of variance as in Hainzl et al. 2006 or with fixed parameters according to Corral 2006

    :param  dtau:   normalized inter-event times
    :param  opt:    type of gamma distribution: if 'norm' (default) std. form with mixed parameters,
     if 'corral' fixed parameters according to Corral 2006
    :return     (1) probability density function of gamma distribution; (2) shape parameter gamma
    """
    if 0 in dtau:
        print('gamma could not be computed due to zeroes in IETs')
        return
    if opt == 'norm':
        beta = np.var(dtau) # usually like that
        y = 1 / beta
        c = 1 / (beta ** y * math.gamma(y))
    elif opt == 'corral':
        c, y, beta = 0.5, 0.67, 1.58

    else:
        print('opt not recognized. Arguments either <<norm>>  or <<corral>>')
        return
    pdf = c * dtau ** (y - 1) * math.e ** (-dtau / beta)
    return pdf, y


def subregion_boundaries(minlat=-24.5, maxlat=-17.98, minlon=-71.5, maxlon=-66.48, overlapx=50,
                         distx=200, overlapy=50, disty=150, indeg=False, strictlims=False, n2s=True):
    """

    Calculates rectangular subregion boundaries

    Parameters (all optional):
        minlat, maxlat, minlon, maxlon in deg
        overlapx,overlapy: overlapping part between two adjacent subregions in x (lon) and y (lat) direction
        distx,disty: rectangle size in x(lon) and y(lat) direction
        indeg (bool): sets dist and overlap units too degree instead of km in geodetic projection, default is False
        strict_lims: bool, if true rectangles can not surpass limits defined by minlat, maxlat e.t.c
        n2s: bool, if true order of edges from north to south

    Returns:
        lons, lats: ndarrays of size nx2 containing left and right boundaries for each subregions

    """
    if not indeg:
        overlapx = obspy.geodetics.kilometers2degrees(overlapx)
        overlapy = obspy.geodetics.kilometers2degrees(overlapy)
        distx = obspy.geodetics.kilometers2degrees(distx)
        disty = obspy.geodetics.kilometers2degrees(disty)
    lons = np.zeros((0,2))
    lats = np.zeros((0,2))
    lat0, lon0 = minlat, minlon
    stepsizex, stepsizey = distx - overlapx, disty - overlapy

    if not strictlims:
        lat1, lon1 = lat0 + disty, lon0 + distx
        while lat1 < maxlat + stepsizey:
            lon0, lon1 = minlon, minlon + distx
            lons = np.append(lons, [[lon0, lon1]], axis=0)
            lats = np.append(lats, [[lat0, lat1]], axis=0)
            while lon1 < maxlon:
                lon0, lon1 = lon0 + stepsizex, lon1 + stepsizex
                lons = np.append(lons, [[lon0, lon1]], axis=0)
                lats = np.append(lats, [[lat0, lat1]], axis=0)
            lat0, lat1 = lat0 + stepsizey, lat1 + stepsizey
    else:
        lat1, lon1 = min(lat0+ disty, maxlat), min(lon0 + distx, maxlon)

        while lat0 + disty < maxlat + 0.8*stepsizey:
            lon0, lon1 = minlon, min(minlon + distx, maxlon)

            while lon0 + distx < maxlon+stepsizex:
                lons = np.append(lons, [[lon0, lon1]], axis=0)
                lats = np.append(lats, [[lat0, lat1]], axis=0)
                lon0, lon1 = lon0 + stepsizex, min(lon1 + stepsizex, maxlon)

            lat0, lat1 = lat0 + stepsizey, min(lat1 + stepsizey, maxlat)
            print('lon1 = ',lon1)

    if n2s:
        return np.flipud(lons), np.flipud(lats)

    return lons, lats


def split_along_edges(quake_set, edges):
    """
    splits events from EarthquakeSubset object into list of subcatalogues corresponding to provinces defined by edges

    :param quake_set: object of type EarthquakeSubset
    :param edges: tuple of two lists describing edge of rectangles longitudes (edges[0]) and latitudes (edges[1])
    or tuple of nx2 arrays with lon0,lon1,lat1,lat2 with four limits of subregion
    :return: list of EarthquakeSubsets with events corresponding subregions defined by edges (1) list of booleans,
    True if subregion defined by edges contains events (2)
    """
    contains_events = []
    four_limits = False
    if type(edges[0]) != list:
        four_limits = edges[0].shape[1] == 2
    return_list = []
    if not four_limits:
        for i in range(len(edges[0]) - 1):
            for j in range(len(edges[1]) - 1):
                subset = quake_set.cut_by_location(edges[0][i], edges[1][j], edges[0][i + 1], edges[1][j + 1])
                if subset is not None:
                    return_list.append(subset)
    else:
        n = edges[0].shape[0]
        contains_events = n * [False]
        for k in range(edges[0].shape[0]):
            subset = quake_set.cut_by_location(edges[0][k, 0], edges[1][k, 0], edges[0][k, 1], edges[1][k, 1])
            if subset is not None:
                return_list.append(subset)
                contains_events[k] = True

    return return_list, np.array(contains_events)


def plot_hists(quakes, suppress_plot=True, edges=None, malpha=0.8, msize=3,
               style=2, legend='all', title=None, subtitles=None, bins=None, n_bins=20, legend_fontsize='xx-small',
               xlim_left=None, xlim_right=None, ylim_bottom=None, ylim_top=None, fig_num=None, logs=True,
               logscalex=True, logscaley=True, unit='seconds', normalize_by_bins=True, bar=False, axes=None):
    """
        plotting histograms of non-normalized inter-event times dt
    input: list of object of type EarthquakefSubset
    suppressplot: supress plt.show() command if value is 'no'
    output: list of tuples containing bin centres, bin edges and weights of histograms
    log: if True, bins are in log scale and density function is plotted (weights divided by bin size)
    Would probably yield similar results to

    :param quakes:          object of type EarthquakeSubset or list of EarthquakeSubsets
    :param suppress_plot:   suppress plt.show() command if value is False
    :param edges:           tuple of seperation lines longitude, latitude along which plot is split,
     or list with length of quakes
    :param malpha:          float indicating transparency of markers (alpha parameter of matplotlib.plot())
    :param msize:           markersize
    :param style:           pre-defined plot() arguments: if 2 (default), markerstyle and colors as indicated in
    markers and colors variable, otherwise please check code for details
    :param legend:          if None: legend suppressed, otherwise show legend (indices of subregions)
    :param title:           string: main title of figure
    :param subtitles:       list with same length as quakes parameter: titles of subplots
    :param bins:            list with bin edges for histogram
    :param n_bins:          number of bins, default 20
    :param xlim_left:       lower limit of x axis
    :param xlim_right:      upper limit of x axis
    :param ylim_bottom:     lower limit of y axis
    :param ylim_top:        upper limit of y axis
    :param fig_num:         name of figure
    :param logs:            boolean, if True use logarithmic bins, otherwise linear
    :param logscalex:       logarithmic x scale
    :param logscaley:       logarithmic y scale
    :param unit:            string showing unit of inter-event times on x axis: 'seconds', 'minutes', 'hours',
    'days' or 'years'
    :param normalize_by_bins:   divide weights by bin-width
    :param bar:             show histogram as bar plot, instead of points/ markers/ graphs
    :param axes:            plot on given axis
    :return:
    """
    unit_in_s = {"seconds": 1, "s": 1, "minutes": 60, "hours": 3600, "days": 86400, "years": 31557600}

    if type(quakes) == EarthquakeSubset:
        quakes = [quakes]
    if quakes is None:
        print('invalid argument: quakes of type NoneType')
        return
    if axes is None:
        fig = plt.figure(figsize=(11, 6)) if fig_num is None else plt.figure(figsize=(11, 6), num=fig_num)
        fig.tight_layout()
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(hspace=0.4)
    n = len(quakes)
    weightlist = []
    if subtitles is None:
        subtitles = n*['']

    if edges is not None:
        if type(edges) is not list:
            edges = n * [edges]
    else:
        edges = n * [None]

    for k, quake in enumerate(quakes):
        if quake is None:
            print('no events contained in given subset')
            pass
        if axes is None:
            ax = plt.subplot(max(1, round(n / 2)), min(2, n), k + 1)
        else:
            ax = axes[k]

        ax.set_title(subtitles[k], size=7, )

        if bins is None:
            if logs:
                bins = np.logspace(0, np.log10(np.max(quake.dt())), n_bins)
            else:
                bins = np.linspace(60, np.max(quake.dt()), n_bins)

        if edges[k] is None:
            weights, bin_edges = np.histogram(quake.dt(), density=normalize_by_bins, bins=bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            if bar:
                ax.bar(bin_centres / unit_in_s[unit], weights)
            else:
                ax.plot(bin_centres / unit_in_s[unit], weights, 'ko', mfc='None', markersize=msize)

            if legend is None:
                ax.legend(['blank'], fontsize=legend_fontsize)
            else:
                ax.legend(legend)
            weightlist.append(weights)
        else:
            wlist = []
            cols, marks = colors(), markers()
            splitsets, grr = split_along_edges(quake, edges[k])
            colors_mod = [cols[ii] for ii in range(len(grr)) if grr[ii]]
            markers_mod = [marks[ii] for ii in range(len(grr)) if grr[ii]]
            for j, splitset in enumerate(splitsets):
                weights, bin_edges = np.histogram(splitset.dt(), density=normalize_by_bins, bins=bins)
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                wlist.append(weights)
                x = bin_centres / unit_in_s[unit]
                if bar:
                    ax.bar(x, weights, edgecolor=colors_mod[j], facecolor=None, alpha=malpha)

                elif style == 0:
                    ax.plot(x, weights, markers_mod[j], mfc='None',
                            color=colors_mod[j], alpha=malpha, markersize=msize)
                elif style == 2:
                    ax.plot(x, weights, marker='o', mfc='None', linewidth=0.7, ls='--',
                            color=colors_mod[j], alpha=malpha, markersize=msize)

                else:
                    ax.plot(x, weights, markers_mod[j], mfc=colors_mod[j],
                            color='None', alpha=malpha, markersize=msize)

                if logscalex:
                    ax.set_xscale('log')
                if logscaley:
                    ax.set_yscale('log')
                xlab = 'non-normalised inter-event time $\Delta t$ in ' + unit
                ax.set_xlabel(xlab, fontsize=6)
                ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
                ax.set_xlim(left=xlim_left, right=xlim_right) # new
                if normalize_by_bins:
                    ax.set_ylabel('density', fontsize=6)
                else:
                    ax.set_ylabel('frequency', fontsize=6)

            weightlist.append(wlist)

        ax.tick_params(labelsize=6)

    if not suppress_plot:
        plt.show()
    if axes is not None:
        fig = None

    return fig, weightlist


def plot_pdf(quakes, suppress_plot=True, edges=None, malpha=0.5, msize=4,
             legend='all', title=None, subtitles=None, size_subs=7, show_gamma='norm',
             xlim_left=10**(-2.8), xlim_right=1e2, ylim_bottom=1e-3, ylim_top=1e3, fig_num=None,
             n_bins=30, figsize=(11, 6), tick_params=8, legend_fontsize='x-small'):
    """
    plot PDF of normalised inter-event times catalogue or list of catalogues, optionally split into provinces

    :param quakes           object of type EarthquakeSubset or list of EarthquakeSubsets
    :param suppress_plot    suppress plt.show() command if value is False
    :param edges            list of 2-tuples of lines seperating subregions
    (lines of longitudes (array), lines  of latitudes (array))   along which catalogue is split;
     subplot will be created for each item (2-tuple) of list and respective subregion setup
    :param malpha           float transparency of plotted points
    :param msize            float indicating marker size
    :param legend           'all', 'part', 'None' allowed, default all. 'all' means every subregion,
                            'part' means just mean/ theoretical distr.
    :param title            string plot caption
    :param subtitles        list of subplot captions
    :param show_gamma       gamma distribution with parameters based on variance shown
    :param xlim_left, xlim_right, ylim_top, ylim_bottom limits of x- and y-axis
    :param fig_num          name of plot
    :param n_bins           number of bins in histogram
    :param figsize          tuple containing width and height of figure
    :param tick_params      labelsize of ticks and tick labels
    :param legend_fontsize  fontsize of legend

    :return figure, 2-tuple containing list of bin centres and corresponding density values
    """

    if type(quakes) == EarthquakeSubset:
        quakes = [quakes]
    if quakes is None:
        return
    fig = plt.figure(figsize=figsize) if fig_num is None else plt.figure(figsize=figsize, num=fig_num)
    fig.tight_layout()
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(hspace=0.5)
    n = len(quakes)
    weightlist = []
    if subtitles is None:
        subtitles = n*['']
    if show_gamma == True:
        show_gamma = 'norm'
    if edges is not None:
        if type(edges) is not list:
            edges = n * [edges]
    else:
        edges = n * [None]

    if legend is not None:
        legend_val = [r'exp($\Delta \tau$)']
        if show_gamma == 'norm' or show_gamma == 'both':
            legend_val.append('gamma as in [Hainzl 2006]')
        if show_gamma == 'corral' or show_gamma == 'both':
            legend_val.append('gamma as in [Corral 2004]')

    for k, quake in enumerate(quakes):

        ax = plt.subplot(max(1, round(n / 2)), min(2, n), k + 1)
        ax.set_title(subtitles[k], size=size_subs, )

        line1, = ax.plot(np.logspace(-3, 3, 100), math.e ** (-np.logspace(-3, 3, 100)), '--', linewidth=1.0)

        dtau = quake.get_dtau()
        if show_gamma == 'norm' or show_gamma == 'both':
            dtau_pos = dtau[dtau > 0]
            try:
                linex, = ax.plot(dtau_pos, compute_gamma(dtau_pos)[0], '--', linewidth=1.0)
            except:
                print('gamma could not be computed')
                show_gamma = False

        if show_gamma == 'corral' or show_gamma == 'both':
            dtau_pos = dtau[dtau > 0]
            try:
                linexx, = ax.plot(dtau_pos, compute_gamma(dtau_pos, opt='corral')[0], '--', linewidth=1.0)
            except:
                print('gamma could not be computed')
                show_gamma = False

        bins = np.logspace(-3, np.log10(np.max(dtau)), n_bins)
        bin_centres = (bins[:-1] + bins[1:]) / 2

        if edges[k] is None:
            weights, bin_edges = np.histogram(quake.get_dtau(), density=True, bins=bins)
            ax.plot(bin_centres, weights, 'ko', mfc='None', markersize=msize)
            if legend != 'None':
                ax.legend(legend_val, fontsize=legend_fontsize)
            weightlist.append(weights)
        else:
            splitsets, grr = split_along_edges(quake, edges[k])
            m_weights = np.zeros_like(bin_centres)
            wlist = []
            cols, marks = colors(), markers()
            colors_mod = [cols[ii] for ii in range(len(grr)) if grr[ii]]
            markers_mod = [marks[ii] for ii in range(len(grr)) if grr[ii]]
            for j, splitset in enumerate(splitsets):
                weights, bin_edges = np.histogram(splitset.get_dtau(), density=True,
                                                  bins=bins)
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                wlist.append(weights)
                ax.plot(bin_centres, weights, 'o', mfc='None',
                        color=colors_mod[j], alpha=malpha, markersize=msize)
                m_weights = m_weights + weights

            m_weights = m_weights / len(splitsets)
            weightlist.append((wlist, m_weights))
            linem, = ax.plot(bin_centres, m_weights, 'k_', markersize=9)

            if legend == 'all':
                ax.legend(legend_val + ['L' + str(ii + 1) for ii in range(len(grr)) if grr[ii]] + ['mean'],
                          fontsize=legend_fontsize)
            elif legend == 'part':
                if show_gamma == 'norm':
                    ax.legend((linem, line1, linex),
                              ('mean', r'exp($\Delta \tau$)',
                               'gamma normal'), fontsize=legend_fontsize)
                elif show_gamma == 'corral':
                    ax.legend((linem, line1, linexx),
                              ('mean', r'exp($\Delta \tau$)',
                               'gamma as in [Corral 2004]'),
                              fontsize=legend_fontsize)
                elif show_gamma == 'both':
                    ax.legend((linem, line1, linex, linexx),
                              ('mean', r'exp($\Delta \tau$)', 'gamma as in [Hainzl 2006]',
                               'gamma as in [Corral 2004]'),
                              fontsize=legend_fontsize)
                else:
                    ax.legend((linem, line1),
                              ('mean', r'exp($\Delta \tau$)'),
                              fontsize=legend_fontsize)

        ax.tick_params(labelsize=tick_params)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\Delta \tau$', fontsize=8)
        plt.xlim(left=xlim_left, right=xlim_right)
        plt.ylim(bottom=ylim_bottom, top=ylim_top)
        plt.ylabel('PDF', fontsize=8)

    if not suppress_plot:
        plt.show()

    return fig, (bins, weightlist)


def plot_on_map_cartopy(quakes, suppress_plot=True, edges=None, contains_quakes=None, show_events=True,
                        alpha_event=0.8, mark='ko', title=None, show_legend=True, map_corners=None, point_size=0.6,
                        fig_num=None, figsize=(11,6), suppress_midpoint=False, axes=None, gridline_kwargs={},
                        legendloc='best'):
    """
        plotting earthquakes on cartopy plot

        :param quakes:          list of Earthquakesets, to be plotted on map
        :param suppress_plot:    boolean, if False plot is shown, default True
        :param edges:           list of 2tuples of ndarrays to be plotted on map, should correspond to subareas
        :param contains_quakes:  list of booleans, indicating if rectangle should be plotted, default None
                                if not None, only  subregions with True value marked
        :param show_events:      boolean, events are plotted if True, default True
        :param alpha_event:     float, transparency of event plots, default 0.8
        :param mark:            marker style of event points, default 'ko'
        :param title:           string, default None
        :param show_legend:      boolean, legend of subarea indications
        :param map_corners:     4tuple defining spatial extent of basemap, (minlon, minlat, maxlon, maxlat) default=(-72., -25., -66., -17.)
        :param point_size:      scaling factor of plotted events, default 0.6
        :param fig_num:         name of figure
        :param figsize:        tuple containing (width,height) of figure
        :param suppress_midpoint: suppress marker to appear on middle of each subregion
        :param axes:          axes object to plot map on
        :param gridline_kwargs: dictionary of optional arguments to feed the cartopy axes.gridlines() function
        :param legendloc: value of argument loc of legend function

        :return:                figure object

        """
    if type(quakes) == EarthquakeSubset:
        quakes = [quakes]

    if edges is not None:
        edge0size, edge1size = edges[0][0, 1] - edges[0][0, 0], edges[1][0, 1] - edges[1][0, 0]
        edgecenters0, edgecenters1 = edges[0][:, 0] + edge0size * 0.5, edges[1][:, 0] + edge1size * 0.5
        lines = []

    if quakes is None:
        return

    if axes is None:
        fig = plt.figure() if fig_num is None else plt.figure(figsize=figsize, num=fig_num)
        fig.tight_layout()
        if title is not None:
            fig.suptitle(title, fontsize=9)
    n = len(quakes)

    for k, quake in enumerate(quakes):
        lat = quake.location[:, 0]
        lon = quake.location[:, 1]
        if axes is None:
            ax = fig.add_subplot(max(1, round(n / 2)), min(2, n), k + 1, projection=ccrs.PlateCarree())
        else:
            ax = axes[k]
            ax.projection = ccrs.PlateCarree()  # probably not necessary
        (x0, y0, x1, y1) = (-72., -25., -66., -17.) if map_corners is None else map_corners
        ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.add_feature(ft.LAND)
        gl = ax.gridlines(draw_labels=True,**gridline_kwargs)
        gl.xlabel_style = {'size': 6, 'color': 'gray'}
        gl.ylabel_style = {'size': 6, 'color': 'gray'}
        if show_events:
            for i in range(len(lon)):
                ax.plot(lon[i], lat[i], mark, mfc='b', alpha=alpha_event, markersize=point_size * quake.magnitudes[i])

        if edges is not None:
            rn = range(len(edges[0]))
            j_style = 0  # Variable pointing at
            for j in rn:
                if contains_quakes is not None:
                    if contains_quakes[k] is not None:
                        if not contains_quakes[k][j]:
                            continue
                if not suppress_midpoint:
                    line, = ax.plot(edgecenters0[j], edgecenters1[j], markers()[j_style],
                                   color=colors()[j_style],
                                    mfc='None', markersize=5)
                    lines.append(line)
                ax.add_patch(
                    pat.Rectangle(
                        (edges[0][j, 0], edges[1][j, 0]),
                        edges[0][j, 1] - edges[0][j, 0], edges[1][j, 1] - edges[1][j, 0],
                        alpha=0.3, edgecolor='k', linewidth=1.5))
                j_style += 1

            ax.add_patch(pat.Rectangle(
                (edges[0][0, 0], edges[1][0, 0]), edge0size, edge1size,
                alpha=0.8, edgecolor='r', facecolor='None',
                linewidth=1, linestyle='--', zorder=3))
            labels = ['L' + str(ii + 1) for ii in range(j_style)]
            if show_legend and not suppress_midpoint:
                ax.legend(lines, labels, fontsize='xx-small', loc=legendloc)

    if not suppress_plot:
        plt.show()
    if axes is None:
        return fig


def ks_test(dtau):
    """
    compute ks_test_Results, called by compute_gamma_function()

    :param dtau:    array of normalised inter-event times
    :return:        2-tuple of object of type scipy.stats.stats.KstestResult
    with gamma (0) and exponential distribution (1) as reference distribution.
    """
    pdf, a = compute_gamma(dtau)    # reminder: a = k = y = shape, theta=beta=scale
    scale = 1/a
    test_y = stats.kstest(dtau, stats.gamma(a,scale=scale).cdf)
    test_e = stats.kstest(dtau, stats.expon.cdf)
    return test_y, test_e


def compute_ks_statistics(quakes, min_events=3, edges=None):
    """
    computes ks test results based on EarthquakeSubset object

    :param quakes:
    :param min_events:
    :param edges:
    :return:
    """
    if edges is None:
        dtau = np.array(list(quakes.get_dtau()))
        dtau = dtau[dtau > 0]
        if len(dtau) >= min_events:
            out_gamma, out_exp = ks_test(dtau)
            return out_exp,out_gamma
        else:
            print('region did not not contain sufficient quakes, therefore statistics not computed')
            return
    else:
        results = []
        splitsets, grr = split_along_edges(quakes, edges)
        for i,splitset in enumerate(splitsets):
            dtau = np.array(list(splitset.get_dtau()))
            dtau = dtau[dtau > 0]
            if len(dtau) >= min_events:
                out_gamma, out_exp = ks_test(dtau)
                results.append((out_exp, out_gamma))
            else:
                print('region',i,'only contains', len(dtau),'quakes, therefore statistics not computed')
                results.append((None, None))
        return results


def plot_ks_results(quakelist, list_edges, subtitles):
    """

    :param quakelist:       List of EarthquakeSubset objects
    :param list_edges:      list of subregion-boundaries
    :param subtitles:       title of (sub-)plots
    :return:                figure
    """
    fig0 = plt.figure()
    fig0.tight_layout()
    n = len(quakelist)
    for i, quakes in enumerate(quakelist):
        out = compute_ks_statistics(quakes, min_events=5, edges=list_edges[i])
        m = len(out)
        p_values_exp, p_values_gamma = np.zeros(m), np.zeros(m)

        for i_edges in range(m):
            p_values_exp[i_edges] = out[i_edges][0][1] if out[i_edges][0] is not None else np.nan
            p_values_gamma[i_edges] = out[i_edges][1][1] if out[i_edges][1] is not None else np.nan
        ax = plt.subplot(n,1,i+1)
        ax.plot(p_values_exp, 'bx')
        ax.plot(p_values_gamma, 'rx')
        ax.set_title(subtitles[i], size=7, )
        ax.grid()
        plt.xlabel('index of subregion')
        plt.ylim(1e-4, 1.3)
        plt.xlim(-0.5, (m-0.5))
        plt.hlines(y=0.05, xmin=-0.5, xmax=(len(out))+0.5, label='0.05', lw=0.7, linestyles='dashed', colors='k')
        plt.xticks(ticks=range(m), labels=['L'+str(i+1) for i in range(m)])
        ax.set_ylabel('KS-test p value', fontsize=7)
        ax.set_xlabel('Subregion index', fontsize=7)
        ax.tick_params(labelsize=8)
        ax.annotate('0.05', xy=(-0.5, 0.05), fontsize='x-small', color='k')
        plt.legend(['exponential distribution', 'gamma distribution'], fontsize='x-small')
        plt.yscale('log')
    return fig0


def load_engdahl_standard(minmags=[5.5, 6, 6.5], region=(-72.5, -69.5, -35, -17), suppress_time_cut=False):
    if region == 'large_region':
        region = (-72.5, -69.5, -35, -17)   # update to -35
    elif region == 'small_region_wide':     # without thrust
        region = (-72, -68., -25.5, -16.8)
    elif region == 'small_region_thrust':
        region = (-72, -69.5, -25.5, -16.8)
    print(type(region))

    (minlon, maxlon, minlat, maxlat) = region  # large region?
    if not suppress_time_cut:
        quakesets_eng = [EarthquakeSubset(load_catalogue_engdahl(), minmag=minmag, max_depth=55).cut_by_location(
            maxlon=maxlon, minlon=minlon, minlat=minlat, maxlat=maxlat).cut_by_time_frame(
            mintime=obspy.UTCDateTime(1964, 1, 1)) for minmag in minmags]
    else:
        quakesets_eng = [EarthquakeSubset(load_catalogue_engdahl(), minmag=minmag, max_depth=55).cut_by_location(
            maxlon=maxlon, minlon=minlon, minlat=minlat, maxlat=maxlat) for minmag in minmags]

    return quakesets_eng


def plot_slab_contour(data=None, types=['2D', '3D'], magnitudes=None, showplot=True, title=None, azim=None, fig=None,
                      axes=None, show_cbar=True):
    """
    :param data:    3xn ndarray with lon,lat,depth values as columns.?? tuple! Depth values must be positive. ??? not tuple??
    :param types:   list of Strings: containing types of plot to be done'2D','3D' for 2D or 3D contour plots
                    by default: both
    :return:        None
    """

    def load_slab(slabdata='data/slab/chile_slab_data.txt'):
        """
        reads slab data file and returns ndarrays of locations

        :param slabdata: data file with lon,lat,depth grid
        :return: tuple (longitude, latitude, depth (positive values))
        """
        df = pd.read_csv(slabdata, header=None, sep="\s+")
        lon, lat, z = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values
        return lon, lat, z

    x, y, z = load_slab()
    xi = np.linspace(min(x), max(x), 500)
    yi = np.linspace(min(y), max(y), 600)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (Xi, Yi))  # negative depth values
    if axes is None:
        fig = plt.figure(figsize=(13, 9))
        fig.tight_layout()
        if title is not None:
            fig.suptitle(title, fontsize=9)
    m_size = 6 if magnitudes is None else 1.4 * magnitudes
    k = 1

    if '2D' in types:
        cm = plt.get_cmap('viridis')  # added!!
        cm.set_bad('k', 1)  # added!!
        if axes is None:
            ax0 = fig.add_subplot(1, len(types), k, projection=ccrs.PlateCarree())
        else:
            ax0 = axes[0]

        (x0, y0, x1, y1) = (-72., -24.5, -66., -18.5)
        ax0.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
        ax0.coastlines('50m')
        ax0.add_feature(ft.LAND)
        gl = ax0.gridlines(draw_labels=True)
        gl.xlabel_style = {'size': 5, 'color': 'gray'}
        gl.ylabel_style = {'size': 5, 'color': 'gray'}

        cont = ax0.contourf(Xi, Yi, Z, np.arange(np.round(np.nanmin(Z)), np.round(np.nanmax(Z)), 10), edgecolors=('k',))
        ax0.contour(Xi, Yi, Z, linestyles='solid', colors='k', linewidths=(0.1,))
        ax0.set_xlabel('Longitude', fontsize=7)
        ax0.set_ylabel('Latitude', fontsize=7)

        if data is not None:  # why not outside the if clause
            ax0.scatter(data[0], data[1], s=m_size, edgecolors='k', linewidths=0.5, facecolor='b', alpha=0.8,
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        k += 1
        if fig is not None and '3D' not in types:
            if show_cbar:
                cbar_ax = fig.add_axes([0.90, 0.40, 0.02, 0.2])  # mal schaun
                col = fig.colorbar(cont, cax=cbar_ax)
                col.set_label('Slab surface depth [km]', fontsize=6)
                col.ax.tick_params(labelsize=6)

    if '3D' in types:
        if axes is None:
            ax1 = fig.add_subplot(1, len(types), k, projection='3d')
        else:
            ax1 = axes[-1]
        ax1.view_init(elev=None, azim=15 if azim is None else azim)
        ax1.set_xlabel('Longitude', fontsize=7)
        ax1.set_ylabel('Latitude', fontsize=7)
        ax1.set_zlabel('z[km]', fontsize=7)
        ax1.tick_params(labelsize=7)
        cm = plt.get_cmap('viridis')
        cm.set_bad('k', 1)
        surf2 = ax1.plot_surface(Xi, Yi, Z, cmap=cm, vmin=np.nanmin(Z), vmax=np.nanmax(Z), linewidth=0,
                                 edgecolors=('None',))

        ax1.clabel(surf2, inline=True, fontsize=50)
        if data is not None:
            ax1.scatter(data[0], data[1], -data[2], c=-data[2], s=5 * m_size, cmap=cm, edgecolors='r', linewidths=0.3,
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        if len(types) == 2:

            ax0.annotate("(a)", xy=(-0.08, 0.9), xycoords="axes fraction")
            ax1.annotate("(b)", xy=(-0.08, 0.9), xycoords="axes fraction")
            if fig is not None:
                fig.subplots_adjust(wspace=0.05)
        if fig is not None:
            if show_cbar:
                col = fig.colorbar(surf2, shrink=0.4, aspect=8)
                col.ax.tick_params(labelsize=2)
                col.set_label('Slab Surface depth [km]', fontsize=6)

    if showplot:
        plt.show()

    return fig


def main():
    return


if __name__ == '__main__':
    main()
