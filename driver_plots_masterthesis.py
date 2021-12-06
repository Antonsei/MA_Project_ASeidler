from iet_analysis import *
import matplotlib.ticker as mticker


def load_4_ipoc_datasets(minmag=2.8):
    """
    Create four EarthquakeSubject objects based on IPOC catalogue with settings:
    1: all earthquakes from catalogue
    2: all earthquakes from catalogue meeting the "thrust-type events" criteria
    3: earthquakes from catalogue after removing several fore- and aftershocks from 2007 Tocopilla and 2014 Iquique
    4: earthquakes after removal of events as in (3) AND meeting the "thrust-type events" criteria

    :param minmag: minimum magnitude threshold of events
    :return: 4-tuple of EarthquakeSubjects (1,2,3,4)
    """
    ipoc_all = EarthquakeSubset(load_catalogue_ipoc(), minmag=minmag)
    ipoc_all_th = EarthquakeSubset(load_catalogue_ipoc(), minmag=minmag,
                                   min_depth=1, max_depth=55).cut_by_location(maxlon=-69.5)
    ipoc_car = EarthquakeSubset(load_catalogue_ipoc_modified(), minmag=minmag)
    ipoc_car_th = EarthquakeSubset(load_catalogue_ipoc_modified(), minmag=minmag,
                                   min_depth=1, max_depth=55).cut_by_location(maxlon=-69.5)
    return ipoc_all, ipoc_all_th, ipoc_car, ipoc_car_th


def make_edges_engdahl_all(small_reg=False, disty=250):
    """
    Auxiliary function to divide dataset into smaller subregions in predefine set-up
    :param small_reg: bool if True: region delimitors (min_lon, max_lon, min_lat, max_lat): (-72, -69.5, -25.5, -17)
    else: (-69.5, -72.5, -35, -17)
    :param disty:   alter extent of subregions in NS direction, default 250
    :return:
    """
    distx = 350
    overlapx = 0
    overlapy = 100
    if small_reg:
        minlat, minlon, maxlat = -24.5, -72, -17.5
    else:
        maxlon, minlon, minlat, maxlat = -69.5, -72.5, -35, -17
    if small_reg:
        maxlon, minlat, minlon, maxlat = -69.5, -24.5, -72, -17.5

    edges_engdahl = subregion_boundaries(distx=distx, disty=disty, overlapx=overlapx, overlapy=overlapy,
                                         maxlon=maxlon, minlon=minlon, minlat=minlat, maxlat=maxlat,
                                         strictlims=True, n2s=True)
    return edges_engdahl


def load_ipoc_subregion_setup():
    edgelist_ipoc = [subregion_boundaries(distx=300, overlapx=0,
                                     disty=[300, 250, 200][i], overlapy=[150, 120, 100][i], maxlon=-69.3,
                                     strictlims=False, n2s=True) for i in range(3)]
    return edgelist_ipoc


def generate_subtitles(ks_stats):
    soustitres = []
    for stats in ks_stats:
        if stats is None:
            soustitres.append(' ')
        else:
            pval_out_1 = r"$\bf{" + "{:.1e}".format(stats[0][1]) + "}$" if stats[0][1] > 1e-20 else r'$\bf{0.0e-20}$'
            pval_out_2 = r"$\bf{" + "{:.1e}".format(stats[1][1]) + "}$" if stats[1][1] > 1e-20 else r'$\bf{0.0e-20}$'
            soustitres.append(
                'KS-test results for Exp (a) and Gamma (b): ' + '\n' + 'p values:  ' + pval_out_1 +
                ' (a) and ' + pval_out_2 + ' (b)')
    return soustitres
# Functions to create and save plots: ---------------------------------------


def create_magnitude_frequency_plots(savefigs=False, path=None, fig_dpi=200):
    """
    Create plots of the magnitude-frequency relationship for (1)

    :param savefigs: save figure in png file
    :param path:     directory where figure should be saved
    :param fig_dpi:  resolution
    :return: None
    """
    if path is None:
        path = ''

    fig_gr_ipoc, axes = plt.subplots(1, 1, sharey=False, figsize=(4, 3))
    axes = [axes]
    fig_gr_ipoc.tight_layout()
    xlims, ylims = (2.0, 8.2), (0.34, 45000)
    for catalogue in load_4_ipoc_datasets(minmag=2)[0:2]:
        gutenberg_richter_plot(catalogue.magnitudes, xlims=xlims, ylims=ylims, ax=axes[0], bins=40, norm=False)
    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.legend(['entire catalogue', 'just thrust events'], fontsize=6)

    fig_gr_eng, ax = plt.subplots(1, 1, sharey=False, figsize=(4, 3))
    fig_gr_eng.tight_layout()
    gutenberg_richter_plot(load_engdahl_standard(minmags=[5.5],
                                                 region=(-72., -69.5, -35, -17), suppress_time_cut=False)[
                               0].magnitudes,
                           xlims=(5.3, 8.1), ax=ax, norm=False)
    ax.tick_params(labelsize=6)
    fig_gr_eng.subplots_adjust(hspace=0.5, wspace=-3)

    if savefigs:
        fig_gr_eng.savefig(fname=path+'gr_engdahl.png', dpi=fig_dpi)
        fig_gr_ipoc.savefig(fname=path+'gr_ipoc.png', dpi=fig_dpi)
    return


def create_seismicity_rate_plots(savefigs=False, path=None, fig_dpi=200):
    """
    plot seismicity rate for:
    (1) different subsets of IPOC catalogue
    (2) Centennial Engdahl catalogue for two regions in N. Chile (Area S, Area L) 1900-2007
    (3) Centennial Engdahl catalogue for two regions in N. Chile (Area S, Area L) 1964-2007

    :param savefigs: bool, if True save figures
    :param path: directory to save figures
    :param fig_dpi: resolution
    :return: None
    """

    if path is None:
        path = ''
    fig_sr_ipoc, (ax1) = plt.subplots(1, 1, sharey=False, figsize=(7, 4))
    fig_sr_ipoc.tight_layout()
    plot_seismicity_rate([d.time for d in load_4_ipoc_datasets()], bins=96, title=None, ax=ax1, logscale=True,
                         plt_kwargs=[{'ls': ['--', '--', '-', '-'][i], 'marker': '.', 'markersize': 3,
                                      'color': ['b', 'g', 'b', 'g'][i]} for i in range(4)])
    ax1.legend(['all events', 'thrust events',
                'all events without major aftershock series', 'thrust events without major aftershock series'],
               fontsize=7)
    ax1.set_ylabel('events/month')

    # Plot Seismicity Rate from Centennial Catalogue! ------------------------------------------------------------------
    fig_sr_eng_1, (ax1) = plt.subplots(1, 1, sharey=False, figsize=(7, 4))
    fig_sr_eng_2, (ax2) = plt.subplots(1, 1, sharey=False, figsize=(7, 4))

    for suppress_time_cut, fig, axis in zip([True, False], [fig_sr_eng_1, fig_sr_eng_2], [ax1, ax2]):
        fig.tight_layout()

        eng_quakes_s, eng_quakes_l = load_engdahl_standard(minmags=[5.5], region=(-72, -69.5, -25.5, -16.8),
                                                           suppress_time_cut=suppress_time_cut)[0], \
                                     load_engdahl_standard(minmags=[5.5], region=(-72.5, -69.5, -35, -17),
                                                           suppress_time_cut=suppress_time_cut)[0]

        plot_seismicity_rate([d.time for d in [eng_quakes_s, eng_quakes_l]],
                             bins=eng_quakes_s.time[-1].year - eng_quakes_s.time[0].year,
                             title=None, ax=axis, logscale=False,
                             plt_kwargs=[{'ls': '-', 'marker': '.', 'markersize': 3, 'color': c} for c in
                                         ['b', 'g']])
        axis.set_ylabel('events/year')
        if suppress_time_cut:
            axis.vlines(x=obspy.UTCDateTime(1966, 1, 1).matplotlib_date, ymin=0, ymax=20, linestyle='--',
                        color='gray',
                        linewidth=0.7)

        axis.legend(['Area S', 'Area L'], fontsize=7)
        axis.set_ylim((0, None))

    if savefigs:
        fig_sr_ipoc.savefig(fname=path + 'oc_ipoc.png', dpi=fig_dpi)
        fig_sr_eng_1.savefig(fname=path + 'oc_eng1.png', dpi=fig_dpi)
        fig_sr_eng_2.savefig(fname=path + 'oc_eng2.png', dpi=fig_dpi)
    return


def create_map_plots_1(savefigs=False, path=None, fig_dpi=200):
    """
    Create several plots with maps:
    (1): map showing events on subduction thrust for a) entire catalgoue and b) thrust-type events
    (2): map showing study areas (Area S, Area L) used in analysis of Centennial catalogue

    :param savefigs: save figure in png file
    :param path:     directory where figure should be saved
    :param fig_dpi:  resolution
    :return: None
    """
    # map showing events on subduction thrust for a) entire catalgoue and b) thrust-type events
    ipoc_4_datasets = load_4_ipoc_datasets()
    data_1, data_2 = [(subset.location[:, 1], subset.location[:, 0], subset.depth)
                      for subset in [ipoc_4_datasets[0], ipoc_4_datasets[1]]]

    fig_maps = plt.figure(figsize=(8, 4))
    axes = [fig_maps.add_subplot(1, 2, k, projection=ccrs.PlateCarree()) for k in [1, 2]]
    fig_maps = plot_slab_contour(data_1, types=['2D'], showplot=False, axes=[axes[0]], fig=fig_maps,
                                 show_cbar=False)
    fig_maps = plot_slab_contour(data_2, types=['2D'], showplot=False, axes=[axes[1]], fig=fig_maps, show_cbar=True)
    fig_maps.subplots_adjust(left=0.05, wspace=0.15, right=0.85)
    axes[0].set_title('a)', loc='left', fontsize=11, pad=9)
    axes[1].set_title('b)', loc='left', fontsize=11, pad=9)

    # map showing Area S and area L on Engdahl plot:

    fig_eng_map = plt.figure(figsize=(4, 6))
    fig_eng_map.tight_layout()
    axes_fig_eng_map = [fig_eng_map.add_subplot(1, 2, k, projection=ccrs.PlateCarree()) for k in [1, 2]]
    for i, reg in enumerate([(-72, -69.5, -24.5, -17), (-72.5, -69.5, -35, -17)]):
        plot_on_map_cartopy(quakes=load_engdahl_standard(minmags=[5.5], region=reg)[0],
                            edges=(np.array([reg[0:2]]), np.array([reg[2:4]])),
                            map_corners=(-73., -36., -67., -16.5),
                            show_legend=False, title="", suppress_midpoint=True, axes=[axes_fig_eng_map[i]],
                            figsize=(9, 5))
    fig_eng_map.text(0.1, 0.9, s='a)')
    fig_eng_map.text(0.54, 0.9, s='b)')
    if savefigs:
        fig_maps.savefig(path + 'maps_ipoc.png', dpi=fig_dpi)
        fig_eng_map.savefig(path + 'maps_eng.png', dpi=fig_dpi)
    return


def create_histograms(savefigs=False, path=None, fig_dpi=200):
    """
    Create histograms of non normalised inter-event times, including total frequency plots and density plots
    IPOC thrust events with different magnitude thresholds (m0 = 3, 4, 5 and 5.5) are used,
    as well as Engdahl Centennial dataset with m0 = 5.5 and 6
    """

    # (1) Plot histograms of non normalised inter-event times for IPOC catalogue:

    # load from earthquake catalogue: all (1), thrust (2), modified (2 sequences removed) (3), modified thrust (4):

    ipoc_all, ipoc_all_th, ipoc_car, ipoc_car_th = load_4_ipoc_datasets()

    edgelist_ipoc = load_ipoc_subregion_setup()     # load subregion setups for IPOC catalogue
    axlim_kwarg_list = list()

    # dictionaries containing axis limits of different plots:
    axlim_kwarg_list.append(
        ({'xlim_left': 1.0e2, 'xlim_right': 3e8, 'ylim_bottom': 0.5, 'ylim_top': 1e3},      # total counts
         {'xlim_left': 1.0e2, 'xlim_right': 3e8, 'ylim_bottom': 1e-10, 'ylim_top': 8e-4}))  # density
    axlim_kwarg_list.append(
        ({'xlim_left': 1.0e2, 'xlim_right': 3e8, 'ylim_bottom': 0.5, 'ylim_top': 3e3},      # mod counts
         {'xlim_left': 1.0e2, 'xlim_right': 3e8, 'ylim_bottom': 1e-10, 'ylim_top': 8e-4}))  # mod density

    for i, catalogue in enumerate([ipoc_all_th, ipoc_car_th]):

        fig_hists_1, axes_1 = plt.subplots(2, 2, figsize=(6, 4))  # figure with small magnitude thresholds 3.0 and 4.0
        fig_hists_2, axes_2 = plt.subplots(2, 2, figsize=(6, 4))  # figure with great magnitude thresholds 5.0 and 5.5
        for fig0, axes0, mag in zip([fig_hists_1, fig_hists_2], [axes_1, axes_2], [[3, 4], [5, 5.5]]):
            for k in range(2):
                catalogue_multimags = [catalogue.cut_by_mag(minmag=mm) for mm in mag]
                alk = axlim_kwarg_list[i][k] if fig0 is fig_hists_1 else {}
                plot_hists(catalogue_multimags, edges=edgelist_ipoc[2],
                           subtitles=[r'$m_0 = $' + str(il.minmag()) for il in catalogue_multimags],
                           bins=np.logspace(2.2, 9, 18),
                           legend='all', legend_fontsize=5,
                           normalize_by_bins=[False, True][k], axes=axes0[k, :], logscaley=True, **alk)
            fig0.subplots_adjust(hspace=0.45, wspace=0.3)

            for k, ax in enumerate(fig0.get_axes()):
                ax.set_title(['1a)', '1b)', '2a)', '2b)'][k], loc='left', fontsize=9, pad=8)

        for ax_tot, ax_dens in zip(axes_2[0, :], axes_2[1, :]):
            ax_tot.set_ylim((0.5, 1e2))
            ax_dens.set_ylim((1e-10, 8e-4))
            ax_tot.set_xlim((1.7e2, 3e8))
            ax_dens.set_xlim((1.7e2, 3e8))
            if catalogue == ipoc_car_th:
                ax_tot.set_xlim(1e3, 3e8)
                ax_dens.set_xlim(1e3, 3e8)

        if savefigs:
            fig_hists_1.savefig(fname=path + 'hists_ipoc_smallmags' + ['all.png', 'dec.png'][i], dpi=fig_dpi)
            fig_hists_2.savefig(fname=path + 'hists_ipoc_greatmags' + ['all.png', 'dec.png'][i], dpi=fig_dpi)

    # (2): Plot histograms of non normalised inter-event times for Engdahl centennial catalogue:

    fig_hists_eng, axes = plt.subplots(2, 2, figsize=(6, 4))
    eng_quakes_s, eng_quakes_l = load_engdahl_standard(minmags=[5.5], region=(-72, -69.5, -24.5, -17),
                                                       suppress_time_cut=False)[0], \
                                 load_engdahl_standard(minmags=[5.5], region=(-72.5, -69.5, -35, -17),
                                                       suppress_time_cut=False)[0]
    edges = make_edges_engdahl_all(small_reg=True)

    for k in range(2):
        catalogue_multimags = [eng_quakes_s.cut_by_mag(minmag=mm) for mm in [5.5, 6]]
        plot_hists(catalogue_multimags, edges=edges,
                   subtitles=[r'$m_0 = $' + str(il.minmag()) for il in catalogue_multimags],
                   title=None, bins=np.logspace(2.2, 9, 18),
                   legend='all', legend_fontsize=5,
                   style=2, normalize_by_bins=[False, True][k], axes=axes[k, :], logscaley=[False, True][k])

    for ax_ob, ax_unt in zip(axes[0, :], axes[1, :]):
        ax_ob.set_ylim((-0.2, 6.2))
        ax_unt.set_ylim((1e-10, 1e-4))

    fig_hists_eng.subplots_adjust(hspace=0.45, wspace=0.3)

    for k, ax in enumerate(fig_hists_eng.get_axes()):
        ax.set_title(['1a)', '1b)', '2a)', '2b)'][k], loc='left', fontsize=9, pad=8)
    if savefigs:
        fig_hists_eng.savefig(path+'hists_eng.png', dpi=fig_dpi)
    return


def create_map_plots_2(savefigs=False, path=None, fig_dpi=200):
    """



    """
    if path is None:
        path = ''
    map_corners_eng = (-72.5, -25, -68.5, -17)

    gridline_kwargs = {'xlocs': mticker.FixedLocator([-88, -72, -70, -50]),
                       'ylocs': mticker.FixedLocator(np.arange(-40, -10, 2)), 'linestyle': ':',
                       'color': 'gray', 'linewidth': 0.5}

    eng_quakes_s, eng_quakes_l = load_engdahl_standard(minmags=[5.5], region=(-72, -69.5, -24.5, -17),
                                                       suppress_time_cut=False)[0], \
                                 load_engdahl_standard(minmags=[5.5], region=(-72.5, -69.5, -35, -17),
                                                       suppress_time_cut=False)[0]
    quakesets = [eng_quakes_s, eng_quakes_s.cut_by_mag(6)]
    map_eng = plot_on_map_cartopy(quakesets, edges=make_edges_engdahl_all(small_reg=True, disty=200),
                                  map_corners=map_corners_eng, show_legend=True, title="",
                                  gridline_kwargs=gridline_kwargs,
                                  figsize=(8, 12), legendloc='right')
    axes = map_eng.get_axes()
    axes[0].get_legend().remove()
    axes[1].get_legend().set_bbox_to_anchor((1.4, 0.5))

    for i, ax in enumerate(axes):
        ax.set_title(r'$m_0 = $' + str(quakesets[i].minmag()), fontsize=8, pad=11)
        ax.set_title(['a)', 'b)'][i], loc='left', fontsize=10, pad=11)

    quakesets = [eng_quakes_l, eng_quakes_l.cut_by_mag(6)]
    map_eng_l = plot_on_map_cartopy(quakesets, legendloc='center right',
                                    edges=make_edges_engdahl_all(small_reg=False, disty=250),
                                    map_corners=(-73., -36., -67., -16.5), show_legend=True, title="",
                                    gridline_kwargs=gridline_kwargs)
    axes = map_eng_l.get_axes()
    for i, ax in enumerate(axes):
        ax.set_title(r'$m_0 = $' + str(quakesets[i].minmag()), fontsize=8, pad=11)
        ax.set_title(['a)', 'b)'][i], loc='left', fontsize=10, pad=11)

    axes[0].get_legend().remove()

    map_eng_l.set_size_inches(7.7, 9)
    posbox = axes[0].get_position()
    axes[1].get_legend().set_bbox_to_anchor((0.9, 0.5))

    ipoc_car_th = EarthquakeSubset(load_catalogue_ipoc_modified(), minmag=2.8,
                                   min_depth=1, max_depth=55).cut_by_location(maxlon=-69.5)

    quakesets = [ipoc_car_th.cut_by_mag(minmag=mag) for mag in [3, 4]]
    edgelist_ipoc = load_ipoc_subregion_setup()

    map_ipoc_th_smallmags = plot_on_map_cartopy(quakesets,
                                                edges=edgelist_ipoc[2], map_corners=map_corners_eng,
                                                show_legend=True,
                                                gridline_kwargs=gridline_kwargs, figsize=(9, 5))
    for i, ax in enumerate(map_ipoc_th_smallmags.get_axes()):
        ax.set_title(r'$m_0 = $' + str(quakesets[i].minmag()), fontsize=9, pad=11)
        ax.set_title(['a)', 'b)'][i], loc='left', fontsize=11, pad=11)

    quakesets = [ipoc_car_th.cut_by_mag(minmag=mag) for mag in [5, 5.5]]
    map_ipoc_th_greatmags = plot_on_map_cartopy([ipoc_car_th.cut_by_mag(minmag=mag) for mag in [5, 5.5]],
                                                edges=edgelist_ipoc[2], map_corners=map_corners_eng,
                                                show_legend=True,
                                                gridline_kwargs=gridline_kwargs, figsize=(9, 5))
    for i, ax in enumerate(map_ipoc_th_greatmags.get_axes()):
        ax.set_title(r'$m_0 = $' + str(quakesets[i].minmag()), fontsize=9, pad=11)
        ax.set_title(['c)', 'd)'][i], loc='left', fontsize=11, pad=11)


    map_ipoc_edges = plt.figure(figsize=(8, 4.5))
    axes = [map_ipoc_edges.add_subplot(1, 3, k, projection=ccrs.PlateCarree()) for k in [1, 2, 3]]
    for i in range(3):
        plot_on_map_cartopy([ipoc_car_th.cut_by_mag(minmag=mag) for mag in [2.8]],
                            edges=edgelist_ipoc[i], map_corners=map_corners_eng, show_legend=True,
                            axes=[axes[i]], gridline_kwargs=gridline_kwargs, show_events=False)
        axes[i].set_title(['a)', 'b)', 'c)'][i], loc='left', fontsize=11, pad=10)

    quakesets = [ipoc_car_th.cut_by_mag(minmag=mag) for mag in [3, 4, 5, 5.5]]
    map_ipoc_th_allmags = plot_on_map_cartopy(quakesets,
                                              edges=edgelist_ipoc[2], map_corners=map_corners_eng,
                                              show_legend=False,
                                              gridline_kwargs=gridline_kwargs, figsize=(9, 10))
    for i, ax in enumerate(map_ipoc_th_allmags.get_axes()):
        ax.set_title(r'$m_0 = $' + str(quakesets[i].minmag()), fontsize=8, pad=11)
        ax.set_title(['a)', 'b)', 'c)', 'd)'][i], loc='left', fontsize=10, pad=11)

    if savefigs:
        map_eng.savefig(path + 'maps_eng_5_6.png', dpi=fig_dpi, figsize=(9, 5))
        map_eng_l.savefig(path + 'maps_eng_l.png', dpi=fig_dpi, figsize=(9, 5))
        map_ipoc_th_smallmags.savefig(path + 'maps_ipoc_car_3_4.png', dpi=fig_dpi)
        map_ipoc_th_greatmags.savefig(path + 'maps_ipoc_car_allmags.png', dpi=fig_dpi)
        map_ipoc_edges.savefig(path + 'maps_ipoc_edges')
    return


def create_pdf_plots(savefigs=False, path=None, fig_dpi=200):
    """
    create probability density plot of various subsets from IPOC and Centennial dataset

    :param savefigs:
    :param path:
    :param fig_dpi:
    :return:
    """
    if path is None:
        path = ''

    ipoc_all, ipoc_all_th, ipoc_car, ipoc_car_th = load_4_ipoc_datasets()
    minmags = [3, 4, 5, 5.5]
    ipoc_multimag_all = [ipoc_all_th.cut_by_mag(minmag=minmag) for minmag in minmags]
    ipoc_multimag_dec = [ipoc_car_th.cut_by_mag(minmag=minmag) for minmag in minmags]
    edgelist_ipoc = load_ipoc_subregion_setup()
    subbies = [r"    $\bf{" + str([300, 250, 200][i]) + "}$ " + "km x " + r"$\bf{300}$ km subregions with $\bf{" +
               str([150, 120, 100][i]) + "}$ " + "km overlap" for i in range(3)]

    labels = ['1', '2', '1', '2']
    for kk, quakeset in enumerate(ipoc_multimag_all):
        fig, _ = plot_pdf(4 * [quakeset], edges=edgelist_ipoc + [None],
                          subtitles=subbies + ['no partitioning into subregions'], legend='all',
                          title=  # 'IET distributions for earthquakes with magnitudes greater than '+
                          r'$m_0 = $' + str(minmags[kk]) + r' (' + str(len(quakeset)) + r' events)',
                          legend_fontsize=6,
                          n_bins=20, show_gamma='both', malpha=0.7, msize=4, size_subs=9)

        filename = 'pdf_' + 'ipoc_minmag_' + str(quakeset.minmag()) + '.png'
        for i, ax in enumerate(fig.get_axes()):
            ax.set_title(labels[kk] + ['a)', 'b)', 'c)', 'd)'][i], loc='left', fontsize=11, pad=5)
        if savefigs:
            fig.savefig(path+filename, dpi=fig_dpi)

    for kk, quakeset in enumerate(ipoc_multimag_dec):
        fig, _ = plot_pdf(4 * [quakeset], edges=edgelist_ipoc + [None],
                          subtitles=subbies + ['no partitioning into subregions'], legend='all',
                          title=r'$m_0 = $' + str(minmags[kk]) + r' (' + str(len(quakeset)) + r' events)',
                          legend_fontsize=6,
                          n_bins=20, show_gamma='both', malpha=0.7, msize=4, size_subs=9)

        for i, ax in enumerate(fig.get_axes()):
            ax.set_title(labels[kk] + ['a)', 'b)', 'c)', 'd)'][i], loc='left', fontsize=10, pad=5)

        filename = 'pdf_ipoc_minmag_carsten' + str(quakeset.minmag()) + '.png'
        if savefigs:
            fig.savefig(path + filename, dpi=fig_dpi)

    # Probability density plots from Engdahl Centennial catalogue

    engdahl_largereg_list = load_engdahl_standard(minmags=[5.5, 6], region=(-72, -69.5, -35, -17))
    fig_engdahl, _ = plot_pdf(engdahl_largereg_list, figsize=(11, 2.7),
                              edges=2 * [make_edges_engdahl_all()],
                              subtitles=[r'$m_0 = 5.5$ ($n = $' + str(len(engdahl_largereg_list[0])) + r'$)$',
                                         r'$m_0 = 6.0$ ($n = $' + str(len(engdahl_largereg_list[1])) + r'$)$'],
                              legend='all', n_bins=20, show_gamma='both',
                              size_subs=9, malpha=0.8, msize=3, legend_fontsize=4.9)
    for i, ax in enumerate(fig_engdahl.get_axes()):
        ax.set_title(['a)', 'b)'][i], loc='left', fontsize=10, pad=5)
        ax.xaxis.set_label_coords(x=0.5, y=-0.09)
    if savefigs:
        fig_engdahl.savefig(path + 'engdahl_pdfs.png', dpi=fig_dpi)

    figs = []
    for i, subset in enumerate([ipoc_all_th.cut_by_mag(minmag=mm) for mm in [3, 4, 5, 5.5]]):
        subsets_ipoc = split_along_edges(subset, edgelist_ipoc[0])[0]
        ks_stats_ipoc = [compute_ks_statistics(subset) for subset in subsets_ipoc]
        subtitles = generate_subtitles(ks_stats_ipoc)
        fig, _ = plot_pdf(subsets_ipoc, title=r'$m_0 = $' + str(subset.minmag()) +
                                              r' (' + str(len(subset)) + r' events)',
                          subtitles=subtitles, show_gamma='norm', size_subs=9)
        filename = 'pdf_ks_ipoc_' + str(subset.minmag()) + '.png'
        fig.text(0.06, 0.51, ['(I)', '(II)', '(I)', '(II)'][i], fontsize=15)
        fig.savefig(path + filename, dpi=fig_dpi)
        figs.append(fig)
    for i, subset in enumerate([ipoc_car_th.cut_by_mag(minmag=mm) for mm in [3, 4, 5, 5.5]]):

        subsets_ipoc = split_along_edges(subset, edgelist_ipoc[0])[0]
        ks_stats_ipoc = [compute_ks_statistics(subset) for subset in subsets_ipoc]
        subtitles = generate_subtitles(ks_stats_ipoc)
        fig, _ = plot_pdf(subsets_ipoc, title=r'$m_0 = $' + str(subset.minmag()) +
                                              r' (' + str(len(subset)) + r' events)',
                          subtitles=subtitles, show_gamma='norm', size_subs=9)
        fig.text(0.06, 0.50, ['(I)', '(II)', '(I)', '(II)'][i], fontsize=15)
        if savefigs:
            filename = 'pdf_ks_ipoc_car_' + str(subset.minmag()) + '.png'
            fig.savefig(path + filename, dpi=fig_dpi)
    return


def create_ks_plot(savefigs=False, path=None, fig_dpi=200):
    """
    plot of p-values of 1-sample ks-test with Exp and Gamma as reference distr. as reference distribution
    for each subregion.
    Dataset is Centennial Engdahl catalogue with magnitude threshold 5.5, events from "Area L"
    """
    if path is None:
        path = ''
    fig = plot_ks_results(load_engdahl_standard(minmags=[5.5], region=(-72.5, -69.5, -35, -17)),
                          [make_edges_engdahl_all()], subtitles=[r'$m_0 = 5.5$'])
    if savefigs:
        fig.savefig(path + 'engdahl_ks_plot.png', dpi=fig_dpi)
    return


if __name__ == '__main__':
    create_histograms(savefigs=False, path='plots/hists/')
    create_magnitude_frequency_plots(savefigs=False, path='plots/gr/')
    create_map_plots_1(savefigs=False, path='plots/maps/')
    create_map_plots_2(savefigs=False, path='plots/maps/')
    create_pdf_plots(savefigs=False, path='plots/pdf/')
    create_seismicity_rate_plots(savefigs=False, path='plots/sr/')
    create_ks_plot(savefigs=False, path='plots/ks_plots/')

    plt.show()
