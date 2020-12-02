""" Summary plots of SHAP values across a whole dataset.
"""

from __future__ import division

import warnings
import numpy as np
from scipy.stats import gaussian_kde
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from shap.plots import colors, labels
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

from shap.common import convert_name, approximate_interactions



def rank_shap(shap_values, X, sorting, onlyrank=True):
    means = X.mean(0)
    indexes = [X[:,i] > means[i] for i in range(len(means))]
    rank = np.array([np.mean(shap_values[indexes[i], i]) for i in range(len(means))])
    rank_sd = np.array([np.std(shap_values[indexes[i], i]) for i in range(len(means))])

    if onlyrank:
        if sorting is None:
            return np.argsort(rank)
        else:
            return np.lexsort((rank, sorting))
    else:
        if sorting is None:
            return np.argsort(rank), rank, rank_sd
        else:
            return np.lexsort((rank, sorting)), rank, rank_sd

# TODO: remove unused title argument / use title argument
def summary_plot(shap_values, features=None, feature_names=None, max_display=None, plot_type=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True, sorting=None,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None, file_name=None,
                 class_inds=None, cmap=None,
                 color_bar_label=labels["FEATURE_VALUE"],
                 # depreciated
                 auto_size_plot=None):
    """Create a SHAP summary plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    cmap : cmp=plt.get_cmap('viridis')

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """

    
    if cmap == None:
        cmap = colors.red_blue
        
    # deprication warnings
    if auto_size_plot is not None:
        warnings.warn("auto_size_plot=False is depricated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    # summary_plot for shape interation value
    # plotting SHAP interaction values
    if not multi_class and len(shap_values.shape) == 3:
        
        if plot_type == "compact_dot":
            new_shap_values = shap_values.reshape(shap_values.shape[0], -1)
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)

            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            return summary_plot(
                new_shap_values, new_features, new_feature_names,
                max_display=max_display, plot_type="dot", color=color, axis_color=axis_color,
                title=title, alpha=alpha, show=show, sort=sort,
                color_bar=color_bar, plot_size=plot_size, class_names=class_names,
                color_bar_label="*" + color_bar_label
            )

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        pl.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_plot(
            proj_shap_values, features[:, sort_inds] if features is not None else None,
            feature_names=feature_names[sort_inds],
            sort=False, show=False, color_bar=False,
            plot_size=None,
            max_display=max_display
        )
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2  # because only off diag effects are split in half
            summary_plot(
                proj_shap_values, features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                pl.xlabel(labels['INTERACTION_VALUE'])
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
        return

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=0), axis=0))
        else:
#             feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
#             feature_order = np.argsort(np.array([np.mean(shap_values[shap_values[:,i] > 0, i]) for i in range(shap_values.shape[1])]))
              feature_order = rank_shap(shap_values, features, sorting)
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    
    
    # summary_plot for shape_value bookmark
    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                           c=cvals, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:

                pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)

    elif plot_type == "violin":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds) * 3

                values = features[:, i]
                window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):

                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], np.ones(shap_values[nan_mask].shape[0]) * pos,
                           color="#777777", vmin=vmin, vmax=vmax, s=9,
                           alpha=alpha, linewidth=0, zorder=1)
                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(shaps[np.invert(nan_mask)], np.ones(shap_values[np.invert(nan_mask)].shape[0]) * pos,
                           cmap=cmap, vmin=vmin, vmax=vmax, s=9,
                           c=cvals, alpha=alpha, linewidth=0, zorder=1)
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        pl.fill_between([xs[i], xs[i + 1]], [pos + ds[i], pos + ds[i + 1]],
                                        [pos - ds[i], pos - ds[i + 1]], color=colors.red_blue_no_bounds(smooth_values[i]),
                                        zorder=2)

        else:
            parts = pl.violinplot(shap_values[:, feature_order], range(len(feature_order)), points=200, vert=False,
                                  widths=0.7,
                                  showmeans=False, showextrema=False, showmedians=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)

    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype(
            'int')  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
        x_points = np.linspace(shap_min, shap_max, num_x_points)

        # loop through each feature and plot:
        for pos, ind in enumerate(feature_order):
            # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
            # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
            feature = features[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            nbins = thesebins.shape[0] - 1
            # order the feature data so we can apply percentiling
            order = np.argsort(feature)
            # x axis is located at y0 = pos, with pos being there for offset
            y0 = np.ones(num_x_points) * pos
            # calculate kdes:
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                # get shap values in this bin:
                shaps = shap_values[order[thesebins[i]:thesebins[i + 1]], ind]
                # if there's only one element, then we can't
                if shaps.shape[0] == 1:
                    warnings.warn(
                        "not enough data in bin #%d for feature %s, so it'll be ignored. Try increasing the number of records to plot."
                        % (i, feature_names[ind]))
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
                # do nothing, but when we've gone with the unqique option, this will matter - e.g. if 99% are male and 1%
                # female, we want the 1% to appear a lot smaller.
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = features.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
            # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
            # whitespace
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = pl.get_cmap(color)(i / (
                        nbins - 1)) if color in pl.cm.datad else color  # if color is a cmap, use it, otherwise use a color
                pl.fill_between(x_points, pos - y, pos + y, facecolor=c)
        pl.xlim(shap_min, shap_max)

    elif not multi_class and plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        pl.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    elif multi_class and plot_type == "bar":
        if class_names is None:
            class_names = ["Class "+str(i) for i in range(len(shap_values))]
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))

        if class_inds is None:
            class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        elif class_inds == "original":
            class_inds = range(len(shap_values))
        for i, ind in enumerate(class_inds):
            global_shap_values = np.abs(shap_values[ind]).mean(0)
            pl.barh(
                y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',
                color=color(i), label=class_names[ind]
            )
            left_pos += global_shap_values[feature_inds]
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        pl.legend(frameon=False, fontsize=12)

    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=1000)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    else:
        pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
    if file_name is not None:
#         pl.tight_layout()
        pl.savefig(file_name, bbox_inches='tight')

        
def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text
    

    
    
    
    
    

def dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
                    interaction_index="auto",
                    color="#1E88E5", axis_color="#333333", cmap=None, sort=False,
                    dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True):
    """ Create a SHAP dependence plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of the classical parital dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.


    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).

    feature_names : list
        Names of the features (length # features).

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).

    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).

    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.

    """

    if cmap is None:
        cmap = colors.red_blue

    

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values, feature_names)

    # guess what other feature as the stongest interaction with the plotted feature
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, shap_values, features)[0]
    interaction_index = convert_name(interaction_index, shap_values, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_plot(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"
    

    # get both the raw and display feature values
    if sort:
#         print(interaction_index)
#         print(features.shape)
    # plot first those points whose binary feature is 0,  so that those with 1 are not covered:
        oinds = np.argsort(features[:,interaction_index])
    else:
        oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
        np.random.shuffle(oinds)    
    
    xv = features[oinds, ind].astype(np.float64)
    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        cv = features[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(np.float), 5)
        chigh = np.nanpercentile(cv.astype(np.float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(np.float))
            chigh = np.nanmax(cv.astype(np.float))
            bounds = np.linspace(clow, chigh, int(chigh - clow + 2))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.ranf(size = len(xv))*jitter_amount) - (jitter_amount/2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[oinds, interaction_index].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(p, ticks=tick_positions, ax=ax)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p, ax=ax)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=11))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()
    
    
    
    
    
    
    
    
    
    

# def dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
#                     interaction_index="auto",
#                     color="#1E88E5", axis_color="#333333", cmap=None,
#                     dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True):
#     """ Create a SHAP dependence plot, colored by an interaction feature.
#     Plots the value of the feature on the x-axis and the SHAP value of the same feature
#     on the y-axis. This shows how the model depends on the given feature, and is like a
#     richer extenstion of the classical parital dependence plots. Vertical dispersion of the
#     data points represents interaction effects. Grey ticks along the y-axis are data
#     points where the feature's value was NaN.
#     Parameters
#     ----------
#     ind : int or string
#         If this is an int it is the index of the feature to plot. If this is a string it is
#         either the name of the feature to plot, or it can have the form "rank(int)" to specify
#         the feature with that rank (ordered by mean absolute SHAP value over all the samples).
#     shap_values : numpy.array
#         Matrix of SHAP values (# samples x # features).
#     features : numpy.array or pandas.DataFrame
#         Matrix of feature values (# samples x # features).
#     feature_names : list
#         Names of the features (length # features).
#     display_features : numpy.array or pandas.DataFrame
#         Matrix of feature values for visual display (such as strings instead of coded values).
#     interaction_index : "auto", None, int, or string
#         The index of the feature used to color the plot. The name of a feature can also be passed
#         as a string. If "auto" then shap.common.approximate_interactions is used to pick what
#         seems to be the strongest interaction (note that to find to true stongest interaction you
#         need to compute the SHAP interaction values).
#     x_jitter : float (0 - 1)
#         Adds random jitter to feature values. May increase plot readability when feature
#         is discrete.
#     alpha : float
#         The transparency of the data points (between 0 and 1). This can be useful to the
#         show density of the data points when using a large dataset.
#     xmin : float or string
#         Represents the lower bound of the plot's x-axis. It can be a string of the format
#         "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
#     xmax : float or string
#         Represents the upper bound of the plot's x-axis. It can be a string of the format
#         "percentile(float)" to denote that percentile of the feature's value used on the x-axis.
#     ax : matplotlib Axes object
#          Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
#          In this case we do not create a Figure, otherwise we do.
#     """

#     if cmap is None:
#         cmap = colors.red_blue

#     if type(shap_values) is list:
#         raise TypeError("The passed shap_values are a list not an array! If you have a list of explanations try " \
#                         "passing shap_values[0] instead to explain the first output class of a multi-output model.")

#     # convert from DataFrames if we got any
#     if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
#         if feature_names is None:
#             feature_names = features.columns
#         features = features.values
#     if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
#         if feature_names is None:
#             feature_names = display_features.columns
#         display_features = display_features.values
#     elif display_features is None:
#         display_features = features

#     if feature_names is None:
#         feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

#     # allow vectors to be passed
#     if len(shap_values.shape) == 1:
#         shap_values = np.reshape(shap_values, len(shap_values), 1)
#     if len(features.shape) == 1:
#         features = np.reshape(features, len(features), 1)

#     ind = convert_name(ind, shap_values, feature_names)

#     # guess what other feature as the stongest interaction with the plotted feature
#     if not hasattr(ind, "__len__"):
#         if interaction_index == "auto":
#             interaction_index = approximate_interactions(ind, shap_values, features)[0]
#         interaction_index = convert_name(interaction_index, shap_values, feature_names)
#     categorical_interaction = False

#     # create a matplotlib figure, if `ax` hasn't been specified.
#     if not ax:
#         figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
#         fig = pl.figure(figsize=figsize)
#         ax = fig.gca()
#     else:
#         fig = ax.get_figure()

#     # plotting SHAP interaction values
#     if len(shap_values.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
#         ind1 = convert_name(ind[0], shap_values, feature_names)
#         ind2 = convert_name(ind[1], shap_values, feature_names)
#         if ind1 == ind2:
#             proj_shap_values = shap_values[:, ind2, :]
#         else:
#             proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

#         # there is no interaction coloring for the main effect
#         if ind1 == ind2:
#             fig.set_size_inches(6, 5, forward=True)

#         # TODO: remove recursion; generally the functions should be shorter for more maintainable code
#         dependence_plot(
#             ind1, proj_shap_values, features, feature_names=feature_names,
#             interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
#             xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
#         )
#         if ind1 == ind2:
#             ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
#         else:
#             ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

#         if show:
#             pl.show()
#         return

#     assert shap_values.shape[0] == features.shape[0], \
#         "'shap_values' and 'features' values must have the same number of rows!"
#     assert shap_values.shape[1] == features.shape[1], \
#         "'shap_values' must have the same number of columns as 'features'!"

#     # get both the raw and display feature values
#     if sort:
#     # plot first those points whose binary feature is 0,  so that those with 1 are not covered:
#         oinds = np.sort(X_train[interaction_index].values)
#     else:
#         oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
#         np.random.shuffle(oinds)

#     xv = encode_array_if_needed(features[oinds, ind])

#     xd = display_features[oinds, ind]
#     s = shap_values[oinds, ind]
#     if type(xd[0]) == str:
#         name_map = {}
#         for i in range(len(xv)):
#             name_map[xd[i]] = xv[i]
#         xnames = list(name_map.keys())

#     # allow a single feature name to be passed alone
#     if type(feature_names) == str:
#         feature_names = [feature_names]
#     name = feature_names[ind]

#     # get both the raw and display color values
#     color_norm = None
#     if interaction_index is not None:
#         interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
#         cv = interaction_feature_values
#         cd = display_features[:, interaction_index]
#         clow = np.nanpercentile(cv.astype(np.float), 5)
#         chigh = np.nanpercentile(cv.astype(np.float), 95)
#         if clow == chigh:
#             clow = np.nanmin(cv.astype(np.float))
#             chigh = np.nanmax(cv.astype(np.float))
#         if type(cd[0]) == str:
#             cname_map = {}
#             for i in range(len(cv)):
#                 cname_map[cd[i]] = cv[i]
#             cnames = list(cname_map.keys())
#             categorical_interaction = True
#         elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
#             categorical_interaction = True

#         # discritize colors for categorical features
#         if categorical_interaction and clow != chigh:
#             clow = np.nanmin(cv.astype(np.float))
#             chigh = np.nanmax(cv.astype(np.float))
#             bounds = np.linspace(clow, chigh, int(chigh - clow + 2))
#             color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

#     # optionally add jitter to feature values
#     if x_jitter > 0:
#         if x_jitter > 1: x_jitter = 1
#         xvals = xv.copy()
#         if isinstance(xvals[0], float):
#             xvals = xvals.astype(np.float)
#             xvals = xvals[~np.isnan(xvals)]
#         xvals = np.unique(xvals) # returns a sorted array
#         if len(xvals) >= 2:
#             smallest_diff = np.min(np.diff(xvals))
#             jitter_amount = x_jitter * smallest_diff
#             xv += (np.random.ranf(size = len(xv))*jitter_amount) - (jitter_amount/2)

#     # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
#     xv_nan = np.isnan(xv)
#     xv_notnan = np.invert(xv_nan)
#     if interaction_index is not None:

#         # plot the nan values in the interaction feature as grey
#         cvals = interaction_feature_values[oinds].astype(np.float64)
#         cvals_imp = cvals.copy()
#         cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
#         cvals[cvals_imp > chigh] = chigh
#         cvals[cvals_imp < clow] = clow
#         p = ax.scatter(
#             xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
#             cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
#             norm=color_norm, rasterized=len(xv) > 500
#         )
#         p.set_array(cvals[xv_notnan])
#     else:
#         p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
#                        alpha=alpha, rasterized=len(xv) > 500)

#     if interaction_index != ind and interaction_index is not None:
#         # draw the color bar
#         if type(cd[0]) == str:
#             tick_positions = [cname_map[n] for n in cnames]
#             if len(tick_positions) == 2:
#                 tick_positions[0] -= 0.25
#                 tick_positions[1] += 0.25
#             cb = pl.colorbar(p, ticks=tick_positions, ax=ax)
#             cb.set_ticklabels(cnames)
#         else:
#             cb = pl.colorbar(p, ax=ax)

#         cb.set_label(feature_names[interaction_index], size=13)
#         cb.ax.tick_params(labelsize=11)
#         if categorical_interaction:
#             cb.ax.tick_params(length=0)
#         cb.set_alpha(1)
#         cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.7) * 20)

#     # handles any setting of xmax and xmin
#     # note that we handle None,float, or "percentile(float)" formats
#     if xmin is not None or xmax is not None:
#         if type(xmin) == str and xmin.startswith("percentile"):
#             xmin = np.nanpercentile(xv, float(xmin[11:-1]))
#         if type(xmax) == str and xmax.startswith("percentile"):
#             xmax = np.nanpercentile(xv, float(xmax[11:-1]))

#         if xmin is None or xmin == np.nanmin(xv):
#             xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
#         if xmax is None or xmax == np.nanmax(xv):
#             xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

#         ax.set_xlim(xmin, xmax)

#     # plot any nan feature values as tick marks along the y-axis
#     xlim = ax.get_xlim()
#     if interaction_index is not None:
#         p = ax.scatter(
#             xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
#             linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
#             vmin=clow, vmax=chigh
#         )
#         p.set_array(cvals[xv_nan])
#     else:
#         ax.scatter(
#             xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
#             linewidth=2, color=color, alpha=alpha
#         )
#     ax.set_xlim(xlim)

#     # make the plot more readable
#     ax.set_xlabel(name, color=axis_color, fontsize=13)
#     ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
#     if title is not None:
#         ax.set_title(title, color=axis_color, fontsize=13)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
#     for spine in ax.spines.values():
#         spine.set_edgecolor(axis_color)
#     if type(xd[0]) == str:
#         ax.set_xticks([name_map[n] for n in xnames])
#         ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=11))
#     if show:
#         with warnings.catch_warnings(): # ignore expected matplotlib warnings
#             warnings.simplefilter("ignore", RuntimeWarning)
#             pl.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    