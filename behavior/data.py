import os
import glob
import numpy as np
import copy


class Markers(object):

    def __init__(self, expt_id, verbose=True, algo='dgp'):
        """
        Args:
            expt_id:
            verbose:
            algo (str): 'dlc' | 'dgp'
        """
        from flygenvectors.utils import get_dirs

        self.expt_id = expt_id
        self.base_data_dir = get_dirs()['data']
        self.algo = algo

        self.markers = {'x': [], 'y': [], 'l': []}
        self.markers_dict = {}

        self.preproc = None
        self.means = {'x': [], 'y': []}
        self.stds = {'x': [], 'y': []}
        self.mins = {'x': [], 'y': []}
        self.maxs = {'x': [], 'y': []}

        self.dtypes = []
        self.dtype_lens = []
        self.skip_idxs = None
        self.idxs_valid = []  # "good" indices
        self.idxs_dict = []  # indices separated by train/test/val

        self.verbose = verbose

    def load_from_csv(self, filename=None):
        from numpy import genfromtxt
        if filename is None:
            if self.algo == 'dlc':
                filename = glob.glob(
                    os.path.join(self.base_data_dir, self.expt_id, '*DeepCut*.csv'))[0]
            elif self.algo == 'dgp':
                filename = os.path.join(
                    self.base_data_dir, 'behavior', 'labels',
                    'resnet-50_ws=%1.1e_wt=%1.1e' % (0, 0),
                    self.expt_id + '_labeled.csv')
            else:
                raise NotImplementedError
        if self.verbose:
            print('loading markers from %s...' % filename, end='')
        dlc = genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
        dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.
        self.markers['x'] = dlc[:, 0::3]
        self.markers['y'] = dlc[:, 1::3]
        self.markers['l'] = dlc[:, 2::3]
        if self.verbose:
            print('done')
            print('total time points: %i' % dlc.shape[0])

    def load_from_h5(self, filename=None):
        """Load from h5 output by DGP."""
        import h5py

        if filename is None:
            filename = os.path.join(
                self.base_data_dir, 'behavior', 'labels',
                'resnet-50_ws=%1.1e_wt=%1.1e' % (0, 0),
                self.expt_id + '_labeled.h5')

        if self.verbose:
            print('loading markers from %s...' % filename, end='')

        with h5py.File(filename, 'r') as f:
            t = f['df_with_missing']['table'][()]
        l = np.concatenate([t[i][1][None, :] for i in range(len(t))])

        self.markers['x'] = l[:, 0::3]
        self.markers['y'] = l[:, 1::3]
        self.markers['l'] = l[:, 2::3]

        if self.verbose:
            print('done')
            print('total time points: %i' % l.shape[0])

    def preprocess(self, preproc_dict):
        self.preproc = copy.deepcopy(preproc_dict)
        for func_str, kwargs in preproc_dict.items():
            if func_str == 'standardize':
                self.standardize(**kwargs)
            elif func_str == 'unitize':
                self.unitize(**kwargs)
            elif func_str == 'filter':
                self.filter(**kwargs)
            else:
                raise ValueError('"%s" is not a valid preprocessing function' % func_str)

    def standardize(self, by_marker=False):
        """subtract off mean and divide by variance across all markers"""
        if self.verbose:
            print('standardizing markers...', end='')

        for c in ['x', 'y']:
            self.means[c] = np.mean(self.markers[c], axis=0)

        if by_marker:
            for c in ['x', 'y']:
                self.stds[c] = np.std(self.markers[c], axis=0)
        else:
            self.stds['x'] = self.stds['y'] = np.std(
                np.concatenate([self.markers['x'], self.markers['y']], axis=0))

        for c in ['x', 'y']:
            self.markers[c] = (self.markers[c] - self.means[c]) / self.stds[c]
        if self.verbose:
            print('done')

    def unitize(self, **kwargs):
        """place each marker (mostly) in [0, 1]"""
        if self.verbose:
            print('unitizing markers...', end='')
        for c in ['x', 'y']:
            self.mins[c] = np.quantile(self.markers[c], 0.05, axis=0)
            self.maxs[c] = np.quantile(self.markers[c], 0.95, axis=0)
            self.markers[c] = (self.markers[c] - self.mins[c]) / (self.maxs[c] - self.mins[c])
        if self.verbose:
            print('done')

    def filter(self, type='median', **kwargs):
        if self.verbose:
            print('applying %s filter to markers...' % type, end='')
        if type == 'median':
            from scipy.signal import medfilt
            kernel_size = 5 if 'kernel_size' not in kwargs else kwargs['kernel_size']
            for c in ['x', 'y']:
                for i in range(self.markers[c].shape[1]):
                    self.markers[c][:, i] = medfilt(
                        self.markers[c][:, i], kernel_size=kernel_size)
        elif type == 'savgol':
            from scipy.signal import savgol_filter
            window_length = 5 if 'window_size' not in kwargs else kwargs['window_size']
            polyorder = 2 if 'order' not in kwargs else kwargs['order']
            for c in ['x', 'y']:
                for i in range(self.markers[c].shape[1]):
                    self.markers[c][:, i] = savgol_filter(
                        self.markers[c][:, i], window_length=window_length, polyorder=polyorder)
        else:
            raise NotImplementedError
        if self.verbose:
            print('done')

    def extract_runs_by_length(self, max_length, return_vals=False, verbose=None):
        """
        Find contiguous chunks of data

        Args:
            max_length (int): maximum length of high likelihood runs; once a
                run surpasses this threshold a new run is started
            return_vals (bool): return list of indices if `True`, otherwise
                store in object as `indxs_valid`
            verbose (bool or NoneType)

        Returns:
            list
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print('extracting runs of length %i...' % max_length, end='')
        n_t = self.markers['x'].shape[0]
        begs = np.arange(0, n_t, max_length)
        ends = np.concatenate([np.arange(max_length, n_t, max_length), [n_t]])
        idxs = [np.arange(begs[i], ends[i]) for i in range(len(begs))]
        if verbose:
            print('done')
            print('extracted %i runs for a total of %i time points' % (
                len(idxs), np.sum([len(i) for i in idxs])))
        if return_vals:
            return idxs
        else:
            self.idxs_valid = idxs

    def split_markers(self, dtypes, dtype_lens):
        if self.verbose:
            print('splitting markers into {}...'.format(dtypes), end='')
        # get train/text/val indices
        self.idxs_dict = split_runs(self.idxs_valid, dtypes, dtype_lens)
        # split markers into train/test/val using index split above
        marker_array = self.get_marker_array()
        self.markers_dict = {dtype: [] for dtype in self.idxs_dict.keys()}
        for dtype, didxs in self.idxs_dict.items():
            for didx in didxs:
                self.markers_dict[dtype].append(marker_array[didx, :])
        if self.verbose:
            print('done')
            for dtype in dtypes:
                print('\t%s: %i time points in %i trials' % (
                    dtype, np.sum([len(i) for i in self.markers_dict[dtype]]),
                    len(self.markers_dict[dtype])))

    def get_marker_array(self):
        """concatenate x/y markers into a single array"""
        return np.concatenate([self.markers['x'], self.markers['y']], axis=1)


def split_runs(indxs, dtypes, dtype_lens):
    """

    Args:
        indxs (list):
        dtypes (list of strs):
        dtype_lens (list of ints):

    Returns:
        dict
    """

    # first sort, then split according to ratio
    i_sorted = np.argsort([len(i) for i in indxs])

    indxs_split = {dtype: [] for dtype in dtypes}
    dtype_indx = 0
    dtype_curr = dtypes[dtype_indx]
    counter = 0
    for indx in reversed(i_sorted):
        if counter == dtype_lens[dtype_indx]:
            # move to next dtype
            dtype_indx = (dtype_indx + 1) % len(dtypes)
            while dtype_lens[dtype_indx] == 0:
                dtype_indx = (dtype_indx + 1) % len(dtypes)
            dtype_curr = dtypes[dtype_indx]
            counter = 0
        indxs_split[dtype_curr].append(indxs[indx])
        counter += 1

    return indxs_split


def preprocess_and_split_data(
        expt_ids, preprocess_list, max_trial_len=1000, algo='dgp', load_from='pkl',
        dtypes=['train', 'test', 'val'], dtype_lens=[8, 1, 1]):
    """Helper function to initialize marker object."""

    if not isinstance(expt_ids, list):
        expt_ids = [expt_ids]

    marker_obj = [Labels(expt_id, algo=algo) for expt_id in expt_ids]

    for n in range(len(marker_obj)):

        if load_from == 'csv':
            marker_obj[n].load_from_csv()
        elif load_from == 'h5':
            marker_obj[n].load_from_h5()
        else:
            raise NotImplementedError('"%s" is not a valid marker file format')

        marker_obj[n].preprocess(preprocess_list)
        marker_obj[n].extract_runs_by_length(max_length=max_trial_len)
        marker_obj[n].split_markers(dtypes=dtypes, dtype_lens=dtype_lens)

    return marker_obj


def shuffle_data(marker_objs, dtype):
    """Randomly interleave data from different sessions."""
    trial_order = [np.random.permutation(len(l.markers_dict[dtype])) for l in marker_objs]
    trial_counters = [0 for _ in range(len(marker_objs))]
    sess_order = np.random.permutation(np.concatenate(
        [[i] * len(l.markers_dict[dtype]) for i, l in enumerate(marker_objs)]))

    data = [None for _ in sess_order]
    for i, sess in enumerate(sess_order):
        c = trial_counters[sess]
        data[i] = marker_objs[sess].markers_dict[dtype][trial_order[sess][c]]
        trial_counters[sess] += 1

    return data, sess_order, trial_order
