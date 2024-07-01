import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

navy = np.array([0, 48, 87])/255
gold = np.array([179, 163, 105])/255
white = np.array([255, 255, 255])/255
lsf_diverging = mpl.colors.LinearSegmentedColormap.from_list(
    'lsf_diverging', 
    [navy, white, gold]
)


class Template(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=np.uint8).view(cls)

    def show(self, ax=None, plot_kwargs=None):
        ''' launch a matplotlib plot of the template

        Parameters
        ----------
        wait : object, optional
            If wait is specified to be anything other than none, then the
            method will not run ``plt.show()`` so that a user may add plot
            elements before showing
        '''
        cmap = mpl.colors.ListedColormap(
            ['#003057', '#B3A369', '#3A5DAE', '#E04F39', '#A4D233'])
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
        if ax is None:
            ax = plt.gca()

        if plot_kwargs is None:
            plot_kwargs = {}
        ax.pcolormesh(self.T, cmap=cmap, norm=norm, **plot_kwargs)
        ax.set_aspect('equal', 'box')


class LevelSetFunction(np.ndarray):
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def create_template(self, threshold=0):
        return Template(
            (self > threshold).astype(int)
        )

    def show(self, ax=None, contours=True, plot_kwargs=None):
        if ax is None:
            ax = plt.gca()

        if plot_kwargs is None:
            plot_kwargs = {}

        cmap = lsf_diverging
        norm = mpl.colors.CenteredNorm(halfrange=1)


        X, Y = np.meshgrid(*[np.linspace(0, 1, s)
                             for s in self.shape])

        ax.pcolormesh(X, Y, self.T, cmap=cmap, norm=norm, **plot_kwargs)
        ax.set_aspect(1/np.divide(*self.shape), 'box')

        if contours:
            levels = np.arange(-1, 1, 0.25)
            colors = ['#E04F39' if np.isclose(l, 0) else 'k' for l in levels]
            qc = ax.contour(X, Y, self.T,
                            levels=levels,
                            colors=colors,
                            linewidths=1)
            ax.clabel(qc, levels=levels, fontsize=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        

class RadialBasisFunction:
    def __init__(self, center, radius, wrap=False):
        self.center = np.array(center)
        self.radius = radius
        self.wrap = wrap

    def distance(self, *x, ord=2):
        ''' returns the distance for any `x` to the center of the function

        Parameters
        ----------
        x : 2xN np.ndarray
        '''

        old_shape = x[0].shape
        x = np.array(x).reshape(len(x), -1).T

        test_dists = []

        offsets = [(0, 0)]
        if self.wrap:
            xs, ys = x[-1] - x[0]
            for n in [-1, 1]:
                offsets.extend([(n*xs, 0), (0, n*ys), (n*xs, n*ys)])

        for offset in offsets:
            test_dists.append(x - self.center + offset)

        test_dists = np.array(test_dists)
        return np.min(
            np.linalg.norm(test_dists/self.radius, axis=-1, ord=ord), axis=0
        ).reshape(old_shape)

    def __repr__(self):
        type_ = type(self)
        module = type_.__module__
        qualname = type_.__qualname__
        return f'{qualname}(center={self.center}, radius={self.radius}, wrap={self.wrap})'

    def __call__(self, *x, ord=2):
        dist = self.distance(*x, ord=ord)
        return np.exp(-dist ** 2)

    def _plot(self, idx, ax=None, patch_kwargs=None, text_kwargs=None):
        if ax is None:
            ax = plt.gca()
        width, height = self.radius * 0.707
        center = self.center
        patch_kw = dict(facecolor='none',
                        edgecolor='k',
                        linewidth=1)

        if patch_kwargs is not None:
            patch_kw.update(patch_kwargs)

        patch = mpatch.Ellipse(
            xy=center,
            width=2*width,
            height=2*height,
            **patch_kw
        )
        ax.add_patch(patch)

        text_kw = dict(va='center',
                       ha='center',
                       backgroundcolor='w')
        if text_kwargs is not None:
            text_kw.update(text_kwargs)

        txt = ax.text(*center, idx, **text_kw)
        txt.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='none'))
        return patch, txt
    
class LevelSetMethod:
    def __init__(self, bases):
        self.bases = bases

    @property
    def nbases(self):
        return len(self.bases)

    @classmethod
    def uniform_grid(cls, n_bases, wrap=False, spacing_factor=0.707):

        radii = 1 / (np.array(n_bases)-1+wrap)*spacing_factor

        bases = []
        for center in itertools.product(*[np.linspace(0, 1, nb, endpoint=not wrap)
                                         for nb in n_bases]):
            bases.append(
                RadialBasisFunction(
                    center=center,
                    radius=radii,
                    wrap=wrap))
        return cls(bases)
    
    def _grid(self, shape):
        return np.meshgrid(
            *[np.linspace(0, 1, s, endpoint=False)
              for s in shape])

    def create_lsf(self, pars, *x, shape=None, ord=2):
        if shape is not None:
            x = self._grid(shape)
        if not len(x):
            raise ValueError('supply mesh grid or shape')

        pars = np.array(pars).flatten()
        assert len(pars) == len(self.bases)
        lsf = 0
        for par, basis in zip(pars, self.bases):
            lsf += par * basis(*x, ord=ord)
        return LevelSetFunction(lsf.T)

    def create_template(self, pars, *x, shape=None, ord=2, threshold=0):
        lsf = self.create_lsf(pars, *x, shape=shape, ord=ord)
        return lsf.create_template(threshold=threshold)