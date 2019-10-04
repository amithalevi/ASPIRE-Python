import logging
import numpy as np
from joblib import Memory
from aspire.image import Image

from aspire.utils.matlab_compat import rand, randi, randn

logger = logging.getLogger(__name__)


# An Xform is anything that implements forward/adjoint methods that takes in a square Image,
# and spits out a square Image
# The 'resolution' parameter tells us the side length of the Image/Volume that the Xform produces as output
class Xform:
    def __init__(self, resolution=np.inf):
        self.resolution = resolution

    def forward(self, im, indices=None):
        if indices is None:
            indices = np.arange(im.n_images)
        return self._forward(im, indices=indices)

    def _forward(self, im, indices):
        raise NotImplementedError

    def adjoint(self, im, indices=None):
        if indices is None:
            indices = np.arange(im.n_images)
        return self._adjoint(im, indices=indices)

    def _adjoint(self, im, indices):
        raise NotImplementedError


class SymmetricXform(Xform):
    def _adjoint(self, im, indices=None):
        return self._forward(im, indices)


class Multiply(SymmetricXform):
    def __init__(self, factor=None, n=None, resolution=np.inf, seed=0):
        super().__init__(resolution=resolution)
        if factor is not None:
            self.multipliers = factor
        else:
            min_, max_ = 2./3, 3./2
            self.multipliers = min_ + rand(n, seed=seed) * (max_ - min_)

    def _forward(self, im, indices):
        return Image(im.asnumpy() * self.multipliers[indices])


class Shift(Xform):
    def __init__(self, shifts=None, n=None, resolution=np.inf, seed=0):
        """
        Initialize a Shift Transform using either a shifts ndarray or n/resolution
        :param shifts: An ndarray of shape (n, 2)
        :param n: Total no. of images expected to pass through this Transform
        :param resolution: Resolution of images expected to pass through this Transform
        """
        super().__init__(resolution=resolution)
        if shifts is not None:
            self.shifts = shifts
            self.n = shifts.shape[0]
        else:
            assert (n is not None) and (resolution != np.inf),\
                "If shifts are not specified, then n and resolution should be."
            self.n = n
            self.shifts = resolution / 16 * randn(2, n, seed=seed).T

    def _forward(self, im, indices):
        return im.shift(self.shifts[indices])

    def _adjoint(self, im, indices):
        return im.shift(-self.shifts[indices])


class DownSample(Xform):
    def _forward(self, im, indices):
        return im.downsample(self.resolution)


class Filter(Xform):
    def __init__(self, filter):
        super().__init__()
        self.filter = filter

    def _forward(self, im, indices):
        return im.filter(self.filter)


class NoiseAdder(Xform):
    def __init__(self, resolution=np.inf, seed=0):
        # TODO: Add ability to specify noise filter
        # TODO: This could be different objects based on indices
        super().__init__(resolution=resolution)
        self.seed = seed

    def _forward(self, im, indices):
        im = im.copy()
        for i, idx in enumerate(indices):
            random_seed = self.seed + 191 * (idx + 1)
            im_s = randn(2 * self.resolution, 2 * self.resolution, seed=random_seed)
            im[:, :, i] += im_s[:self.resolution, :self.resolution]

        return im


class IndexedXform(Xform):
    def __init__(self, unique_xforms, indices):
        assert np.min(indices) >= 0
        assert np.max(indices) < len(unique_xforms)

        resolution = min(*[xform.resolution for xform in unique_xforms])
        super().__init__(resolution=resolution)

        self.n_indices = len(indices)
        self.indices = indices
        self.n_xforms = len(unique_xforms)
        self.unique_xforms = unique_xforms

        # A list of references to individual Xform objects, with possibly multiple references pointing to
        # the same Xform object.
        self.xforms = [unique_xforms[i] for i in indices]

    def _forward(self, im, indices):
        # Ensure that we will be able to apply all transformers to the image
        assert self.n_indices >= im.n_images, f'Can process Image object of max depth {self.n_indices}. Got {im.n_images}.'

        im_data = im.asnumpy().copy()

        # For each individual transformation
        for i in range(self.n_xforms):
            # Get the indices corresponding to that transformation
            idx = np.where(self.indices == i)[0]
            # For the incoming Image object, find out which transformation indices are applicable
            idx = np.intersect1d(idx, indices)
            # For the transformation indices we found, find the indices in the Image object that we'll use
            im_data_indices = np.where(np.isin(indices, idx))[0]
            # Apply the transformation to the selected indices in the Image object
            if len(im_data_indices) > 0:
                im_data[:, :, im_data_indices] = self.unique_xforms[i].forward(Image(im_data[:, :, im_data_indices])).asnumpy()
        return Image(im_data)

    def _adjoint(self, im, indices):
        # Ensure that we will be able to apply all transformers to the image
        assert self.n_indices >= im.n_images, f'Can process Image object of max depth {self.n_indices}. Got {im.n_images}.'

        im_data = im.asnumpy().copy()

        # For each individual transformation
        for i in range(self.n_xforms):
            # Get the indices corresponding to that transformation
            idx = np.where(self.indices == i)[0]
            # For the incoming Image object, find out which transformation indices are applicable
            idx = np.intersect1d(idx, indices)
            # For the transformation indices we found, find the indices in the Image object that we'll use
            im_data_indices = np.where(np.isin(indices, idx))[0]
            # Apply the transformation to the selected indices in the Image object
            if len(im_data_indices) > 0:
                im_data[:, :, im_data_indices] = self.unique_xforms[i].adjoint(Image(im_data[:, :, im_data_indices])).asnumpy()
        return Image(im_data)


class RandomIndexedXform(IndexedXform):
    def __init__(self, unique_xforms, n=None, seed=0):
        if n is None:
            n = len(unique_xforms)
        indices = randi(len(unique_xforms), n, seed=seed) - 1
        super().__init__(unique_xforms=unique_xforms, indices=indices)


# Global function that is capable of being cached using joblib
def _apply_transform(xform, im, indices, adjoint=False):
    if not adjoint:
        logger.info('  Applying ' + str(xform))
        return xform.forward(im, indices=indices)
    else:
        logger.info('  Applying Adjoint ' + str(xform))
        return xform.adjoint(im, indices=indices)


class Pipeline(Xform):
    def __init__(self, xforms=[], memory=None):
        self.xforms = xforms
        self.memory = memory

        resolution = min(*[xform.resolution for xform in xforms])

        Xform.__init__(self, resolution=resolution)

    def _forward(self, im, indices):

        memory = Memory(location=self.memory, verbose=0)
        _apply_transform_cached = memory.cache(_apply_transform)

        logger.info('Applying transformations in pipeline')
        for xform in self.xforms:
            im = _apply_transform_cached(xform, im, indices, False)
        logger.info('All transformations applied')

        return im

    def _adjoint(self, im, indices):

        memory = Memory(location=self.memory, verbose=0)
        _apply_transform_cached = memory.cache(_apply_transform)

        logger.info('Applying transformations in pipeline')
        for xform in self.xforms[::-1]:
            im = _apply_transform_cached(xform, im, indices, True)
        logger.info('All transformations applied')

        return im
