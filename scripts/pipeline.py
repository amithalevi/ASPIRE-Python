import logging
import numpy as np
import pickle

from aspire.source.relion import RelionSource
from aspire.basis.fb_3d import FBBasis3D
from aspire.estimation.mean import MeanEstimator
from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.estimation.covar import CovarianceEstimator

logger = logging.getLogger('aspire')


if __name__ == '__main__':

    total_images = 1024
    batch_size = 1024
    resolution = 8
    num_volumes = 2
    num_eigs = 16

    # source = Simulation(
    #     n=total_images,
    #     C=num_volumes,
    #     L=resolution,
    #     seed=0
    # )

    source = RelionSource(
        'E:\\yan_wu\\All_class001_r1_ct1_data.star',
        pixel_size=1.388,
        max_rows=1000
    )

    source.downsample(resolution)

    noise_estimator = WhiteNoiseEstimator(source, batchSize=batch_size)
    # Estimate the noise variance. This is needed for the covariance estimation step below.
    noise_variance = noise_estimator.estimate()
    logger.info(f'Noise Variance = {noise_variance}')

    with open('noise_filter.pkl', 'wb') as f:
        pickle.dump(noise_estimator.filter, f)
    #
    # noise_variance = 0.00795280732423889
    #
    # source.whiten(noise_estimator.filter)
    #
    # basis = FBBasis3D((resolution, resolution, resolution))
    # mean_estimator = MeanEstimator(source, basis, batch_size=batch_size)
    # mean_est = mean_estimator.estimate()
    #
    # # Passing in a mean_kernel argument to the following constructor speeds up some calculations
    # covar_estimator = CovarianceEstimator(source, basis, mean_kernel=mean_estimator.kernel)
    # covar_est = covar_estimator.estimate(mean_est, noise_variance)
    # np.save('covar_est.npy', covar_est)
    #
    # # noise_filter = Filter(ScalarFilter(value=1, power=0.5))
    #
    # # xforms = [
    # #     RandomIndexedXform(
    # #         [Filter(RadialCTFFilter(defocus=d)) for d in np.linspace(1.5e4, 2.5e4, 7)],
    # #         n=total_images, seed=0),
    # #     Shift(n=total_images, resolution=resolution),
    # #     Multiply(n=total_images, resolution=resolution),
    # #     NoiseAdder(resolution=resolution),
    # #     noise_filter
    # # ]
    #
    # # xforms = [
    # #     DownSample(resolution=8)
    # # ]
    # #
    # # pipeline = Pipeline(source, xforms, memory='my_cache_dir')
    #
    # # print('Running Pipeline for noise estimation')
    # # noise_estimator = WhiteNoiseEstimator(pipeline, batchSize=batch_size)
    # # noise_variance = noise_estimator.estimate()
    # # print(noise_variance)
    #
    # # ----------------------------------------------------------
    # # Needed by MeanEstimator/CovarianceEstimator
    # # TODO: Which part of the pipeline should be setting these?
    # # ----------------------------------------------------------
    # # pipeline.filters = [xform.filter for xform in xforms[0].xforms]
    # # pipeline.offsets = xforms[1].shifts
    # # pipeline.amplitudes = xforms[2].multipliers
    # # pipeline.angles = source.angles
    # # ----------------------------------------------------------
    #
    # # basis = FBBasis3D((pipeline.resolution, pipeline.resolution, pipeline.resolution))
    # # mean_estimator = MeanEstimator(pipeline, basis, batch_size=batch_size)
    # # print('Running Pipeline for mean volume estimation')
    # # mean_est = mean_estimator.estimate()
    #
    # # Passing in a mean_kernel argument to the following constructor speeds up some calculations
    # # covar_estimator = CovarianceEstimator(pipeline, basis, mean_kernel=mean_estimator.kernel)
    # # print('Running Pipeline for volume variance')
    # # covar_est = covar_estimator.estimate(mean_est, noise_variance)
    # #
    # # pipeline.save('out/mystar.star', batch_size=120, overwrite=True)
