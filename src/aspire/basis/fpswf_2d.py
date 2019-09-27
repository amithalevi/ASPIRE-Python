import logging
import numpy as np

from scipy.special import jn
from aspire.basis.pswf_2d import PSWFBasis2D


from aspire.basis.basis_func import leggauss_0_1

logger = logging.getLogger(__name__)

class FPSWFBasis2D(PSWFBasis2D):
    """
    Define a derived class using the Prolate Spheroidal Wave Function (PSWF) basis for mapping 2D images.
    The numerical evaluation for 2D PSWFs at arbitrary points in the unit disk is based on the fast method
    described in the papers as below:
        1) Boris Landa and Yoel Shkolnisky, "Steerable principal components for space-frequency localized images",
        SIAM J. Imag. Sci. 10, 508-534 (2017).
        2) Boris Landa and Yoel Shkolnisky, "Approximation scheme for essentially bandlimited and space-concentrated
        functions on a disk", Appl. Comput. Harmon. Anal. 43, 381-403 (2017).
        3) Yoel Shkolnisky, "Prolate spheroidal wave functions on a disc-Integration and approximation of
        two-dimensional bandlimited functions", Appl. Comput. Harmon. Anal. 22, 235-256 (2007).
    """

    def _build(self):

        logger.info('Expanding 2D images in the direct method using PSWF basis functions.')

        # initial the whole set of PSWF basis functions based on the bandlimit and eps error.
        self.bandlimit = self.beta * np.pi * self.resolution
        self.d_vec_all, self.alpha_all, self.lengths = self.init_pswf_func2d(self.bandlimit, eps=np.spacing(1))

        # generate_the 2D grid and corresponding indices inside the disc.
        self.generate_grid()

        # precompute the basis functions in 2D grids
        self.precomp()

        # calculate total number of basis functions
        # self.basis_count = self.k_max[0] + sum(2 * self.k_max[1:])

        # obtain a 2D grid to represent basis functions
        # self.basis_coords = unique_coords_nd(self.N, self.d)

        # generate 1D indices for basis functions
        # self._indices = self.indices()

    def precomp(self):
        # find max alpha for each N
        max_ns = []
        a = np.square(float(self.beta * self.resolution) / 2)
        m = 0
        alpha_all = []
        while True:
            alpha = self.alpha_all[m]

            lambda_var = a * np.square(np.absolute(alpha))
            gamma = np.sqrt(np.absolute(lambda_var / (1 - lambda_var)))

            n_end = np.where(gamma <= self.truncation)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                max_ns.extend([n_end])
                alpha_all.extend(alpha[:n_end])
                m += 1

        a, b, c, d, e, f = self._generate_pswf_quad(4 * self.resolution, 2 * self.bandlimit, 1e-16, 1e-16, 1e-16)

        self.pswf_radial_quad = self.evaluate_pswf2d_all(d, np.zeros(len(d)), max_ns)
        self.quad_rule_pts_x = a
        self.quad_rule_pts_y = b
        self.quad_rule_wts = c
        self.radial_quad_pts = d
        self.quad_rule_radial_wts = e
        self.num_angular_pts = f
        self.angular_frequency = np.repeat(np.arange(len(max_ns)), max_ns).astype('float')
        self.radian_frequency = np.concatenate([range(1, l + 1) for l in max_ns]).astype('float')
        self.alpha_nn = np.array(alpha_all)

        # pre computing variables for forward
        us_fft_pts = np.column_stack((self.quad_rule_pts_x, self.quad_rule_pts_y))
        us_fft_pts = self.bandlimit / (self.resolution * np.pi * 2) * us_fft_pts  # for pynfft
        blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max =\
            self._pswf_integration_sub_routine()

        self.us_fft_pts = us_fft_pts
        self.blk_r = blk_r
        self.num_angular_pts = num_angular_pts
        self.r_quad_indices = r_quad_indices
        self.numel_for_n = numel_for_n
        self.indices_for_n = indices_for_n
        self.n_max = n_max
        self.size_x = len(self.points_inside_circle)

    def evaluate_t(self, images):
        # start and finish are for the threads option in the future
        images_shape = images.shape
        start = 0

        # if we got several images
        if len(images_shape) == 3:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], images_shape[2]), order='F')
            finish = images_shape[2]

        # else we got only one image
        else:
            flattened_images = images.reshape((images_shape[0] * images_shape[1], 1), order='F')
            finish = 1

        flattened_images = flattened_images[self.points_inside_circle_vec, :]

        nfft_res = self._compute_nfft_potts(flattened_images, start, finish)
        coefficients = self._pswf_integration(nfft_res)
        return coefficients

    def _generate_pswf_quad(self, n, bandlimit, phi_approximate_error, lambda_max, epsilon):
        radial_quad_points, radial_quad_weights = self._generate_pswf_radial_quad(n, bandlimit, phi_approximate_error,
                                                                                  lambda_max)

        num_angular_points = np.ceil(np.e * radial_quad_points * bandlimit / 2 - np.log(epsilon)).astype('int') + 1

        for i in range(len(radial_quad_points)):
            ang_error_vec = np.absolute(jn(range(1, 2 * num_angular_points[i] + 1),
                                           bandlimit * radial_quad_points[i]))

            num_angular_points[i] = self._sum_minus_cumsum_smaller_eps(ang_error_vec, epsilon)
            if num_angular_points[i] % 2 == 1:
                num_angular_points[i] += 1

        temp = 2 * np.pi / num_angular_points

        t = 2

        quad_rule_radial_weights = temp * radial_quad_points * radial_quad_weights
        quad_rule_weights = np.repeat(quad_rule_radial_weights, repeats=num_angular_points)
        quad_rule_pts_r = np.repeat(radial_quad_points, repeats=(num_angular_points / t).astype('int'))
        quad_rule_pts_theta = np.concatenate([temp[i] * np.arange(num_angular_points[i] / t)
                                              for i in range(len(radial_quad_points))])

        pts_x = quad_rule_pts_r * np.cos(quad_rule_pts_theta)
        pts_y = quad_rule_pts_r * np.sin(quad_rule_pts_theta)

        return pts_x, pts_y, quad_rule_weights, radial_quad_points, quad_rule_radial_weights, num_angular_points

    def _generate_pswf_radial_quad(self, n, bandlimit, phi_approximate_error, lambda_max):
        x, w = leggauss_0_1(20 * n)

        big_n = 0

        x_as_mat = x.reshape((len(x), 1))

        alpha_n, d_vec, approx_length = self.pswf_func2d(big_n, n, bandlimit, phi_approximate_error, x, w)

        cut_indices = np.where(bandlimit / 2 / np.pi * np.absolute(alpha_n) < lambda_max)[0]

        if len(cut_indices) == 0:
            k = len(alpha_n)
        else:
            k = cut_indices[0]

        if k % 2 == 0:
            k = k + 1

        range_array = np.arange(approx_length).reshape((1, approx_length))

        idx_for_quad_nodes = int((k + 1) / 2)
        num_quad_pts = idx_for_quad_nodes - 1

        phi_zeros = find_initial_nodes(x, n, bandlimit / 2, phi_approximate_error, idx_for_quad_nodes)

        def phi_for_quad_weights(t):
            return np.dot(t_x_mat2(t, big_n, range_array, approx_length), d_vec[:, :k - 1])

        b = np.dot(w * np.sqrt(x), phi_for_quad_weights(x_as_mat))

        a = phi_for_quad_weights(phi_zeros.reshape((len(phi_zeros), 1))).transpose() * np.sqrt(phi_zeros)
        init_quad_weights = lstsq(a, b)
        init_quad_weights = init_quad_weights[0]
        tolerance = 1e-16

        def obj_func(quad_rule):
            q = quad_rule.reshape((len(quad_rule), 1))
            temp = np.dot((phi_for_quad_weights(q[:num_quad_pts]) * np.sqrt(q[:num_quad_pts])).transpose(),
                      q[num_quad_pts:])
            temp = temp.reshape(temp.shape[0])
            return temp - b

        arr_to_send = np.concatenate((phi_zeros, init_quad_weights))
        quad_rule_final = least_squares(obj_func, arr_to_send, xtol=tolerance, ftol=tolerance, max_nfev=1000)
        quad_rule_final = quad_rule_final.x
        quad_rule_pts = quad_rule_final[:num_quad_pts]
        quad_rule_weights = quad_rule_final[num_quad_pts:]
        return quad_rule_pts, quad_rule_weights


    def find_initial_nodes(x, n, bandlimit, phi_approximate_error, idx_for_quad_nodes):
        big_n = 0

        d_vec, approx_length, range_array = pswf_2d_minor_computations(big_n, n, bandlimit, phi_approximate_error)

        def phi_for_quad_nodes(t):
            return np.dot(t_x_mat(t, big_n, range_array, approx_length), d_vec[:, idx_for_quad_nodes - 1])

        fun_vec = phi_for_quad_nodes(x)
        sign_flipping_vec = np.where(np.sign(fun_vec[:-1]) != np.sign(fun_vec[1:]))[0]
        phi_zeros = np.zeros(idx_for_quad_nodes - 1)

        tmp = phi_for_quad_nodes(x)
        for i, j in enumerate(sign_flipping_vec[:idx_for_quad_nodes - 1]):
            new_zero = x[j] - tmp[j] * \
                       (x[j + 1] - x[j]) / (tmp[j + 1] - tmp[j])
            phi_zeros[i] = new_zero

        phi_zeros = np.array(phi_zeros)
        return phi_zeros


    def _sum_minus_cumsum_smaller_eps(self, x, eps):
        y = np.cumsum(np.flipud(x))
        return len(y) - np.where(y > eps)[0][0] + 1

    def __pswf_integration_sub_routine(self):

        t = 2

        num_angular_pts = (self.num_angular_pts / t).astype('int')

        r_quad_indices = [0]
        r_quad_indices.extend(num_angular_pts)
        r_quad_indices = np.cumsum(r_quad_indices, dtype='int')

        n_max = int(max(self.angular_frequency) + 1)

        numel_for_n = np.zeros(n_max, dtype='int')
        for i in range(n_max):
            numel_for_n[i] = np.count_nonzero(self.angular_frequency == i)

        indices_for_n = [0]
        indices_for_n.extend(numel_for_n)
        indices_for_n = np.cumsum(indices_for_n, dtype='int')

        blk_r = [0] * n_max
        temp_const = self.bandlimit / (2 * np.pi * self.resolution)
        for i in range(n_max):
            blk_r[i] = temp_const * self.pswf_radial_quad[:, indices_for_n[i] + np.arange(numel_for_n[i])].T

        return blk_r, num_angular_pts, r_quad_indices, numel_for_n, indices_for_n, n_max


    # use np.outer instead of x as mat
    def t_x_mat2(x, n, j, approx_length): return np.power(x, n + 0.5).dot(np.sqrt(2 * (2 * j + n + 1))) * \
                                                p_n(approx_length - 1, n, 0, 1 - 2 * np.square(x))

    def _compute_nfft_potts(self, images, start, finish):
        x = self.us_fft_pts
        n = self.size_x
        points_inside_circle = self.points_inside_circle
        num_images = finish - start

        # pynufft
        # m = x.shape[0]
        # nufft_obj = NUFFT_cpu()
        # nufft_obj.plan(x, (n, n), (2*n, 2*n), (10, 10))
        # shift = np.exp(x * fast_model.resolution * 1j)
        # shift = np.sum(shift, axis=1)

        # gal nufft
        # m = x.shape[1]
        # nufft_obj = py_nufft.factory('nufft')

        # pynfft
        m = x.shape[0]
        plan = NFFT(N=[n, n], M=m)
        plan.x = x
        plan.precompute()

        images_nufft = np.zeros((m, num_images), dtype='complex128')
        current_image = np.zeros((n, n))
        for i in range(start, finish):
            current_image[points_inside_circle] = images[:, i]
            plan.f_hat = current_image
            images_nufft[:, i - start] = plan.trafo()

        return images_nufft

    def _pswf_integration(self, images_nufft):
        num_images = images_nufft.shape[1]
        n_max_float = float(self.n_max) / 2
        r_n_eval_mat = np.zeros((len(self.radial_quad_pts), self.n_max, num_images), dtype='complex128')

        for i in range(len(self.radial_quad_pts)):
            curr_r_mat = images_nufft[self.r_quad_indices[i]: self.r_quad_indices[i] + self.num_angular_pts[i], :]
            curr_r_mat = np.concatenate((curr_r_mat, np.conj(curr_r_mat)))
            fft_plan = pyfftw.builders.fft(curr_r_mat, axis=0, overwrite_input=True, auto_contiguous=True,
                                           auto_align_input=False, avoid_copy=True, planner_effort='FFTW_ESTIMATE')
            angular_eval = fft_plan() * self.quad_rule_radial_wts[i]

            r_n_eval_mat[i, :, :] = np.tile(angular_eval, (int(max(1, np.ceil(n_max_float / self.num_angular_pts[i]))),
                                                           1))[:self.n_max, :]

        r_n_eval_mat = r_n_eval_mat.reshape((len(self.radial_quad_pts) * self.n_max, num_images), order='F')
        coeff_vec_quad = np.zeros((len(self.angular_frequency), num_images), dtype='complex128')
        m = len(self.pswf_radial_quad)
        for i in range(self.n_max):
            coeff_vec_quad[self.indices_for_n[i] + np.arange(self.numel_for_n[i]), :] =\
                np.dot(self.blk_r[i], r_n_eval_mat[i * m:(i + 1)*m, :])

        return coeff_vec_quad



