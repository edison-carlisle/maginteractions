#!/usr/bin/env python
"""Class to find the magnetic interactions of a spin system from mPDF data."""

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Voronoi
from scipy.special import kn

from diffpy.mpdf.magutils import calculate_avg_spin_magnitude


class MagInteractions:
    """
    Stores and computes information about the magnetic interactions
    of a spin system.

    Args:
        magstruc (MagStructure object): provides information about the
            magnetic structure. Must have arrays of atoms and spins.
        temperatures (list): list of temperatures to perform calculations
            at. The temperatures must be in the paramagnetic regime of the
            material.
        spin_squared (float): length of the spin vector squared. E.g. S(S+1)
            for a system with quantum spin number S.
        j_ij (list): list of the magnetic interactions in order of increasing
            neighbor distance. E.g. [J1, J2, J3] for a system with interactions
            up to the 3rd nearest neighbor.
        r_max (float): maximum distance to consider for magnetic interactions.
            Used to determine how many neighbors to include in the interaction
            matrix.
        spin_dim (int): dimensionality of the spin vectors in the spin system.
            E.g. 3 for a Heisenberg system, 2 for an XY system, 1 for an Ising
            system.
        lat_vecs (array): 3x3 array of the lattice vectors. If not provided,
            will be calculated from magstruc.
        basis_vecs (array): Nx3 array of the basis vectors. If not provided,
            will be calculated from magstruc.
        occ_avg (float): average occupancy of the magnetic atoms. If not
            provided, will be calculated from magstruc.

    """

    def __init__(self, magstruc=None, temperatures=None, spin_squared=1.0,
                 j_ij=None, r_max=10.0, spin_dim=3, lat_vecs=None,
                 basis_vecs=None, occ_avg=1.0):
        if temperatures is None:
            temperatures = [1.0]
        if j_ij is None:
            j_ij = [1.0]

        if magstruc is None:
            # "Manual" mode: no MagStructure object provided. Only the
            # reciprocal-lattice geometry and the k-space interaction
            # matrix (make_interaction_matrix, calc_j_q) are usable in
            # this mode. Methods that need real-space neighbor
            # information (r_nn, g_nn, calculate_orf_real_space,
            # calculate_lmop, etc.) require a magstruc object and will
            # raise an AttributeError if called without one.
            self.magstruc = None

            self.lat_vecs = lat_vecs
            if basis_vecs is None:
                self.basis_vecs = None
                self.N = 0
            else:
                self.basis_vecs = basis_vecs
                self.N = len(basis_vecs)

            self.occ_avg = occ_avg

        else:
            self.magstruc = magstruc.copy()
            self.struc_idxs = []
            occ_sum = 0
            for magspecies in self.magstruc.species.values():
                self.struc_idxs.extend(magspecies.strucIdxs)
                occ_sum += magspecies.occ
            self.occ_avg = occ_sum / len(self.magstruc.species)

            self.lat_vecs = self.magstruc.struc.lattice.base
            self.basis_vecs = self.magstruc.struc.xyz_cartn[self.struc_idxs]

            self.N = len(self.basis_vecs)

        if self.lat_vecs is not None:
            self.recip_lats = np.zeros_like(self.lat_vecs)
            self.uc_vol = np.linalg.det(self.lat_vecs)
            self.bz_vol = (2 * np.pi) ** 3 / self.uc_vol
            for i in range(3):
                self.recip_lats[i] = np.cross(self.lat_vecs[(i + 1) % 3],
                                              self.lat_vecs[(i + 2) % 3])
                self.recip_lats[i] *= 2 * np.pi / self.uc_vol
        else:
            self.recip_lats = None
            self.uc_vol = None
            self.bz_vol = None

        self.temperatures = np.asarray(temperatures, dtype=float)
        self.n_temps = len(self.temperatures)
        self.orfs = np.zeros(self.n_temps)
        self.spin_squared = spin_squared
        self.j_ij = list(j_ij)
        self.r_max = r_max
        self.spin_dim = spin_dim

        self.damping_matrices = None

        if self.magstruc is not None:
            r_nn_arg = np.argmin(np.apply_along_axis(
                np.linalg.norm, 1,
                self.magstruc.atoms[1:] - self.magstruc.atoms[0]))
            self.r_nn_vec = (self.magstruc.atoms[r_nn_arg + 1]
                             - self.magstruc.atoms[0])
            self.r_nn = np.linalg.norm(self.r_nn_vec)
            self.g_nn = np.sqrt(self.magstruc.gfactors[r_nn_arg + 1]
                                * self.magstruc.gfactors[0])
            self.g = np.mean(self.magstruc.gfactors)
            self.vector_mag = calculate_avg_spin_magnitude(self.magstruc)
        else:
            self.r_nn_vec = None
            self.r_nn = None
            self.g_nn = None
            self.g = None
            self.vector_mag = None

    def make_interaction_matrix(self):
        """constructs the interaction matrix Jij(R) for the system based on the input
        parameters. The interaction matrix is stored as a 5D array, where the first
        two dimensions correspond to the basis vectors (i, j) and the last three
        dimensions correspond to the lattice vectors (R)."""

        intercell_dists = [np.linalg.norm(self.lat_vecs - self.basis_vecs[i],
                                          axis=1) for i in range(self.N)]
        intercell_dist = np.min(intercell_dists, axis=0)
        max_dist = np.ceil(self.r_max / intercell_dist).astype(int)
        uc_tuple = tuple(2 * max_dist - 1)
        pairwise_distance = np.zeros((self.N, self.N) + uc_tuple)
        self.j_mat = np.zeros_like(pairwise_distance)

        for rx in range(1 - max_dist[0], max_dist[0]):
            for ry in range(1 - max_dist[1], max_dist[1]):
                for rz in range(1 - max_dist[2], max_dist[2]):
                    for i in range(self.N):
                        for j in range(self.N):
                            r_vec = np.array([rx, ry, rz])
                            distance_vec = (self.basis_vecs[i] - self.basis_vecs[j]
                                            + r_vec @ self.lat_vecs)
                            pairwise_distance[i, j, rx, ry, rz] = np.linalg.norm(distance_vec)
        pairwise_distance = np.round(pairwise_distance, 5)
        self.neighbor_distances = sorted(set(pairwise_distance.flatten()))

        for i in range(len(self.j_ij)):
            self.j_mat += np.where(pairwise_distance == self.neighbor_distances[i + 1],
                                   self.j_ij[i], 0.0)

    def calc_chi_0(self, temperatures=None):
        """calculates the curie susceptibility, chi_0, for the system at each temperature.
        If temperatures is provided as an argument, calculates chi_0 at those temperatures;
        otherwise, calculates chi_0 at the temperatures stored as an attribute of the class.

        Args:
            temperatures (array, optional): Array of temperatures at which to calculate chi_0.

        Returns:
            array: Array of curie susceptibilities at the specified temperatures.
        """

        if temperatures is None:
            temperatures = self.temperatures

        return self.spin_squared / (self.spin_dim * temperatures)

    def calc_j_q(self, q_vecs, diagonalize=True, store=False, return_eig_vecs=False):
        """calculates the Fourier transform of the interaction matrix, J(q), for a list
        of q-vectors. If diagonalize is True, also diagonalizes J(q) and returns the
        eigenvalues and eigenvectors. If store is True, stores the calculated J(q) and
        eigenvalues/eigenvectors as attributes of the class. If return_eig_vecs is True,
        returns both the eigenvalues and eigenvectors; otherwise, returns only the
        eigenvalues.

        Args:
            q_vecs (list): list of q-vectors to calculate J(q) for.
            diagonalize (bool): whether to diagonalize J(q) and return
                eigenvalues/eigenvectors.
            store (bool): whether to store the calculated J(q) and eigenvalues/
                eigenvectors as attributes of the class.
            return_eig_vecs (bool): whether to return both eigenvalues and eigenvectors
                (if diagonalize is True) or just eigenvalues.

        Returns:
            j_q (array): if diagonalize is False, returns an array of shape (n_q, N, N)
                containing the calculated J(q) matrices for each q-vector. If diagonalize
                is True and return_eig_vecs is False, returns an array of shape (n_q, N)
                containing the eigenvalues of J(q) for each q-vector. If diagonalize is
                True and return_eig_vecs is True, returns a tuple of two arrays: the first
                array has shape (n_q, N) and contains the eigenvalues of J(q) for each
                q-vector, and the second array has shape (n_q, N, N) and contains the
                corresponding eigenvectors.
        """
        q_vecs = np.array(q_vecs).reshape((-1, 3))
        n_q = len(q_vecs)

        rx_max, ry_max, rz_max = self.j_mat.shape[2:]
        rx_max = int((rx_max + 1) / 2)
        ry_max = int((ry_max + 1) / 2)
        rz_max = int((rz_max + 1) / 2)

        j_q = np.zeros((n_q, self.N, self.N), dtype='complex128')
        j_mu_q = np.zeros((n_q, self.N))  # eigenvectors of j_q
        u_q = np.zeros((n_q, self.N, self.N))  # eigenvalues of j_q

        for k, q_vec in enumerate(q_vecs):
            for rx in range(1 - rx_max, rx_max):
                for ry in range(1 - ry_max, ry_max):
                    for rz in range(1 - rz_max, rz_max):
                        r_vec = np.array([rx, ry, rz]) @ self.lat_vecs
                        phase = np.exp(-1j * np.dot(q_vec, r_vec))
                        j_q[k] += self.j_mat[:, :, rx, ry, rz] * phase

            if diagonalize:
                eig_vals, eig_vecs = np.linalg.eigh(j_q[k])
                j_mu_q[k] = eig_vals.astype('float64')
                u_q[k] = eig_vecs.astype('float64')

        if diagonalize:
            if store:
                self.q_vecs = q_vecs
                self.j_q = j_mu_q
                self.u_q = u_q

            if return_eig_vecs:
                return j_mu_q.T, u_q.T
            else:
                return j_mu_q.T

        else:
            j_q = j_q.astype('float64')

            if store:
                self.q_vecs = q_vecs
                self.j_q = j_q
            return j_q

    def filter_spin_positions(self, r_max=None):
        """filters the spin positions in magstruc to only include those within r_max of the spins
        at the indices specified in magstruc.calcIdxs. This is used to speed up calculations
        by only including spins that are relevant for the interactions being considered.

        Args:
            r_max (float): maximum distance from the spins at the indices specified in
                magstruc.calcIdxs to include in the filtered spin positions. If None, uses
                the r_max attribute of the class.
        """

        if r_max is None:
            r_max = self.r_max

        calc_idxs = self.magstruc.calcIdxs

        indices_to_keep = set(calc_idxs)

        for uu in calc_idxs:
            ri = self.magstruc.atoms[uu]
            rj = self.magstruc.atoms

            d_xyz = rj - ri
            d1_xyz = np.sqrt(np.sum(d_xyz ** 2, axis=1))

            indices_to_keep.update(np.where(d1_xyz <= r_max)[0])

        kept_idxs = sorted(indices_to_keep)
        old_to_new = {old: new for new, old in enumerate(kept_idxs)}
        calc_idxs_new = [old_to_new[uu] for uu in calc_idxs]

        self.magstruc.atoms = self.magstruc.atoms[kept_idxs]
        self.magstruc.spins = self.magstruc.spins[kept_idxs]

        self.magstruc.calcIdxs = calc_idxs_new

    def calculate_orf_real_space(self, lmop=1.0, corr_length=0.0, damping_matrix=None,
                                 correlation_method='simple', r=None, r_bin=None,
                                 r_step=0.01, r_min=0.0, extended_r_min=4.0,
                                 extended_r_max=4.0):
        """calculates the Onsager reaction field (ORF) for the system from local spin
        correlations based on the input parameters. Only calculates the ORF at a single
        temperature, corresponding to the temperature for which ordered_scale and
        corr_length or damping_matrix (if correlation_method is 'full') are
        calculated.

        Args:
            lmop (float): local magnetic order parameter (LMOP) of the system at the
                temperature for which to calculate the ORF.
            corr_length (float): exponential magnetic correlation length of the
                structure; default is 0, which is considered to be infinite.
                This is important to get the linear term right for samples
                with nonzero net magnetization.
            damping_matrix (array): 3 x 3 array representing the damping matrix for
                the system, where m is the dimensionality of the correlations. If
                correlation_method is 'full', this must be provided. If
                correlation_method is not 'full', this is not needed and will be
                ignored if provided.
            correlation_method (str): determines how the calculation should
                be done if the correlation length is finite. If 'full', actual
                spin magnitudes are adjusted according to the damping matrix.
            r_step (float): step size for r-grid of calculated spin correlations.
            r_min (float): minimum value of r for which spin correlations should be
                calculated.
            extended_r_min (float): extension of the r-grid on which the spin
                correlations is calculated to properly account for contribution of
                pairs just before the boundary.
            extended_r_max (float): extension of the r-grid on which the spin
                correlations is calculated to properly account for contribution of
                pairs just outside the boundary.

        Returns:
            orf (float): calculated value of the ORF for the system.
        """

        original_damping = getattr(self.magstruc, 'dampingMat', None)
        damping_mat_was_set = False

        if correlation_method == 'full' and damping_matrix is not None:
            self.magstruc.dampingMat = damping_matrix
            damping_mat_was_set = True
        elif corr_length != 0.0:
            lmop *= np.exp(self.r_nn / 2.0 / corr_length)

        xyz = self.magstruc.atoms
        s_xyz = self.magstruc.spins
        calc_idxs = self.magstruc.calcIdxs

        # set up real space grid
        if r is None:
            r = np.arange(r_min - extended_r_min, self.r_max + extended_r_max + r_step,
                          r_step)
            r = np.round(r, decimals=6)

            # don't calculate for negative r
            if r[0] < 0:
                start_idx = np.argmin(np.abs(r))
                if r[start_idx] < 0:
                    start_idx += 1
                r = r[start_idx:]

        if r_bin is None:
            r_bin = np.concatenate([r - r_step / 2, [r[-1] + r_step / 2]])

        s1 = np.zeros(len(r))

        if isinstance(calc_idxs, str) and calc_idxs == 'all':
            calc_idxs = np.arange(len(xyz))

        for uu in calc_idxs:
            ri = xyz[uu]
            rj = xyz
            si = s_xyz[uu]
            if correlation_method == 'full':
                sj = self.magstruc.generateScaledSpins(uu)
                sj = np.where(np.isnan(sj), s_xyz, sj)
            else:
                sj = s_xyz

            d_xyz = rj - ri
            d1_xyz = np.sqrt(np.sum(d_xyz ** 2, axis=1))

            sisj = np.sum(si * sj, axis=1)
            s1 += np.histogram(d1_xyz, bins=r_bin, weights=sisj)[0]

        j_r = np.concatenate([[0.0], self.j_ij, np.zeros(len(self.neighbor_distances)
                                                         - len(self.j_ij) - 1)])

        s2 = np.histogram(self.neighbor_distances, bins=r_bin, weights=j_r)[0]

        scale = (lmop / self.g / self.vector_mag) ** 2 / (self.spin_squared * len(calc_idxs))

        if correlation_method != 'full' and corr_length != 0.0:
            scale = scale * np.exp(-r / corr_length)

        orf_r = s1 * s2 * scale

        orf = np.sum(orf_r)

        # Restore the original state before returning
        if damping_mat_was_set:
            if original_damping is not None:
                self.magstruc.dampingMat = original_damping
            else:
                del self.magstruc.dampingMat

        return orf

    def calculate_orf_reciprocal_space(self, n_bz_vecs=2, n_recip_lats=2,
                                       store=True):
        """calculates the Onsager reaction field (ORF) for the system at each
        temperature from the Fourier transform of the interaction matrix, J(q),
        based on the input parameters.

        Args:
            n_bz_vecs (int): used for sampling vectors from the first
                Brillouin zone (BZ) to calculate J(q). n_bz_vecs corresponds
                to the number of points sampled in between two BZ vertices
                (including the vertices themselves).
            n_recip_lats (int): number of reciprocal lattice points to generate
                in each direction for constructing the BZ.
            store (bool): whether to store the calculated ORF values as an
                attribute of the class. Default is True.
        """

        # vectors inside 1BZ
        # reciprocal lattice
        recip_lat_pts = []
        for h in range(-n_recip_lats, n_recip_lats + 1):
            for k in range(-n_recip_lats, n_recip_lats + 1):
                for ll in range(-n_recip_lats, n_recip_lats + 1):
                    pt = np.array([h, k, ll]) @ self.recip_lats
                    recip_lat_pts.append(pt)
        recip_lat_pts = np.array(recip_lat_pts)

        # build 1BZ
        vor = Voronoi(recip_lat_pts)
        origin_index = np.where(np.all(recip_lat_pts == [0, 0, 0], axis=1))[0][0]
        bz_region = vor.regions[vor.point_region[origin_index]]
        bz_vertices = vor.vertices[bz_region]

        bz_pts = np.array([np.mean(combo, axis=0)
                           for combo in combinations_with_replacement(bz_vertices, n_bz_vecs)])

        bz_pts = np.round(bz_pts, 14)
        bz_pts = np.unique(bz_pts, axis=0)

        # Sample J(q) from BZ
        self.calc_j_q(bz_pts, diagonalize=True, store=True)

        orfs = np.zeros(self.n_temps)

        for i in range(self.n_temps):
            chi_0 = self.calc_chi_0(temperatures=self.temperatures[i])

            max_orf = np.max(self.j_q)
            min_orf = max_orf - 1 / chi_0
            orf_guess = np.mean([min_orf, max_orf])

            def orf_constraint(orf):
                orf_sum = np.sum((1 - chi_0 * (self.j_q - orf[0]))**(-1))
                return orf_sum - self.N * len(bz_pts)

            bounds = (min_orf, max_orf)
            orfs[i] = least_squares(orf_constraint, orf_guess,
                                    bounds=bounds).x[0]

        if store:
            self.orfs = orfs

        return orfs

    def calculate_j_hessian(self, q_vec, mode=-1, hstep=1e-4):
        """calculates the Hessian of J(q) at a given q-vector for a specified mode
        using finite differences.

        Args:
            q_vec (array): 3D array representing the q-vector at which to calculate
                the Hessian.
            mode (int): index of the mode for which to calculate the Hessian. Default
                is -1, which corresponds to the mode with the largest eigenvalue.
            hstep (float): step size for finite difference calculation. Default is
                1e-4.

        Returns:
            j_hessian (array): 3x3 array representing the Hessian of J(q) at the
                specified q-vector and mode.
        """
        j_hessian = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                if i == j:
                    h_vec = np.zeros(3)
                    h_vec[i] = hstep
                    j_hessian[i, i] = (self.calc_j_q(q_vec + h_vec)[mode]
                                       - 2 * self.calc_j_q(q_vec)[mode]
                                       + self.calc_j_q(q_vec - h_vec)[mode]) / hstep**2
                else:
                    hi = np.zeros(3)
                    hj = np.zeros(3)

                    hi[i] = hstep
                    hj[j] = hstep

                    j_hessian[i, j] = 0.25 / hstep**2 * (self.calc_j_q(q_vec + hi + hj)[mode]
                                                         - self.calc_j_q(q_vec - hi + hj)[mode]
                                                         - self.calc_j_q(q_vec + hi - hj)[mode]
                                                         + self.calc_j_q(q_vec - hi - hj)[mode])

        return j_hessian

    def calculate_correlation_subspace(self, ord_vec, dim=1, mode=-1, j_hessian=None):
        """calculates the subspace of the correlation function that corresponds to the
        specified dimensionality of magnetic correlations based on the Hessian of J(q)
        at the ordering vector.

        Args:
            ord_vec (array): 3D array representing the magnetic ordering vector.
            dim (int): dimensionality of the magnetic correlations to consider. Must
                be 1 or 2. Default is 1.
            mode (int): index of the mode for which to calculate the Hessian. Default
                is -1, which corresponds to the mode with the largest eigenvalue.
            j_hessian (array): square array representing the Hessian of J(q) at the
                ordering vector. If not provided, will be calculated using the
                calculate_j_hessian method.


        Returns:
            If dim is 1, returns a tuple containing:
                j_hessian_1d (float): the eigenvalue of the Hessian of J(q)
                    corresponding to the 1D subspace of correlations.
                subspace_vec (array): 3D array representing the eigenvector of the
                    Hessian of J(q) corresponding to the 1D subspace of correlations.
            If dim is 2, returns a tuple containing:
                j_hessian_2d (array): 2x2 array representing the eigenvalues of the
                    Hessian of J(q) corresponding to the 2D subspace of correlations.
                subspace_vecs (array): 3x2 array representing the eigenvectors of the
                    Hessian of J(q) corresponding to the 2D subspace of correlations.
        """

        if j_hessian is None:
            j_hessian = self.calculate_j_hessian(ord_vec, mode=mode)

        j_hessian_evals, j_hessian_evecs = np.linalg.eig(j_hessian)

        if dim == 1:
            max_arg = np.argmax(np.abs(j_hessian_evals))

            j_hessian_1d = j_hessian_evals[max_arg]
            subspace_vec = j_hessian_evecs[:, [max_arg]]

            self.bz_len = np.linalg.norm(self.recip_lats @ subspace_vec)

            j_hessian_1d = np.array([[j_hessian_1d]])

            return j_hessian_1d, subspace_vec

        elif dim == 2:
            arg_list = [0, 1, 2]
            min_arg = np.argmin(np.abs(j_hessian_evals))
            arg_list.remove(min_arg)

            j_hessian_2d = np.diag(j_hessian_evals[arg_list])
            subspace_vecs = j_hessian_evecs[:, arg_list]

            self.bz_area = np.linalg.norm(np.cross(self.recip_lats @ subspace_vecs[:, 0],
                                                   self.recip_lats @ subspace_vecs[:, 1]))

            return j_hessian_2d, subspace_vecs

        else:
            raise ValueError(f'dim must be 1 or 2 for calculate_correlation_subspace; got {dim}.')

    def calculate_damping_matrix(self, ord_vec=None, mode=-1, isotropic=False,
                                 j_0=None, orfs=None, temperatures=None,
                                 j_hessian=None, store=False):
        """calculates the damping matrix for the system.

        Args:
            ord_vec (array): 3D array representing the magnetic ordering vector.
            mode (int): index of the mode for which to calculate the Hessian of J(q).
                Default is -1, which corresponds to the mode with the largest eigenvalue.
            isotropic (bool): whether to calculate an isotropic damping matrix. If True,
                the calculated damping matrix will be isotropic with the same determinant
                as the anisotropic damping matrix. Default is False.
            j_0 (float): the value of J(q) at the ordering vector. If not provided, will be
                calculated. If j_0 is provided, ord_vec and mode are not needed but
                j_hessian is required.
            orfs (array): array representing the Onsager reaction field for the system at
                each temperature. If not provided, will use the orfs attribute of the class.
            temperatures (array): array representing the temperatures at which to calculate
                the damping matrix. If not provided, will use the temperatures attribute of
                the class.
            j_hessian (array): square array representing the Hessian of J(q) at the
                ordering vector. If not provided, will be calculated using the
                calculate_j_hessian method.
            store (bool): whether to store the calculated damping matrices as an attribute
                of the class. Default is False.

        Returns:
            array: n_tempsxmxm array representing the damping matrix for the system
                at each temperature where m is the dimensionality of the Hessian.
        """

        if orfs is None:
            orfs = self.orfs

        if temperatures is None:
            temperatures = self.temperatures

        if orfs.shape[0] != temperatures.shape[0]:
            raise ValueError('Number of ORF values must match number of temperatures.')

        if j_0 is None:
            if ord_vec is None:
                raise ValueError(
                    'If j_0 is not provided, ord_vec must be provided to calculate j_0.')
            j_0 = self.calc_j_q(ord_vec)[mode]

        if j_hessian is None:
            if ord_vec is None:
                raise ValueError(
                    'If j_hessian is not provided, ord_vec must be provided '
                    'to calculate j_hessian.')
            j_hessian = self.calculate_j_hessian(ord_vec, mode=mode)

        chi_0 = self.calc_chi_0(temperatures=temperatures)
        c_0 = 1 / chi_0 - j_0 + orfs

        inverse_damping_matrices = np.einsum('i,jk->ijk', 1 / c_0, -0.5 * j_hessian)
        damping_matrices = np.linalg.inv(inverse_damping_matrices)

        if isotropic:
            damping_amplitude = np.linalg.det(
                damping_matrices)**(1 / np.shape(damping_matrices)[1])
            if len(temperatures) == 1:
                damping_matrices = np.array([damping_amplitude * np.eye(3)])
            else:
                damping_matrices = np.array([damping_amplitude[i] * np.eye(3)
                                             for i in range(len(temperatures))])

        if store:
            self.damping_matrices = damping_matrices

        return damping_matrices

    def calc_3d_damping_matrix(self, damping_matrix=None, subspace_vecs=None):
        """calculates the full 3D damping matrix from the damping matrix in the subspace of
        correlations and the basis vectors of the subspace.

        Args:
            damping_matrix (array): n_tempsxmxm array representing the damping matrices in the
                subspace of correlations, where m is the dimensionality of the subspace.
            subspace_vecs (array): 3xm array representing the basis vectors of the
                subspace of correlations.
        Returns:
            array: 3x3 array representing the full damping matrix.
        """

        if subspace_vecs is None:
            raise ValueError(
                'Subspace eigenvectors must be provided to calculate full 3D damping matrix.')

        if len(damping_matrix.shape) == 3:
            if damping_matrix.shape[1] != subspace_vecs.shape[1]:
                raise ValueError(
                    'Damping matrix dimensionality does not match '
                    'number of subspace eigenvectors.')
        elif damping_matrix.shape[0] != subspace_vecs.shape[1]:
            raise ValueError(
                'Damping matrix dimensionality does not match '
                'number of subspace eigenvectors.')

        damping_matrix_3d = subspace_vecs @ damping_matrix @ subspace_vecs.T

        # The background term ensures that correlations outside the subspace are
        # sufficiently damped so they don't contribute to the ORF. The factor of
        # 1e4 × the maximum eigenvalue is chosen to be large enough to suppress
        # out-of-subspace contributions without causing numerical issues.
        background_damping_amplitude = np.max(
            np.linalg.eigvals(damping_matrix_3d), axis=1) * 1e4
        background_damping_amplitude = np.atleast_1d(background_damping_amplitude)
        background_damping_matrix = np.identity(3) - subspace_vecs @ subspace_vecs.T

        background_damping = np.einsum('i,jk->ijk', background_damping_amplitude,
                                       background_damping_matrix)

        damping_matrix_3d = (damping_matrix_3d + background_damping).real

        return damping_matrix_3d

    def calculate_lmop(self, n_ord_vec=1, dim=3, damping_matrices=None,
                       use_stored_damping_matrices=False, ord_vec=None, mode=-1,
                       j_0=None, orfs=None, temperatures=None, j_hessian=None,
                       subspace_vecs=None, store=False):
        """calculates the local magnetic order parameter (LMOP) for the system.

        Args:
            n_ord_vec (int): number of ordering vectors equivalent to the given
                ordering vector.
            dim (int): dimensionality of the correlation calculation.
            damping_matrices (array): array representing the damping matrices for the system
                at each temperature. If not provided, will be calculated using the
                calculate_damping_matrix method.
            use_stored_damping_matrices (bool): whether to use the damping matrices stored
                as an attribute of the class. If False, will calculate the damping
                matrices using the calculate_damping_matrix method. Default is False. Only
                used if damping_matrices is not provided as an argument; if damping_matrices
                is provided, this argument is ignored.
            ord_vec (array): 3D array representing the magnetic ordering vector. Only needed
                if damping_matrices is not provided as an argument and
                use_stored_damping_matrices is False.
            mode (int): index of the mode for which to calculate the Hessian of J(q).
                Default is -1, which corresponds to the mode with the largest eigenvalue.
            j_0 (float): the value of J(q) at the ordering vector. Only required if
                damping_matrices is not provided as an argument and
                use_stored_damping_matrices is False. If not provided, will be calculated.
                If j_0 is provided, ord_vec and mode are not needed but j_hessian is
                required.
            orfs (array): array representing the Onsager reaction field for the system at
                each temperature. If not provided, will use the orfs attribute of the class.
            temperatures (array): array representing the temperatures at which to calculate
                the LMOP. If not provided, will use the temperatures attribute of the class.
            j_hessian (array): square array representing the Hessian of J(q) at the
                ordering vector. If not provided, will be calculated using the
                calculate_j_hessian method.
            subspace_vecs (array): 3x(dim) array representing the basis vectors of the
                subspace in which the correlations are calculated.
            store (bool): whether to store the calculated LMOPs as an attribute of the
                class. Default is False.

        Returns:
            lmops (array): local magnetic order parameters for the system at each
                temperature.
        """

        if orfs is None:
            orfs = self.orfs

        if temperatures is None:
            temperatures = self.temperatures

        if orfs.shape[0] != temperatures.shape[0]:
            raise ValueError('Number of ORF values must match number of temperatures.')

        if j_hessian is None:
            if ord_vec is None:
                raise ValueError(
                    'If j_hessian is not provided, ord_vec must be provided '
                    'to calculate j_hessian.')
            j_hessian = self.calculate_j_hessian(ord_vec, mode=mode)

        if j_hessian.shape[0] != dim:
            raise ValueError(
                'Hessian dimensionality does not match specified correlation dimensionality.')

        if dim not in (1, 2, 3):
            raise ValueError(f'dim must be 1, 2, or 3; got {dim}.')

        if dim != 3 and subspace_vecs is None:
            raise ValueError(
                'Subspace eigenvectors must be provided for 1D and 2D correlation calculations.')

        if damping_matrices is None:
            if not use_stored_damping_matrices or self.damping_matrices is None:
                if j_0 is None:
                    if ord_vec is None:
                        raise ValueError(
                            'If j_0 is not provided, ord_vec must be provided to calculate j_0.')
                    j_0 = self.calc_j_q(ord_vec)[mode]

                damping_matrices = self.calculate_damping_matrix(
                    j_0=j_0, j_hessian=j_hessian,
                    temperatures=temperatures, orfs=orfs, store=store)
            else:
                damping_matrices = self.damping_matrices

        if dim == 1:
            r_nn = np.abs(self.r_nn_vec @ subspace_vecs)  # project r_nn onto 1D subspace
            chi_0 = self.calc_chi_0(temperatures=temperatures)

            c_0 = 1 / chi_0 - j_0 + orfs
            decay_term = np.exp(-r_nn * np.sqrt(damping_matrices)).ravel()
            prefactor = np.sqrt(2) * np.pi * n_ord_vec / self.bz_len / self.N
            local_correlations = (prefactor * temperatures * decay_term
                                  / np.sqrt(-j_hessian * c_0))
            lmops = self.g_nn * np.sqrt(np.abs(local_correlations)) / self.occ_avg

        elif dim == 2:
            r_nn = self.r_nn_vec @ subspace_vecs  # project r_nn onto 2D subspace

            decay_term = kn(0, np.sqrt(r_nn @ damping_matrices @ r_nn))
            prefactor = 4 * np.pi * n_ord_vec / self.bz_area / self.N
            local_correlations = (prefactor * temperatures * decay_term
                                  / np.sqrt(np.prod(j_hessian)))
            lmops = self.g_nn * np.sqrt(np.abs(local_correlations)) / self.occ_avg

        else:  # dim == 3
            decay_term = np.exp(
                -np.sqrt(self.r_nn_vec @ damping_matrices @ self.r_nn_vec))
            prefactor = (2 * np.pi) ** 2 * n_ord_vec / self.bz_vol / self.N
            local_correlations = (
                prefactor * temperatures * decay_term
                / np.sqrt(-np.linalg.det(j_hessian))
                / np.sqrt(-self.r_nn_vec @ np.linalg.inv(j_hessian) @ self.r_nn_vec))
            lmops = self.g_nn * np.sqrt(np.abs(local_correlations)) / self.occ_avg

        lmops = lmops.ravel()

        if store:
            self.lmops = lmops

        return lmops

    def calculate_correlation_parameters(self, ord_vec, n_ord_vec=1, mode=-1, hstep=1e-4,
                                         dim=3, calc_orf=True, sum_rule_orf=False,
                                         isotropic=False, n_bz_vecs=2, n_recip_lats=2):
        """calculates the parameters related to the magnetic correlations in the system,
        including the subspace of the correlation function corresponding to the specified
        dimensionality of correlations, the damping matrix, and the local magnetic order
        parameter (LMOP) at each temperature. The ORF can also be calculated as part of
        this function, either using the sum rule or by directly calculating the ORF from
        the real space correlations.

        Args:
            ord_vec (array): 3D array representing the magnetic ordering vector.
            n_ord_vec (int): number of ordering vectors equivalent to the given ordering
                vector.
            mode (int): index of the mode for which to calculate the Hessian of J(q)
                at the ordering vector. Default is -1, which corresponds to the mode with
                the largest eigenvalue.
            hstep (float): step size for finite difference calculation of the Hessian of J(q).
            dim (int): dimensionality of the magnetic correlations to consider. Must be 1, 2,
                or 3. Default is 3.
            calc_orf (bool): whether to calculate the Onsager reaction field (ORF) for the
                system as part of this function. Default is True.
            sum_rule_orf (bool): whether to calculate the ORF using the sum rule (True) or by
                directly calculating the ORF from the real space correlations (False). Default
                is False. Only used if calc_orf is True.
            isotropic (bool): whether to calculate an isotropic damping matrix. If True,
                the calculated damping matrix will be isotropic. Default is False.
            n_bz_vecs (int): used for sampling vectors from the first Brillouin zone (BZ) to
                calculate J(q) if sum_rule_orf is True. n_bz_vecs corresponds to the number
                of points sampled in two BZ vertices (including the vertices themselves). Only
                used if sum_rule_orf is True.
            n_recip_lats (int): number of reciprocal lattice points to generate in each
                direction for constructing the BZ. Only used if sum_rule_orf is True.

        Returns:
            None: the calculated parameters are stored as attributes of the class (self.orfs,
                self.lmops, and self.damping_matrices)."""

        j_0 = self.calc_j_q(ord_vec)[mode]

        j_hessian = self.calculate_j_hessian(ord_vec, mode=mode, hstep=hstep)

        if dim != 3:
            j_hessian, subspace_vecs = self.calculate_correlation_subspace(ord_vec, dim=dim,
                                                                           mode=mode,
                                                                           j_hessian=j_hessian)
        else:
            subspace_vecs = None

        if calc_orf:
            if sum_rule_orf:
                self.calculate_orf_reciprocal_space(n_bz_vecs=n_bz_vecs,
                                                    n_recip_lats=n_recip_lats,
                                                    store=True)
            else:
                for i in range(self.n_temps):

                    def orf_residual(orf):
                        temperature = self.temperatures[i]
                        temperature = np.array([temperature])

                        damping_matrix = self.calculate_damping_matrix(j_0=j_0,
                                                                       isotropic=isotropic,
                                                                       orfs=orf,
                                                                       temperatures=temperature,
                                                                       j_hessian=j_hessian,
                                                                       store=False)

                        lmop = self.calculate_lmop(j_0=j_0, n_ord_vec=n_ord_vec, dim=dim,
                                                   orfs=orf, temperatures=temperature,
                                                   damping_matrices=damping_matrix,
                                                   use_stored_damping_matrices=False,
                                                   j_hessian=j_hessian,
                                                   subspace_vecs=subspace_vecs, store=False)

                        if dim != 3:
                            damping_matrix = self.calc_3d_damping_matrix(
                                damping_matrix=damping_matrix, subspace_vecs=subspace_vecs)

                        damping_matrix = damping_matrix[0]  # make it 2D instead of 3D since we're
                        # only calculating for one temperature at a time

                        if isotropic:
                            corr_length = (self.r_nn /
                                           np.sqrt(self.r_nn_vec @ damping_matrix @ self.r_nn_vec))
                            orf_calc = self.calculate_orf_real_space(lmop=lmop,
                                                                     corr_length=corr_length)
                        else:
                            orf_calc = self.calculate_orf_real_space(lmop=lmop,
                                                                     damping_matrix=damping_matrix,
                                                                     correlation_method='full')

                        return orf - orf_calc

                    chi_0 = self.calc_chi_0(temperatures=self.temperatures[i])
                    max_orf = j_0
                    min_orf = max_orf - 1 / chi_0
                    orf_guess = np.mean([min_orf, max_orf])
                    bounds = (min_orf, max_orf)

                    orf_result = least_squares(orf_residual, orf_guess, bounds=bounds,
                                               method='trf')
                    self.orfs[i] = orf_result.x[0]

        self.calculate_damping_matrix(j_0=j_0, isotropic=isotropic, j_hessian=j_hessian,
                                      store=True)

        self.calculate_lmop(n_ord_vec=n_ord_vec, dim=dim, use_stored_damping_matrices=True,
                            j_0=j_0, j_hessian=j_hessian, subspace_vecs=subspace_vecs,
                            store=True)

        if dim != 3:
            self.damping_matrices = self.calc_3d_damping_matrix(
                damping_matrix=self.damping_matrices, subspace_vecs=subspace_vecs)