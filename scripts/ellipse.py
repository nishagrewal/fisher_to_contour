import numpy as np
from matplotlib.patches import Ellipse
from modules.fisher import *
import modules.cosmogrid as cosmogrid

class CosmoFisher:
    def __init__(self,p1, p2, method, name, path='recon', f_points='all'):
        self.cosmo_params = cosmogrid.cosmo_params
        self.p1 = p1
        self.p2 = p2
        self.p_list = list(self.cosmo_params.keys())
        self.method = method
        self.name = name
        self.path = path
        if f_points=='all':
            self.F = calc_fisher_all(self.p_list, method, name, path)
        if f_points=='top':
            self.F = calc_fisher_top(self.p_list, method, name, path)
        if f_points=='bottom':
            self.F = calc_fisher_bottom(self.p_list, method, name, path)
        if f_points=='outside':
            self.F = calc_fisher_outside(self.p_list, method, name, path)


    def remove_marginalised_params(self, matrix, p_keep):
        idxs = [self.p_list.index(p) for p in p_keep if p in self.p_list]
        matrix = matrix[idxs][:, idxs]
        return matrix

    def fisher_inverse_covariance(self):
        inv_cov = np.linalg.inv(self.F) 
        inv_cov_rm = self.remove_marginalised_params(inv_cov, [self.p1,self.p2])
        return inv_cov_rm

    def ellipse(self, nstd=1, **kwargs):
        # center of ellipse
        center = [self.cosmo_params[self.p1]['value'], self.cosmo_params[self.p2]['value']]

        # build ellipse
        F_cov = self.fisher_inverse_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(F_cov)
        order = eigenvalues.argsort()[::-1]
        vals, vecs = eigenvalues[order], eigenvectors[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)

        # standard deviation for each parameter
        stdv = np.sqrt(np.diag(F_cov))

        return ellip,center,stdv 

    def S8_std_dev(self,alpha=0.5):
        # get S8 standard deviation
        # only works for Om and s8 Fisher matrix
        Om = self.cosmo_params['Om']['value']
        s8 = self.cosmo_params['s8']['value']
        M = np.array([[1, 0],
                    [-alpha * (s8 / Om), (0.3 / Om) ** alpha]])    # transformation matrix
        
        # get Om-s8 maatrix
        f_inv = self.fisher_inverse_covariance()
        f = np.linalg.inv(f_inv) 

        # transform F to get S8 ellipse
        F_inv_cov = np.matmul(M.T, np.matmul(f, M))
        F_cov = np.linalg.inv(F_inv_cov)
        S8_std = np.sqrt(F_cov[1, 1])
        return S8_std
        
        # # need to remove marginalised params from F
        # f = self.remove_marginalised_params(self.F, ['Om','s8'])

        # # transform F to get S8 standard deviation
        # F_new = np.matmul(M.T, np.matmul(f, M))
        # cov = np.linalg.inv(F_new)
        # S8_std = np.sqrt(cov[1, 1])
        # return S8_std

    def S8_ellipse(self, nstd=1, **kwargs):
        # get S8 ellipse
        # only works for Om and s8 Fisher matrix
        Om = self.cosmo_params['Om']['value']
        s8 = self.cosmo_params['s8']['value']
        center = [Om,s8]

        M = np.array([[1, 0],
                    [-0.5 * (s8 / Om), (0.3 / Om) ** 0.5]])    # transformation matrix

        # get Om-s8 maatrix
        f_inv = self.fisher_inverse_covariance()
        f = np.linalg.inv(f_inv) 

        # transform F to get S8 ellipse
        F_inv_cov = np.matmul(M.T, np.matmul(f, M))
        F_cov = np.linalg.inv(F_inv_cov)

        # build ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(F_cov)
        order = eigenvalues.argsort()[::-1]
        vals, vecs = eigenvalues[order], eigenvectors[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)

        # standard deviation for each parameter
        stdv = np.sqrt(np.diag(F_cov))

        return ellip,center,stdv 












# class CosmoFisher:
#     def __init__(self,p1, p2, p_list, method, name):
#         self.cosmo_params = {
#             'Om': {'value': 0.26, 'step_size': 0.01},
#             's8': {'value': 0.84, 'step_size': 0.015},
#             'w0': {'value': -1, 'step_size': 0.05},
#             'ns': {'value': 0.9649, 'step_size': 0.02},
#             'Ob': {'value': 0.0493, 'step_size': 0.001},
#             'H0': {'value': 67.3, 'step_size': 2}
#         }
#         self.p1 = p1
#         self.p2 = p2
#         self.p_list = p_list
#         self.method = method
#         self.name = name
#         self.F = calc_fisher(p_list, method, name)


#     def remove_marginalised_params(self, matrix, p_keep):
#         idxs = [self.p_list.index(p) for p in p_keep if p in self.p_list]
#         matrix = matrix[idxs][:, idxs]
#         return matrix

#     def fisher_inverse_covariance(self):
#         cov = np.linalg.inv(self.F)
#         f_inv_cov = self.remove_marginalised_params(cov, [self.p1,self.p2])
#         return f_inv_cov
    
#     def std_dev(self):
#         # only works for Om and s8
#         Om = self.cosmo_params['Om']['value']
#         s8 = self.cosmo_params['s8']['value']
#         M = np.array([[1, 0],
#                     [-0.5 * (s8 / Om), (0.3 / Om) ** 0.5]])    # transformation matrix
        
#         # need to remove marginalised params from F
#         f = self.remove_marginalised_params(self.F, ['Om','s8'])
#         F_new = np.matmul(M.T, np.matmul(f, M))
#         std = np.sqrt(np.linalg.inv(F_new))
#         Om_std = std[0, 0]
#         s8_std = std[1, 1]
#         return Om_std, s8_std

#     def ellipse(self, nstd=1, **kwargs):
#         # center of ellipse
#         center = [self.cosmo_params[self.p1]['value'], self.cosmo_params[self.p2]['value']]

#         # build ellipse
#         F_cov = self.fisher_inverse_covariance()
#         eigenvalues, eigenvectors = np.linalg.eigh(F_cov)
#         order = eigenvalues.argsort()[::-1]
#         vals, vecs = eigenvalues[order], eigenvectors[:, order]
#         theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#         width, height = 2 * nstd * np.sqrt(vals)
#         ellip = Ellipse(xy=center, width=width, height=height, angle=theta, **kwargs)

#         # standard deviation of ellipse
#         if self.p1=='Om' and self.p2=='s8':
#             stdv = self.std_dev()

#         return ellip,center,stdv
