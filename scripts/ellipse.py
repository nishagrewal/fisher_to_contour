import numpy as np
from matplotlib.patches import Ellipse

class CosmoFisher:
    def __init__(self, p1, p2, fisher_matrix, cosmo_params):
        self.p1 = p1
        self.p2 = p2
        self.fisher_matrix = fisher_matrix 
        self.cosmo_params = cosmo_params # must be a dictionary with keys 'value' and 'label'?


    def remove_marginalised_params(self, matrix, p_keep):
        '''
        Remove rows and columns from the Fisher matrix corresponding to the parameters that have been marginalised over.
        
        Parameters
        ----------
        matrix : array
            Fisher matrix
        p_keep : list
            List of parameters to keep in the Fisher matrix
        
        Returns
        -------
        matrix : array
            Fisher matrix with rows and columns removed for marginalised parameters
        '''
        p_list = list(self.cosmo_params.keys())
        idxs = [p_list.index(p) for p in p_keep if p in p_list]
        matrix = matrix[idxs][:, idxs]
        return matrix


    def fisher_inverse_covariance(self):
        '''
        Return the inverse of the Fisher matrix, then removing the marginalised parameters.

        Returns
        -------
        inv_cov_rm : array
            Inverse of the Fisher matrix covariance
        '''
        inv_cov = np.linalg.inv(self.fisher_matrix) 
        inv_cov_rm = self.remove_marginalised_params(inv_cov, [self.p1,self.p2])
        return inv_cov_rm


    def ellipse(self, nstd=1, **kwargs):
        '''
        Create an ellipse centered at the best fit values of the parameters.
        Note: this only works for 2 parameters.
        
        Parameters
        ----------
        nstd : float
            Number of standard deviations to use for the ellipse size
        kwargs : dict
            Keyword arguments for the Ellipse object
        
        Returns
        -------
        ellipse : Ellipse
            Ellipse object
        center : list
            Center of the ellipse
        stdv : list
            Standard deviation for each parameter
        '''

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

        return ellip, center, stdv 


    def S8_ellipse(self, alpha=0.5, nstd=1, **kwargs):
        '''
        Create an Omega_m-S_8 ellipse centered at the best fit values of the parameters.

        Parameters
        ----------
        alpha : float
            Power law index
        nstd : float
            Number of standard deviations to use for the ellipse size
        kwargs : dict
            Keyword arguments for the Ellipse object
        
        Returns
        -------
        ellipse : Ellipse
            Ellipse object
        center : list
            Center of the ellipse
        stdv : list
            Standard deviation for each parameter
        '''

        if 'Om' not in self.cosmo_params or 's8' not in self.cosmo_params:
            raise ValueError('Om and s8 must be in the cosmo_params dictionary')

        Om = self.cosmo_params['Om']['value']
        s8 = self.cosmo_params['s8']['value']
        center = [Om,s8]

        M = np.array([[1, 0],
                    [-alpha * (s8 / Om), (0.3 / Om) ** alpha]])    # transformation matrix

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

        return ellip, center, stdv 
