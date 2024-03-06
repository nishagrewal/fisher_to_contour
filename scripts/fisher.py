import numpy as np
import h5py
import cosmogrid as cosmogrid

cosmo_params = cosmogrid.cosmo_params
recon_path = cosmogrid.recon_path

def cut_and_reshape(data):
    slicing_dict = {
        "v0": (1, 10), 
        "v1": (11, 20),
        "v2": (21, 30)
    }
    # slicing_dict = {
    #     "v0": (2, 10), 
    #     "v1": (11, 20),
    #     "v2": (21, 30)
    # }
    if len(data.shape) == 3:
        slices = [data[:,:,start:end] for start, end in slicing_dict.values()]
        mfs = np.concatenate(slices, axis=2)
        return np.reshape(mfs, (mfs.shape[0], -1))
    elif len(data.shape) == 2: # dont really need this
        slices = [data[:,start:end] for start, end in slicing_dict.values()]
        return np.concatenate(slices, axis=1).flatten()

def fisher(s_list, i_cov):
    matrix_size = len(s_list)
    fisher_matrix = np.zeros((matrix_size, matrix_size))

    for i, p1 in enumerate(s_list):
        for j, p2 in enumerate(s_list):
            fisher_matrix[i, j] = np.einsum('i,ij,j', p1, i_cov, p2)

    return fisher_matrix


def param_func_all(p,method,name,map_num,fid_cut,path='recon'):
    
    # param value
    param = cosmo_params[p]['value']

    # define param delta
    delta = cosmo_params[p]['step_size']

    # load MF files
    param_m = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_m{name}.hdf5', 'r')['mf'][0:map_num]
    param_p = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_p{name}.hdf5', 'r')['mf'][0:map_num]

    # getting the average MF values (more maps = less noise, closer to truth)
    param_m_mean = np.mean(param_m,axis=0)
    param_p_mean = np.mean(param_p,axis=0)

    # cut out edge values to eliminate linear dependencies (adjacent points with same value)
    # flatten to combine MFs from different redshifts
    param_p_cut = cut_and_reshape(param_p_mean)
    param_m_cut = cut_and_reshape(param_m_mean)

    # get partial difference
    x = [param - delta, param, param + delta]
    y = np.array([param_m_cut, fid_cut, param_p_cut])

    # get slopes for each MF observable using all 3 points
    s_a = np.polyfit(x,y,1)[0]

    return s_a


def calc_fisher_all(p_list,method,name,path,map_num=5_000):

    '''
    Inputs:
    --------
    p_list: list of parameters to calculate fisher matrix for
    method: reconstruction method
    name: noise and or/nor mask
    map_num: number of maps

    Returns:
    --------
    fisher matrix
    '''

    # load fiducial MF files
    fid = h5py.File(f'{recon_path}/ngrewal/recon/mf/{method}/fiducial{name}.hdf5', 'r')["mf"][0:map_num]

    # remove edge values to eliminate linear dependencies (adjacent points with same value)
    fid_for_cov = cut_and_reshape(fid)

    # scale factor to DES area (in sq deg)
    sf = 5000/(15**2) #np.sqrt(5000/(15**2))     

    # covariance   
    cov = np.cov(fid_for_cov.T) / sf

    # inverse covariance
    i_cov = np.linalg.inv(cov)       

    # getting the average MF values for each cosmology (more maps = less noise, closer to truth)
    fid_mean = np.mean(fid,axis=0)

    # cut out edge values to eliminate linear dependencies (adjacent points with same value)
    # flatten to combine MFs from different redshifts
    fid_cut = cut_and_reshape(fid_mean)

    # get a list of slopes from the param_func function
    s_list = [param_func_all(p,method,name,map_num,fid_cut,path) for p in p_list]
    
    # get fisher matrix
    return fisher(s_list, i_cov)




# ## TOP TWO POINTS
# def param_func_top(p,method,name,map_num,fid_cut,path='recon'):
    
#     # param value
#     param = cosmo_params[p]['value']

#     # define param delta
#     delta = cosmo_params[p]['step_size']

#     # load MF files
#     param_m = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_m{name}.hdf5', 'r')['mf'][0:map_num]
#     param_p = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_p{name}.hdf5', 'r')['mf'][0:map_num]

#     # getting the average MF values (more maps = less noise, closer to truth)
#     param_m_mean = np.mean(param_m,axis=0)
#     param_p_mean = np.mean(param_p,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     param_p_cut = cut_and_reshape(param_p_mean)
#     param_m_cut = cut_and_reshape(param_m_mean)

#     # get partial difference
#     x = [param, param + delta]
#     y = np.array([fid_cut, param_p_cut])

#     # get slopes for each MF observable using all 3 points
#     s_a = np.polyfit(x,y,1)[0]

#     return s_a


# def calc_fisher_top(p_list,method,name,path,map_num=5_000):

#     '''
#     Inputs:
#     --------
#     p_list: list of parameters to calculate fisher matrix for
#     method: reconstruction method
#     name: noise and or/nor mask
#     map_num: number of maps

#     Returns:
#     --------
#     fisher matrix
#     '''

#     # load fiducial MF files
#     fid = h5py.File(f'{recon_path}/ngrewal/recon/mf/{method}/fiducial{name}.hdf5', 'r')["maps"][0:map_num]

#     # remove edge values to eliminate linear dependencies (adjacent points with same value)
#     fid_for_cov = cut_and_reshape(fid)

#     # scale factor to DES area (in sq deg)
#     sf = np.sqrt(5000/(15**2))     

#     # covariance   
#     cov = np.cov(fid_for_cov.T) / sf

#     # inverse covariance
#     i_cov = np.linalg.inv(cov)       

#     # getting the average MF values for each cosmology (more maps = less noise, closer to truth)
#     fid_mean = np.mean(fid,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     fid_cut = cut_and_reshape(fid_mean)

#     # get a list of slopes from the param_func function
#     s_list = [param_func_top(p,method,name,map_num,fid_cut,path) for p in p_list]
    
#     # get fisher matrix
#     return fisher(s_list, i_cov)



# ## BOTTOM TWO POINTS
# def param_func_bottom(p,method,name,map_num,fid_cut,path='recon'):
    
#     # param value
#     param = cosmo_params[p]['value']

#     # define param delta
#     delta = cosmo_params[p]['step_size']

#     # load MF files
#     param_m = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_m{name}.hdf5', 'r')['maps'][0:map_num]
#     param_p = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_p{name}.hdf5', 'r')['maps'][0:map_num]

#     # getting the average MF values (more maps = less noise, closer to truth)
#     param_m_mean = np.mean(param_m,axis=0)
#     param_p_mean = np.mean(param_p,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     param_p_cut = cut_and_reshape(param_p_mean)
#     param_m_cut = cut_and_reshape(param_m_mean)

#     # get partial difference
#     x = [param - delta, param]
#     y = np.array([param_m_cut, fid_cut])

#     # get slopes for each MF observable using all 3 points
#     s_a = np.polyfit(x,y,1)[0]

#     return s_a


# def calc_fisher_bottom(p_list,method,name,path,map_num=5_000):

#     '''
#     Inputs:
#     --------
#     p_list: list of parameters to calculate fisher matrix for
#     method: reconstruction method
#     name: noise and or/nor mask
#     map_num: number of maps

#     Returns:
#     --------
#     fisher matrix
#     '''

#     # load fiducial MF files
#     fid = h5py.File(f'{recon_path}/ngrewal/recon/mf/{method}/fiducial{name}.hdf5', 'r')["maps"][0:map_num]

#     # remove edge values to eliminate linear dependencies (adjacent points with same value)
#     fid_for_cov = cut_and_reshape(fid)

#     # scale factor to DES area (in sq deg)
#     sf = np.sqrt(5000/(15**2))     

#     # covariance   
#     cov = np.cov(fid_for_cov.T) / sf

#     # inverse covariance
#     i_cov = np.linalg.inv(cov)       

#     # getting the average MF values for each cosmology (more maps = less noise, closer to truth)
#     fid_mean = np.mean(fid,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     fid_cut = cut_and_reshape(fid_mean)

#     # get a list of slopes from the param_func function
#     s_list = [param_func_bottom(p,method,name,map_num,fid_cut,path) for p in p_list]
    
#     # get fisher matrix
#     return fisher(s_list, i_cov)


# ## OUTSIDE TWO POINTS

# def param_func_outside(p,method,name,map_num,fid_cut,path='recon'):
    
#     # param value
#     param = cosmo_params[p]['value']

#     # define param delta
#     delta = cosmo_params[p]['step_size']

#     # load MF files
#     param_m = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_m{name}.hdf5', 'r')['maps'][0:map_num]
#     param_p = h5py.File(f'{recon_path}/ngrewal/{path}/mf/{method}/delta_{p}_p{name}.hdf5', 'r')['maps'][0:map_num]

#     # getting the average MF values (more maps = less noise, closer to truth)
#     param_m_mean = np.mean(param_m,axis=0)
#     param_p_mean = np.mean(param_p,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     param_p_cut = cut_and_reshape(param_p_mean)
#     param_m_cut = cut_and_reshape(param_m_mean)

#     # get partial difference
#     x = [param - delta, param + delta]
#     y = np.array([param_m_cut, param_p_cut])

#     # get slopes for each MF observable using all 3 points
#     s_a = np.polyfit(x,y,1)[0]

#     return s_a


# def calc_fisher_outside(p_list,method,name,path,map_num=5_000):

#     '''
#     Inputs:
#     --------
#     p_list: list of parameters to calculate fisher matrix for
#     method: reconstruction method
#     name: noise and or/nor mask
#     map_num: number of maps

#     Returns:
#     --------
#     fisher matrix
#     '''

#     # load fiducial MF files
#     fid = h5py.File(f'{recon_path}/ngrewal/recon/mf/{method}/fiducial{name}.hdf5', 'r')["maps"][0:map_num]

#     # remove edge values to eliminate linear dependencies (adjacent points with same value)
#     fid_for_cov = cut_and_reshape(fid)

#     # scale factor to DES area (in sq deg)
#     sf = np.sqrt(5000/(15**2))     

#     # covariance   
#     cov = np.cov(fid_for_cov.T) / sf

#     # inverse covariance
#     i_cov = np.linalg.inv(cov)       

#     # getting the average MF values for each cosmology (more maps = less noise, closer to truth)
#     fid_mean = np.mean(fid,axis=0)

#     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
#     # flatten to combine MFs from different redshifts
#     fid_cut = cut_and_reshape(fid_mean)

#     # get a list of slopes from the param_func function
#     s_list = [param_func_outside(p,method,name,map_num,fid_cut,path) for p in p_list]
    
#     # get fisher matrix
#     return fisher(s_list, i_cov)




































































# # def fisher(p1,p2,i_cov):
# #     return [[np.einsum('i,ij,j',p1,i_cov,p1), np.einsum('i,ij,j',p1,i_cov,p2)],
# #             [np.einsum('i,ij,j',p2,i_cov,p1), np.einsum('i,ij,j',p2,i_cov,p2)]] 

# # def calc_fisher(p1,p2,method,name,map_num):

# #     param1 = cosmo_params[p1]['value']
# #     param2 = cosmo_params[p2]['value']

# #     # define param deltas
# #     delta1 = cosmo_params[p1]['step_size']
# #     delta2 = cosmo_params[p2]['step_size']

# #     # load MF files
# #     fid = h5py.File(f'/disk01/ngrewal/jax/mf/{method}/fiducial{name}.hdf5', 'r')["maps"][0:map_num]
# #     param1_m = h5py.File(f'/disk01/ngrewal/jax/mf/{method}/delta_{p1}_m{name}.hdf5', 'r')['maps'][0:map_num]
# #     param1_p = h5py.File(f'/disk01/ngrewal/jax/mf/{method}/delta_{p1}_p{name}.hdf5', 'r')['maps'][0:map_num]
# #     param2_m = h5py.File(f'/disk01/ngrewal/jax/mf/{method}/delta_{p2}_m{name}.hdf5', 'r')['maps'][0:map_num]
# #     param2_p = h5py.File(f'/disk01/ngrewal/jax/mf/{method}/delta_{p2}_p{name}.hdf5', 'r')['maps'][0:map_num]

# #     # inverse covariance (30x30 matrix) 
# #     fid_for_cov = cut_and_reshape(fid)
# #     cov = np.cov(fid_for_cov.T)
# #     i_cov = np.linalg.inv(cov)

# #     # getting the average MF values for each cosmology (more maps = less noise, closer to truth)
# #     fid_mean = np.mean(fid,axis=0)
# #     param1_m_mean = np.mean(param1_m,axis=0)
# #     param1_p_mean = np.mean(param1_p,axis=0)
# #     param2_m_mean = np.mean(param2_m,axis=0)
# #     param2_p_mean = np.mean(param2_p,axis=0)

# #     # cut out edge values to eliminate linear dependencies (adjacent points with same value)
# #     # flatten to combine MFs from different redshifts
# #     fid_cut = cut_and_reshape(fid_mean)
# #     param1_p_cut = cut_and_reshape(param1_p_mean)
# #     param1_m_cut = cut_and_reshape(param1_m_mean)
# #     param2_p_cut = cut_and_reshape(param2_p_mean)
# #     param2_m_cut = cut_and_reshape(param2_m_mean)

# #     # get partial difference for param 1
# #     x1 = [param1 - delta1, param1, param1 + delta1]
# #     y1 = np.array([param1_m_cut, fid_cut, param1_p_cut])

# #     # get partial difference for param 2
# #     x2 = [param2 - delta2, param2, param2 + delta2]  
# #     y2 = np.array([param2_m_cut, fid_cut, param2_p_cut])

# #     # get slopes for each MF observable using all 3 points
# #     s1_a = np.polyfit(x1,y1,1)[0]
# #     s2_a = np.polyfit(x2,y2,1)[0]

# #     # get fisher matrix
# #     return fisher(s1_a,s2_a,i_cov)