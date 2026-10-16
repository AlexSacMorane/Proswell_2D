# this file is used to inform the main code about the parameters of the simulation.
# authors: Alexandre Sac-Morane, alexandre.sac-morane@enpc.fr

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Librairies
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Functions
#------------------------------------------------------------------------------------------------------------------------------------------#

def shear_modulus(E, ν):
    '''
    Compute the shear modulus from the Young's modulus and the Poisson's ratio.
    '''
    return E / (2 * (1 + ν))

#---------------------------------------

def first_lame_coefficient(μ, ν):
    '''
    Compute the first Lame coefficient from the shear modulus and the Poisson's ratio.
    '''
    return 2.0 * μ * ν / (1.0 - 2.0 * ν)

#---------------------------------------

def bulk_modulus(μ, ν):
    '''
    Define the bulk modulus from the shear modulus and the Poisson's ratio.
    '''
    λ = first_lame_coefficient(μ, ν)
    return λ + 2.0 * μ / 3 

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Parameters
#------------------------------------------------------------------------------------------------------------------------------------------#

def get_parameters():
    '''
    Define the parameters used in the simulation.
    '''
    # initialize the dictionaries
    dict_user = {}
    dict_sample = {}

    # ---------------------------------------------------------------------
    # Mesh description
    # an unit cell of 1-1-1 is considered

    dict_user['dim_x'] = 100
    dict_user['dim_y'] = 100
    dict_user['dim_z'] = 1
    if dict_user['dim_z'] != 1 : # check it is 2D
        raise ValueError("The simulation is 2D, please set dim_z = 1")

    # compute the mean size of the mesh
    if dict_user['dim_z'] == 1:
        m_size_mesh = (0.5*1/dict_user['dim_x'] + 0.5*1/dict_user['dim_y'])
    else :
        m_size_mesh = (1/dict_user['dim_x']/3 + 1/dict_user['dim_y']/3 + 1/dict_user['dim_z']/3)

    # ---------------------------------------------------------------------
    # Grain geometry definition

    # size of the grain
    dict_user['radius'] = 0.3

    # number of mesh in the interface
    n_int = 3
    # the interface thickness
    dict_user['w_int'] = m_size_mesh*n_int

    # ---------------------------------------------------------------------
    # first-layer algorithm

    # number of increments
    dict_user['n_ite'] = 40

    # save the current increment
    dict_sample['i_ite'] = 1

    # method for the remaping
    # trimesh, pyvista  
    dict_user['method_construction_new_map'] = 'pyvista'

    # ---------------------------------------------------------------------
    # FFT simulation 
    
    # K = E.G/(3*(3.G-E)) = 2.G.(1+v)/(3*(1-2.v)) = E/(3*(1-2.v))
    # G = E/(2*(1+v)) = 3.K.(1-2.v)/(2*(1+v)) = 3.K.E/(9.K-E)
    # E = 9.K.G/(3.K+G) = 3.K.(1-2.v) = 2.G.(1+v)
    # v = E/(2.G) - 1 = (3.K-E)/(6.K) = (3.K-2.G)/(2.(3.K+G))

    # define the constitutive parameters for the simulation
    # 0: pore
    # 1: anhydrite
    # 2: gypsum
    dict_user['E'] =  np.array([   0,   70,   10])
    dict_user['nu'] = np.array([0.49,  0.3, 0.25])
    dict_user["μ"] =  np.array([   0, 26.9,  4.0])
    #dict_user["κ"] =  np.array([ 2.2, 58.3,  6.7])
    dict_user["κ"] =  np.array([ 0, 58.3,  6.7])
    dict_user['biot'] =        [  1,   0]
    dict_user['alpha'] =       [  0,   1]
    dict_user['alpha0'] = dict_user['alpha'].copy()

    # interpolation of the mechanical parameters for the anhydrite-gypsum
    dict_user['flag_anhydrite_gypsum_interpolation'] = False

    # define the hydration increment
    dict_user['dhyd_ite'] = np.array([0.01])

    # define the tolerance for the resolution
    dict_user['tol_fft'] = 0.001

    # ---------------------------------------------------------------------
    # output
    
    # FFT results (strain, stress, displacement)
    dict_user['flag_data_tiff'] = False

    # pp the displacement reconstruction (Brisard, 2017)
    dict_user['flag_print_un_mesh'] = False 

    # verify the volumetric deformation in the hydrating phase
    dict_user['flag_check_eps_vol'] = False

    # interpolation error during the interpolation of the next mesh from the previous one
    dict_user['flag_print_plot_error_interpolation'] = False 

    # ---------------------------------------------------------------------
    # trackers initialization

    # volume of the phase 1
    dict_sample['L_vol_phi1'] = []
    # volume of the phase 1 (> 0.1)
    dict_sample['L_vol_phi1_01'] = []
    # check the volume of the phase 1 and phase 1 (> 0.1)
    if dict_user['κ'][0] == 0:
        dict_sample['L_vol_phi1_expected'] = []
        dict_sample['L_vol_phi1_01_expected'] = []
    # volume of control
    dict_sample['L_vol_unit'] = []

    # hydration
    dict_sample['L_hyd'] = []

    # performance computing
    # fft, def_disp, int_mesh, prep_step, total
    dict_sample['L_tac_tic'] = [0, 0, 0, 0, 0]

    # ---------------------------------------------------------------------
    # outputs

    return dict_user, dict_sample