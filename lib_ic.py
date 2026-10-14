# this file is used to describe the initial conditions.
# authors: Alexandre Sac-Morane, alexandre.sac-morane@enpc.fr

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Librairies
#------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------ #
# Functions
#------------------------------------------------------------------------------------------------------------------------------------------#

def generateIC(dict_user, dict_sample):
    '''
    Generate the IC for the simulation
    '''
    # preparation of the mineral maps
    dict_sample['L_phi'] = np.array([ np.ones((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'])), # pore
                                   np.zeros((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'])), # mineral 1
                                  ])

    # initialize the position of the interface
    if dict_user['method_construction_new_map'] == 'trimesh':
        dict_sample['L_i_XYZ_interface'] = []

    # iterate on the mesh
    for i_x in range(dict_user['dim_x']):
        x_i = (i_x+0.5)/dict_user['dim_x']
        for i_y in range(dict_user['dim_y']):
            y_i = (i_y+0.5)/dict_user['dim_y']
            for i_z in range(dict_user['dim_z']):
                z_i = (i_z+0.5)/dict_user['dim_z']

                # compute the distance to the center
                sdf = dict_user['radius'] - np.linalg.norm(np.array([0.5, 0.5, 0.5])-np.array([x_i, y_i, z_i]))

                # determine the gray scale
                if sdf > dict_user['w_int']/2: # inside the grain
                    dict_sample['L_phi'][1][i_x, i_y, i_z] = 1
                elif sdf < -dict_user['w_int']/2: # outside the grain
                    dict_sample['L_phi'][1][i_x, i_y, i_z] = 0
                else : # in the interface
                    dict_sample['L_phi'][1][i_x, i_y, i_z] = 0.5 + sdf/dict_user['w_int']    

                # reduce the pore variable
                dict_sample['L_phi'][0][i_x, i_y, i_z] = dict_sample['L_phi'][0][i_x, i_y, i_z] - dict_sample['L_phi'][1][i_x, i_y, i_z]
                # verify 
                if dict_sample['L_phi'][0][i_x, i_y, i_z] < 0 :
                    raise ValueError('Error in the IC generation')

                if dict_user['method_construction_new_map'] == 'trimesh':
                    # save the interface
                    if abs(sdf) < dict_user['w_int']:
                        dict_sample['L_i_XYZ_interface'].append([i_x, i_y, i_z]) 

    # initialization of the coordinates map of the cubes that compose the mesh
    dict_sample['M_L_coord_cubeMesh'] = np.zeros((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'], 8, 3))
    for i_x in range(dict_user['dim_x']):
        for i_y in range(dict_user['dim_y']):
            for i_z in range(dict_user['dim_z']):
                dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z,:,:] = [[  (i_x)/dict_user['dim_x'],     i_y/dict_user['dim_y'],     i_z/dict_user['dim_z']],\
                                                                        [(i_x+1)/dict_user['dim_x'],     i_y/dict_user['dim_y'],     i_z/dict_user['dim_z']],\
                                                                        [(i_x+1)/dict_user['dim_x'], (i_y+1)/dict_user['dim_y'],     i_z/dict_user['dim_z']],\
                                                                        [  (i_x)/dict_user['dim_x'], (i_y+1)/dict_user['dim_y'],     i_z/dict_user['dim_z']],\
                                                                        [  (i_x)/dict_user['dim_x'],     i_y/dict_user['dim_y'], (i_z+1)/dict_user['dim_z']],\
                                                                        [(i_x+1)/dict_user['dim_x'],     i_y/dict_user['dim_y'], (i_z+1)/dict_user['dim_z']],\
                                                                        [(i_x+1)/dict_user['dim_x'], (i_y+1)/dict_user['dim_y'], (i_z+1)/dict_user['dim_z']],\
                                                                        [  (i_x)/dict_user['dim_x'], (i_y+1)/dict_user['dim_y'], (i_z+1)/dict_user['dim_z']]]
    # save as the undeformed configuration
    dict_sample['M_L_coord_cubeMesh0'] = dict_sample['M_L_coord_cubeMesh'].copy()
    
    # initialization of an unit volume
    dict_sample['v_unit'] = 1

    # initialize the strain and stress maps
    dict_sample['L_eps'] = np.zeros((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'], 6))
    dict_sample['L_sig'] = np.zeros((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'], 6))

    return