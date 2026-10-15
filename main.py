#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import shutil, os, skfmm, time
import numpy as np
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt

# from own code
from parameters import get_parameters
from lib_numpy_to_vtk import write_vtk_structured_points, write_multiple_vtk_structured_points
from lib_ic import generateIC
from lib_reconstruct_displacement_brisard import reconstruct_displacement, reconstruct_displacement_unknownStressPolarization
from lib_construction_new_map import construction_new_map

# load the FFT solver
from fft_solver import adaptive_eyre_milton_thermoporoelasticity
solver = adaptive_eyre_milton_thermoporoelasticity
model = solver.__name__

#-------------------------------------------------------------------------------
#Functions
#-------------------------------------------------------------------------------

def mk_new_dir(foldername):
    '''
    Create a new folder (erase the preexisting, if it exists).
    '''
    if Path(foldername).exists():
        shutil.rmtree(foldername)
    os.mkdir(foldername)

#---------------------------------------

def rebuild_M_sample(dict_user, dict_sample):
    '''
    Rebuild the sample from the list of the mineral maps.
    '''
    # initialization of the sample
    M_sample = np.zeros_like(dict_sample['L_phi'][0])

    # iterate on the mesh
    for i_x in range(dict_user['dim_x']):
        for i_y in range(dict_user['dim_y']):
            for i_z in range(dict_user['dim_z']):
                for i_phi in range(len(dict_sample['L_phi'])):
                    M_sample[i_x, i_y, i_z] = M_sample[i_x, i_y, i_z] + i_phi*dict_sample['L_phi'][i_phi][i_x, i_y, i_z]

    return M_sample

#---------------------------------------

def convert_to_tiff(field):
    '''
    convert a field into a .tiff compatible data
    '''
    if len(field.shape) == 3:
        return np.asarray(np.swapaxes(field, 0, 2), dtype=np.float32)
    else:
        return np.asarray(field, dtype=np.float32)

#-------------------------------------------------------------------------------
# User
#-------------------------------------------------------------------------------

dict_user, dict_sample = get_parameters()

#-------------------------------------------------------------------------------
# Prepare the simulation
#-------------------------------------------------------------------------------

# generate new folders for the simulation
mk_new_dir('vtk')
mk_new_dir('result')
if dict_user['flag_data_tiff'] or dict_user['flag_print_plot_error_interpolation']:
    mk_new_dir('fft')

#-------------------------------------------------------------------------------
# Initial conditions
#-------------------------------------------------------------------------------

# generate an artificial sample
generateIC(dict_user, dict_sample)

# write the sample in a vtk file 
write_multiple_vtk_structured_points(filename='vtk/sample_hyd0.vtk',\
                                        L_arrays=[np.transpose(rebuild_M_sample(dict_user, dict_sample), (2, 1, 0)),\
                                                np.transpose(dict_sample['L_eps'][..., 0], (2, 1, 0)),\
                                                np.transpose(dict_sample['L_eps'][..., 1], (2, 1, 0)),\
                                                np.transpose(dict_sample['L_eps'][..., 2], (2, 1, 0)),\
                                                np.transpose((dict_sample['L_eps'][..., 0]+dict_sample['L_eps'][..., 1]+dict_sample['L_eps'][..., 2]), (2, 1, 0)),\
                                                np.transpose(dict_sample['L_sig'][..., 0], (2, 1, 0)),\
                                                np.transpose(dict_sample['L_sig'][..., 1], (2, 1, 0)),\
                                                np.transpose(dict_sample['L_sig'][..., 2], (2, 1, 0)),\
                                                np.transpose((dict_sample['L_sig'][..., 0]+dict_sample['L_sig'][..., 1]+dict_sample['L_sig'][..., 2])/3, (2, 1, 0)),\
                                                np.transpose(np.sqrt(0.5*((dict_sample['L_sig'][..., 0]-dict_sample['L_sig'][..., 1])**2+(dict_sample['L_sig'][..., 1]-dict_sample['L_sig'][..., 2])**2+(dict_sample['L_sig'][..., 2]-dict_sample['L_sig'][..., 0])**2)), (2, 1, 0)),\
                                                ],\
                                        L_names = ['sample', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_v', 'sig_xx', 'sig_yy', 'sig_zz', 'sig_p', 'sig_VM'],\
                                        spacing=(1/dict_user['dim_x'], 1/dict_user['dim_y'], 1/dict_user['dim_z']),\
                                        origin=(0.5/dict_user['dim_x'], 0.5/dict_user['dim_y'], 0.5/dict_user['dim_z'])) 

# pp for the tracker
phi1_01 = (dict_sample['L_phi'][1].copy() > 0.1)*1

# compute the initial volume of the hydrating phase
dict_sample['L_vol_phi1'].append(np.sum(dict_sample['L_phi'][1])*(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']))
dict_sample['L_vol_phi1_01'].append(np.sum(phi1_01)*(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']))
if dict_user['κ'][0] == 0:
    dict_sample['L_vol_phi1_expected'].append(dict_sample['L_vol_phi1'][-1])
    dict_sample['L_vol_phi1_01_expected'].append(dict_sample['L_vol_phi1_01'][-1])
dict_sample['L_vol_unit'].append(dict_sample['v_unit'])

# compute the hydration degree
dict_sample['L_hyd'].append((dict_sample['v_unit']-1)/(1.5-1)*100)

#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------

# iteration
while dict_sample['i_ite'] <= dict_user['n_ite'] and dict_sample['L_hyd'][-1] < 100:

    # user
    print('\nIteration', dict_sample['i_ite'], '/', dict_user['n_ite'], '\n')

    #---------------------------------------
    # compute the increment of the hydration
    
    dict_sample["dhyd_ite_mask"] = np.ones_like(dict_sample['L_phi'][1])
    #for i_x in range(int(dict_user['dim_x']/2)):
    #    dict_sample["dhyd_ite_mask"][i_x, :, :] = 0
    
    #---------------------------------------
    # Preparation

    # performance computing
    tic_fft = time.perf_counter()

    # prepare the parameters for the FFT resolution
    params = {}
    # read the microstructures
    params["L_φ"] = dict_sample['L_phi']
    # define the constitutive parameters for the simulation
    #params["E"] = np.array([ dict_user['E'][0], dict_user['E'][1]*(1-dict_sample['L_hyd'][-1]/100) + dict_user['E'][2]*dict_sample['L_hyd'][-1]/100])
    params["ν"] = np.array([ dict_user['nu'][0], dict_user['nu'][1]*(1-dict_sample['L_hyd'][-1]/100) + dict_user['nu'][2]*dict_sample['L_hyd'][-1]/100])
    params["μ"] = np.array([ dict_user['μ'][0], dict_user['μ'][1]*(1-dict_sample['L_hyd'][-1]/100) + dict_user['μ'][2]*dict_sample['L_hyd'][-1]/100])
    params["κ"] = np.array([ dict_user['κ'][0], dict_user['κ'][1]*(1-dict_sample['L_hyd'][-1]/100) + dict_user['κ'][2]*dict_sample['L_hyd'][-1]/100])
    params["biot"] = np.array([ dict_user['biot'][0], dict_user['biot'][1]])
    params["alpha"] = np.array([ dict_user['alpha'][0], dict_user['alpha'][1]])
    # apply the mask for the hydration
    params["dhyd_ite_mask"] = dict_sample["dhyd_ite_mask"].copy()
    # define the macroscopic solicitation
    params["p_macro"] = np.array([0.0]) 
    params["T_macro"] = dict_user['dhyd_ite'].copy()
    params["ε_macro"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # define the tolerance for the resolution
    params["e"] = dict_user['tol_fft']

    #---------------------------------------
    # FFT solver

    # call the FFT solver
    grid_sig, grid_eps, grid_stress_polarization = solver(**params)

    # save data in .tiff
    if dict_user['flag_data_tiff']:
        # stress
        for i in range(len(params["ε_macro"])):
            image = convert_to_tiff(grid_sig[..., i])
            io.imsave(f'fft/sigma{i}.tiff', image)
        io.imsave(f'fft/sigma_p.tiff', (convert_to_tiff(grid_sig[..., 0])+convert_to_tiff(grid_sig[..., 1])+convert_to_tiff(grid_sig[..., 2]))/3)
        # strain
        for i in range(len(params["ε_macro"])):
            image = convert_to_tiff(grid_eps[..., i])
            io.imsave(f'fft/epsilon{i}.tiff', image)
        io.imsave(f'fft/epsilon_v.tiff', convert_to_tiff(grid_eps[..., 0])+convert_to_tiff(grid_eps[..., 1])+convert_to_tiff(grid_eps[..., 2]))

    # update the stress and strain maps
    dict_sample['L_sig'] = dict_sample['L_sig'] + grid_sig.copy()
    dict_sample['L_eps'] = dict_sample['L_eps'] + grid_eps.copy()

    # apply the hydration for the volume of control
    dict_sample['v_unit'] = dict_sample['v_unit']*(1+2*dict_user['alpha'][1]*dict_user['dhyd_ite'][0])

    # performance computing
    tac_fft = time.perf_counter()
    dict_sample['L_tac_tic'][0] = dict_sample['L_tac_tic'][0] + tac_fft-tic_fft
    dict_sample['L_tac_tic'][-1] = dict_sample['L_tac_tic'][-1] + tac_fft-tic_fft

    #---------------------------------------
    # Compute the displacement from the deformation

    # user
    print('\nCompute and apply displacement from deformation')

    # performance computing
    tic_def_disp = time.perf_counter()

    # build the material grid
    # required if the function reconstruct_displacement_unknownStressPolarization is used
    grid_mat = np.empty((dict_user['dim_x'], dict_user['dim_y'], dict_user['dim_z'], 2))
    # iterate on the coordinate
    for i_x in range(dict_user['dim_x']):
        for i_y in range(dict_user['dim_y']):
            for i_z in range(dict_user['dim_z']):
                # plan the next work
                if len(dict_sample['L_phi']) > 2:
                    raise ValueError('Check the interpolation of the parameters')
                # iterate on the minerals
                for i_phi in range(1, len(dict_sample['L_phi'])):
                    grid_mat[i_x, i_y, i_z, 0] = params["κ"][0]*(1-dict_sample['L_phi'][i_phi][i_x, i_y, i_z]) + params["κ"][1]*dict_sample['L_phi'][i_phi][i_x, i_y, i_z]
                    grid_mat[i_x, i_y, i_z, 1] = params["μ"][0]*(1-dict_sample['L_phi'][i_phi][i_x, i_y, i_z]) + params["μ"][1]*dict_sample['L_phi'][i_phi][i_x, i_y, i_z]

    # apply a prefactor on reference material
    prefactor = 1000
    # compute displacement from (Brisard, 2017)
    grid_u = reconstruct_displacement_unknownStressPolarization(grid_eps, grid_mat, (1, 1, 1), (prefactor*params["κ"][1], prefactor*params["μ"][1]))

    # seem to have an error for the 3D
    if dict_user['dim_z'] > 1:
        raise ValueError('check that (Brisard, 2017) work in 3D')
    grid_u[:,:,:, 2] = 0

    # save data in .tiff
    if dict_user['flag_data_tiff']:
        print('write data into output files')
        # displacement
        for i in range(3):
            image = convert_to_tiff(grid_u[..., i])
            io.imsave(f'fft/u{i}.tiff', image)
        io.imsave(f'fft/un.tiff', np.sqrt(convert_to_tiff(grid_u[..., 0]**2)+convert_to_tiff(grid_u[..., 1]**2)+convert_to_tiff(grid_u[..., 2]**2)))

    # pp grid_u
    grid_u_pp = np.pad(grid_u, pad_width=((1, 0), (1, 0), (1, 0), (0, 0)), mode='edge')
        
    # write the displacement in a vtk
    write_disps_vtk_structured_points(filename='vtk/disps_hyd'+str(dict_sample['i_ite']-1)+'.vtk',\
                                        u_x=np.transpose(grid_u_pp[..., 0], (2, 1, 0)),\
                                        u_y=np.transpose(grid_u_pp[..., 1], (2, 1, 0)),\
                                        u_z=np.transpose(grid_u_pp[..., 2], (2, 1, 0)),\
                                        spacing=(1/dict_user['dim_x'], 1/dict_user['dim_y'], 1/dict_user['dim_z']),\
                                        #origin=( 1/dict_user['dim_x'], 1/dict_user['dim_y'], 1/dict_user['dim_z'])) 
                                        origin=( 0, 0, 0)) 

    #---------------------------------------
    # Apply deformation

    # prepare the next configuration
    dict_sample['M_L_coord_cubeMesh'] = np.zeros_like(dict_sample['M_L_coord_cubeMesh'])
    # iteration over coordinates x
    for i_x in range(dict_user['dim_x']):
        # iteration over coordinates y
        for i_y in range(dict_user['dim_y']):
            # iteration over coordinates z
            for i_z in range(dict_user['dim_z']):          

                # update the coordination of the mesh
                dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z, :, :] = [np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][0]) + np.array(grid_u[i_x-1, i_y-1, i_z-1]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][1]) + np.array(grid_u[  i_x, i_y-1, i_z-1]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][2]) + np.array(grid_u[  i_x,   i_y, i_z-1]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][3]) + np.array(grid_u[i_x-1,   i_y, i_z-1]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][4]) + np.array(grid_u[i_x-1, i_y-1,   i_z]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][5]) + np.array(grid_u[  i_x, i_y-1,   i_z]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][6]) + np.array(grid_u[  i_x,   i_y,   i_z]),\
                                                                          np.array(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z][7]) + np.array(grid_u[i_x-1,   i_y,   i_z])]

                # prepare a post-processing (norm of u vs the mesh)
                L_u = [np.linalg.norm(grid_u[i_x-1, i_y-1, i_z-1]), np.linalg.norm(grid_u[i_x, i_y-1, i_z-1]), np.linalg.norm(grid_u[i_x, i_y, i_z-1]), np.linalg.norm(grid_u[i_x, i_y, i_z-1]),\
                        np.linalg.norm(grid_u[i_x-1, i_y-1,   i_z]), np.linalg.norm(grid_u[i_x, i_y-1,   i_z]), np.linalg.norm(grid_u[i_x, i_y,   i_z]), np.linalg.norm(grid_u[i_x, i_y, i_z])]
                # add a condition for the pp
                if dict_user['dim_z'] > 1:
                    raise ValueError('Modify the pp (check the mesh size in 3D)')
                # find min, mean, max
                if i_x == 0 and i_y == 0 and i_z == 0: # init
                    un_mesh_min = min(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                    un_mesh_mean = np.mean(L_u)/(dict_user['dim_x']*dict_user['dim_y']*dict_user['dim_z'])  / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                    un_mesh_max = max(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                else : # current
                    if min(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y']) < un_mesh_min:
                        un_mesh_min = min(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                    un_mesh_mean = un_mesh_mean + np.mean(L_u)/(dict_user['dim_x']*dict_user['dim_y']*dict_user['dim_z'])  / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                    if un_mesh_max < max(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y']):
                        un_mesh_max = max(L_u) / (1/2*1/dict_user['dim_x'] + 1/2*1/dict_user['dim_y'])
                
    # user
    if dict_user['flag_print_un_mesh']:
        print('Ratio |u|/s_mesh :', un_mesh_min, '(min),', un_mesh_mean, '(mean),', un_mesh_max, '(max)')

    # performance computing
    tac_def_disp = time.perf_counter()
    dict_sample['L_tac_tic'][1] = dict_sample['L_tac_tic'][1] + tac_def_disp-tic_def_disp
    dict_sample['L_tac_tic'][-1] = dict_sample['L_tac_tic'][-1] + tac_def_disp-tic_def_disp

    #---------------------------------------
    #  Verify the volumetric deformation

    if dict_user['flag_check_eps_vol']:
        # initialize the pp
        m_vol_eps = 0
        # iteration over coordinates x
        for i_x in range(dict_user['dim_x']):
            # iteration over coordinates y
            for i_y in range(dict_user['dim_y']):
                # iteration over coordinates z
                for i_z in range(dict_user['dim_z']):   
                    m_vol_eps = m_vol_eps + (grid_eps[i_x, i_y, i_z, 0] + grid_eps[i_x, i_y, i_z, 1] + grid_eps[i_x, i_y, i_z, 2])*dict_sample['L_phi'][1][i_x, i_y, i_z]/np.sum(dict_sample['L_phi'][1])

        # user
        print('\nVolumetric deformation (eps - mean) in the hydrating phase :', m_vol_eps)

        # import python library
        import trimesh
        # define the faces of a cube for trimesh
        faces_cube = np.array([
            [0, 1, 2, 3],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [4, 5, 6, 7]
        ])
        # initialize the pp
        m_vol_disp = 0
        # iteration over coordinates x
        for i_x in range(dict_user['dim_x']):
            # iteration over coordinates y
            for i_y in range(dict_user['dim_y']):
                # iteration over coordinates z
                for i_z in range(dict_user['dim_z']):   
                    # mesh the mesh of the cube
                    mesh_cubeMesh = trimesh.Trimesh(vertices=dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z, :, :], faces=faces_cube)
                    mesh_cubeMesh.fix_normals() # fix the orientation of the normals
                    m_vol_disp = m_vol_disp + (mesh_cubeMesh.volume-(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']))/(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z'])*dict_sample['L_phi'][1][i_x, i_y, i_z]/np.sum(dict_sample['L_phi'][1])

        # user
        print('Volume deformation (disp - mean) in the hydrating phase :', m_vol_disp)
        if dict_user["κ"][0] == 0:
            print('Volume deformation (expected) : ', 2*dict_user['alpha'][1]*dict_user['dhyd_ite'][0])

    #---------------------------------------
    # Construction of the new map

    # user
    print('\ninterpolate the new configuration (undeformed) from the previous mesh (deformed)')

    # performance computing
    tic_int_mesh = time.perf_counter()

    # Contruction the new map from the previous map.
    construction_new_map(dict_user, dict_sample)

    # performance computing
    tac_int_mesh = time.perf_counter()
    dict_sample['L_tac_tic'][2] = dict_sample['L_tac_tic'][2] + tac_int_mesh-tic_int_mesh
    dict_sample['L_tac_tic'][-1] = dict_sample['L_tac_tic'][-1] + tac_int_mesh-tic_int_mesh

    #---------------------------------------
    # output

    # write the sample in a vtk file
    write_multiple_vtk_structured_points(filename='vtk/sample_hyd'+str(dict_sample['i_ite'])+'.vtk',\
                                         L_arrays=[np.transpose(rebuild_M_sample(dict_user, dict_sample), (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_eps'][..., 0], (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_eps'][..., 1], (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_eps'][..., 2], (2, 1, 0)),\
                                                   np.transpose((dict_sample['L_eps'][..., 0]+dict_sample['L_eps'][..., 1]+dict_sample['L_eps'][..., 2]), (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_sig'][..., 0], (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_sig'][..., 1], (2, 1, 0)),\
                                                   np.transpose(dict_sample['L_sig'][..., 2], (2, 1, 0)),\
                                                   np.transpose((dict_sample['L_sig'][..., 0]+dict_sample['L_sig'][..., 1]+dict_sample['L_sig'][..., 2])/3, (2, 1, 0)),\
                                                   np.transpose(np.sqrt(0.5*((dict_sample['L_sig'][..., 0]-dict_sample['L_sig'][..., 1])**2+(dict_sample['L_sig'][..., 1]-dict_sample['L_sig'][..., 2])**2+(dict_sample['L_sig'][..., 2]-dict_sample['L_sig'][..., 0])**2)), (2, 1, 0)),\
                                                   ],\
                                         L_names = ['sample', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_v', 'sig_xx', 'sig_yy', 'sig_zz', 'sig_p', 'sig_VM'],\
                                         spacing=(1/dict_user['dim_x'], 1/dict_user['dim_y'], 1/dict_user['dim_z']),\
                                         origin=(0.5/dict_user['dim_x'], 0.5/dict_user['dim_y'], 0.5/dict_user['dim_z']))  

    # pp for the tracker
    phi1_01 = (dict_sample['L_phi'][1].copy() > 0.1)*1
    
    # track the volume of the hydrating phase
    dict_sample['L_vol_phi1'].append(np.sum(dict_sample['L_phi'][1])*(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']))
    dict_sample['L_vol_phi1_01'].append(np.sum(phi1_01)*(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']))
    if dict_user['κ'][0] == 0:
        dict_sample['L_vol_phi1_expected'].append((1 + 2*dict_user['alpha'][1]*dict_user['dhyd_ite'][0])*dict_sample['L_vol_phi1_expected'][-1])
        dict_sample['L_vol_phi1_01_expected'].append((1 + 2*dict_user['alpha'][1]*dict_user['dhyd_ite'][0])*dict_sample['L_vol_phi1_01_expected'][-1])
    dict_sample['L_vol_unit'].append(dict_sample['v_unit'])

    # track the hydration
    dict_sample['L_hyd'].append((dict_sample['v_unit']-1)/(1.5-1)*100)

    #---------------------------------------
    # preparation of the next step

    # performance computing
    tic_prep_step = time.perf_counter()

    if dict_user['method_construction_new_map'] == 'trimesh':
        # plan the next work
        if len(dict_sample['L_phi']) > 2:
            raise ValueError('Check the update of the interface')
        # initialization of the interface
        dict_sample['L_i_XYZ_interface'] = []
        # compute the signed distance function
        M_sdf = skfmm.distance(dict_sample['L_phi'][1]-0.5, dx = np.array([1/dict_user['dim_x'], 1/dict_user['dim_y'], 1/dict_user['dim_z']]))
        # iterate on the mesh
        for i_x in range(dict_user['dim_x']):
            for i_y in range(dict_user['dim_y']):
                for i_z in range(dict_user['dim_z']):
                    # update the interface 
                    if abs(M_sdf[i_x, i_y, i_z]) < dict_user['w_int']:
                        dict_sample['L_i_XYZ_interface'].append([i_x, i_y, i_z]) 

    # change the parameters to consider the initial state as a reference ??


    # incrementation
    dict_sample['i_ite'] = dict_sample['i_ite'] + 1

    # performance computing
    tac_prep_step = time.perf_counter()
    dict_sample['L_tac_tic'][3] = dict_sample['L_tac_tic'][3] + tac_prep_step-tic_prep_step
    dict_sample['L_tac_tic'][-1] = dict_sample['L_tac_tic'][-1] + tac_prep_step-tic_prep_step

#-------------------------------------------------------------------------------
# close
#-------------------------------------------------------------------------------

# user
print('\nEnd of the simulation')

# iteration
# volume evolution of the hydrating phase
'''fig, ax = plt.subplots(1,1, figsize=(16, 9))
ax.plot(dict_sample['L_vol_phi1'], linewidth=6, color='r')
if dict_user['κ'][0] == 0:
    ax.plot(dict_sample['L_vol_phi1_expected'], linewidth=4, color='r', linestyle='dashed')
ax.set_xlabel('iterations (-)', fontsize=25)
ax.set_ylabel('volume of hydrating phase (-)', fontsize=25, color='r')
ax.tick_params(axis='both', width=4, length=6, labelsize=20)
# volume evolution of the unit control
axb = ax.twinx()
axb.plot(dict_sample['L_vol_unit'], linewidth=6, color='b')
axb.set_ylabel('volume of control (-)', fontsize=25, color='b')
axb.tick_params(axis='y', width=4, length=6, labelsize=20)
fig.tight_layout()
fig.savefig('result/fig_iteration_volHydPhi.png')
plt.close(fig)'''

# iteration
# volume evolution of the hydrating phase versus the expected volume
'''fig, ax = plt.subplots(1,1, figsize=(16, 9))
ax.plot(dict_sample['L_vol_phi1'], linewidth=6, color='r', label='computed')
if dict_user['κ'][0] == 0:
    ax.plot(dict_sample['L_vol_phi1_expected'], linewidth=6, color='k', label='expected', linestyle='dashed')
ax.set_xlabel('iterations (-)', fontsize=25)
ax.set_ylabel('volume of hydrating phase (-)', fontsize=25, color='r')
ax.tick_params(axis='both', width=4, length=6, labelsize=20)
ax.legend(fontsize=25)
fig.tight_layout()
fig.savefig('result/fig_iteration_volHydPhi_expected.png')
plt.close(fig)'''

# hydration
# volume evolution of the hydrating phase
fig, ax = plt.subplots(1,1, figsize=(16, 9))
ax.plot(dict_sample['L_hyd'], dict_sample['L_vol_phi1'], linewidth=6, color='r', label='computed')
if dict_user['κ'][0] == 0:
    ax.plot(dict_sample['L_hyd'], dict_sample['L_vol_phi1_expected'], linewidth=4, color='k', linestyle='dashed', label='expected')
ax.set_xlabel('hydration (%)', fontsize=25)
ax.set_ylabel('volume of hydrating phase (-)', fontsize=25)
ax.tick_params(axis='both', width=4, length=6, labelsize=20)
ax.legend(fontsize=25)
# volume evolution of the unit control
#axb = ax.twinx()
#axb.plot(dict_sample['L_hyd'], dict_sample['L_vol_unit'], linewidth=6, color='b')
#axb.tick_params(axis='y', width=4, length=6, labelsize=20)
#axb.set_ylabel('volume of control (-)', fontsize=25, color='b')
fig.tight_layout()
fig.savefig('result/fig_hydration_volHydPhi.png')
plt.close(fig)

# time performance
fig, ax = plt.subplots(1,1,figsize=(16,9))
ax.pie(
    dict_sample['L_tac_tic'][:-1],
    labels=['fft', 'def_disp', 'int_mesh', 'prep_step'],
    colors=[  'r',        'b',        'g',         'm'],
    autopct='%1.1f%%',  # print percentages with 1 decimale
)
ax.set_title("Distribution of the time "+str(int(dict_sample['L_tac_tic'][-1]))+" s", fontsize=25)
fig.tight_layout()
fig.savefig('result/fig_distribution_time.png')


# user
if dict_user['κ'][0] == 0:
    print('\n(simulated-expected)/expected*100')
    #print('simulated :',dict_sample['L_vol_phi1'])
    #print('expected :',dict_sample['L_vol_phi1_expected'])
    print('(phi) :', (np.array(dict_sample['L_vol_phi1'] - np.array(dict_sample['L_vol_phi1_expected'])))/np.array(dict_sample['L_vol_phi1_expected'])*100)
    print('(phi > 0.1):', (np.array(dict_sample['L_vol_phi1_01'] - np.array(dict_sample['L_vol_phi1_01_expected'])))/np.array(dict_sample['L_vol_phi1_01_expected'])*100)


# for Alex
print()
print('Is the cristallization induce some stress in the hydrating phase ?')
print('maybe yes as there is cracks')
