"""
Remap a mesh (undeformed) from a previous one (deformed)

Different strategies are employed.
"""
#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

import trimesh
from skimage import io
import numpy as np
import pyvista as pv

#-------------------------------------------------------------------------------
#Functions
#-------------------------------------------------------------------------------

def convert_to_tiff(field):
    '''
    convert a field into a .tiff compatible data
    '''
    if len(field.shape) == 3:
        return np.asarray(np.swapaxes(field, 0, 2), dtype=np.float32)
    else:
        return np.asarray(field, dtype=np.float32)

#---------------------------------------
    
def construction_new_map(dict_user, dict_sample):
    '''
    Contruction the new map from the previous map.

    A selection is made here, depending on user input
    '''
    # based on the use of trimesh
    if dict_user['method_construction_new_map'] == 'trimesh':
        construction_new_map_trimesh(dict_user, dict_sample)
    # based on the use of pyvista
    elif dict_user['method_construction_new_map'] == 'pyvista':
        construction_new_map_pyvista(dict_user, dict_sample)

    # method unavailable
    else : 
        raise ValueError('Select a method available for remaping')

#---------------------------------------

def construction_new_map_trimesh(dict_user, dict_sample):
    '''
    Contruction the new map from the previous map, based on the use of trimesh.
    '''
    # define the faces of a cube for trimesh
    faces_cube = np.array([
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [4, 5, 6, 7]
    ])

    # iterate on the phases
    for phi in dict_sample['L_phi'][1:]:
        # save previous map (index 0)
        phi0 = phi.copy()

        # prepare a pp of the result
        L_S_V_overlap = []
        M_error = np.ones_like(phi0)

        # iterate over the interface of the undeformed mesh (next one) 
        for i_XYZ_interface in dict_sample['L_i_XYZ_interface']:
            # user 
            if dict_sample['L_i_XYZ_interface'].index(i_XYZ_interface) % 300 == 0:
                print(dict_sample['L_i_XYZ_interface'].index(i_XYZ_interface)+1, '/', len(dict_sample['L_i_XYZ_interface']), '('+str(round((dict_sample['L_i_XYZ_interface'].index(i_XYZ_interface)+1)/len(dict_sample['L_i_XYZ_interface'])*100,2))+' %)')

            # read the coordinates
            i_x_next = i_XYZ_interface[0]
            i_y_next = i_XYZ_interface[1]
            i_z_next = i_XYZ_interface[2]
                            
            # mesh the undeformed cube
            mesh_cubeMesh_next = trimesh.Trimesh(vertices=dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, :, :], faces=faces_cube)
            mesh_cubeMesh_next.fix_normals() # fix the orientation of the normals

            # read the limits of the cube (undeformed)
            x_min_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 0, 0]
            x_max_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 1, 0]
            y_min_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 0, 1]
            y_max_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 2, 1]
            z_min_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 0, 2] 
            z_max_cube_next = dict_sample['M_L_coord_cubeMesh0'][i_x_next, i_y_next, i_z_next, 4, 2]

            # determination of the interpolation limits
            i_x_m = max(                   0, i_x_next-2)
            i_x_p = min(dict_user['dim_x']-1, i_x_next+2)
            i_y_m = max(                   0, i_y_next-2)
            i_y_p = min(dict_user['dim_y']-1, i_y_next+2)
            i_z_m = max(                   0, i_z_next-2)
            i_z_p = min(dict_user['dim_z']-1, i_z_next+2)
            # could be smarter by limiting the research based on the maximum deformation (~ un_mesh)

            # initialization of the estimation of the value phi
            phi_deformed = 0
            S_V_overlap = 0

            # iterate on the mesh in the interpolation limits
            for i_x_prev in range(i_x_m, i_x_p+1):
                for i_y_prev in range(i_y_m, i_y_p+1):
                    for i_z_prev in range(i_z_m, i_z_p+1):

                        # read the limits of the previous cube (deformed)                    
                        x_min_cube_prev = min([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 0, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 3, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 4, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 7, 0]])
                        x_max_cube_prev = max([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 1, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 2, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 5, 0], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 6, 0]])
                        y_min_cube_prev = min([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 0, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 1, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 4, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 5, 1]])
                        y_max_cube_prev = min([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 2, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 3, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 6, 1], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 7, 1]])
                        z_min_cube_prev = min([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 0, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 1, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 2, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 3, 2]]) 
                        z_max_cube_prev = max([dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 4, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 5, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 6, 2], dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, 7, 2]])

                        # determine if the previous mesh (deformed) is in the next mesh (undeformed)
                        if (((x_min_cube_next <= x_min_cube_prev and x_min_cube_prev <= x_max_cube_next)  or  \
                                (x_min_cube_next <= x_max_cube_prev and x_max_cube_prev <= x_max_cube_next)) and \
                            ((y_min_cube_next <= y_min_cube_prev and y_min_cube_prev <= y_max_cube_next)  or  \
                                (y_min_cube_next <= y_max_cube_prev and y_max_cube_prev <= y_max_cube_next)) and \
                            ((z_min_cube_next <= z_min_cube_prev and z_min_cube_prev <= z_max_cube_next)  or  \
                                (z_min_cube_next <= z_max_cube_prev and z_max_cube_prev <= z_max_cube_next))) or \
                            (((x_min_cube_prev <= x_min_cube_next and x_min_cube_next <= x_max_cube_prev)  or  \
                                (x_min_cube_prev <= x_max_cube_next and x_max_cube_next <= x_max_cube_prev)) and \
                            ((y_min_cube_prev <= y_min_cube_next and y_min_cube_next <= y_max_cube_prev)  or  \
                                (y_min_cube_prev <= y_max_cube_next and y_max_cube_next <= y_max_cube_prev)) and \
                            ((z_min_cube_prev <= z_min_cube_next and z_min_cube_next <= z_max_cube_prev)  or  \
                                (z_min_cube_prev <= z_max_cube_next and z_max_cube_next <= z_max_cube_prev))) :

                            # mesh the cubes
                            mesh_cubeMesh_prev = trimesh.Trimesh(vertices=dict_sample['M_L_coord_cubeMesh'][i_x_prev, i_y_prev, i_z_prev, :, :], faces=faces_cube)
                            mesh_cubeMesh_prev.fix_normals()

                            # compute the intersection
                            inter_cube_cube = trimesh.boolean.intersection([mesh_cubeMesh_prev, mesh_cubeMesh_next], engine="manifold")

                            # verify that the intersection exists and compute its volume
                            if inter_cube_cube is None or inter_cube_cube.is_empty:
                                V_overlap = 0.0

                            else:
                                V_overlap = inter_cube_cube.volume

                            # compute the weighted sum (for a mean)
                            phi_deformed = phi_deformed + V_overlap*phi0[i_x_prev, i_y_prev, i_z_prev]
                            S_V_overlap = S_V_overlap + V_overlap

            # compute the mean
            if S_V_overlap > 0:
                phi_deformed = phi_deformed/S_V_overlap
            else :
                phi_deformed = phi0[i_x_prev, i_y_prev, i_z_prev].copy() 
            # save it
            phi[i_x_next, i_y_next, i_z_next] = phi_deformed

            # pp the data
            L_S_V_overlap.append(S_V_overlap)
            if S_V_overlap < 0.5*(1/dict_user['dim_x']*1/dict_user['dim_y']*1/dict_user['dim_z']):
                M_error[i_x_next, i_y_next, i_z_next] = S_V_overlap

        # user
        if dict_user['flag_print_plot_error_interpolation']:
            print('(min)', min(L_S_V_overlap), '(mean)', np.mean(L_S_V_overlap), '(max)', max(L_S_V_overlap))
            image = convert_to_tiff(M_error)
            io.imsave(f'fft/error.tiff', image)

#---------------------------------------

def construction_new_map_pyvista(dict_user, dict_sample):
    '''
    Contruction the new map from the previous map, based on the use of pyvista.
    
    source: https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.interpolate
    '''
    # initialization of the coordinates for the undeformed mesh
    coord_x0 = []
    coord_y0 = []
    coord_z0 = []
    # iterate on the mesh
    for i_x in range(dict_user['dim_x']):
            for i_y in range(dict_user['dim_y']):
                for i_z in range(dict_user['dim_z']):
                    coord_x0.append(np.mean(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z, :, 0]))
                    coord_y0.append(np.mean(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z, :, 1]))
                    coord_z0.append(np.mean(dict_sample['M_L_coord_cubeMesh0'][i_x, i_y, i_z, :, 2]))   
    # convert to numpy array
    coord_x0 = np.array(coord_x0)
    coord_y0 = np.array(coord_y0)
    coord_z0 = np.array(coord_z0)

    # build the pyvista mesh
    nodes0 = np.column_stack((coord_x0, coord_y0, coord_z0,))
    mesh0 = pv.PolyData(nodes0)

    if len(dict_sample['L_phi']) > 2:
        raise ValueError('The remaping is only available for two phases')

    # iterate on the phases
    for phi in dict_sample['L_phi'][1:]:
        # initialization of the coordinates for the deformed mesh
        coord_x = []
        coord_y = []
        coord_z = []
        value = []
        sig00 = []
        sig11 = []
        sig22 = []
        sig01 = []
        sig02 = []
        sig12 = []
        eps00 = []
        eps11 = []
        eps22 = []
        eps01 = []
        eps02 = []
        eps12 = []
        # iterate on the mesh
        for i_x in range(dict_user['dim_x']):
                for i_y in range(dict_user['dim_y']):
                    for i_z in range(dict_user['dim_z']):
                        coord_x.append(np.mean(dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z, :, 0]))
                        coord_y.append(np.mean(dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z, :, 1]))
                        coord_z.append(np.mean(dict_sample['M_L_coord_cubeMesh'][i_x, i_y, i_z, :, 2]))   
                        value.append(dict_sample['L_phi'][1][i_x, i_y, i_z]) # for the first mineral
                        sig00.append(dict_sample['L_sig'][i_x, i_y, i_z, 0])
                        sig11.append(dict_sample['L_sig'][i_x, i_y, i_z, 1])
                        sig22.append(dict_sample['L_sig'][i_x, i_y, i_z, 2])
                        sig01.append(dict_sample['L_sig'][i_x, i_y, i_z, 3])
                        sig02.append(dict_sample['L_sig'][i_x, i_y, i_z, 4])
                        sig12.append(dict_sample['L_sig'][i_x, i_y, i_z, 5])
                        eps00.append(dict_sample['L_eps'][i_x, i_y, i_z, 0])
                        eps11.append(dict_sample['L_eps'][i_x, i_y, i_z, 1])
                        eps22.append(dict_sample['L_eps'][i_x, i_y, i_z, 2])
                        eps01.append(dict_sample['L_eps'][i_x, i_y, i_z, 3])
                        eps02.append(dict_sample['L_eps'][i_x, i_y, i_z, 4])
                        eps12.append(dict_sample['L_eps'][i_x, i_y, i_z, 5])
        # convert to numpy array
        coord_x = np.array(coord_x)
        coord_y = np.array(coord_y)
        coord_z = np.array(coord_z)
        value = np.array(value)
        sig00 = np.array(sig00)
        sig11 = np.array(sig11)
        sig22 = np.array(sig22)
        sig01 = np.array(sig01)
        sig02 = np.array(sig02)
        sig12 = np.array(sig12)
        eps00 = np.array(eps00)
        eps11 = np.array(eps11)
        eps22 = np.array(eps22)
        eps01 = np.array(eps01)
        eps02 = np.array(eps02)
        eps12 = np.array(eps12)

        # build the pyvista mesh
        nodes = np.column_stack((coord_x, coord_y, coord_z,))
        mesh = pv.PolyData(nodes)
        # assign the values to the mesh
        mesh.point_data['phi'] = value
        mesh.point_data['sig00'] = sig00
        mesh.point_data['sig11'] = sig11
        mesh.point_data['sig22'] = sig22
        mesh.point_data['sig01'] = sig01
        mesh.point_data['sig02'] = sig02
        mesh.point_data['sig12'] = sig12
        mesh.point_data['eps00'] = eps00
        mesh.point_data['eps11'] = eps11
        mesh.point_data['eps22'] = eps22
        mesh.point_data['eps01'] = eps01
        mesh.point_data['eps02'] = eps02
        mesh.point_data['eps12'] = eps12

        # determine the parameters for the interpolation
        radius_inter = 2*(1/dict_user['dim_x'] + 1/dict_user['dim_y'])/2
        # interpolate the values from the deformed mesh to the undeformed mesh
        mesh0_interpolated = mesh0.interpolate(mesh, radius=radius_inter, sharpness=2)

        # iterate on the mesh
        for i_x in range(dict_user['dim_x']):
                for i_y in range(dict_user['dim_y']):
                    for i_z in range(dict_user['dim_z']):
                        # assign the interpolated values to the phi field
                        phi[i_x, i_y, i_z] = mesh0_interpolated.point_data['phi'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        # update the stress maps
                        dict_sample['L_sig'][i_x, i_y, i_z, 0] = mesh0_interpolated.point_data['sig00'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_sig'][i_x, i_y, i_z, 1] = mesh0_interpolated.point_data['sig11'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_sig'][i_x, i_y, i_z, 2] = mesh0_interpolated.point_data['sig22'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_sig'][i_x, i_y, i_z, 3] = mesh0_interpolated.point_data['sig01'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_sig'][i_x, i_y, i_z, 4] = mesh0_interpolated.point_data['sig02'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_sig'][i_x, i_y, i_z, 5] = mesh0_interpolated.point_data['sig12'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        # update the strain maps
                        dict_sample['L_eps'][i_x, i_y, i_z, 0] = mesh0_interpolated.point_data['eps00'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_eps'][i_x, i_y, i_z, 1] = mesh0_interpolated.point_data['eps11'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_eps'][i_x, i_y, i_z, 2] = mesh0_interpolated.point_data['eps22'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_eps'][i_x, i_y, i_z, 3] = mesh0_interpolated.point_data['eps01'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_eps'][i_x, i_y, i_z, 4] = mesh0_interpolated.point_data['eps02'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]
                        dict_sample['L_eps'][i_x, i_y, i_z, 5] = mesh0_interpolated.point_data['eps12'][i_x*dict_user['dim_y']*dict_user['dim_z'] + i_y*dict_user['dim_z'] + i_z]