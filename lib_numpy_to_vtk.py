"""Write a legacy VTK STRUCTURED_POINTS file from a NumPy array.

Usage:
  - Import `write_vtk_structured_points` and call with a 2D or 3D NumPy array.
  - The array should be shaped (Z, Y, X) for 3D or (Y, X) for 2D.

This writes an ASCII legacy .vtk file which is readable by ParaView and other VTK tools.
"""
from __future__ import annotations
import array
import numpy as np
from typing import Tuple


_DTYPE_MAP = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
    np.dtype('int32'): 'int',
    np.dtype('int64'): 'long',
    np.dtype('uint8'): 'unsigned_char',
    np.dtype('int8'): 'char',
}


def _map_dtype(dtype: np.dtype) -> str:
    dt = np.dtype(dtype)
    if dt in _DTYPE_MAP:
        return _DTYPE_MAP[dt]
    if np.issubdtype(dt, np.floating):
        return 'double'
    if np.issubdtype(dt, np.integer):
        return 'int'
    raise ValueError(f'Unsupported dtype: {dtype}')

#-------------------------------------------------------------------------------

def write_vtk_structured_points(
    filename: str,
    array: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scalar_name: str = 'scalars',
    binary: bool = False,
):
    """Write `array` to a legacy ASCII VTK file as STRUCTURED_POINTS.

    Parameters
    - filename: target .vtk filename
    - array: 2D (Y,X) or 3D (Z,Y,X) NumPy array. Use shape (Z,Y,X) for stacks.
    - spacing: voxel spacing (sx, sy, sz)
    - origin: origin coordinates (ox, oy, oz)
    - scalar_name: name for the scalar field

    The data are written in ASCII with X varying fastest (C-order ravel).
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('array must be a NumPy ndarray')

    if array.ndim == 2:
        # interpret as (Y, X) -> make (Z=1, Y, X)
        arr = array[np.newaxis, :, :]
    elif array.ndim == 3:
        raise ValueError('3D array must be (Z,Y,X)')
    else:
        raise ValueError('array must be 2D or 3D (Y,X) or (Z,Y,X)')

    zdim, ydim, xdim = arr.shape
    npoints = xdim * ydim * zdim

    # VTK expects X fastest; for arr shaped (Z,Y,X), ravel(order='C') does that
    flat = arr.ravel(order='C')

    vtk_dtype = _map_dtype(arr.dtype)

    mode = 'wb' if binary else 'w'

    header_lines = [
        '# vtk DataFile Version 3.0',
        'VTK output from numpy_to_vtk.py',
        'BINARY' if binary else 'ASCII',
        'DATASET STRUCTURED_POINTS',
        f'DIMENSIONS {xdim} {ydim} {zdim}',
        f'ORIGIN {origin[0]} {origin[1]} {origin[2]}',
        f'SPACING {spacing[0]} {spacing[1]} {spacing[2]}',
        f'POINT_DATA {npoints}',
        f'SCALARS {scalar_name} {vtk_dtype}',
        'LOOKUP_TABLE default',
    ]

    if binary:
        # For binary VTK legacy files, data must be big-endian (network order)
        # Map to numpy dtype strings with big-endian specifiers
        if arr.dtype == np.dtype('float32'):
            be_dtype = '>f4'
        elif arr.dtype == np.dtype('float64'):
            be_dtype = '>f8'
        elif arr.dtype == np.dtype('int32'):
            be_dtype = '>i4'
        elif arr.dtype == np.dtype('int64'):
            be_dtype = '>i8'
        elif arr.dtype == np.dtype('uint8'):
            be_dtype = '>u1'
        elif arr.dtype == np.dtype('int8'):
            be_dtype = '>i1'
        else:
            # Fallback: choose big-endian of same kind/size if possible
            be_dtype = '>' + str(arr.dtype)

        with open(filename, mode) as f:
            for line in header_lines:
                f.write((line + '\n').encode('utf-8'))

            # write binary blob
            data_bytes = arr.astype(be_dtype).ravel(order='C').tobytes()
            f.write(data_bytes)
    else:
        with open(filename, mode) as f:
            for line in header_lines:
                f.write(line + '\n')

            # Write data values, keep lines reasonably short
            per_line = 9
            for i in range(0, flat.size, per_line):
                chunk = flat[i : i + per_line]
                f.write(' '.join(map(str, chunk.tolist())) + '\n')

#-------------------------------------------------------------------------------

def write_multiple_vtk_structured_points(
    filename: str,
    L_arrays: list[np.ndarray],
    L_names: list[str],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Write `array` to a legacy ASCII VTK file as STRUCTURED_POINTS.

    Parameters
    - filename: target .vtk filename
    - L_array: 2D (Y,X) or 3D (Z,Y,X) NumPy arrays. Use shape (Z,Y,X) for stacks.
    - L_scalar_name: names for the scalar fields
    - spacing: voxel spacing (sx, sy, sz)
    - origin: origin coordinates (ox, oy, oz)

    The data are written in ASCII with X varying fastest (C-order ravel).
    """
    header_lines = [
        '# vtk DataFile Version 3.0',
        'VTK output from numpy_to_vtk.py',
        'ASCII',
        'DATASET STRUCTURED_POINTS',
    ]
    # write the header
    with open(filename, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')


    for i_array in range(len(L_arrays)):
        # read the array and check the dimension
        if L_arrays[i_array].ndim == 2:
            # interpret as (Y, X) -> make (Z=1, Y, X)
            arr = L_arrays[i_array][np.newaxis, :, :]
        elif L_arrays[i_array].ndim == 3:
            arr = L_arrays[i_array]
        else :
            raise ValueError('array must be 2D or 3D (Y,X) or (Z,Y,X)')

        if i_array == 0:
            # read the dimensions
            zdim, ydim, xdim = arr.shape
            npoints = xdim * ydim * zdim
            header_lines_extend=[
                f'DIMENSIONS {xdim} {ydim} {zdim}',
                f'ORIGIN {origin[0]} {origin[1]} {origin[2]}',
                f'SPACING {spacing[0]} {spacing[1]} {spacing[2]}',
                f'POINT_DATA {npoints}',
            ]
            # write the header
            with open(filename, 'a') as f:
                for line in header_lines_extend:
                    f.write(line + '\n')

        # VTK expects X fastest; for arr shaped (Z,Y,X), ravel(order='C') does that
        flat = arr.ravel(order='C')

        vtk_dtype = _map_dtype(arr.dtype)

        header_lines_array = [
        f'SCALARS {L_names[i_array]} {vtk_dtype}',
        'LOOKUP_TABLE default',]
        
        with open(filename, 'a') as f:
            # write the header
            for line in header_lines_array:
                f.write(line + '\n')

            # Write data values, keep lines reasonably short
            per_line = 9
            for i in range(0, flat.size, per_line):
                chunk = flat[i : i + per_line]
                f.write(' '.join(map(str, chunk.tolist())) + '\n')


if __name__ == '__main__':
    # demo: create a small 3D ramp and write sample.vtk in current folder
    import os

    demo = np.zeros((16, 32, 48), dtype=np.float32)
    # shape (Z, Y, X)
    Z, Y, X = demo.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                demo[z, y, x] = x + y * 0.5 + z * 0.2

    out_ascii = os.path.join(os.path.dirname(__file__), 'sample.vtk')
    out_bin = os.path.join(os.path.dirname(__file__), 'sample_bin.vtk')
    write_vtk_structured_points(out_ascii, demo, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=False)
    write_vtk_structured_points(out_bin, demo, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), binary=True)
    print('Wrote demo VTK to', out_ascii)
    print('Wrote demo binary VTK to', out_bin)
