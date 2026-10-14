#!/usr/bin/env python3
"""
Reconstruct a periodic Q1 displacement field from a cell-wise strain field.

Based on Brisard (2017), "Reconstructing displacements from the solution
to the periodic Lippmann–Schwinger equation discretized on a uniform grid".

Input
-----
strain : ndarray
    2-D: shape (N1,N2,3), components [e11,e22,e12]
    3-D: shape (N1,N2,N3,6), components
          [e11,e22,e33,e12,e13,e23]

material : ndarray
    Isotropic material at each cell, shape (...,2), containing
    [Bulk modulus K, shear modulus G].

reference : tuple (K0, G0)
    Homogeneous reference material C0 used in the auxiliary problem.

lengths : tuple
    Physical dimensions of the periodic cell, e.g. (Lx,Ly) or (Lx,Ly,Lz).

The implementation follows Procedure 1 of the paper:
    sigma_p = (C-C0):epsilon
    K_k uhat_k = - sigma_p_hat_k · B_k
    uhat_0 = 0
and then inverse FFT gives the periodic nodal displacement.

Important:
    This is the paper's reconstruction, not simply a pointwise integration
    of strain. It assumes the strain is a cell-wise representation on a
    periodic uniform grid and uses the Q1 finite-element reconstruction.

For a strain field that is already known to be compatible, a simpler
Fourier least-squares integration can also be used; that is not the
procedure implemented here.
"""

import numpy as np

def _voigt_to_tensor_2d(e):
    """[e11,e22,e12] -> symmetric 2x2 tensor."""
    out = np.empty(e.shape[:-1] + (2, 2), dtype=e.dtype)
    out[..., 0, 0] = e[..., 0]
    out[..., 1, 1] = e[..., 1]
    out[..., 0, 1] = e[..., 2]
    out[..., 1, 0] = e[..., 2]
    return out


def _voigt_to_tensor_3d(e):
    """[e11,e22,e33,e12,e13,e23] -> symmetric 3x3 tensor."""
    out = np.empty(e.shape[:-1] + (3, 3), dtype=e.dtype)
    out[..., 0, 0] = e[..., 0]
    out[..., 1, 1] = e[..., 1]
    out[..., 2, 2] = e[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = e[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = e[..., 4]
    out[..., 1, 2] = out[..., 2, 1] = e[..., 5]
    return out


def _lame_from_Enu(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _lame_from_KG(K, G):
    mu = G
    lam = K - 2 * G / 3
    return mu, lam


def _stress_polarization(strain, material, reference):
    """
    sigma_p = (C-C0):epsilon.

    For 2-D plane strain, epsilon_33=0 is used in the constitutive law,
    while only the in-plane stress components are retained.
    """
    d = strain.shape[-1]
    K0, G0 = reference
    mu0, lam0 = _lame_from_KG(K0, G0)

    K = material[..., 0]
    G = material[..., 1]
    mu, lam = _lame_from_KG(K, G)

    # Reference and local tensors differ by Delta(mu), Delta(lambda).
    dmu = mu - mu0
    dlam = lam - lam0

    # Constitutive law : sigma_ij = 2 mu eps_ij + lambda tr(eps) delta_ij.
    # Plane-strain for 2D 
        
    # 2D
    if d == 3:
        eps = _voigt_to_tensor_2d(strain)
        tr = eps[..., 0, 0] + eps[..., 1, 1]

        sig = np.zeros_like(eps, dtype=np.result_type(strain, material, float))
        sig[..., 0, 0] = 2*dmu*eps[..., 0, 0] + dlam*tr
        sig[..., 1, 1] = 2*dmu*eps[..., 1, 1] + dlam*tr
        sig[..., 0, 1] = sig[..., 1, 0] = 2*dmu*eps[..., 0, 1]
        return sig

    # 3D
    if d == 6:
        eps = _voigt_to_tensor_3d(strain)
        tr = np.trace(eps, axis1=-2, axis2=-1)

        eye = np.eye(3)
        sig = 2*dmu[..., None, None]*eps + dlam[..., None, None]*tr[..., None, None]*eye
        return sig

    raise ValueError("Last strain dimension must be 3 (2-D) or 6 (3-D).")


def _B_vector(k, N, L):
    """
    Modal strain-displacement vector B_k from Brisard Eq. (40).

    The result is complex and has d components. It is the vector such that
        epsilon_hat ~ sym(u_hat outer B_k)
    for the cell-averaged Q1 strain.
    """
    d = len(N)
    k = np.asarray(k, dtype=float)
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    z = np.sum(k / N)
    phase = np.exp(1j * np.pi * z)

    B = np.empty(d, dtype=complex)
    for i in range(d):
        B[i] = (
            2j * N[i] / L[i]
            * np.sin(np.pi * k[i] / N[i])
            * np.prod([
                np.cos(np.pi * k[j] / N[j])
                for j in range(d) if j != i
            ])
            * phase
        )
    return B

def _H_matrix(k, N, L):
    """
    H_k from Brisard Appendix B.3, Eqs. (B.15)-(B.17).

    This is the Q1 modal stiffness kernel. The closed-form expressions are
    implemented for d=2 and d=3.
    """
    d = len(N)
    k = np.asarray(k, dtype=float)
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    # h_i = L_i/N_i
    h = L / N
    z = 2.0 * np.pi * k / N

    def phi(x):
        return 2.0 * (1.0 - np.cos(x))

    def chi(x):
        return (2.0 + np.cos(x)) / 3.0

    def psi(x):
        return np.sin(x)

    H = np.zeros((d, d), dtype=float)

    # B.15 / B.16
    if d == 2:
        H[0,0] = (phi(z[0])*chi(z[1])) / (h[0]**2)
        H[0,1] = H[1,0] = (psi(z[0])*psi(z[1])) / (h[0]*h[1])
        H[1,1] = (chi(z[0])*phi(z[1])) / (h[1]**2)
    elif d == 3:
        H[0,0] = (phi(z[0])*chi(z[1])*chi(z[2])) / (h[0]**2)
        H[0,1] = H[1,0] = (psi(z[0])*psi(z[1])*chi(z[2])) / (h[0]*h[1])
        H[0,2] = H[2,0] = (psi(z[0])*chi(z[1])*psi(z[2])) / (h[0]*h[2])
        H[1,1] = (chi(z[0])*phi(z[1])*chi(z[2])) / (h[1]**2)
        H[1,2] = H[2,1] = (chi(z[0])*psi(z[1])*psi(z[2])) / (h[1]*h[2])
        H[2,2] = (chi(z[0])*chi(z[1])*phi(z[2])) / (h[2]**2)

    return H


def _modal_stiffness(k, N, L, K0, G0):
    """
    K_k from Brisard Eq. (46):
        K_k = mu0 tr(H_k) I + mu0/(1-2 nu0) H_k.
    """
    d = len(N)
    mu0, _ = _lame_from_KG(K0, G0)
    nu0 = (3*K0 - 2*G0) / (2*(3*K0 + G0))
    H = _H_matrix(k, N, L)
    return mu0 * np.trace(H) * np.eye(d) + mu0 / (1.0 - 2.0*nu0) * H


def _sigma_hat_dot_B(sigma_hat, B):
    """Return vector sigma_hat @ B."""
    return np.einsum("...ij,...j->...i", sigma_hat, B)


def reconstruct_displacement(strain, sigma_p, lengths, reference):
    """
    Main reconstruction routine.

    Parameters
    ----------
    strain : ndarray
        2-D (...,3) or 3-D (...,6) cell strain field.
    sigma_p : ndarray
        2-D (...,3) or 3-D (...,6) cell stress polarization field.
    lengths : sequence
        Cell dimensions.
    reference : (E0, nu0)
        Homogeneous reference material.

    Returns
    -------
    u : ndarray
        Nodal periodic displacement field. Its shape is
        (N1,N2,d) or (N1,N2,N3,d).
        The mean displacement is zero, as in Eq. (55).
    """
    raise ValueError("work on this function")


    strain = np.asarray(strain, dtype=float)

    if strain.ndim == 3 and strain.shape[-1] == 3:
        d = 2
    elif strain.ndim == 4 and strain.shape[-1] == 6:
        d = 3
    else:
        raise ValueError(
            "Expected 2D strain (...,3) or 3D strain (...,6). "
        f"Got shape {strain.shape}."
        )

    N = strain.shape[:-1]
    if len(lengths) != len(N):
        raise ValueError("lengths must have one entry per spatial dimension")

    E0, nu0 = reference
    raise ValueError("reference must be (K0, G0)")

    # adapt the stress polarization to a tensor form
    if d == 3:
        sigma_p_adapt = np.zeros((sigma_p.shape[0], sigma_p.shape[1], sigma_p.shape[2], 3, 3))
        sigma_p_adapt[..., 0, 0] = sigma_p[..., 0].copy()
        sigma_p_adapt[..., 1, 1] = sigma_p[..., 1].copy()
        sigma_p_adapt[..., 2, 2] = sigma_p[..., 2].copy()
        sigma_p_adapt[..., 0, 1] = sigma_p[..., 3].copy()
        sigma_p_adapt[..., 1, 0] = sigma_p[..., 3].copy()
        sigma_p_adapt[..., 0, 2] = sigma_p[..., 4].copy()
        sigma_p_adapt[..., 2, 0] = sigma_p[..., 4].copy()
        sigma_p_adapt[..., 1, 2] = sigma_p[..., 5].copy()
        sigma_p_adapt[..., 2, 1] = sigma_p[..., 5].copy()

    # FFT convention used by numpy.fft.fft is the same sign convention
    # as the paper's DFT definition.
    sigma_hat = np.zeros(sigma_p_adapt.shape, dtype=complex)
    for i in range(d):
        for j in range(d):
            sigma_hat[..., i, j] = np.fft.fftn(sigma_p_adapt[..., i, j])

    u_hat = np.zeros(N + (d,), dtype=complex)

    # Zero-frequency mode: impose zero mean displacement (Eq. 55).
    for k in np.ndindex(*N):
        if all(ki == 0 for ki in k):
            continue

        B = _B_vector(k, N, lengths)
        K = _modal_stiffness(k, N, lengths, E0, nu0)

        # no body forces
        # F = 0

        # Eq. (54), with no body forces:
        # K uhat = - sigma_hat · B
        rhs = -_sigma_hat_dot_B(sigma_hat[k], B)

        # K should be positive definite for all nonzero modes.
        u_hat[k] = np.linalg.solve(K, rhs)

    # Inverse DFT. numpy's ifftn includes the 1/prod(N) normalization.
    u = np.empty(N + (d,), dtype=float)
    for i in range(d):
        u[..., i] = np.fft.ifftn(u_hat[..., i]).real

    # I notice that the displacement sign should be inverted
    u = -u

    return u

def reconstruct_displacement_unknownStressPolarization(strain, material, lengths, reference):
    """
    Main reconstruction routine.

    Parameters
    ----------
    strain : ndarray
        2-D (...,3) or 3-D (...,6) cell strain field.
    material : ndarray
        Same grid shape as strain without the last component, with
        material[...,0] = E and material[...,1] = nu.
    lengths : sequence
        Cell dimensions.
    reference : (K0, G0)
        Homogeneous reference material.

    Returns
    -------
    u : ndarray
        Nodal periodic displacement field. Its shape is
        (N1,N2,d) or (N1,N2,N3,d).
        The mean displacement is zero, as in Eq. (55).
    """
    strain = np.asarray(strain, dtype=float)
    material = np.asarray(material, dtype=float)

    if strain.ndim == 3 and strain.shape[-1] == 3:
        d = 2
    elif strain.ndim == 4 and strain.shape[-1] == 6:
        d = 3
    else:
        raise ValueError(
            "Expected 2D strain (...,3) or 3D strain (...,6). "
        f"Got shape {strain.shape}."
        )

    N = strain.shape[:-1]
    if len(lengths) != len(N):
        raise ValueError("lengths must have one entry per spatial dimension")

    if material.shape != N + (2,):
        raise ValueError(
            "material must have shape strain.shape[:-1] + (2,), "
            "with [Bulk modulus, Shear modulus] at every cell."
        )

    # Step 1 of the paper, after obtaining the polarization from the
    # read the reference material
    K0, G0 = reference
    # supplied strain field:
    sigma_p = _stress_polarization(strain, material, reference)

    # FFT convention used by numpy.fft.fft is the same sign convention
    # as the paper's DFT definition.
    sigma_hat = np.zeros(sigma_p.shape, dtype=complex)
    for i in range(d):
        for j in range(d):
            sigma_hat[..., i, j] = np.fft.fftn(sigma_p[..., i, j])

    u_hat = np.zeros(N + (d,), dtype=complex)

    # Zero-frequency mode: impose zero mean displacement (Eq. 55).
    for k in np.ndindex(*N):
        if all(ki == 0 for ki in k):
            continue

        B = _B_vector(k, N, lengths)
        K = _modal_stiffness(k, N, lengths, K0, G0)

        # no body forces
        # F = 0

        # Eq. (54), with no body forces:
        # K uhat = - sigma_hat · B
        rhs = -_sigma_hat_dot_B(sigma_hat[k], B)

        # K should be positive definite for all nonzero modes.
        u_hat[k] = np.linalg.solve(K, rhs)

    # Inverse DFT. numpy's ifftn includes the 1/prod(N) normalization.
    u = np.empty(N + (d,), dtype=float)
    for i in range(d):
        u[..., i] = np.fft.ifftn(u_hat[..., i]).real

    # I notice that the displacement sign should be inverted
    u = -u

    return u


def strain_from_nodal_displacement(u, lengths):
    """
    Optional validation helper.

    Reconstruct the cell-average strain corresponding to the nodal
    displacement using the same Q1 modal relation used in the paper.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[:-1]
    d = u.shape[-1]

    u_hat = np.empty(N + (d,), dtype=complex)
    for i in range(d):
        u_hat[..., i] = np.fft.fftn(u[..., i])

    if d == 2:
        out = np.zeros(N + (3,), dtype=complex)
        for k in np.ndindex(*N):
            B = _B_vector(k, N, lengths)
            uu = u_hat[k]
            out[k][0] = uu[0] * B[0]
            out[k][1] = uu[1] * B[1]
            out[k][2] = 0.5 * (uu[0]*B[1] + uu[1]*B[0])
        result = np.empty_like(out.real)
    else:
        out = np.zeros(N + (6,), dtype=complex)
        for k in np.ndindex(*N):
            B = _B_vector(k, N, lengths)
            uu = u_hat[k]
            out[k][0] = uu[0] * B[0]
            out[k][1] = uu[1] * B[1]
            out[k][2] = uu[2] * B[2]
            out[k][3] = 0.5 * (uu[0]*B[1] + uu[1]*B[0])
            out[k][4] = 0.5 * (uu[0]*B[2] + uu[2]*B[0])
            out[k][5] = 0.5 * (uu[1]*B[2] + uu[2]*B[1])
        result = np.empty_like(out.real)

    for i in range(out.shape[-1]):
        result[..., i] = np.fft.ifftn(out[..., i]).real
    return result


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Minimal example: 2-D periodic cell
    # ---------------------------------------------------------------
    N1, N2 = 64, 64
    Lx, Ly = 1.0, 1.0

    # mesh in the X direction
    L_X = (np.arange(N1) + 0.5) * Lx/N1

    # Example heterogeneous isotropic material:
    # material[...,0] = Young's modulus
    # material[...,1] = Poisson ratio
    material = np.empty((N1, N2, 2))
    for i_X in range(len(L_X)):
        X = L_X[i_X]
        if 0.25 <= X and X <= 0.75:
            material[i_X, :, 0] = 1.0
            material[i_X, :, 1] = 0.3
        else:
            material[i_X, :, 0] = 0.001
            material[i_X, :, 1] = 0.0

    # Example strain field: replace this by your UGPLS strain result.
    strain = np.zeros((N1, N2, 3))
    # A compatible-looking test strain (small perturbation around zero).
    strain[..., 1] = 0
    strain[..., 2] = 0
    for i_X in range(len(L_X)):
        X = L_X[i_X]
        if 0.25 <= X and X <= 0.75:
            strain[i_X, :, 0] = 1
        else:
            strain[i_X, :, 0] = -1

    # Homogeneous reference medium C0.
    reference = (100.0, 0.)

    u = reconstruct_displacement(
        strain,
        material,
        lengths=(Lx, Ly),
        reference=reference,
    )

    dx = L_X[1] - L_X[0]
    for i_x in range(strain.shape[0]):
        X = L_X[i_x]
        if i_x == 0:
            u_theo = [strain[i_x, 0, 0]*dx]
        else:
            u_theo.append(u_theo[-1]+strain[i_x, 0, 0]*dx)


    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ax.plot(L_X+dx/2, u[:,0,0], color='b')
    ax.plot(L_X+dx/2, u_theo, color='b', linestyle='dotted')
    ax.set_ylabel('displacement (-)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    axb = ax.twinx()
    axb.plot(L_X, strain[:,0,0], color='r')
    axb.set_ylabel('strain (-)', color='r')
    axb.tick_params(axis='y', labelcolor='r')
    #plt.show()
    plt.close()


