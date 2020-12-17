import numpy as np
from vectorized_integrators.ds_dt import compute_ds_dt
from numba import jit

# Adams-Bashforth integration constants
ab = np.array([1901 / 720, 2774 / 720, 2616 / 720, 1274 / 720, 251 / 720])


# Evaluate the fifth step using function values from the past 4 steps
@jit(nopython=True)
def fifth_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz):
    return ab[0] * d_fx - ab[1] * sx[3, :] + ab[2] * sx[2, :] - ab[3] * sx[1, :] + ab[4] * sx[0, :], \
           ab[0] * d_fy - ab[1] * sy[3, :] + ab[2] * sy[2, :] - ab[3] * sy[1, :] + ab[4] * sy[0, :], \
           ab[0] * d_fz - ab[1] * sz[3, :] + ab[2] * sz[2, :] - ab[3] * sz[1, :] + ab[4] * sz[0, :]


# Evaluate the fourth step using function values from the past 3 steps
@jit(nopython=True)
def fourth_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz):
    return (55 / 24) * d_fx - (59 / 24) * sx[3, :] + (37 / 24) * sx[2, :] - (9 / 24) * sx[1, :], \
           (55 / 24) * d_fy - (59 / 24) * sy[3, :] + (37 / 24) * sy[2, :] - (9 / 24) * sy[1, :], \
           (55 / 24) * d_fz - (59 / 24) * sz[3, :] + (37 / 24) * sz[2, :] - (9 / 24) * sz[1, :]


# Evaluate the third step using function values from the past 2 steps
@jit(nopython=True)
def third_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz):
    return (23 / 12) * d_fx - (16 / 12) * sx[3, :] + (5 / 12) * sx[2, :], \
           (23 / 12) * d_fy - (16 / 12) * sy[3, :] + (5 / 12) * sy[2, :], \
           (23 / 12) * d_fz - (16 / 12) * sz[3, :] + (5 / 12) * sz[2, :]


# Evaluate the second step using function values from the past step
@jit(nopython=True)
def second_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz):
    return (3 / 2) * d_fx - (1 / 2) * sx[3, :], \
           (3 / 2) * d_fy - (1 / 2) * sy[3, :], \
           (3 / 2) * d_fz - (1 / 2) * sz[3, :]


# Integrate one step using fifth order Adams Bashforth
# This is a linear multistep method (https://en.wikipedia.org/wiki/Linear_multistep_method)
@jit(nopython=True)
def ad_bs_step(p_x, p_y, p_z, b_eff_x, b_eff_y, b_eff_z, b_rands_x,
               b_rands_y, b_rands_z, lam, gamma, dt, spin, sx, sy, sz):
    d_fx, d_fy, d_fz = compute_ds_dt(p_x, p_y, p_z, b_eff_x, b_eff_y, b_eff_z, 0, 0, 0, lam, gamma)

    # Gradually increase steps in the Adams Bashforth method
    if np.isnan(sx[3, :]).any():
        d_sx, d_sy, d_sz = d_fx, d_fy, d_fz
    elif np.isnan(sx[2, :]).any():
        d_sx, d_sy, d_sz = second_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz)
    elif np.isnan(sx[1, :]).any():
        d_sx, d_sy, d_sz = third_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz)
    elif np.isnan(sx[0, :]).any():
        d_sx, d_sy, d_sz = fourth_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz)
    else:
        d_sx, d_sy, d_sz = fifth_ad_bs_step(d_fx, d_fy, d_fz, sx, sy, sz)

    # Move the values along so we keep continuing
    sx[0, :], sx[1, :], sx[2, :], sx[3, :] = sx[1, :], sx[2, :], sx[3, :], d_fx
    sy[0, :], sy[1, :], sy[2, :], sy[3, :] = sy[1, :], sy[2, :], sy[3, :], d_fy
    sz[0, :], sz[1, :], sz[2, :], sz[3, :] = sz[1, :], sz[2, :], sz[3, :], d_fz

    # Calculate the random energy added
    # This is based on temperature
    d_spin_rand_x = gamma * spin * (b_rands_z * p_x - b_rands_y * p_z)
    d_spin_rand_y = gamma * spin * (b_rands_x * p_z - b_rands_z * p_x)
    d_spin_rand_z = gamma * spin * (b_rands_y * p_x - b_rands_x * p_y)

    # Take the step and calculate the final position
    p_x += d_sx * dt + d_spin_rand_x
    p_y += d_sy * dt + d_spin_rand_y
    p_z += d_sz * dt + d_spin_rand_z

    return p_x, p_y, p_z, sx, sy, sz
