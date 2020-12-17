from numba import jit


# compute_ds_dt is the function we need to evaluate next steps
@jit(nopython=True)
def compute_ds_dt(p_x, p_y, p_z, b_eff_x, b_eff_y, b_eff_z, b_rands_x, b_rands_y, b_rands_z, lam, gamma):
    par1 = p_y * b_eff_z * lam - b_eff_y * lam * p_z - b_eff_x - b_rands_x
    ds_dt_x = gamma * (b_eff_x * p_y * p_y * lam + p_y * (b_eff_z + b_rands_z - b_eff_y * lam * p_x) +
                       p_z * (p_z * b_eff_x * lam - b_eff_z * lam * p_x - b_eff_y - b_rands_y))
    ds_dt_y = -gamma * (-b_eff_y * p_x * p_x * lam + p_x * (p_y * b_eff_x * lam + b_eff_z + b_rands_z) + p_z * par1)
    ds_dt_z = gamma * (b_eff_z * p_x * p_x * lam + p_x * (-p_z * b_eff_x * lam + b_eff_y + b_rands_y) + p_y * par1)

    return ds_dt_x, ds_dt_y, ds_dt_z
