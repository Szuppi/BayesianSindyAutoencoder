import os
import sys

import numpy as np
from scipy.integrate import odeint
from scipy.signal import savgol_filter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sindy import library_size

# -----------------------------------------------------------------------------
# Nested mixer cache: ensures depth-1 ⊂ depth-2 ⊂ depth-3 ⊂ depth-4
# -----------------------------------------------------------------------------
_MIXER_CACHE = {}


def _mixer_cache_key(seed, n_points, max_layers, cond_thresh, leaky_alpha, add_bias):
    # round floats to avoid float-key weirdness
    return (
        int(seed),
        int(n_points),
        int(max_layers),
        float(cond_thresh),
        float(leaky_alpha),
        bool(add_bias),
    )


def _col_normalize(W):
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)


def _sample_well_conditioned_W(rng, dim=3, cond_thresh=50.0, max_tries=20000):
    for _ in range(max_tries):
        W = rng.uniform(-1.0, 1.0, size=(dim, dim)).astype(np.float32)
        W = _col_normalize(W).astype(np.float32)
        if np.linalg.cond(W) < cond_thresh:
            return W
    raise RuntimeError(
        f"Failed to sample well-conditioned W (cond<{cond_thresh}). "
        f"Try increasing cond_thresh or max_tries."
    )


def _get_or_create_nested_mixer(
    seed,
    n_points,
    max_layers,
    cond_thresh,
    leaky_alpha,
    add_bias,
):
    """
    Returns dict with:
      - Ws_full: list of length max_layers (each 3x3)
      - A: (n_points, 3)
      - b: (n_points,)
    """
    key = _mixer_cache_key(seed, n_points, max_layers, cond_thresh, leaky_alpha, add_bias)
    if key in _MIXER_CACHE:
        return _MIXER_CACHE[key]

    rng = np.random.RandomState(int(seed))

    Ws_full = [
        _sample_well_conditioned_W(rng, dim=3, cond_thresh=float(cond_thresh))
        for _ in range(int(max_layers))
    ]

    n = int(n_points)
    A = (rng.randn(n, 3).astype(np.float32) / np.sqrt(3.0)).astype(np.float32)
    b = (
        rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32)
        if bool(add_bias)
        else np.zeros((n,), dtype=np.float32)
    )

    mixer = {"Ws_full": Ws_full, "A": A, "b": b}
    _MIXER_CACHE[key] = mixer
    return mixer


def finite_diff_first(y, dt):
    """
    y: (T, D)
    returns dy: (T, D)
    """
    dy = np.empty_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    dy[0]    = (y[1] - y[0]) / dt
    dy[-1]   = (y[-1] - y[-2]) / dt
    return dy


def finite_diff_second(y, dt):
    """
    y: (T, D)
    returns ddy: (T, D)
    """
    ddy = np.empty_like(y)
    ddy[1:-1] = (y[2:] - 2.0*y[1:-1] + y[:-2]) / (dt**2)
    # boundary copy
    ddy[0]  = ddy[1]
    ddy[-1] = ddy[-2]
    return ddy


def maybe_smooth(y, window=11, poly=3):
    """
    Savitzky-Golay smoothing along time axis.
    y: (T, D)
    """
    if window is None:
        return y
    # window must be odd and <= T
    T = y.shape[0]
    w = int(window)
    if w >= T:
        w = T - 1 if (T - 1) % 2 == 1 else T - 2
    if w < 5:
        return y
    if w % 2 == 0:
        w += 1
    return savgol_filter(y, window_length=w, polyorder=poly, axis=0, mode="interp")


def get_lorenz_data(
    n_ics,
    noise_strength=0,
    smooth=True,
    sg_window=11,
    sg_poly=3,
    seed=0,
    input_dim=128,
    t_step=0.01,
    n_layers=4,          # <-- ADD THIS
    cond_thresh=50.0,    # <-- (optional) expose if you want
    leaky_alpha=0.2,     # <-- (optional) expose if you want
    add_bias=True,       # <-- (optional) expose if you want
    nested=False,
    max_layers=4,
    normalization=None
):
    t = np.arange(0, 5, t_step)
    dt = t[1] - t[0]

    ic_means = np.array([0, 0, 25])
    ic_widths = 2*np.array([36, 48, 41])
    rng = np.random.RandomState(seed)
    ics = ic_widths*(rng.rand(n_ics, 3) - .5) + ic_means

    data = generate_lorenz_data(
        ics, t, n_points=input_dim,
        linear=False,
        normalization=normalization, #np.array([1/40, 1/40, 1/40]),
        seed=seed,

        # existing knobs
        noise_strength=noise_strength,
        smooth=smooth,
        sg_window=sg_window,
        sg_poly=sg_poly,

        # --- IMPORTANT: make mixer depth configurable ---
        n_layers=int(n_layers),

        # (optional pass-through)
        cond_thresh=float(cond_thresh),
        leaky_alpha=float(leaky_alpha),
        add_bias=bool(add_bias),

        # nested mixer
        nested=bool(nested),
        max_layers=int(max_layers)
    )

    # flatten for TF pipeline
    data['x']   = data['x'].reshape((-1, input_dim))
    data['dx']  = data['dx'].reshape((-1, input_dim))
    data['ddx'] = data['ddx'].reshape((-1, input_dim))

    return data


def lorenz_coefficients(normalization, poly_order=3, sigma=10., beta=8/3, rho=28):
    """
    Generate the SINDy coefficient matrix for the Lorenz system.

    Arguments:
        normalization - 3-element list of array specifying scaling of each Lorenz variable
        poly_order - Polynomial order of the SINDy model.
        sigma, beta, rho - Parameters of the Lorenz system
    """
    Xi = np.zeros((library_size(3,poly_order),3))
    Xi[1,0] = -sigma
    Xi[2,0] = sigma*normalization[0]/normalization[1]
    Xi[1,1] = rho*normalization[1]/normalization[0]
    Xi[2,1] = -1
    Xi[6,1] = -normalization[1]/(normalization[0]*normalization[2])
    Xi[3,2] = -beta
    Xi[5,2] = normalization[2]/(normalization[0]*normalization[1])
    return Xi


def simulate_lorenz(z0, t, sigma=10., beta=8/3, rho=28.):
    """
    Simulate the Lorenz dynamics.

    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        sigma, beta, rho - Lorenz parameters

    Returns:
        z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    f = lambda z,t : [sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]]
    df = lambda z,dz,t : [sigma*(dz[1] - dz[0]),
                          dz[0]*(rho - z[2]) + z[0]*(-dz[2]) - dz[1],
                          dz[0]*z[1] + z[0]*dz[1] - beta*dz[2]]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i],dt*i)
        ddz[i] = df(z[i], dz[i], dt*i)
    return z, dz, ddz


def generate_lorenz_data(
    ics,
    t,
    n_points,
    linear=True,                 # kept for signature compatibility; unused here
    normalization=None,
    sigma=10,
    beta=8/3,
    rho=28,
    seed=0,
    # --- new knobs (safe defaults keep old behavior: clean x, FD derivatives) ---
    noise_strength=0.0,          # std of additive Gaussian noise applied to x ONLY
    smooth=True,                 # if True, smooth x before finite differences
    sg_window=11,                # Savitzky-Golay window length (odd)
    sg_poly=3,                   # Savitzky-Golay poly order
    cond_thresh=50.0,            # max condition number allowed per MLP layer weight
    n_layers=4,                  # number of square linear layers in the invertible-ish MLP
    leaky_alpha=0.2,             # leaky ReLU slope
    add_bias=True,               # include bias in final linear map to R^n
    *,
    nested=False,        # <-- ADD
    max_layers=4,        # <-- ADD (the "full" depth you nest within)
):
    """
    Generate high-dimensional Lorenz dataset with DCL-style mixing and FD derivatives.

    Pipeline:
        z(t) in R^3  --(invertible-ish MLP h: R^3->R^3)--> h(z)
                    --(linear map A: R^3->R^n [+ b])--> x_clean(t) in R^n
                    --(+ Gaussian noise)--> x_noisy(t)
                    --(optional Savitzky-Golay smoothing)--> x_used(t)
                    --(finite differences)--> dx_used(t), ddx_used(t)

    Outputs:
        data['x_nl']   = x_used (what AE-SINDy should train/test on)
        data['dx_nl']  = finite-diff dx from x_used
        data['ddx_nl'] = finite-diff ddx from x_used

    Also includes:
        data['x_clean'], data['x_noisy'], data['x_used'] for debugging.
    """

    ics = np.asarray(ics, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    n_ics = ics.shape[0]
    n_steps = t.size
    if n_steps < 3:
        raise ValueError("Need at least 3 time points for finite differences.")
    dt = float(t[1] - t[0])

    # -----------------------
    # 1) Simulate latent Lorenz
    # -----------------------
    d = 3
    z = np.zeros((n_ics, n_steps, d), dtype=np.float32)
    dz = np.zeros_like(z)
    ddz = np.zeros_like(z)
    for i in range(n_ics):
        zi, dzi, ddzi = simulate_lorenz(ics[i], t, sigma=sigma, beta=beta, rho=rho)
        z[i] = zi.astype(np.float32)
        dz[i] = dzi.astype(np.float32)
        ddz[i] = ddzi.astype(np.float32)

    if normalization is not None:
        norm = np.asarray(normalization, dtype=np.float32).reshape((1, 1, 3))
        z *= norm
        dz *= norm
        ddz *= norm

    # -----------------------
    # 2) Build DCL-style invertible-ish MLP h: R^3 -> R^3 (fixed random weights) -- nested or non-nested
    # -----------------------
    n_layers = int(n_layers)
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if nested and int(max_layers) < n_layers:
        raise ValueError(f"max_layers ({max_layers}) must be >= n_layers ({n_layers}) when nested=True")

    n = int(n_points) # -- input dim

    if nested:
        # sample a full mixer once, then slice Ws_full[:n_layers]
        mixer = _get_or_create_nested_mixer(
            seed=seed,
            n_points=n_points,
            max_layers=max_layers,
            cond_thresh=cond_thresh,
            leaky_alpha=leaky_alpha,
            add_bias=add_bias,
        )
        Ws = mixer["Ws_full"][:n_layers]
        A = mixer["A"]
        b = mixer["b"]
    else:
        # original behavior: sample exactly n_layers matrices + A,b in this call
        rng = np.random.RandomState(int(seed))
        Ws = [
            _sample_well_conditioned_W(rng, dim=3, cond_thresh=float(cond_thresh))
            for _ in range(n_layers)
        ]
        A = (rng.randn(n, 3).astype(np.float32) / np.sqrt(3.0)).astype(np.float32)
        b = (
            rng.uniform(-0.5, 0.5, size=(n,)).astype(np.float32)
            if bool(add_bias)
            else np.zeros((n,), dtype=np.float32)
        )

    def _leaky_relu(x):
        return np.where(x >= 0.0, x, float(leaky_alpha) * x)

    def h_mlp(z_batch_2d):
        xh = z_batch_2d
        for li, W in enumerate(Ws):
            xh = xh @ W.T
            if li < len(Ws) - 1:
                xh = _leaky_relu(xh)
        return xh

    # -----------------------
    # 3) Compute clean high-dimensional x = A h(z) + b
    # -----------------------
    x_clean = np.zeros((n_ics, n_steps, n), dtype=np.float32)
    for i in range(n_ics):
        hz = h_mlp(z[i])              # (n_steps,3)
        x_clean[i] = hz @ A.T + b     # (n_steps,n)

    # -----------------------
    # 4) Add noise to x ONLY (optional), then optional smoothing
    # -----------------------
    rng_noise = np.random.RandomState(int(seed) + 12345)

    if noise_strength and noise_strength > 0.0:
        x_noisy = (x_clean + noise_strength * rng_noise.randn(*x_clean.shape).astype(np.float32)).astype(np.float32)
    else:
        x_noisy = x_clean

    if smooth:
        x_used = np.empty_like(x_noisy)
        for i in range(n_ics):
            x_used[i] = maybe_smooth(x_noisy[i], window=sg_window, poly=sg_poly).astype(np.float32)
    else:
        x_used = x_noisy

    # -----------------------
    # 5) Finite-difference derivatives computed FROM x_used
    # -----------------------
    dx_used = np.empty_like(x_used)
    ddx_used = np.empty_like(x_used)
    for i in range(n_ics):
        dx_used[i] = finite_diff_first(x_used[i], dt).astype(np.float32)
        ddx_used[i] = finite_diff_second(x_used[i], dt).astype(np.float32)

    # -----------------------
    # 6) True SINDy coefficients for Lorenz (in latent space)
    # -----------------------
    if normalization is None:
        sindy_coefficients = lorenz_coefficients([1, 1, 1], sigma=sigma, beta=beta, rho=rho)
    else:
        sindy_coefficients = lorenz_coefficients(np.asarray(normalization), sigma=sigma, beta=beta, rho=rho)

    # -----------------------
    # 7) Package dict (AE-SINDy compatibility)
    # -----------------------
    data = {}
    data["t"] = t
    data["z"] = z
    data["dz"] = dz
    data["ddz"] = ddz
    data["sindy_coefficients"] = sindy_coefficients.astype(np.float32)

    # "nonlinear observed" that AE-SINDy will use
    data["x_nl"] = x_used
    data["dx_nl"] = dx_used
    data["ddx_nl"] = ddx_used

    # keep these too (some codepaths might expect them)
    data["x"] = x_used
    data["dx"] = dx_used
    data["ddx"] = ddx_used

    # debug/diagnostics
    data["x_clean"] = x_clean
    data["x_noisy"] = x_noisy
    data["x_used"] = x_used

    # mixing params for reproducibility
    data["mixing_seed"] = int(seed)
    data["mixing_n_layers"] = int(n_layers)
    data["mixing_nested"] = bool(nested)
    data["mixing_Ws"] = Ws
    data["mixing_A"] = A
    data["mixing_b"] = b

    if nested:
        # store full Ws so you can verify nesting
        data["mixing_Ws_full"] = _get_or_create_nested_mixer(
            seed=seed,
            n_points=n_points,
            max_layers=max_layers,
            cond_thresh=cond_thresh,
            leaky_alpha=leaky_alpha,
            add_bias=add_bias,
        )["Ws_full"]

    return data

'''
TRUE SYSTEM:
   dz0/dt = -10.000*z0 + 10.000*z1
   dz1/dt = 28.000*z0 + -1.000*z1 + -40.000*z0*z2
   dz2/dt = -2.667*z2 + 40.000*z0*z1
IDENTIFIED SYSTEM:
   dz0/dt = 7.232*1 + -2.470*z0 + 4.000*z1*z2 + -0.080*z2*z2
   dz1/dt = -3.254*z0 + -9.666*z1 + -8.767*z2
   dz2/dt = -2.309*1 + -10.085*z0*z1 + -0.148*z1*z1*z2
'''