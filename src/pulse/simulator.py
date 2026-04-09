"""
CUORE pulse simulator based on the nonlinear electro-thermal model.

Reference: CUORE Collaboration, "An Energy-dependent Electro-thermal Response
Model of CUORE Cryogenic Calorimeter", arXiv:2205.04549

Implements the simplified 3-node thermal circuit (Figure 5) with:
  - Crystal+NTD-Ge lattice node (T_c)
  - NTD-Ge electron node (T_e)
  - PTFE support node (T_t)
  - Electrical node (V_bol)

ODE system: Eq. 2.21 with q-factor correction (Eq. 4.1).
Solved with SciPy solve_ivp.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import sosfilt
from dataclasses import dataclass

# --- Physical constants ---
K_B = 1.380649e-23       # J/K
E_CHARGE = 1.602176634e-19  # C
KEV_TO_J = 1e3 * E_CHARGE

# Internal ODE sampling rate (Hz). High enough to resolve fast transients
# before Bessel filtering. The output is decimated to f_sample.
F_INTERNAL = 10000


@dataclass
class DetectorParams:
    """Detector parameters for a CUORE-like channel.

    Randomized parameters have defaults at the midpoint of Table 5 ranges.
    Fixed parameters come from Tables 1, 3, 6 (Data-set I).
    """
    # --- Randomized parameters ---
    R0: float = 0.548          # [Ohm]  NTD-Ge resistance prefactor
    T0: float = 7.18           # [K]    characteristic temperature
    lambda0: float = 7.9e-9    # [m·K^0.5] hopping length
    q: float = -33.0           # second-order correction factor
    g_ec: float = 0.098        # [W/K^a_ec] electron-crystal conductivity coeff
    a_ec: float = 5.43         # electron-crystal conductivity power exponent
    C_p: float = 512e-12       # [F] parasitic capacitance
    T_base: float = 0.012      # [K] heat-sink temperature

    # --- Fixed parameters ---
    gamma: float = 0.5         # Shklovskii-Efros exponent
    g_ct: float = 1e-8         # [W/K] crystal-PTFE conductivity (a_ct=1)
    a_ct: float = 1.0          # crystal-PTFE power exponent (fixed)
    g_ts: float = 1e-8         # [W/K] PTFE-sink conductivity (a_ts=1)
    a_ts: float = 1.0          # PTFE-sink power exponent (fixed)
    c_e: float = 8.6e-11       # [J/K^2] electron heat capacity prefactor
    c_c: float = 2.2e-3        # [J/K^4] crystal heat capacity prefactor
    c_t: float = 2.1e-6        # [J/K^2] PTFE heat capacity prefactor
    R_bias: float = 60e9       # [Ohm] bias resistor
    V_bias: float = 4.71       # [V] total bias voltage
    gain: float = 5150.0       # amplifier gain
    W: float = 2.9e-3          # [m] thermistor width
    f_bessel: float = 120.0    # [Hz] Bessel filter cutoff
    bessel_order: int = 6


# Truncated Gaussian distributions: (mean, sigma, lower_bound, upper_bound)
# None means no bound on that side.
PARAM_DISTS = {
    'R0':      (0.548,    0.15,    0.0,  None),
    'T0':      (7.18,     0.40,    0.0,  None),
    'lambda0': (7.9e-9,   5.9e-9,  0.0,  None),
    'q':       (-33.0,    10.0,    None, None),
    'g_ec':    (0.098,    0.072,   0.0,  None),
    'a_ec':    (5.43,     0.38,    5.0,  6.0),
    'C_p':     (512e-12,  18e-12,  0.0,  None),
    'T_base':  (0.012,    0.002,   0.0,  None),
}


def sample_params(rng: np.random.Generator) -> DetectorParams:
    """Sample detector parameters from truncated Gaussian distributions.

    Rejection sampling: resample until all physical constraints are met.
    """
    params = DetectorParams()
    for name, (mean, sigma, lower, upper) in PARAM_DISTS.items():
        while True:
            value = rng.normal(mean, sigma)
            ok_lower = (lower is None) or (value > lower)
            ok_upper = (upper is None) or (value < upper)
            if ok_lower and ok_upper:
                break
        setattr(params, name, value)
    return params


@dataclass
class EquilibriumState:
    """Self-consistent equilibrium state of the detector."""
    V_bol: float   # [V] bolometer voltage
    T_e: float     # [K] electron temperature
    T_c: float     # [K] crystal temperature
    T_t: float     # [K] PTFE temperature
    R_bol: float   # [Ohm] bolometer resistance
    P_e: float     # [W] self-heating power


def thermistor_R(T_e: float, V_bol: float, params: DetectorParams) -> float:
    """NTD-Ge resistance R(T_e, V_bol) with E-field correction (Eq. 2.15)."""
    C_lam = E_CHARGE * params.lambda0 / (K_B * params.W)
    R = params.R0 * np.exp((params.T0 / T_e) ** params.gamma)
    R *= np.exp(-C_lam * V_bol / T_e ** 1.5)
    return R


def find_equilibrium(params: DetectorParams) -> EquilibriumState:
    """Find self-consistent equilibrium by fixed-point iteration.

    At equilibrium all time derivatives vanish. The thermal chain is serial:
    electrons -> crystal -> PTFE -> heat-sink, with self-heating as the source.
    """
    T_s = params.T_base
    T_e = T_s * 1.01  # initial guess

    for _ in range(500):
        # Inner loop: self-consistent V_bol for given T_e
        V_bol = params.V_bias * 1e-3  # initial guess
        for _ in range(200):
            R = thermistor_R(T_e, V_bol, params)
            V_new = params.V_bias * R / (params.R_bias + R)
            if abs(V_new - V_bol) < 1e-15:
                V_bol = V_new
                break
            V_bol = V_new

        R = thermistor_R(T_e, V_bol, params)
        P_e = V_bol ** 2 / R

        # Steady-state temperature chain (a_ct = a_ts = 1):
        # P_e = g_ts*(T_t - T_s) => T_t = P_e/g_ts + T_s
        # P_e = g_ct*(T_c - T_t) => T_c = P_e/g_ct + T_t
        # P_e = g_ec*(T_e^a - T_c^a) => T_e = (P_e/g_ec + T_c^a)^(1/a)
        T_t = P_e / params.g_ts + T_s
        T_c = P_e / params.g_ct + T_t
        T_e_new = (P_e / params.g_ec + T_c ** params.a_ec) ** (1.0 / params.a_ec)

        if abs(T_e_new - T_e) < 1e-14:
            T_e = T_e_new
            break
        # Damped update for stability
        T_e = 0.5 * T_e + 0.5 * T_e_new

    # Final consistent values
    R = thermistor_R(T_e, V_bol, params)
    V_bol = params.V_bias * R / (params.R_bias + R)
    P_e = V_bol ** 2 / R

    return EquilibriumState(V_bol=V_bol, T_e=T_e, T_c=T_c, T_t=T_t,
                            R_bol=R, P_e=P_e)


def is_valid_equilibrium(params: DetectorParams, eq: EquilibriumState,
                         max_r_ratio: float = 0.02,
                         max_loop_gain: float = 0.8) -> bool:
    """Check whether the equilibrium is in a physical operating regime.

    Rejects parameter sets where:
    1. R_bol / R_bias > max_r_ratio  (not in constant-current regime)
    2. Electro-thermal loop gain L > max_loop_gain  (feedback unstable)
    """
    # Check 1: resistance ratio
    if eq.R_bol / params.R_bias > max_r_ratio:
        return False

    # Check 2: loop gain  L = P_e * eta / (G_ec_lin * T_e)
    eta = params.gamma * (params.T0 / eq.T_e) ** params.gamma
    G_ec_lin = params.g_ec * params.a_ec * eq.T_e ** (params.a_ec - 1)
    if G_ec_lin <= 0:
        return False
    loop_gain = eq.P_e * eta / (G_ec_lin * eq.T_e)
    if loop_gain > max_loop_gain:
        return False

    return True


def _compute_alpha(delta_te: float, delta_vbol: float,
                   params: DetectorParams, eq: EquilibriumState) -> float:
    """Resistance perturbation exponent: R = R_eq * exp(-alpha).

    First order from Eq. 2.18-2.19, second order with q from Eq. 4.1.
    """
    Te = eq.T_e
    Ve = eq.V_bol
    gamma = params.gamma

    # Temperature sensitivity: eta = gamma * (T0/Te)^gamma
    eta = gamma * (params.T0 / Te) ** gamma

    # E-field coupling: C_A = e*lambda0*V_bol / (k_B*W*Te^(gamma+1))
    C_lam = E_CHARGE * params.lambda0 / (K_B * params.W)
    C_A = C_lam * Ve / Te ** (gamma + 1)

    u = delta_te / Te                                    # relative T perturbation
    w = delta_vbol / Ve if abs(Ve) > 1e-30 else 0.0      # relative V perturbation

    # First order
    alpha = (eta - (gamma + 1) * C_A) * u + C_A * w

    # Second order (Eq. 4.1)
    alpha += (-0.75 * eta
              + (gamma + 1) * (gamma + 2) / 2.0 * C_A
              + params.q) * u ** 2
    alpha += -(gamma + 1) * C_A * u * w

    return alpha


def _ode_rhs(t, y, params: DetectorParams, eq: EquilibriumState):
    """Right-hand side of the perturbation ODE (Eq. 2.21).

    State vector y = [delta_v_bol, delta_t_e, delta_t_c, delta_t_t].
    """
    dv, dte, dtc, dtt = y

    T_e = max(eq.T_e + dte, 1e-6)
    T_c = max(eq.T_c + dtc, 1e-6)
    T_t = max(eq.T_t + dtt, 1e-6)
    T_s = params.T_base

    # Heat capacities at current temperatures
    Ce = params.c_e * T_e
    Cc = params.c_c * T_c ** 3
    Ct = params.c_t * T_t

    # Resistance perturbation
    alpha = _compute_alpha(dte, dv, params, eq)
    ea = np.exp(alpha)

    R_eq = eq.R_bol
    V_eq = eq.V_bol
    P_eq = eq.P_e

    # Thermal power perturbations: delta_P = P(current) - P(equilibrium)
    dP_ec = (params.g_ec * (T_e ** params.a_ec - T_c ** params.a_ec)
             - params.g_ec * (eq.T_e ** params.a_ec - eq.T_c ** params.a_ec))

    dP_ct = (params.g_ct * (T_c ** params.a_ct - T_t ** params.a_ct)
             - params.g_ct * (eq.T_c ** params.a_ct - eq.T_t ** params.a_ct))

    dP_ts = (params.g_ts * (T_t ** params.a_ts - T_s ** params.a_ts)
             - params.g_ts * (eq.T_t ** params.a_ts - T_s ** params.a_ts))

    # Electrical equation (Eq. 2.21, first row)
    dv_dot = (-dv * (ea / R_eq + 1.0 / params.R_bias)
              - V_eq / R_eq * (ea - 1.0)) / params.C_p

    # Electron thermal equation (Eq. 2.20 / 2.21, second row)
    dte_dot = (P_eq * (ea - 1.0)
               + 2.0 * dv * V_eq / R_eq * ea
               + dv ** 2 / R_eq * ea
               - dP_ec) / Ce

    # Crystal thermal equation (Eq. 2.21, third row)
    dtc_dot = (dP_ec - dP_ct) / Cc

    # PTFE thermal equation (Eq. 2.21, fourth row)
    dtt_dot = (dP_ct - dP_ts) / Ct

    return [dv_dot, dte_dot, dtc_dot, dtt_dot]


def _make_bessel_sos(params: DetectorParams, fs: float):
    """6th-order Bessel low-pass filter as second-order sections."""
    from src.basics.filters import make_bessel_sos
    return make_bessel_sos(params.f_bessel, params.bessel_order, fs)


def simulate_pulse(energy_kev: float, params: DetectorParams,
                   eq: EquilibriumState,
                   duration: float = 4.0,
                   t_onset: float = 0.3,
                   f_sample: float = 1000.0) -> tuple:
    """Simulate a single detector pulse.

    Parameters
    ----------
    energy_kev : float
        Deposited energy in keV (must be > 0).
    params : DetectorParams
        Detector parameters.
    eq : EquilibriumState
        Pre-computed equilibrium state for these params.
    duration : float
        Total window duration [s].
    t_onset : float
        Time of energy deposition within the window [s].
    f_sample : float
        Output sampling rate [Hz].

    Returns
    -------
    t : ndarray, shape (n_samples,)
        Time array [s].
    v : ndarray, shape (n_samples,)
        Bolometer voltage pulse [V], positive-going, amplified + filtered.
    """
    # Internal high-rate time grid
    n_internal = int(duration * F_INTERNAL)
    n_pre = int(t_onset * F_INTERNAL)
    t_ode_end = duration - t_onset

    # Initial condition: instantaneous crystal temperature kick
    E_joules = energy_kev * KEV_TO_J
    C_c_eq = params.c_c * eq.T_c ** 3
    dTc0 = E_joules / C_c_eq

    # ODE time points (after onset)
    n_post = n_internal - n_pre
    t_eval = np.linspace(0, t_ode_end, n_post, endpoint=False)

    # Solve
    sol = solve_ivp(
        _ode_rhs, [0, t_ode_end], [0.0, 0.0, dTc0, 0.0],
        args=(params, eq),
        method='Radau',
        t_eval=t_eval,
        rtol=1e-8, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Build full internal waveform (zeros before onset)
    v_internal = np.zeros(n_internal)
    v_internal[n_pre:n_pre + len(sol.y[0])] = sol.y[0]  # delta_v_bol

    # Invert (delta_v is negative for energy deposition) and amplify
    v_internal = -v_internal * params.gain

    # Bessel filter at internal rate
    sos = _make_bessel_sos(params, F_INTERNAL)
    v_internal = sosfilt(sos, v_internal)

    # Decimate to output sample rate
    dec = int(F_INTERNAL / f_sample)
    v_out = v_internal[::dec]
    n_out = int(duration * f_sample)
    v_out = v_out[:n_out]
    t_out = np.arange(n_out) / f_sample

    return t_out, v_out


def simulate_pileup(energies_kev: list, onsets: list,
                    params: DetectorParams, eq: EquilibriumState,
                    duration: float = 4.0,
                    f_sample: float = 1000.0) -> tuple:
    """Simulate a pileup event using linear superposition.

    Each pulse is solved independently and the results are summed before
    applying the Bessel filter.

    Parameters
    ----------
    energies_kev : list of float
        Energy of each pulse [keV].
    onsets : list of float
        Onset time of each pulse [s].
    params : DetectorParams
        Detector parameters (shared for all pulses).
    eq : EquilibriumState
        Pre-computed equilibrium.
    duration : float
        Total window duration [s].
    f_sample : float
        Output sampling rate [Hz].

    Returns
    -------
    t : ndarray, shape (n_samples,)
    v : ndarray, shape (n_samples,)
    """
    n_internal = int(duration * F_INTERNAL)
    v_sum = np.zeros(n_internal)

    for E, t_on in zip(energies_kev, onsets):
        n_pre = int(t_on * F_INTERNAL)
        t_ode_end = duration - t_on
        n_post = n_internal - n_pre
        t_eval = np.linspace(0, t_ode_end, n_post, endpoint=False)

        E_joules = E * KEV_TO_J
        C_c_eq = params.c_c * eq.T_c ** 3
        dTc0 = E_joules / C_c_eq

        sol = solve_ivp(
            _ode_rhs, [0, t_ode_end], [0.0, 0.0, dTc0, 0.0],
            args=(params, eq),
            method='Radau',
            t_eval=t_eval,
            rtol=1e-8, atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed for E={E} keV: {sol.message}")

        v_sum[n_pre:n_pre + len(sol.y[0])] += sol.y[0]

    # Invert and amplify
    v_sum = -v_sum * params.gain

    # Bessel filter
    sos = _make_bessel_sos(params, F_INTERNAL)
    v_sum = sosfilt(sos, v_sum)

    # Decimate
    dec = int(F_INTERNAL / f_sample)
    n_out = int(duration * f_sample)
    v_out = v_sum[::dec][:n_out]
    t_out = np.arange(n_out) / f_sample

    return t_out, v_out
