"""D2Q9 Hermite-moment machinery for the Neural Collision Operator (NCO).

Background — Bedrunka et al., Phys. Rev. E 112, 055308 (2025)
-------------------------------------------------------------
The NCO is a multiple-relaxation-time (MRT) collision operator written in
moment space [paper Eq. (7)]:

    Gamma(f, theta) = -M^{-1} S_theta (M f - m^eq),

so the post-collision populations are

    f_post = f - M^{-1} S_theta (M f - m^eq).

Here ``M`` transforms populations into Hermite moments ``m = M f`` ordered by
polynomial order ``n`` [paper Eq. (12)], and ``S_theta`` is a diagonal matrix of
relaxation rates.  Moments are relaxed group-wise by order so that the operator
stays equivariant under the lattice symmetry group:

  * order 0 (density) and order 1 (momentum) are conserved -> rate fixed to 1,
  * order 2 (stress / shear) -> rate ``omega_nu`` set by the viscosity,
  * orders >= 3 (ghost moments) -> rates predicted by the neural network.

This module provides the D2Q9 analogue of the paper's D3Q27 construction: the
Hermite transform matrix, its inverse, the per-row order grouping, and the
Keras layers that perform the moment transform and the MRT collision.
"""

import numpy as np
import tensorflow as tf
import keras

from lbm_ml.lattice.stencil import LB_stencil

# ---------------------------------------------------------------------------
# D2Q9 Hermite moment basis (constant, built once at import time)
# ---------------------------------------------------------------------------
_c, _w, _CS2, _compute_feq = LB_stencil()
_CX = _c[:, 0].astype(np.float64)
_CY = _c[:, 1].astype(np.float64)

# Rows are the Hermite polynomials H^(n)(c_i) evaluated at each lattice
# velocity, grouped by order n [paper Eq. (12)].  In 2D the nine moments split
# as 1 + 2 + 3 + 2 + 1 across orders n = 0..4.
_M = np.array(
    [
        np.ones(9),                              # n0: density            (conserved)
        _CX,                                     # n1: x-momentum         (conserved)
        _CY,                                     # n1: y-momentum         (conserved)
        _CX * _CX - _CS2,                        # n2: stress xx          (shear, omega_nu)
        _CY * _CY - _CS2,                        # n2: stress yy          (shear, omega_nu)
        _CX * _CY,                               # n2: stress xy          (shear, omega_nu)
        (_CX * _CX - _CS2) * _CY,                # n3: ghost xxy          (network)
        _CX * (_CY * _CY - _CS2),                # n3: ghost xyy          (network)
        (_CX * _CX - _CS2) * (_CY * _CY - _CS2),  # n4: ghost xxyy         (network)
    ]
)
_MINV = np.linalg.inv(_M)

# Polynomial order of each moment row, used to relax equal-order moments with a
# single shared rate (the condition for equivariance in the paper).
MOMENT_ORDERS: tuple[int, ...] = (0, 1, 1, 2, 2, 2, 3, 3, 4)

# Orders whose relaxation rate the neural network predicts (everything above the
# viscosity-controlled shear moments).  For D2Q9 these are orders 3 and 4, so
# the network outputs two rates.  Each row in MOMENT_ORDERS equal to one of
# these orders shares that order's predicted rate.
HIGHER_ORDERS: tuple[int, ...] = (3, 4)
N_RATE_GROUPS: int = len(HIGHER_ORDERS)

# Count of moment rows per higher-order group, in HIGHER_ORDERS order.
_GROUP_SIZES: tuple[int, ...] = tuple(MOMENT_ORDERS.count(n) for n in HIGHER_ORDERS)
# Number of conserved (rate==1) rows and shear (rate==omega_nu) rows.
_N_CONSERVED: int = sum(1 for n in MOMENT_ORDERS if n <= 1)
_N_SHEAR: int = MOMENT_ORDERS.count(2)


# ---------------------------------------------------------------------------
# Keras layers
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="lbm")
class MomentTransform(keras.layers.Layer):
    """Map populations to Hermite moments: ``m = M f`` (per sample).

    Input  : populations, shape (batch, 9)
    Output : moments, shape (batch, 9), ordered by polynomial order n.
    """

    def call(self, f):
        M = tf.cast(tf.constant(_M), f.dtype)
        return tf.matmul(f, M, transpose_b=True)


@keras.saving.register_keras_serializable(package="lbm")
class SigmoidSafety(keras.layers.Layer):
    """Bound raw network outputs to relaxation rates in (0.5, 1.0) [paper Eq. (18)].

        tau_n = 1 / (2 (1 + e^x)) + 0.5

    The lower bound 0.5 guarantees numerical stability; the upper bound 1.0
    corresponds to relaxing the ghost moments fully to equilibrium (regularized
    LBM).  Embedding these physical bounds in the architecture stabilises
    training and makes the operator robust by construction.
    """

    def call(self, x):
        return 1.0 / (2.0 * (1.0 + tf.exp(x))) + 0.5


@keras.saving.register_keras_serializable(package="lbm")
class NCOCollision(keras.layers.Layer):
    """MRT collision in moment space [paper Eq. (7)].

        f_post = f - M^{-1} S_theta (M f - m^eq)

    The conserved (order 0/1) moments relax at rate 1 and the shear (order 2)
    moments at ``omega_nu``; the network supplies the higher-order rates.
    Because the conserved-moment differences ``(M f - m^eq)`` are zero, mass and
    momentum are conserved exactly regardless of the predicted rates.

    Call signature
    --------------
    fpre          : pre-collision populations, shape (batch, 9)
    higher_rates  : network-predicted rates, shape (batch, N_RATE_GROUPS)
                    (one rate per higher-order group, already bounded)
    """

    def __init__(self, omega_nu: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.omega_nu = omega_nu

    def call(self, fpre, higher_rates):
        dt = fpre.dtype
        M = tf.cast(tf.constant(_M), dt)
        Minv = tf.cast(tf.constant(_MINV), dt)
        cx = tf.cast(tf.constant(_CX), dt)
        cy = tf.cast(tf.constant(_CY), dt)
        w = tf.cast(tf.constant(_w), dt)
        cs2 = tf.cast(tf.constant(_CS2), dt)

        # -- Macroscopic moments from the pre-collision populations --
        rho = tf.reduce_sum(fpre, axis=-1, keepdims=True)
        ux = tf.reduce_sum(fpre * cx, axis=-1, keepdims=True) / rho
        uy = tf.reduce_sum(fpre * cy, axis=-1, keepdims=True) / rho

        # -- Discrete equilibrium f^eq (same 2nd-order form as the stencil) --
        cu = (ux * cx + uy * cy) / cs2
        uu = (ux * ux + uy * uy) / cs2
        feq = w * rho * (1.0 + cu + 0.5 * (cu * cu - uu))

        # -- Moment-space relaxation --
        m = tf.matmul(fpre, M, transpose_b=True)
        meq = tf.matmul(feq, M, transpose_b=True)

        batch = tf.shape(fpre)[0]
        conserved = tf.ones((batch, _N_CONSERVED), dt)
        shear = self.omega_nu * tf.ones((batch, _N_SHEAR), dt)
        # Repeat each predicted rate across its order group (e.g. order 3 has
        # two moments that share one rate) to preserve equivariance.
        higher = [
            tf.repeat(higher_rates[:, k : k + 1], size, axis=-1)
            for k, size in enumerate(_GROUP_SIZES)
        ]
        S = tf.concat([conserved, shear, *higher], axis=-1)

        delta = S * (m - meq)
        return fpre - tf.matmul(delta, Minv, transpose_b=True)

    def get_config(self):
        config = super().get_config()
        config.update({"omega_nu": self.omega_nu})
        return config
