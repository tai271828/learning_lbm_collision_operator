import numpy as np
import tensorflow as tf
import keras

from lbm_ml.lattice.stencil import LB_stencil


def LBrot90(f, k=1):
    """Rotate the D2Q9 population vector by k×90° counter-clockwise.

    f : tensor of shape (batch, 9)
    k : number of 90° rotation steps (positive = CCW)
    """
    # Index 0 (rest) is unchanged.
    # Indices 1–4 (axis-aligned) and 5–8 (diagonal) each cycle as a group.
    return tf.concat(
        [f[:, 0, None], tf.roll(f[:, 1:5], k, axis=-1), tf.roll(f[:, 5:], k, axis=-1)],
        axis=-1,
    )


def LBmirror(f):
    """Reflect the D2Q9 population vector across the x-axis (swap North↔South).

    This swaps direction indices so that populations moving in the +y direction
    are exchanged with their -y counterparts:
        2 (N) ↔ 4 (S),  5 (NE) ↔ 8 (SE),  6 (NW) ↔ 7 (SW)
    """
    return tf.concat(
        [
            f[:, 0, None],  # rest — unchanged
            f[:, 1, None],  # East — unchanged (on mirror axis)
            f[:, 4, None],  # was South → now North
            f[:, 3, None],  # West — unchanged (on mirror axis)
            f[:, 2, None],  # was North → now South
            f[:, 8, None],  # was SE → now NE
            f[:, 7, None],  # was SW → now NW
            f[:, 6, None],  # was NW → now SW
            f[:, 5, None],  # was NE → now SE
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# D4 symmetry helpers
# ---------------------------------------------------------------------------
# The square lattice has the dihedral symmetry group D4: 4 rotations (0°, 90°,
# 180°, 270°) and 4 reflections.  A physically correct collision operator must
# be equivariant under these 8 transforms — if you rotate the input populations
# by 90°, the output should rotate by 90° too.
#
# Pattern (group-equivariant lift/pool):
#   1. D4Symmetry  — "lift": given one input, produce all 8 group-transformed
#      copies so the network sees every orientation.
#   2. Process each copy through the same (shared-weight) sub-network.
#   3. D4AntiSymmetry — "project": undo the transform on each output and
#      average, so the final result is invariant (or equivariant) by construction.


@keras.saving.register_keras_serializable(package="lbm")
class D4Symmetry(keras.layers.Layer):
    """Lift a single population vector to all 8 D4-transformed copies.

    Input  : tensor of shape (batch, 9)
    Output : list of 8 tensors, each of shape (batch, 9), corresponding to
             0°, 90°, 180°, 270° rotations and their x-axis mirror images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return [
            x,                          # identity (0°)
            LBrot90(x, k=1),            # 90° CCW
            LBrot90(x, k=2),            # 180°
            LBrot90(x, k=3),            # 270° CCW
            LBmirror(x),                # mirror
            LBmirror(LBrot90(x, k=1)),  # mirror ∘ 90°
            LBmirror(LBrot90(x, k=2)),  # mirror ∘ 180°
            LBmirror(LBrot90(x, k=3)),  # mirror ∘ 270°
        ]


@keras.saving.register_keras_serializable(package="lbm")
class D4AntiSymmetry(keras.layers.Layer):
    """Undo each D4 transform on the corresponding processed output.

    This is the inverse of D4Symmetry: it maps the 8 transformed outputs back
    to the original orientation so they can be meaningfully averaged.

    Input  : list of 8 tensors (one per group element), shape (batch, 9) each
    Output : list of 8 tensors in the canonical (identity) orientation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return [
            x[0],                        # identity — no transform needed
            LBrot90(x[1], k=-1),         # undo 90° CCW → rotate 90° CW
            LBrot90(x[2], k=-2),         # undo 180°
            LBrot90(x[3], k=-3),         # undo 270° CCW
            LBmirror(x[4]),              # mirror is its own inverse
            LBrot90(LBmirror(x[5]), k=-1),
            LBrot90(LBmirror(x[6]), k=-2),
            LBrot90(LBmirror(x[7]), k=-3),
        ]


@keras.saving.register_keras_serializable(package="lbm")
class AlgReconstruction(keras.layers.Layer):
    """Recover the full 9-component population from a symmetry-reduced prediction.

    Background
    ----------
    The D4 symmetry of the square lattice means that some of the 9 populations
    are not independent: once 6 of the 9 are known, the remaining 3 can be
    derived from the conservation laws (mass and two momentum components):
        Σ_i f_i         = rho   (mass)
        Σ_i f_i c_{ix}  = rho*ux (x-momentum)
        Σ_i f_i c_{iy}  = rho*uy (y-momentum)

    The network therefore only predicts a reduced set of populations (fpred).
    This layer uses the three conservation constraints to algebraically solve
    for the three missing components (indices 2, 5, 8) and reconstructs the
    full post-collision population.

    Parameters
    ----------
    fpre  : pre-collision populations  (batch, 9) — provides the reference values
    fpred : network output             (batch, 9) — the predicted correction

    Returns
    -------
    Tensor of shape (batch, 9) — the physically consistent post-collision populations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, fpre, fpred):
        # Difference between predicted and pre-collision populations
        df = fpred - fpre

        # Solve for the three constrained directions (2, 5, 8) so that
        # mass and momentum are conserved exactly.
        # These linear expressions come from substituting the stencil weights
        # and velocity vectors into the three conservation equations and solving
        # for df[2], df[5], df[8] given the other six df values.
        df2 = -(df[:, 0] + 2 * df[:, 3] + df[:, 4] + 2 * df[:, 6] + 2 * df[:, 7])
        df5 = 0.5 * (df[:, 0] + 3 * df[:, 3] + 2 * df[:, 4] + 2 * df[:, 6] + 4 * df[:, 7] - df[:, 1])
        df8 = -0.5 * (df[:, 0] + df[:, 1] + df[:, 3] + 2 * df[:, 4] + 2 * df[:, 7])

        # Reassemble the full correction vector with the reconstructed directions
        df = tf.concat(
            [
                df[:, 0, None],
                df[:, 1, None],
                df2[:, None],  # reconstructed
                df[:, 3, None],
                df[:, 4, None],
                df5[:, None],  # reconstructed
                df[:, 6, None],
                df[:, 7, None],
                df8[:, None],  # reconstructed
            ],
            axis=-1,
        )

        # Add the correction back to the pre-collision state
        return fpre + df


# ===========================================================================
# Lattice-Equivariant Neural Networks (LENN) — Ortali & Gabbana et al. (2025)
# ===========================================================================
# The GAVG pattern above (D4Symmetry / D4AntiSymmetry) enforces equivariance
# *externally*: it runs a shared sub-network on all 8 transformed copies of the
# input and averages.  Its cost scales with the size of the symmetry group.
#
# LENNs instead bake equivariance into *each individual layer* by constraining
# the layer's affine weights.  Lattice symmetries act on populations as
# permutations P (Eq. 15); an affine map A·x + b is equivariant w.r.t. the
# group iff
#       A P = P A        and        b = P b      ∀ P ∈ {generators}      (Eq. 24)
# The solution is a *parameter-sharing* pattern: A is constant on the orbits of
# index pairs (i, j) under the group, and b is constant on the orbits of single
# indices.  For D2Q9 this collapses the 9×9 + 9 = 90 free parameters down to
# 15 + 3 = 18 (Eq. 25-26), exactly the degrees of freedom reported in the paper.


def d2q9_generators() -> list[list[int]]:
    """Permutation generators (R, S) of the D4 group on the D2Q9 stencil.

    Each generator is given in "source-index" form: applying permutation ``p``
    to a population vector ``f`` yields ``(p·f)[i] = f[p[i]]``.  These match the
    explicit LBrot90 (90° rotation R) and LBmirror (reflection S) maps above.
    """
    perm_R = [0, 4, 1, 2, 3, 8, 5, 6, 7]  # 90° rotation  (cf. LBrot90, k=1)
    perm_S = [0, 1, 4, 3, 2, 8, 7, 6, 5]  # x-axis mirror (cf. LBmirror)
    return [perm_R, perm_S]


def build_group(generators: list[list[int]], Q: int) -> list[tuple[int, ...]]:
    """Generate the full permutation group from its generators by BFS closure.

    Composition uses ``compose(a, b)[i] = a[b[i]]`` (right-multiplication by the
    generators), which reaches every element of the group for a finite group.
    """
    identity = tuple(range(Q))
    group: set[tuple[int, ...]] = {identity}
    frontier = [identity]
    while frontier:
        p = frontier.pop()
        for g in generators:
            new = tuple(p[g[i]] for i in range(Q))
            if new not in group:
                group.add(new)
                frontier.append(new)
    return list(group)


def pair_orbits(group: list[tuple[int, ...]], Q: int) -> np.ndarray:
    """Map each index pair (i, j) to its orbit id under the group action.

    The number of distinct orbits equals the number of independent weights
    #A_q in the equivariant weight matrix (15 for D2Q9, cf. Eq. 26).
    """
    orbit_id = -np.ones((Q, Q), dtype=np.int64)
    nxt = 0
    for i in range(Q):
        for j in range(Q):
            if orbit_id[i, j] != -1:
                continue
            for p in group:
                orbit_id[p[i], p[j]] = nxt
            nxt += 1
    return orbit_id


def point_orbits(group: list[tuple[int, ...]], Q: int) -> np.ndarray:
    """Map each index i to its orbit id under the group action.

    The number of distinct orbits equals the number of independent bias
    weights #b_q (3 for D2Q9: {rest}, {axis-aligned}, {diagonal}).
    """
    orbit_id = -np.ones((Q,), dtype=np.int64)
    nxt = 0
    for i in range(Q):
        if orbit_id[i] != -1:
            continue
        for p in group:
            orbit_id[p[i]] = nxt
        nxt += 1
    return orbit_id


@keras.saving.register_keras_serializable(package="lbm")
class LatticeEquivariantLayer(keras.layers.Layer):
    """A single lattice-equivariant (LENN) layer — Ortali & Gabbana et al. (2025).

    Generalises the equivariant affine operator (S1, Eq. 24) to multi-channel
    "population features" x ∈ R^{Q×C} (S2-S3, Eq. 31-38) and implements the
    memory-efficient parametrisation of S4: only the independent weights
    Ã ∈ R^{#A_q × C_in × C_out} and b̃ ∈ R^{#b_q × C_out} are stored, and the
    full tensors are reconstructed on the fly by scattering them over the
    precomputed orbit pattern (the Λ / Δ tensors of Eq. 27/37).

    Input  : (batch, Q, C_in)   population feature
    Output : (batch, Q, C_out)  population feature  (after activation σ)

    Equivariance L(P x) = P L(x) holds by construction for every P in the
    lattice symmetry group, so no group averaging is required.
    """

    def __init__(
        self,
        channels_out: int,
        activation=None,
        use_bias: bool = True,
        generators: list[list[int]] | None = None,
        Q: int = 9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels_out = int(channels_out)
        self.activation = keras.activations.get(activation)
        self.use_bias = bool(use_bias)
        self.Q = int(Q)
        if generators is None:
            generators = d2q9_generators()
        self._generators = [[int(v) for v in g] for g in generators]

        group = build_group(self._generators, self.Q)
        self._pair_orbit = pair_orbits(group, self.Q)    # (Q, Q)
        self._point_orbit = point_orbits(group, self.Q)  # (Q,)
        self.n_A = int(self._pair_orbit.max()) + 1        # #A_q  (15 for D2Q9)
        self.n_b = int(self._point_orbit.max()) + 1       # #b_q  (3  for D2Q9)

    def build(self, input_shape):
        c_in = int(input_shape[-1])
        # Ã: the independent weights (one per index-pair orbit, per channel pair)
        self.A_tilde = self.add_weight(
            name="A_tilde",
            shape=(self.n_A, c_in, self.channels_out),
            initializer="he_uniform",
            trainable=True,
        )
        if self.use_bias:
            # b̃: one independent bias per point orbit, per output channel
            self.b_tilde = self.add_weight(
                name="b_tilde",
                shape=(self.n_b, self.channels_out),
                initializer="zeros",
                trainable=True,
            )
        # Orbit patterns as constant gather indices (Λ and Δ of Eq. 27/37).
        self._pair_idx = tf.constant(self._pair_orbit, dtype=tf.int32)
        self._point_idx = tf.constant(self._point_orbit, dtype=tf.int32)
        super().build(input_shape)

    def call(self, x):
        # Reconstruct the full equivariant weight tensor A_{ijbc} by scattering
        # the independent weights over the orbit pattern: (Q, Q, C_in, C_out).
        A_full = tf.gather(self.A_tilde, self._pair_idx)
        # A x_{ic} = Σ_{j,b} A_{ijbc} x_{jb}   (Eq. 34/38)
        y = tf.einsum("ijbc,njb->nic", A_full, x)
        if self.use_bias:
            b_full = tf.gather(self.b_tilde, self._point_idx)  # (Q, C_out)
            y = y + b_full[None, :, :]
        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.Q, self.channels_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "channels_out": self.channels_out,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "generators": self._generators,
                "Q": self.Q,
            }
        )
        return cfg


@keras.saving.register_keras_serializable(package="lbm")
class ConservationCorrection(keras.layers.Layer):
    """Isotropic mass/momentum conservation correction (Eq. 40-42).

    Given the pre-collision populations ``fpre`` and a raw post-collision
    prediction ``fpost``, the collision increment Ω = fpost − fpre is shifted by
    an isotropic correction
        Ω_i ← Ω_i + κ₁ + κ₂ c_{ix} + κ₃ c_{iy}
    with the κ's chosen so that mass (Σ Ω_i) and momentum (Σ Ω_i c_i) both
    vanish.  Unlike the index-specific AlgReconstruction used by the GAVG
    branches, this correction is built purely from lattice-isotropic projections
    and therefore preserves the D4 equivariance of a LENN — it is applied once,
    to the single equivariant output, with no averaging needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c, _, _, _ = LB_stencil()
        self._cx = c[:, 0].astype(np.float64)
        self._cy = c[:, 1].astype(np.float64)
        # Normalisation denominators: Q for mass, Σ c_x² for momentum.
        self._n_mass = float(c.shape[0])
        self._n_mom = float(np.sum(c[:, 0] ** 2))

    def call(self, fpre, fpost):
        cx = tf.cast(tf.constant(self._cx), fpost.dtype)
        cy = tf.cast(tf.constant(self._cy), fpost.dtype)

        omega = fpost - fpre
        k1 = -tf.reduce_sum(omega, axis=-1, keepdims=True) / self._n_mass
        k2 = -tf.reduce_sum(omega * cx, axis=-1, keepdims=True) / self._n_mom
        k3 = -tf.reduce_sum(omega * cy, axis=-1, keepdims=True) / self._n_mom
        omega = omega + k1 + k2 * cx + k3 * cy
        return fpre + omega
