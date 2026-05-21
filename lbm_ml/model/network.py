from collections.abc import Callable

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

from lbm_ml.lattice.symmetry import (
    D4Symmetry,
    D4AntiSymmetry,
    AlgReconstruction,
    LatticeEquivariantLayer,
    ConservationCorrection,
)
from lbm_ml.model.losses import rmsre

# ---------------------------------------------------------------------------
# Inner sub-networks
# ---------------------------------------------------------------------------


def sequential_model(Q=9, n_hidden_layers=2, n_per_layer=50, activation="relu", ll_activation="linear", bias=False):
    """Plain feed-forward inner network (no skip connections)."""
    model = Sequential(
        [
            keras.Input(shape=(Q,)),
            Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"),
        ]
    )
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"))
    model.add(Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform"))
    return model


def resnet_sequential_model(
    Q=9, n_hidden_layers=2, n_per_layer=50, activation="relu", ll_activation="linear", bias=False
):
    """Residual inner network: project → residual blocks → project back.

    Each residual block is a two-layer bottleneck:
        x_new = W₂(activation(W₁·x)) + x
    The second Dense (W₂) has no activation, so its output can be any sign.
    This lets the skip connection genuinely correct in either direction —
    unlike a single-layer relu block where relu(W·x) ≥ 0 forces the hidden
    state to only grow, crippling the network's expressiveness.
    """
    inp = keras.Input(shape=(Q,))

    # Project input to hidden dimension
    x = Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform")(inp)

    # Two-layer residual blocks: activate → linear projection → add skip
    for _ in range(n_hidden_layers):
        residual = x
        x = Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform")(x)
        # No activation here: output can be negative, so skip can correct either way
        x = Dense(n_per_layer, activation=None, use_bias=bias, kernel_initializer="he_uniform")(x)
        x = layers.Add()([x, residual])

    # Project back to Q populations
    out = Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform")(x)

    return keras.Model(inputs=inp, outputs=out)


# ---------------------------------------------------------------------------
# D4-equivariant wrappers
# ---------------------------------------------------------------------------


def _wrap_d4(
    sub_model_fn, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias
) -> keras.Model:
    """Wrap any inner sub-network factory in the D4-equivariant lift/pool pattern."""
    the_input = keras.Input(shape=(Q,))

    sub = sub_model_fn(Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)

    input_lst = D4Symmetry()(the_input)
    output_lst = [sub(x) for x in input_lst]
    output_lst = [AlgReconstruction()(input_lst[k], x) for k, x in enumerate(output_lst)]
    output_lst = D4AntiSymmetry()(output_lst)

    the_output = layers.Average()(output_lst)
    model = keras.Model(inputs=the_input, outputs=the_output)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def create_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """D4-equivariant network with a plain feed-forward inner sub-network.

    Architecture:
      1. Lift input to all 8 D4-transformed copies (D4Symmetry).
      2. Pass each copy through the same shared-weight sequential sub-network.
      3. Enforce conservation laws (AlgReconstruction) on each branch output.
      4. Undo each transform (D4AntiSymmetry) then average.
    """
    return _wrap_d4(sequential_model, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)


def create_resnet_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """D4-equivariant network with a residual inner sub-network.

    Identical equivariant wrapper as create_model; the inner sub-network uses
    skip connections (ResNet-style) instead of a plain sequential stack.
    """
    return _wrap_d4(
        resnet_sequential_model, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias
    )


# ---------------------------------------------------------------------------
# Lattice-Equivariant Neural Network (LENN) — Ortali & Gabbana et al. (2025)
# ---------------------------------------------------------------------------


def lenn_core(
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 10,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = True,
) -> keras.Model:
    """Stack of lattice-equivariant layers acting on population features.

    The single-channel input population vector x ∈ R^Q is lifted to a
    multi-channel population feature R^{Q×n_per_layer}, transformed through
    ``n_hidden_layers`` equivariant layers, and projected back to one channel
    (R^{Q×1}).  Every layer is equivariant by construction (parameter sharing),
    so the whole stack is equivariant without any group averaging.

    ``n_per_layer`` plays the role of the channel count C in the paper (cf. the
    LENN architecture column of Table 1, e.g. [1, 1, 8, 8, 10] in 2D).
    """
    inp = keras.Input(shape=(Q,))
    x = layers.Reshape((Q, 1))(inp)

    # Lift to C channels, then run the equivariant hidden stack.
    x = LatticeEquivariantLayer(n_per_layer, activation=activation, use_bias=bias, Q=Q)(x)
    for _ in range(n_hidden_layers):
        x = LatticeEquivariantLayer(n_per_layer, activation=activation, use_bias=bias, Q=Q)(x)

    # Project back to a single-channel population vector.
    x = LatticeEquivariantLayer(1, activation=None, use_bias=bias, Q=Q)(x)
    x = layers.Reshape((Q,))(x)

    # Component-wise final activation: softmax over the Q populations is itself
    # permutation-equivariant (the normalising sum is permutation-invariant),
    # so it does not break the lattice equivariance of the network.
    x = layers.Activation(ll_activation)(x)
    return keras.Model(inputs=inp, outputs=x)


def create_lenn_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 10,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = True,
) -> keras.Model:
    """Lattice-equivariant neural network (LENN).

    Architecture:
      1. Run the equivariant LENN core (lift → equivariant stack → project).
      2. Enforce mass/momentum conservation with the isotropic, equivariance-
         preserving ConservationCorrection (Eq. 40-42).

    Unlike the GAVG-based ``d4equivariant``/``resnet`` models, equivariance here
    is built into each layer's weights, so there is no 8-way lift/average.
    """
    inp = keras.Input(shape=(Q,))
    core = lenn_core(Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)
    out = ConservationCorrection()(inp, core(inp))
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ---------------------------------------------------------------------------
# Model registry — maps name → factory function
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable] = {
    "d4equivariant": create_model,
    "resnet": create_resnet_model,
    "lenn": create_lenn_model,
}
