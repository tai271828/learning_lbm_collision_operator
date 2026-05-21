from collections.abc import Callable

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

from lbm_ml.lattice.symmetry import D4Symmetry, D4AntiSymmetry, AlgReconstruction
from lbm_ml.lattice.moments import (
    MomentTransform, SigmoidSafety, NCOCollision, N_RATE_GROUPS,
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
# Neural Collision Operator (Bedrunka et al., Phys. Rev. E 112, 055308)
# ---------------------------------------------------------------------------


def _inner_rate_network(Q, n_hidden_layers, n_per_layer, activation, n_out):
    """Shallow feed-forward net mapping moments → raw higher-order rates.

    Matches the paper's standard sub-network: two hidden layers of 20 nodes
    each, separated by ReLU, with biases (1064 parameters for the D3Q27 case).
    Output is linear; bounding to (0.5, 1.0) is done afterwards by SigmoidSafety.
    """
    model = Sequential([keras.Input(shape=(Q,))])
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_layer, activation=activation, use_bias=True, kernel_initializer="he_uniform"))
    model.add(Dense(n_out, activation="linear", use_bias=True))
    return model


def create_nco_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 20,
    activation: str = "relu",
    ll_activation: str = "linear",  # accepted for registry compatibility; unused
    bias: bool = True,
) -> keras.Model:
    """Neural Collision Operator with an invariant network [paper Sec. II.B].

    The network predicts the higher-order relaxation rates of an MRT collision
    operator; the conserved and shear moments use fixed rates. It is made
    *invariant* to the lattice symmetry group by averaging a shared sub-network
    over all group-transformed moment inputs [Eq. (15)], and its outputs are
    bounded to (0.5, 1.0) by a sigmoid safety layer [Eq. (18)].

    Architecture:
      1. Lift the input populations to all 8 D4-transformed copies (D4Symmetry).
      2. Transform each copy to Hermite moments m(g) = M·g·f (MomentTransform).
      3. Pass each through the same shared sub-network → raw rates per branch.
      4. Average the branches → group-invariant raw rates [Eq. (15)].
      5. Bound to (0.5, 1.0) via the sigmoid safety layer [Eq. (18)].
      6. Apply the MRT collision f_post = f − M⁻¹ S_θ(M f − m^eq) (NCOCollision).
    """
    the_input = keras.Input(shape=(Q,))

    inner = _inner_rate_network(Q, n_hidden_layers, n_per_layer, activation, N_RATE_GROUPS)
    moment = MomentTransform()

    f_copies = D4Symmetry()(the_input)                      # 8 group-transformed populations
    rate_branches = [inner(moment(fc)) for fc in f_copies]  # shared-weight sub-network
    raw_rates = layers.Average()(rate_branches)             # invariant rates, Eq. (15)
    higher_rates = SigmoidSafety()(raw_rates)               # bound to (0.5, 1.0), Eq. (18)
    the_output = NCOCollision()(the_input, higher_rates)    # MRT collision, Eq. (7)

    model = keras.Model(inputs=the_input, outputs=the_output)
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ---------------------------------------------------------------------------
# Model registry — maps name → factory function
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable] = {
    "d4equivariant": create_model,
    "resnet": create_resnet_model,
    "nco": create_nco_model,
}
