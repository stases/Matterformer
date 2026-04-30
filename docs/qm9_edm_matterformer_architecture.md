# QM9 EDM Matterformer Architecture Overview

This document describes the Matterformer QM9 EDM model implemented primarily in
`src/matterformer/models/qm9.py`, `src/matterformer/models/transformer.py`,
`src/matterformer/models/attention.py`, `src/matterformer/models/embeddings.py`,
`src/matterformer/geometry/adapters.py`, and `src/matterformer/tasks/edm.py`.

The main focus is the architecture: how scalar atom features `x`, coordinates
`pos`, padding masks, and the diffusion noise level `sigma` enter the model, how
they pass through the transformer trunk, and how atom and coordinate outputs are
read out. EDM preconditioning and loss are included because they define what the
network is actually asked to predict.

## 1. Notation And Input Shapes

For the native Matterformer QM9 path, tensors are padded by molecule:

| Symbol | Code name | Shape | Meaning |
| --- | --- | --- | --- |
| \(B\) | batch size | scalar | number of molecules |
| \(N\) | max atoms | scalar | max atom count in padded batch |
| \(C_x\) | `atom_channels` | scalar | atom feature channels, usually 5 atom one-hot channels plus optional charge |
| \(d\) | `d_model` | scalar | token hidden width |
| \(H\) | `n_heads` | scalar | number of attention heads |
| \(L\) | `n_layers` | scalar | number of transformer blocks |
| \(x\) | `atom_noisy` | \((B,N,C_x)\) | noised and preconditioned atom features entering `QM9EDMModel` |
| \(r\) | `coords_noisy` | \((B,N,3)\) | noised and preconditioned coordinates entering `QM9EDMModel` |
| \(p\) | `pad_mask` | \((B,N)\) | `True` means padding, `False` means a real atom |
| \(\sigma\) | `sigma` | \((B,)\) or \((B,1)\) | one diffusion noise level per molecule |

It is useful to define the valid-atom indicator

$$
m_{bi} = 1 - \mathbb{1}\{p_{bi} = \text{True}\}.
$$

Most operations explicitly zero padded atoms. Coordinate centering and coordinate
mean subtraction always use only valid atoms.

For a running example, suppose

$$
B=2,\quad N=29,\quad C_x=6,\quad d=768,\quad H=12,\quad L=19.
$$

Then `atom_noisy` has shape \((2,29,6)\), `coords_noisy` has shape
\((2,29,3)\), the token sequence entering the transformer has shape
\((2,29,768)\), and MHA heads have per-head width \(d_h = d/H = 64\).

## 2. Mental Map

Native Matterformer QM9 EDM path:

```text
clean QM9 batch
  atom types, optional charges, centered coordinates
          |
          v
EDMLoss samples sigma and adds Gaussian noise
          |
          v
EDMPreconditioner scales x and pos by c_in
          |
          v
QM9EDMModel
  sigma_condition = log(sigma) / 4
  x (+ optional sigma channel) -> atom_proj -> tokens
  pos -> optional coordinate token embedding -> add to tokens
  sigma_condition -> optional TimeEmbedder -> AdaLN conditioning vector
  pos, sigma -> optional RoPE / geometry bias / simplicial geometry inputs
          |
          v
TransformerTrunk: L pre-norm attention + MLP blocks
          |
          v
atom_head -> atom_delta
coord_head -> coord_delta
          |
          v
EDMPreconditioner converts residual deltas into denoised x and pos
```

Platonic adapter path:

```text
Platonic ragged tensors
  x:   (M, C_x)
  pos: (M, 3)
  batch: (M,)
          |
          v
Platonic EDMPrecond appends c_noise = log(sigma)/4 to scalar node features
          |
          v
MatterformerQM9Adapter
  recovers per-graph sigma from appended c_noise
  packs ragged nodes to padded tensors
          |
          v
same QM9EDMModel as above
          |
          v
unpack padded outputs back to ragged Platonic format
```

The adapter does not define a separate architecture. It only translates between
Platonic's ragged graph signature and Matterformer's padded QM9 signature.

## 3. EDM Training Target And Preconditioning

### 3.1 Clean Data Representation

The native QM9 EDM loss builds clean atom features as

$$
x_0 =
\left[
\frac{\mathrm{onehot}(\mathrm{atom\ type})}{s_x},
\frac{q}{s_q}
\right],
$$

where the charge column is present only when `use_charges=True`. The defaults in
`scripts/train_qm9_edm.py` are

$$
s_x = \texttt{atom_feature_scale} = 4,\qquad
s_q = \texttt{charge_feature_scale} = 8.
$$

Coordinates are centered per molecule:

$$
\bar r_b = \frac{\sum_i m_{bi} r_{bi}}{\sum_i m_{bi}},
\qquad
r^{\mathrm{centered}}_{bi} = m_{bi}(r_{bi} - \bar r_b).
$$

This means the target coordinate distribution has zero center of mass for every
molecule.

### 3.2 Noise Injection

For each molecule, the loss samples

$$
\log \sigma_b \sim \mathcal{N}(p_{\mathrm{mean}}, p_{\mathrm{std}}^2),
$$

with default \(p_{\mathrm{mean}}=-1.2\) and \(p_{\mathrm{std}}=1.2\). Atom features
receive independent Gaussian noise:

$$
x_{\sigma,b i} = x_{0,b i} + \sigma_b \epsilon^x_{b i}.
$$

Coordinate noise is also Gaussian, but it is centered per molecule before being
added:

$$
\epsilon^r_b \sim \mathcal{N}(0,I),\qquad
\tilde\epsilon^r_b = \mathrm{center}(\epsilon^r_b),
$$

$$
r_{\sigma,b} =
\mathrm{center}(r_{0,b} + \sigma_b \tilde\epsilon^r_b).
$$

This keeps all noised coordinates in the same centered coordinate gauge as the
clean data.

### 3.3 EDM Coefficients

`EDMPreconditioner` computes Karras-style coefficients

$$
c_{\mathrm{skip}}(\sigma)
=
\frac{\sigma_{\mathrm{data}}^2}{\sigma^2 + \sigma_{\mathrm{data}}^2},
$$

$$
c_{\mathrm{out}}(\sigma)
=
\frac{\sigma \sigma_{\mathrm{data}}}{\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}},
$$

$$
c_{\mathrm{in}}(\sigma)
=
\frac{1}{\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}}.
$$

The model does not see the raw noised tensors directly. It sees

$$
x_{\mathrm{in}} = c_{\mathrm{in}} x_\sigma,
\qquad
r_{\mathrm{in}} = \mathrm{center}(c_{\mathrm{in}} r_\sigma).
$$

`QM9EDMModel` returns residual-like deltas
\(\Delta x_\theta, \Delta r_\theta\). The preconditioner interprets them as
corrections to the scaled input:

$$
\hat x_{\mathrm{clean,scaled}} =
x_{\mathrm{in}} - \Delta x_\theta,
$$

$$
\hat r_{\mathrm{clean,scaled}} =
\mathrm{center}(r_{\mathrm{in}} - \Delta r_\theta).
$$

The final denoised prediction is

$$
D_x(x_\sigma,\sigma)
=
c_{\mathrm{skip}}x_\sigma
+
c_{\mathrm{out}}\hat x_{\mathrm{clean,scaled}},
$$

$$
D_r(r_\sigma,\sigma)
=
\mathrm{center}\left(
c_{\mathrm{skip}}r_\sigma
+
c_{\mathrm{out}}\hat r_{\mathrm{clean,scaled}}
\right).
$$

The loss is weighted MSE:

$$
w(\sigma) =
\frac{\sigma^2+\sigma_{\mathrm{data}}^2}
{(\sigma\sigma_{\mathrm{data}})^2},
$$

optionally clamped by `max_loss_weight`, and then applied to atom and coordinate
MSE over valid atoms.

Intuition: the neural network is the residual part of an EDM denoiser. The skip
path already carries a sigma-dependent fraction of the noisy input, while the
network predicts the correction needed in the preconditioned coordinate system.

## 4. QM9EDMModel Forward Pass

`QM9EDMModel.forward(atom_noisy, coords_noisy, pad_mask, sigma, lattice=None)`
does the following:

```text
sigma -> c = log(sigma) / 4

atom_noisy
  optionally concatenate c per atom
  Linear(C_x or C_x+1 -> d)
        |
        v
token_features h_0: (B, N, d)

coords_noisy
  optional centered coordinate embedding
        |
        v
additive coord token features: (B, N, d)

c
  optional TimeEmbedder(c, c)
        |
        v
conditioning vector: (B, d)

h_0, conditioning vector, coords, sigma
        |
        v
TransformerTrunk
        |
        v
h_L: (B, N, d)

h_L -> atom_head  -> atom_delta:  (B, N, C_x)
h_L -> coord_head -> coord_delta: (B, N, 3)
```

There are five main architectural entry points for coordinates:

1. Coordinates are scaled and centered before the model by `EDMPreconditioner`.
2. Coordinates may be embedded and added to token features.
3. Coordinates may define RoPE phases in MHA.
4. Coordinates may define geometry features for attention bias or simplicial
   attention structure.
5. Coordinates also enter the relative-vector coordinate readout when
   `coord_head_mode="equivariant"`.

These mechanisms are independent knobs, except that MHA RoPE requires
`attn_type="mha"`.

## 5. Noise Conditioning Modes

The scalar used inside the model is

$$
c = \frac{\log(\sigma)}{4}.
$$

`noise_conditioning` is canonicalized to a tuple containing zero or more of
`"concat"` and `"adaln"`.

### 5.1 `noise_conditioning="concat"`

The model appends \(c_b\) to every valid atom feature:

$$
x'_{bi} = [x_{bi}, c_b].
$$

Then

$$
h^{(0)}_{bi} = W_{\mathrm{atom}}x'_{bi} + b_{\mathrm{atom}}.
$$

Padding rows receive a zero sigma feature before projection and are masked again
inside the trunk.

This is the closest mode to Platonic EDM preconditioning, where \(c\) is
concatenated to scalar node features before calling the backbone.

### 5.2 `noise_conditioning="adaln"`

The model builds a global conditioning vector

$$
e_b = \mathrm{TimeEmbedder}(c_b,c_b) \in \mathbb{R}^d.
$$

`TimeEmbedder` makes sinusoidal embeddings of both arguments, concatenates them,
and applies an MLP. In QM9 EDM both inputs are currently the same value \(c_b\).

Each transformer block uses this vector to produce six AdaLN vectors:

$$
(\Delta\mu_a,\Delta s_a,g_a,\Delta\mu_m,\Delta s_m,g_m)
=
\mathrm{MLP}_{\mathrm{AdaLN}}(e_b).
$$

The attention and MLP residuals become

$$
\tilde h = \mathrm{LN}(h)(1+\Delta s_a) + \Delta\mu_a,
$$

$$
h \leftarrow h + g_a \odot \mathrm{Attn}(\tilde h),
$$

$$
\tilde h = \mathrm{LN}(h)(1+\Delta s_m) + \Delta\mu_m,
$$

$$
h \leftarrow h + g_m \odot \mathrm{MLP}(\tilde h).
$$

The final AdaLN linear layer is zero-initialized. Therefore at initialization
the shifts, scales, and gates are all zero, and attention/MLP residual branches
start gated off. This stabilizes startup, but also means the block initially
passes information mostly through the residual stream until the gates learn to
open.

### 5.3 `noise_conditioning="concat,adaln"`

Both paths are active:

```text
c enters token features as an extra scalar channel
c also enters every transformer block through AdaLN
```

This is the default legacy behavior when `noise_conditioning=None` and
`concat_sigma_condition` is not disabled.

### 5.4 `noise_conditioning="none"`

Neither concat nor AdaLN is active. The transformer blocks become plain pre-norm
residual transformer blocks:

$$
h \leftarrow h + \mathrm{Attn}(\mathrm{LN}(h)),
\qquad
h \leftarrow h + \mathrm{MLP}(\mathrm{LN}(h)).
$$

One subtlety: if geometry bias is enabled, the geometry bias modules still use
\(\sigma\) for their noise gate. Thus `"none"` removes direct token/AdaLN noise
conditioning, but may not remove every architectural dependence on \(\sigma\)
unless `use_geometry_bias=False`.

### 5.5 Legacy Flag

The old `concat_sigma_condition` flag is still accepted as a compatibility
input:

| Inputs | Canonical result |
| --- | --- |
| `noise_conditioning=None`, `concat_sigma_condition=True` or unset | `("concat", "adaln")` |
| `noise_conditioning=None`, `concat_sigma_condition=False` | `("adaln",)` |
| `noise_conditioning="concat"` | `("concat",)` |
| `noise_conditioning="adaln"` | `("adaln",)` |
| `noise_conditioning="concat,adaln"` or `"both"` | `("concat", "adaln")` |
| `noise_conditioning="none"` | `()` |

## 6. How Coordinates Enter The Model

Coordinates are the most overloaded input. They are not just another feature
column. They affect token construction, attention logits, attention value
transport, and coordinate readout depending on the selected options.

### 6.1 Centering

The EDM preconditioner centers coordinates before the model. Separately, the
coordinate token embedder also centers coordinates before embedding:

$$
r^{\mathrm{emb}} = \mathrm{center}(r_{\mathrm{in}}).
$$

For the additive coordinate embedding only, `coord_embed_normalize=True` further
divides by the masked RMS radius

$$
s_b =
\sqrt{
\frac{\sum_i m_{bi}\lVert r_{bi}^{\mathrm{emb}}\rVert^2}
{\sum_i m_{bi}}
},
\qquad
\tilde r_{bi} = \frac{r_{bi}^{\mathrm{emb}}}{\max(s_b,10^{-8})}.
$$

This normalization does not affect the coordinates passed to RoPE, geometry
bias, or coordinate heads. It only affects additive coordinate token embeddings.

### 6.2 `coord_embed_mode="none"`

No coordinate token embedding is added. Coordinates may still enter through
MHA RoPE, geometry bias, simplicial attention masks/biases, and coordinate
heads.

### 6.3 `coord_embed_mode="fourier"`

For each coordinate component and each frequency \(f=1,\dots,K\),

$$
\phi_{\mathrm{fourier}}(r)
=
\left[
\sin(2\pi f r_x),\cos(2\pi f r_x),
\sin(2\pi f r_y),\cos(2\pi f r_y),
\sin(2\pi f r_z),\cos(2\pi f r_z)
\right]_{f=1}^K.
$$

This has input dimension \(6K\). It is then projected through

$$
E_r(r) = W_2\,\mathrm{SiLU}(W_1\phi(r)+b_1)+b_2
\in \mathbb{R}^d,
$$

and added to token features:

$$
h^{(0)}_{bi} \leftarrow h^{(0)}_{bi} + E_r(r_{bi}).
$$

This embedding is axis-dependent. It gives the network direct access to
coordinates in the current frame; it is not itself rotation-equivariant.

### 6.4 `coord_embed_mode="rff"`

The fixed random Fourier feature mode samples a non-trainable projection

$$
P \in \mathbb{R}^{3\times K},\qquad
P_{jk}\sim \mathcal{N}(0,\sigma_{\mathrm{rff}}^2).
$$

Then

$$
z = rP,\qquad
\phi_{\mathrm{rff}}(r) = [\sin(2\pi z),\cos(2\pi z)].
$$

The feature dimension is \(2K\), followed by the same MLP to \(d\).

### 6.5 `coord_embed_mode="learnable_rff"`

The learnable RFF mode uses a trainable projection

$$
P \in \mathbb{R}^{3\times K},
$$

initialized as approximately \(P_{jk}\sim\mathcal{N}(0,1/\sigma_{\mathrm{rff}}^2)\).
The features are

$$
z = rP,\qquad
\phi_{\mathrm{learnable}}(r)
=
\sqrt{\frac{2}{2K}}[\sin(z),\cos(z)].
$$

This is the mode selected by aliases such as `"coords"`, `"on"`,
`"learned_rff"`, and `"fieldformer_rff"`.

### 6.6 `coord_embed_mode="rope"` Alias

`coord_embed_mode="rope"`, `"mha_rope"`, or `"rotary"` does not create an
additive coordinate embedding. It canonicalizes to

```text
coord_embed_mode = "none"
mha_position_mode = "rope"
```

This is important: RoPE changes attention internals, not the token input
features.

## 7. Geometry Features

`TransformerTrunk` can compute geometry features whenever a geometry adapter is
configured. QM9 uses `NonPeriodicGeometryAdapter` by default.

### 7.1 Nonperiodic Geometry

Given coordinates \(r\), the adapter builds pairwise displacements

$$
\delta_{ij} = r_i - r_j,
$$

distances

$$
\rho_{ij} = \lVert \delta_{ij}\rVert_2,
\qquad
\rho^2_{ij} = \lVert \delta_{ij}\rVert_2^2,
$$

and an RMS molecular radius

$$
s =
\sqrt{
\frac{\sum_i m_i\lVert r_i\rVert^2}
{\sum_i m_i}
}.
$$

The normalized distance is

$$
\hat\rho_{ij} = \frac{\rho_{ij}}{\max(s,10^{-8})}.
$$

The pair mask is

$$
M_{ij}=m_i m_j.
$$

Invalid pair features are zeroed.

### 7.2 Periodic Geometry

The same transformer stack can also accept `PeriodicGeometryAdapter` for MOF-like
periodic data. It expects a lattice tensor of shape \((B,6)\), converts that
latent lattice representation to a metric tensor, enumerates periodic image
offsets, and uses the nearest image displacement. The normalized distance is
scaled by the average cell scale rather than the molecular RMS radius.

QM9 uses nonperiodic geometry, but the attention bias code is written so that
periodic features can append lattice/global geometry terms.

## 8. Transformer Trunk

The trunk is a stack of \(L\) identical blocks plus a final layer norm:

```text
for block in blocks:
    h = h.masked_fill(pad_mask, 0)
    h = block(h, cond, masks, geometry inputs, positions)

h = LayerNorm(h)
h = h.masked_fill(pad_mask, 0)
```

Each block contains:

```text
LayerNorm
Attention: either MHA or 2-simplicial attention
Residual add, optionally AdaLN-gated
LayerNorm
MLP: Linear(d -> mlp_ratio*d) + GELU + Dropout + Linear(... -> d)
Residual add, optionally AdaLN-gated
```

The attention implementation is selected by `attn_type`.

## 9. Multi-Head Attention Path

Set

```text
attn_type = "mha"
```

to use standard pairwise multi-head self-attention. Without RoPE, the block uses
`torch.nn.MultiheadAttention`. With RoPE, it uses the custom
`RotaryMultiheadAttention`.

### 9.1 Plain MHA

For head \(h\),

$$
Q = HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

The attention logits are

$$
\ell_{ij}^{(h)}
=
\frac{\langle Q_i^{(h)},K_j^{(h)}\rangle}{\sqrt{d_h}}
+
b_{ij}^{(h)}
+
\mathrm{mask}_{ij},
$$

where \(b_{ij}^{(h)}\) is optional geometry bias.

### 9.2 MHA Geometry Bias: `mha_geom_bias_mode="standard"`

When `use_geometry_bias=True`, `attn_type="mha"`, and
`mha_geom_bias_mode="standard"`, the model builds a pairwise per-head bias from
geometry:

$$
b_{ij}^{(h)} =
g_h(\sigma)\left[
-\mathrm{softplus}(a_h)\hat\rho_{ij}
+
\mathrm{EdgeMLP}_h(\psi_{ij})
\right].
$$

The edge feature \(\psi_{ij}\) contains an RBF expansion of normalized distance
and the raw pair displacement:

$$
\psi_{ij} = [\mathrm{RBF}(\hat\rho_{ij}),\delta_{ij}]
$$

plus periodic Fourier/lattice features for periodic adapters.

The geometry MLP final layer is zero-initialized, so at initialization the
learned edge term is zero and only the distance slope term contributes. The
distance slope is negative by construction, so large distances initially reduce
attention logits.

The noise gate is

$$
g_h(\sigma)
=
\mathrm{sigmoid}\left(\alpha_h(-\log\sigma)+\beta_h\right),
\qquad
\alpha_h=\mathrm{softplus}(\alpha^{\mathrm{raw}}_h).
$$

At high noise, \(-\log\sigma\) is smaller, so the gate can learn to suppress
geometry bias when geometry is unreliable.

### 9.3 MHA Geometry Bias: `mha_geom_bias_mode="factorized_marginal"`

This mode reuses the same factorized geometric terms used by simplicial
attention:

$$
u_{ij}^{(h)},\qquad v_{ik}^{(h)},\qquad w_{jk}^{(h)}.
$$

Simplicial attention would apply these to triplet logits over \((j,k)\):

$$
B_{ijk}^{(h)} =
g_i^{(h)}(\sigma)\left(u_{ij}^{(h)}+v_{ik}^{(h)}+w_{jk}^{(h)}\right).
$$

For MHA, the code collapses this triplet bias to a pairwise bias by a log-sum-exp
marginal over \(k\). Conceptually:

$$
b_{ij}^{(h)}
\approx
g_i^{(h)}u_{ij}^{(h)}
+
\log\left(
\frac{1}{|\mathcal{N}_i|}
\sum_{k\in\mathcal{N}_i}
\exp\left(g_i^{(h)}(v_{ik}^{(h)}+w_{jk}^{(h)})\right)
\right).
$$

This lets ordinary MHA receive a bias that resembles the factorized
two-simplex geometry signal without actually attending over all pairs
\((j,k)\) as values.

## 10. MHA 3D RoPE

Set

```text
attn_type = "mha"
mha_position_mode = "rope"
```

to use coordinate-dependent rotary position embeddings.

### 10.1 Frequency Construction

For each head \(h\) and feature pair \(f\), the RoPE module has a 3D frequency
vector

$$
\omega_{hf}\in\mathbb{R}^3.
$$

The default frequencies are fixed spiral directions with magnitudes increasing
up to `mha_rope_freq_sigma`. If `mha_rope_learned_freqs=True`, the frequency
vectors are trainable.

For token coordinate \(r_i\), the phase is

$$
\theta_{b h i f} = \langle r_{b i}, \omega_{h f}\rangle.
$$

For non-atom/global tokens, `_build_mha_positions` pads positions with zeros.
QM9 EDM has only atom tokens, so this is usually just the atom coordinate array.

### 10.2 Rotary Operation

The first even number of head features are interpreted as 2D pairs. For a pair
\((a,b)\),

$$
\mathrm{Rot}_\theta(a,b)
=
(a\cos\theta-b\sin\theta,\; a\sin\theta+b\cos\theta).
$$

The code applies this rotation to \(Q\) and \(K\):

$$
\tilde Q_i = \mathrm{Rot}_{\theta_i}(Q_i),
\qquad
\tilde K_i = \mathrm{Rot}_{\theta_i}(K_i).
$$

### 10.3 `mha_rope_use_key`

If

```text
mha_rope_use_key = True
```

then keys are learned from token features:

$$
K_i = h_i W_K.
$$

If

```text
mha_rope_use_key = False
```

then keys are set to all ones before RoPE:

$$
K_i = \mathbf{1},
\qquad
\tilde K_i = \mathrm{Rot}_{\theta_i}(\mathbf{1}).
$$

This makes the key side purely positional. The query still depends on token
features, so logits still depend on atom/scalar context through \(Q_i\). This is
closer to the Platonic RoPE style where positional keys can come from a constant
carrier rather than from learned token keys.

Implementation detail: the current module still allocates the \(W_K\) slice in
`in_proj_weight`, but it is unused and receives no gradients when
`mha_rope_use_key=False`.

The flag is more general than either behavior alone: `True` gives learned token
keys, `False` gives positional all-ones keys.

### 10.4 `mha_rope_on_values`

If

```text
mha_rope_on_values = False
```

then values are ordinary learned values:

$$
V_i = h_i W_V.
$$

If

```text
mha_rope_on_values = True
```

then values are also RoPE-rotated before attention and inverse-rotated after
attention:

$$
\tilde V_i = \mathrm{Rot}_{\theta_i}(V_i),
$$

$$
O_i = \sum_j \alpha_{ij}\tilde V_j,
\qquad
\hat O_i = \mathrm{Rot}_{-\theta_i}(O_i).
$$

This gives the value path a coordinate-dependent transport operation. It is
again closer to the Platonic value path than the previous Matterformer MHA RoPE
mode, which applied RoPE only to \(Q/K\).

### 10.5 Platonic-Style MHA RoPE Configuration

A Matterformer configuration that approximates the relevant Platonic RoPE
choices is:

```text
attn_type = "mha"
mha_position_mode = "rope"
mha_rope_use_key = false
mha_rope_on_values = true
use_geometry_bias = false
coord_embed_mode = "none" or "rope"
noise_conditioning = "concat"
coord_head_mode = "direct"
```

This does not turn Matterformer into PlatonicTransformer. It only makes the MHA
RoPE key/value conventions more similar. The rest of the block, token mixing,
normalization, MLP, and output heads remain Matterformer.

## 11. Two-Simplicial Attention Path

Set

```text
attn_type = "simplicial"
```

to use `TwoSimplicialAttention`. Instead of attending from a query token \(i\)
to a single key token \(j\), each query attends over ordered token pairs
\((j,k)\).

### 11.1 Projection Shapes

The attention layer projects hidden features into five tensors:

$$
q_i,\quad k^{(1)}_j,\quad v^{(1)}_j,\quad k^{(2)}_k,\quad v^{(2)}_k.
$$

Each has shape \((B,H,N,d_h)\). The query is scaled by \(d_h^{-1/2}\).

### 11.2 Logits And Values

The core triplet logit is

$$
\ell_{ijk}^{(h)}
=
\sum_{a=1}^{d_h}
q_{ia}^{(h)}
k_{ja}^{(1,h)}
k_{ka}^{(2,h)}
+
B_{ijk}^{(h)}
+
\mathrm{mask}_{ijk}.
$$

The softmax is over all valid ordered pairs \((j,k)\):

$$
\alpha_{ijk}^{(h)}
=
\frac{\exp(\ell_{ijk}^{(h)})}
{\sum_{j',k'}\exp(\ell_{ij'k'}^{(h)})}.
$$

The base output is

$$
o_i^{(h)}
=
\sum_{j,k}
\alpha_{ijk}^{(h)}
\left(v_j^{(1,h)} \odot v_k^{(2,h)}\right).
$$

This is more expressive than pairwise MHA because the attention distribution can
represent interactions among the query atom \(i\), a first context atom \(j\),
and a second context atom \(k\).

### 11.3 Simplicial Masks

The simplicial mask has three pieces:

| Mask | Shape | Meaning |
| --- | --- | --- |
| `query_valid` | \((B,N)\) | whether query \(i\) is a valid token |
| `pair_key_valid` | \((B,N)\) | whether a token may appear as \(j\) or \(k\) |
| `pair_valid` | optional \((B,N,N)\) | whether the pair \((j,k)\) is allowed |

For QM9 atom-only sequences, pair validity follows the atom padding mask and the
geometry pair mask.

### 11.4 `simplicial_geom_mode="none"`

The model still uses two-simplicial attention, but no learned geometry bias is
added to the triplet logits. Masks still prevent padded atoms from participating.

### 11.5 `simplicial_geom_mode="factorized"`

The geometry module builds three pairwise bias tensors:

$$
u_{ij}^{(h)},\qquad v_{ik}^{(h)},\qquad w_{jk}^{(h)}.
$$

The triplet bias is

$$
B_{ijk}^{(h)}
=
g_i^{(h)}(\sigma)
\left(
u_{ij}^{(h)} + v_{ik}^{(h)} + w_{jk}^{(h)}
\right).
$$

The spoke terms \(u\) and \(v\) are built from features such as
\(\delta_{ij}\) and \(\mathrm{RBF}(\hat\rho_{ij})\). The pair term \(w\) is built
from pair-distance features. Final layers are zero-initialized, so this branch
starts as no extra learned logit bias.

### 11.6 `simplicial_geom_mode="angle_residual"`

This includes the factorized terms above and adds a direct triplet residual
built from geometric angles. For query \(i\), context atoms \(j,k\),

$$
\cos\theta_{ijk}
=
\frac{\delta_{ij}^{\top}G\delta_{ik}}
{\lVert\delta_{ij}\rVert_G\lVert\delta_{ik}\rVert_G},
$$

where \(G\) is the metric tensor. For nonperiodic QM9, \(G=I\). The residual MLP
receives RBF features for \(d_{ij}\), RBF features for \(d_{ik}\), and
\(\cos\theta_{ijk}\), then returns a per-head triplet logit residual.

### 11.7 `simplicial_geom_mode="angle_low_rank"`

This includes the factorized terms and adds a low-rank angle residual:

$$
B^{\mathrm{angle}}_{ijk}
=
g_i(\sigma)
\frac{1}{\sqrt R}
\sum_{r=1}^R
L_{ijr}R_{ikr}.
$$

This is cheaper than materializing a full angle MLP over all triplets. The code
zero-initializes only one side of the low-rank product so gradients can still
flow through the product branch at initialization.

### 11.8 `simplicial_message_mode="low_rank"`

This modifies the value/message path, not only the logits. It builds low-rank
factors over \((i,j)\) and \((i,k)\), combines them under the attention weights,
and projects the resulting coefficients through a learned per-head basis:

$$
c_{ir}
=
\frac{1}{\sqrt R}
\sum_{j,k}
\alpha_{ijk}
L_{ijr}R_{ikr},
$$

$$
o_i \leftarrow o_i + \sum_r c_{ir}B_r.
$$

This gives the attention output an additional geometry-conditioned message term.

### 11.9 Torch And Triton Backends

`simplicial_impl` can be:

| Value | Behavior |
| --- | --- |
| `"torch"` | always use the PyTorch chunked implementation |
| `"triton"` | require the Triton implementation |
| `"auto"` | use Triton when available and supported, otherwise PyTorch |

`simplicial_precision` controls the Triton core precision mode:
`"bf16_tc"`, `"tf32"`, or `"ieee_fp32"`.

## 12. Geometry Bias Gating By Noise

Both MHA geometry bias and simplicial geometry bias can use the same sigma gate:

$$
g_h(\sigma)=\mathrm{sigmoid}(\alpha_h(-\log\sigma)+\beta_h).
$$

This is enabled for `QM9EDMModel` because EDM denoising has very different
geometric reliability across noise levels. At small \(\sigma\), geometry should
usually be meaningful. At large \(\sigma\), coordinates are dominated by noise,
so the model may learn to reduce geometry bias.

The regression model disables this gate because there is no diffusion sigma.

## 13. Coordinate Output Heads

The trunk output \(h_i\in\mathbb{R}^d\) is converted into coordinate deltas by
one of two heads.

### 13.1 Shared Centering Rule

Both coordinate heads subtract the masked mean of the predicted delta:

$$
\Delta r_i
\leftarrow
\Delta r_i
-
\frac{\sum_j m_j \Delta r_j}{\sum_j m_j}.
$$

Then padded rows are zeroed again.

This enforces

$$
\sum_i m_i\Delta r_i = 0.
$$

Why this matters:

1. The clean coordinates and noisy coordinates are always centered.
2. EDM preconditioning recenters coordinates again after applying the delta.
3. A global translation delta is unidentifiable under centered training data.
4. Removing the mean prevents the coordinate head from using capacity to chase a
   center-of-mass degree of freedom that the data pipeline will immediately
   remove.

The tradeoff is that the model cannot output a net translation. For QM9 EDM that
is intentional, because the generative coordinate gauge is centered.

### 13.2 `coord_head_mode="direct"`

The direct coordinate head is

```text
LayerNorm(d)
Linear(d -> d)
SiLU
Linear(d -> 3)
mean-subtract over valid atoms
```

In equations:

$$
\Delta r_i^{\mathrm{raw}}
=
W_2\,\mathrm{SiLU}(W_1\mathrm{LN}(h_i)+b_1)+b_2.
$$

Then the masked mean is subtracted.

The final linear layer is zero-initialized, so initially

$$
\Delta r_i = 0
$$

for every atom. This makes the coordinate denoising path initially rely on the
EDM skip/preconditioned input path rather than on random coordinate corrections.

The direct head is not rotation-equivariant by construction. It can learn a
useful vector output in a fixed or augmented frame, but a rotation of the input
does not algebraically force the output to rotate the same way.

### 13.3 `coord_head_mode="equivariant"`

The relative-vector head uses pairwise displacement vectors from geometry:

$$
\delta_{ij} = r_i-r_j.
$$

It first expands normalized distances with radial basis functions:

$$
\mathrm{RBF}_k(\hat\rho_{ij})
=
\exp\left[-\gamma(\hat\rho_{ij}-\mu_k)^2\right].
$$

Then it builds pair inputs

$$
z_{ij} = [h_i,h_j,\mathrm{RBF}(\hat\rho_{ij})],
$$

and scalar pair weights

$$
a_{ij}=\mathrm{PairMLP}(z_{ij}).
$$

Invalid pairs and diagonal terms are zeroed. The coordinate delta is

$$
\Delta r_i^{\mathrm{raw}}
=
\frac{1}{\max(\sum_j m_j,1)}
\sum_j a_{ij}\delta_{ij}.
$$

Then the masked mean is subtracted.

This head has a relative-vector structure: it can only move atoms by weighted
sums of interatomic displacement vectors. If the scalar weights \(a_{ij}\) are
rotation-invariant, then the output is rotation-equivariant. However, the full
model is only formally equivariant if the hidden features and weights are also
produced in a rotation-compatible way. Direct coordinate embeddings, raw
coordinate-dependent MLP biases, and ordinary scalar hidden features can break
strict SO(3) equivariance even though the head itself uses relative vectors.

Like the direct head, the final pair-head linear layer is zero-initialized, so
initial coordinate deltas are zero.

### 13.4 Head Aliases

The model canonicalizes several names:

| Input name | Canonical mode |
| --- | --- |
| `"equivariant"` | `"equivariant"` |
| `"relative"` | `"equivariant"` |
| `"pair"` or `"pairwise"` | `"equivariant"` |
| `"direct"` | `"direct"` |
| `"non_relative"` | `"direct"` |
| `"non-relative"` | `"direct"` |
| `"non_equivariant"` | `"direct"` |
| `"non-equivariant"` | `"direct"` |

## 14. Atom Output Head

The atom head is

```text
LayerNorm(d)
Linear(d -> d)
SiLU
Linear(d -> C_x)
```

or

$$
\Delta x_i =
W_2\,\mathrm{SiLU}(W_1\mathrm{LN}(h_i)+b_1)+b_2.
$$

The atom head final layer is not zero-initialized in the current implementation.
Padded atom outputs are zeroed before returning.

The preconditioner uses atom deltas as

$$
\hat x_{\mathrm{clean,scaled}} =
x_{\mathrm{in}} - \Delta x_\theta.
$$

So a positive atom delta means "subtract this from the scaled noisy atom feature
before applying the EDM output coefficients."

## 15. Regression Model Variant

`QM9RegressionModel` reuses the same trunk machinery but has a different task
interface.

Architecture:

```text
atom_types -> Embedding
centered, RMS-normalized coords -> fixed RFF coord embedding
add atom and coord embeddings
optional CLS token
learned null conditioning vector
TransformerTrunk
pool: CLS, sum, or mean
MLP scalar head -> prediction
```

Key differences from EDM:

1. There is no diffusion sigma.
2. The trunk uses `LearnedNullConditioning` rather than `TimeEmbedder`.
3. Geometry bias has `use_noise_gate=False`.
4. The output is a scalar property, not atom and coordinate denoising deltas.

This is useful because many architecture modules are shared, but the EDM details
above apply specifically to `QM9EDMModel`.

## 16. Sampling

`edm_sampler` samples molecule sizes, creates padded masks, initializes atom
features and coordinates from Gaussian noise, and follows a Karras noise
schedule:

$$
t_i =
\left[
\sigma_{\max}^{1/\rho}
+
\frac{i}{S-1}
\left(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\right)
\right]^\rho.
$$

At each step it optionally adds churn noise, calls the preconditioned denoiser,
and computes the EDM ODE derivative:

$$
d_x = \frac{x_{\mathrm{hat}} - D_x(x_{\mathrm{hat}},t_{\mathrm{hat}})}
{t_{\mathrm{hat}}},
\qquad
d_r = \frac{r_{\mathrm{hat}} - D_r(r_{\mathrm{hat}},t_{\mathrm{hat}})}
{t_{\mathrm{hat}}}.
$$

It uses Euler for the step and Heun correction except at the final step. After
every coordinate update it recenters coordinates and zeros padded atom features.

Atom types are decoded by argmax over the atom-type feature channels. Charges,
if present, are decoded by multiplying the charge channel by
`charge_feature_scale` and rounding.

## 17. Platonic Adapter Details

The Platonic pipeline calls its backbone with ragged tensors:

```text
scalars: (M, C_x + 1)  # x plus c_noise
pos:     (M, 3)
batch:   (M,)
```

where \(M\) is the total number of atoms across all molecules in the batch, and
the final scalar channel is

$$
c_{\mathrm{noise}} = \frac{\log\sigma}{4}.
$$

`MatterformerQM9Adapter` does this:

1. Split `scalars` into atom features and the appended noise channel.
2. Recover one \(\sigma_b\) per graph by averaging `log_sigma_over_4` over nodes
   of that graph and exponentiating:

   $$
   \sigma_b =
   \exp\left(
   4\frac{\sum_{i\in b}c_i}{|\{i:i\in b\}|}
   \right).
   $$

3. Pack ragged `x` and `pos` into padded tensors.
4. Call `QM9EDMModel(atom_padded, pos_padded, pad_mask, sigma)`.
5. Unpack outputs back to ragged format.
6. Return coordinate deltas as shape \((M,1,3)\), matching Platonic's vector
   output convention.

Important distinction: Platonic's EDM preconditioner has already concatenated
\(c_{\mathrm{noise}}\) to the scalar input. The adapter strips that channel to
recover \(\sigma\). If Matterformer is configured with
`noise_conditioning="concat"`, then `QM9EDMModel` concatenates the same
\(\log\sigma/4\) value again inside its own padded atom projection. That is
intentional for matching the native Matterformer conditioning interface.

## 18. Option Summary

### 18.1 Core Width And Depth

| Option | Meaning |
| --- | --- |
| `d_model` | hidden token width \(d\) |
| `n_heads` | number of attention heads \(H\) |
| `n_layers` | number of transformer blocks \(L\) |
| `mlp_ratio` | hidden expansion factor in each block MLP |
| `dropout` | MLP dropout |
| `attn_dropout` | attention dropout |

### 18.2 Attention Type

| Option | Values | Effect |
| --- | --- | --- |
| `attn_type` | `"mha"` | pairwise multi-head attention |
| `attn_type` | `"simplicial"` | two-simplicial attention over ordered pairs \((j,k)\) |

### 18.3 MHA Position And Geometry

| Option | Values | Effect |
| --- | --- | --- |
| `mha_position_mode` | `"none"` | no MHA positional RoPE |
| `mha_position_mode` | `"rope"` | apply 3D RoPE to MHA \(Q/K\) |
| `mha_rope_freq_sigma` | float | maximum/default scale of RoPE frequencies |
| `mha_rope_learned_freqs` | bool | make RoPE frequencies trainable |
| `mha_rope_use_key` | bool | learned token keys if true, all-ones positional keys if false |
| `mha_rope_on_values` | bool | rotate values and inverse-rotate outputs |
| `mha_geom_bias_mode` | `"standard"` | distance plus edge MLP pairwise bias |
| `mha_geom_bias_mode` | `"factorized_marginal"` | marginalize simplicial factorized geometry terms into MHA pair bias |

### 18.4 Simplicial Geometry

| Option | Values | Effect |
| --- | --- | --- |
| `simplicial_geom_mode` | `"none"` | no learned geometry logit bias |
| `simplicial_geom_mode` | `"factorized"` | add \(u_{ij}+v_{ik}+w_{jk}\) factorized triplet bias |
| `simplicial_geom_mode` | `"angle_residual"` | factorized bias plus full angle residual callback |
| `simplicial_geom_mode` | `"angle_low_rank"` | factorized bias plus low-rank angle residual |
| `simplicial_angle_rank` | int | rank for low-rank angle residual |
| `simplicial_message_mode` | `"none"` | no extra value/message residual |
| `simplicial_message_mode` | `"low_rank"` | add low-rank geometry-conditioned message residual |
| `simplicial_message_rank` | int | rank for low-rank message residual |
| `simplicial_impl` | `"auto"`, `"torch"`, `"triton"` | backend choice |
| `simplicial_precision` | `"bf16_tc"`, `"tf32"`, `"ieee_fp32"` | Triton precision mode |

### 18.5 Coordinate Input

| Option | Values | Effect |
| --- | --- | --- |
| `coord_embed_mode` | `"none"` | no additive coordinate token embedding |
| `coord_embed_mode` | `"fourier"` | deterministic Fourier features of coordinates |
| `coord_embed_mode` | `"rff"` | fixed random Fourier features |
| `coord_embed_mode` | `"learnable_rff"` | trainable random Fourier projection |
| `coord_embed_mode` | `"rope"` | alias for MHA RoPE, no additive embedding |
| `coord_n_freqs` | int | Fourier frequency count or default RFF dim |
| `coord_rff_dim` | int or none | explicit RFF feature count |
| `coord_rff_sigma` | float | RFF scale |
| `coord_embed_normalize` | bool | RMS-normalize centered coords before additive coord embedding |

### 18.6 Coordinate Output

| Option | Values | Effect |
| --- | --- | --- |
| `coord_head_mode` | `"direct"` | per-token MLP to xyz delta, then mean subtraction |
| `coord_head_mode` | `"equivariant"` | pair-weighted sum of relative displacement vectors, then mean subtraction |
| `pair_hidden_dim` | int | hidden width of pair MLP for equivariant coordinate head |
| `pair_n_rbf` | int | number of RBF distance features in equivariant coordinate head |
| `pair_rbf_max` | float | maximum RBF center value |

### 18.7 Noise Conditioning

| Option | Values | Effect |
| --- | --- | --- |
| `noise_conditioning` | `"concat"` | append \(\log\sigma/4\) to atom input channels |
| `noise_conditioning` | `"adaln"` | use TimeEmbedder and AdaLN conditioning in blocks |
| `noise_conditioning` | `"concat,adaln"` | use both |
| `noise_conditioning` | `"none"` | use neither, except geometry bias may still use sigma gate |
| `concat_sigma_condition` | bool or none | legacy compatibility flag |

### 18.8 Geometry On/Off

| Option | Effect |
| --- | --- |
| `use_geometry_bias=True` | enable MHA geometry bias or simplicial geometry/message modules according to attention type |
| `use_geometry_bias=False` | disable learned geometry bias/message modules |

Even with `use_geometry_bias=False`, coordinates can still enter through
coordinate embeddings, MHA RoPE, and coordinate heads.

## 19. Common Configurations

### 19.1 Native Simplicial Matterformer

```text
attn_type = "simplicial"
simplicial_geom_mode = "factorized"
simplicial_message_mode = "low_rank"
coord_embed_mode = "none"
coord_head_mode = "equivariant"
noise_conditioning = "concat,adaln"
use_geometry_bias = true
```

Mental model:

```text
x + sigma channel -> tokens
sigma -> AdaLN
pos -> geometry features -> simplicial factorized bias and message residual
trunk -> relative-vector coordinate head
```

This is the most geometry-heavy Matterformer route.

### 19.2 MHA RoPE With Learned Keys

```text
attn_type = "mha"
mha_position_mode = "rope"
mha_rope_use_key = true
mha_rope_on_values = false
coord_embed_mode = "none" or "rope"
coord_head_mode = "direct" or "equivariant"
```

Mental model:

```text
Q/K depend on token features
Q/K are phase-rotated by coordinates
values are ordinary learned values
```

This is standard learned-token MHA with coordinate-aware attention logits.

### 19.3 Platonic-Style MHA RoPE Matterformer

```text
attn_type = "mha"
mha_position_mode = "rope"
mha_rope_use_key = false
mha_rope_on_values = true
use_geometry_bias = false
coord_embed_mode = "none" or "rope"
coord_head_mode = "direct"
noise_conditioning = "concat"
```

Mental model:

```text
token features -> learned queries and values
keys are coordinate-rotated all-ones carriers
values are coordinate-rotated and transported back by inverse RoPE
coordinate deltas come from a zero-initialized direct xyz head
```

This is designed to test whether the Matterformer trunk behaves closer to the
trivial Platonic setting, while still keeping Matterformer's block and head
implementation.

### 19.4 Additive Coordinate Embedding Ablation

```text
coord_embed_mode = "fourier" or "rff" or "learnable_rff"
coord_embed_normalize = true or false
```

Mental model:

```text
pos -> coordinate Fourier/RFF MLP -> add to atom token features
```

This gives every atom token direct coordinate features before any attention.
It can be useful, but it is a more frame-dependent way of injecting coordinates
than relative-distance geometry terms.

## 20. Practical Remarks

### 20.1 What Is Zero-Initialized?

The following branches start at zero effect:

1. Direct coordinate head final linear layer.
2. Equivariant pair coordinate head final linear layer.
3. Geometry edge/factorized/angle MLP final layers, with a special one-sided
   zero init for low-rank products.
4. AdaLN shift/scale/gate projection final layer, when AdaLN is enabled.

The atom head is not zero-initialized.

### 20.2 What Does Mean Subtraction Buy?

Mean subtraction in coordinate deltas keeps the model in the same centered gauge
as the data and sampler. It removes a degree of freedom that the pipeline does
not train or sample: global translation. For centered molecule generation, this
is usually the right constraint.

The downside is that the model cannot use its coordinate head to shift a whole
molecule. If a future task has absolute positions, external fields, or periodic
origins where a global translation has meaning, this should become a task-level
option. For current QM9 EDM, disabling it would mostly introduce an output mode
that the preconditioner and sampler immediately erase by recentering.

### 20.3 Direct Head Versus Relative-Vector Head

The direct head is simple and matches the Platonic adapter's direct vector
readout style more closely, but it has no built-in rotation equivariance. The
relative-vector head is more geometrically structured because it predicts scalar
weights on displacement vectors. It is the safer choice when the upstream
features are intended to behave like scalar geometric features.

### 20.4 MHA RoPE Is Not Geometry Bias

MHA RoPE changes how attention logits are formed from coordinates by rotating
head features. Geometry bias adds explicit pairwise or marginalized triplet
logit terms derived from distances, displacements, and optional lattice
features. These can be combined in principle, but they are different mechanisms:

```text
RoPE:          coordinate-dependent representation of Q/K/(optional V)
Geometry bias: coordinate-derived additive logit bias
Coord embed:   coordinate-derived additive token input
Coord head:    coordinate-derived output vector construction
```

### 20.5 `mha_rope_use_key=false` Does Not Remove Token Dependence

With all-ones keys, the key side is positional, but queries and values still
come from token features. Therefore attention still depends on atom identity,
charge, sigma-conditioning, and previous hidden layers. What is removed is the
learned token-content key projection.

### 20.6 `noise_conditioning="concat"` Versus `"adaln"`

Concat conditioning exposes sigma as an ordinary scalar feature at the input.
AdaLN conditioning modulates every block and gates residual branches. They are
not equivalent. Concat lets the network learn sigma-dependent hidden features
from the start. AdaLN gives every layer an explicit global sigma control knob,
but the current zero-init means the residual branches initially start closed.

### 20.7 What The Model Outputs In One Sentence

For each molecule and sigma, Matterformer predicts atom and coordinate residual
deltas in the EDM-preconditioned coordinate system, masks padded atoms, forces
coordinate deltas to have zero mean, and lets the EDM wrapper turn those deltas
into denoised atom features and centered coordinates.
