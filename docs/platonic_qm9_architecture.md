# PlatonicTransformer QM9 Architecture Overview

This document describes the PlatonicTransformer QM9 generation architecture from
`/home/thadziv/GitHub/PlatonicTransformers`. It focuses on the two Platonic QM9
baselines we have been comparing:

1. The main tetrahedral baseline: `solid_name="tetrahedron"`.
2. The trivial Platonic baseline: `solid_name="trivial"`.

The relevant source files are:

| File | Role |
| --- | --- |
| `configs/qm9_gen.yaml` | QM9 generation defaults for the Platonic runs |
| `mains/main_qm9_gen.py` | Lightning module wiring dataset, EDM loss, sampler, and backbone |
| `platonic_transformers/diffusion/edm.py` | Karras EDM preconditioning, loss, and sampler |
| `platonic_transformers/models/platoformer/platoformer.py` | top-level `PlatonicTransformer` |
| `platonic_transformers/models/platoformer/groups.py` | finite symmetry groups |
| `platonic_transformers/models/platoformer/io.py` | scalar/vector lifting, pooling, scalar/vector readout |
| `platonic_transformers/models/platoformer/linear.py` | group-constrained `PlatonicLinear` |
| `platonic_transformers/models/platoformer/block.py` | pre-norm residual block |
| `platonic_transformers/models/platoformer/conv.py` | `PlatonicConv` interaction/attention layer |
| `platonic_transformers/models/platoformer/rope.py` | group-equivariant RoPE |
| `platonic_transformers/models/platoformer/ape.py` | optional group-equivariant absolute position encoding |

## 1. What The QM9 Platonic Model Is

For QM9 generation, PlatonicTransformer is used as a node-level EDM denoiser. It
takes noised atom scalar features and noised atom positions, and returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `scalars_out` | \((M,C_x)\) | per-node scalar correction, later subtracted from preconditioned scalar input |
| `vecs_out` | \((M,1,3)\) | per-node coordinate/vector correction, later subtracted from preconditioned positions |

Here \(M\) is the total number of atoms across all molecules in a ragged batch.
Unlike Matterformer, the Platonic QM9 pipeline is primarily ragged/graph-mode,
not padded:

| Symbol | Code name | Shape | Meaning |
| --- | --- | --- | --- |
| \(M\) | total nodes | scalar | total atoms across all graphs |
| \(B\) | number of graphs | scalar | number of molecules |
| \(C_x\) | `out_scalar` | scalar | atom feature channels, usually 5 atom types plus optional charge |
| \(x\) | `x` | \((M,C_x)\) | clean or noised atom features |
| \(r\) | `pos` | \((M,3)\) | clean or noised atom positions |
| \(b\) | `batch` | \((M,)\) | graph index for every atom |
| \(\sigma\) | `sigma_per_graph` | \((B,1)\) | diffusion noise level per molecule |

For the QM9 generation config with charges enabled:

```text
C_x = 6                 # 5 atom one-hot channels + 1 formal charge channel
input_dim = C_x + 1     # extra EDM c_noise channel appended before the backbone
output_dim = C_x
output_dim_vec = 1      # one 3D coordinate correction vector per atom
```

The main model defaults are:

```text
hidden_dim = 1152
num_layers = 14
num_heads = 72
solid_name = "tetrahedron"
attention = true
rope_sigma = 4.0
ape_sigma = null
learned_freqs = true
freq_init = "spiral"
use_key = false
rope_on_values = true
attention_backend = "flash"   # or scatter fallback
scalar_task_level = "node"
vector_task_level = "node"
```

The trivial baseline uses the same architectural knobs but changes:

```text
solid_name = "trivial"
```

That one change is conceptually large: it removes the nontrivial group axis,
removes tetrahedral weight sharing, and removes exact tetrahedral equivariance.

## 2. End-To-End Mental Map

```text
QM9 clean ragged batch
  x: atom one-hot (+ optional charge)
  pos: coordinates
  batch: graph id per atom
        |
        v
EDMLoss
  subtract per-graph coordinate mean
  scale atom/charge channels
  sample one sigma per graph
  add scalar and coordinate noise
        |
        v
EDMPrecond
  c_in scales x and pos
  c_noise = log(sigma) / 4
  concatenate c_noise to scalar node features
        |
        v
PlatonicTransformer
  lift scalars/vectors into group-indexed feature fields
  PlatonicLinear input embedder
  optional PlatonicAPE (off in QM9 baseline)
  14 PlatonicBlocks:
    group-wise norm
    PlatonicConv with group-equivariant RoPE
    group-wise norm
    PlatonicLinear FFN
  scalar and vector readouts
        |
        v
EDMPrecond
  subtract model scalar/vector outputs from preconditioned inputs
  combine with c_skip and c_out
        |
        v
denoised atom features and positions
```

The architectural core is the group axis. Features are not just vectors of size
`hidden_dim`. They are interpreted as

$$
h_i \in \mathbb{R}^{G\times C},
\qquad
d = G C,
$$

then flattened to shape \((M,d)\) for storage.

For the two baselines:

| Baseline | Group order \(G\) | Hidden total \(d\) | Channels per group \(C=d/G\) | Total heads | Heads per group | Head dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tetrahedron | 12 | 1152 | 96 | 72 | 6 | 16 |
| trivial | 1 | 1152 | 1152 | 72 | 72 | 16 |

The head dimension is the same, but the representation is organized very
differently.

## 3. EDM Preconditioning In The Platonic Pipeline

### 3.1 Clean Data And Noise

The Platonic QM9 `EDMLoss` first centers positions per graph:

$$
\bar r_b =
\frac{1}{|\{i:b_i=b\}|}
\sum_{i:b_i=b} r_i,
\qquad
r_i \leftarrow r_i - \bar r_{b_i}.
$$

Atom features are scaled:

$$
x_i =
\left[
\frac{\mathrm{onehot}(\mathrm{atom}_i)}{s_x},
\frac{q_i}{s_q}
\right],
$$

where the default scales are

$$
s_x=4,\qquad s_q=8.
$$

One noise level is sampled per molecule:

$$
\log\sigma_b \sim \mathcal{N}(P_{\mathrm{mean}},P_{\mathrm{std}}^2),
$$

with default \(P_{\mathrm{mean}}=-1.2\) and \(P_{\mathrm{std}}=1.2\).

Noised features are

$$
x_{\sigma,i}=x_i+\sigma_{b_i}\epsilon^x_i,
$$

and noised positions are

$$
r_{\sigma,i}=r_i+\sigma_{b_i}\tilde\epsilon^r_i,
$$

where \(\tilde\epsilon^r\) is Gaussian coordinate noise with its per-graph mean
subtracted.

### 3.2 Preconditioner Coefficients

`EDMPrecond` computes

$$
c_{\mathrm{skip}} =
\frac{\sigma_{\mathrm{data}}^2}{\sigma^2+\sigma_{\mathrm{data}}^2},
$$

$$
c_{\mathrm{out}} =
\frac{\sigma\sigma_{\mathrm{data}}}
{\sqrt{\sigma^2+\sigma_{\mathrm{data}}^2}},
$$

$$
c_{\mathrm{in}} =
\frac{1}{\sqrt{\sigma_{\mathrm{data}}^2+\sigma^2}},
$$

and

$$
c_{\mathrm{noise}}=\frac{\log\sigma}{4}.
$$

The backbone input is

$$
x_{\mathrm{in},i}=c_{\mathrm{in},b_i}x_{\sigma,i},
\qquad
r_{\mathrm{in},i}=c_{\mathrm{in},b_i}r_{\sigma,i}.
$$

Then \(c_{\mathrm{noise}}\) is concatenated to scalar node features:

$$
s_i = [x_{\mathrm{in},i}, c_{\mathrm{noise},b_i}].
$$

This is the only explicit noise conditioning in the Platonic backbone. There is
no AdaLN or time embedding path in the Platonic QM9 baseline.

### 3.3 Backbone Output Convention

The backbone is called as

```python
scalars_out, vecs_out = model(scalars_in, pos_in, batch, vec=None)
```

with

```text
scalars_in: (M, C_x + 1)
pos_in:     (M, 3)
vec:        None
```

The preconditioner interprets the model outputs by subtraction:

$$
F_x = x_{\mathrm{in}} - \mathrm{scalars\_out},
$$

$$
F_r = r_{\mathrm{in}} - \mathrm{vecs\_out.squeeze(1)}.
$$

The final denoised outputs are

$$
D_x =
c_{\mathrm{skip}}x_\sigma + c_{\mathrm{out}}F_x,
$$

$$
D_r =
c_{\mathrm{skip}}r_\sigma + c_{\mathrm{out}}F_r.
$$

So the Platonic backbone output is in the same "subtract this from the
preconditioned input" convention used by the Matterformer adapter. The vector
head output is not explicitly mean-centered inside `PlatonicTransformer`; the
training target is centered, so nonzero center-of-mass output is learned against
the loss rather than hard-projected away.

## 4. The Group Axis

### 4.1 What A Group Is In This Code

`groups.py` stores each symmetry group as a finite list of orthogonal matrices:

$$
\mathcal{G}=\{R_0,\dots,R_{G-1}\},
\qquad
R_g^\top R_g=I,\qquad
\det(R_g)=\pm 1.
$$

For the main QM9 baseline:

```text
solid_name = "tetrahedron"
G = 12
```

The 12 matrices are the proper rotational symmetry group of a tetrahedron.

For the trivial baseline:

```text
solid_name = "trivial"
G = 1
R_0 = I_3
```

The group object also precomputes:

| Quantity | Meaning |
| --- | --- |
| `elements` | tensor of group matrices \(R_g\), shape \((G,3,3)\) |
| `inverse_indices` | index of \(g^{-1}\) for every group element |
| `cayley_table` | multiplication table: product of two group elements as an index |

The Cayley table is used to build group-constrained weights.

### 4.2 What The Group Axis Represents

After lifting, every node feature is a field over the group:

$$
h_i = (h_{i0},h_{i1},\dots,h_{i,G-1}),
\qquad
h_{ig}\in\mathbb{R}^C.
$$

In code this is stored as a flat vector:

$$
h_i^{\mathrm{flat}}\in\mathbb{R}^{G C}.
$$

The intended action of a group element is a permutation of the group index. The
exact left/right convention is encoded by the Cayley table, but the key point is:
physical rotations in the chosen finite group act by permuting the group axis and
rotating vector channels. `PlatonicLinear`, `PlatonicConv`, normalization, and
readout are written to commute with that action.

For tetrahedron, this gives exact equivariance to the 12 tetrahedral rotations:

$$
\mathrm{scalar}(x,Rr)=\mathrm{scalar}(x,r),
$$

$$
\mathrm{vector}(x,Rr)=R\,\mathrm{vector}(x,r),
\qquad R\in \mathcal{G}_{\mathrm{tetra}}.
$$

For trivial, the only exact group element is \(I\), so this guarantee is only
the identity statement.

### 4.3 Important Limitation

Tetrahedral PlatonicTransformer is not exactly equivariant to all rotations in
\(\mathrm{SO}(3)\). It is exactly equivariant to the finite tetrahedral group.
QM9 training still uses random \(\mathrm{SO}(3)\) rotation augmentation, so the
model can learn broader rotational behavior statistically, but the algebraic
guarantee is finite-group equivariance.

## 5. Lifting Scalars And Vectors

The first architectural step in `PlatonicTransformer.forward` is

```python
x = lift(x, vec, self.group)
```

### 5.1 Scalar Lift

Scalar features are copied across the group axis:

$$
\mathrm{lift}_{\mathrm{scalar}}(x)_{igc} = x_{ic}.
$$

For QM9 generation, `vec=None`, so this scalar lift is the only lift used at the
input.

Shapes:

| Baseline | Input `scalars_in` | After scalar lift | Flattened |
| --- | --- | --- | --- |
| tetrahedron | \((M,7)\) | \((M,12,7)\) | \((M,84)\) |
| trivial | \((M,7)\) | \((M,1,7)\) | \((M,7)\) |

The 7 channels are 6 atom/charge channels plus \(c_{\mathrm{noise}}\).

### 5.2 Vector Lift

The generic model can also accept input vectors `vec` of shape
\((M,C_v,3)\). It lifts a vector by expressing it in every group frame:

$$
\mathrm{lift}_{\mathrm{vector}}(v)_{igc:}
=
R_g^\top v_{ic}
$$

up to the code's row/column convention. In code:

```python
frames = group.elements
torch.einsum("gji,...cj->...gci", frames, x)
```

QM9 generation passes `input_dim_vec=0`, so there are no input vector channels.
The vector machinery matters mainly at readout, where coordinate corrections are
produced.

### 5.3 Input Embedder

After lifting, the model applies a group-constrained linear map:

```python
self.x_embedder =
    PlatonicLinear((input_dim + input_dim_vec * spatial_dim) * G,
                   hidden_dim,
                   solid_name,
                   bias=False)
```

For QM9:

| Baseline | Lifted input dim | Hidden dim |
| --- | ---: | ---: |
| tetrahedron | \(7\times12=84\) | 1152 |
| trivial | \(7\times1=7\) | 1152 |

For tetrahedron, the hidden representation is \((M,12,96)\) conceptually. For
trivial, it is \((M,1,1152)\).

## 6. PlatonicLinear

`PlatonicLinear` is the weight-sharing mechanism that makes ordinary linear
layers equivariant over the finite group.

### 6.1 Parameterization

For input shape \((...,G C_{\mathrm{in}})\) and output shape
\((...,G C_{\mathrm{out}})\), it stores a kernel

$$
K \in \mathbb{R}^{G\times C_{\mathrm{out}}\times C_{\mathrm{in}}}.
$$

It then constructs the full block matrix

$$
W \in \mathbb{R}^{G C_{\mathrm{out}}\times G C_{\mathrm{in}}}
$$

using the Cayley table:

$$
W_{h,g} = K_{g^{-1}h}.
$$

Here \(W_{h,g}\) is the block mapping input group channel \(g\) to output group
channel \(h\). This tied block-circulant structure is the finite-group analogue
of a convolution.

The bias has shape \((C_{\mathrm{out}})\), not \((G C_{\mathrm{out}})\), and is
shared across the group axis:

$$
y_{h} \leftarrow y_{h}+b.
$$

That shared bias is necessary for equivariance.

### 6.2 Parameter Count Consequence

For a dense ordinary linear map from \(d\) to \(d\), parameter count is roughly
\(d^2\). For `PlatonicLinear` with total width \(d=GC\), the kernel has

$$
G C^2 = \frac{d^2}{G}
$$

parameters.

This is why the tetrahedral model with \(G=12\) is much smaller than the
trivial model at the same total hidden width:

| Baseline | Actual trainable parameters |
| --- | ---: |
| tetrahedron 1152/14/72 | 17,289,129 |
| trivial 1152/14/72 | 207,266,697 |

The trivial model has \(G=1\), so `PlatonicLinear` reduces to an ordinary
unshared linear layer.

## 7. Optional Absolute Position Encoding

`PlatonicTransformer` can add `PlatonicAPE` after the input embedder:

```python
x = x + self.ape(pos) if self.ape is not None else x
```

QM9 generation sets:

```text
ape_sigma = null
```

so APE is off in both the tetrahedral and trivial QM9 baselines.

When enabled, `PlatonicAPE` samples base frequencies and rotates them by all
group elements:

$$
\Omega_g = R_g \Omega.
$$

It projects positions against every rotated frequency set:

$$
\theta_{igf} = r_i^\top \Omega_{gf},
$$

and emits grouped sinusoidal features

$$
[\cos\theta_{igf},\sin\theta_{igf}]_{g,f}.
$$

This is an additive coordinate embedding, but it is not part of the active QM9
generation baseline. In active QM9 runs, geometry enters the backbone through
RoPE in `PlatonicConv`, plus centering/rotation augmentation in the diffusion
pipeline.

## 8. PlatonicBlock

The trunk is a stack of `num_layers` `PlatonicBlock`s:

```text
for layer in layers:
    x = layer(x, pos, batch, avg_num_nodes)
```

Each block is pre-norm:

```text
group-wise norm
PlatonicConv interaction
dropout / optional LayerScale / optional DropPath
residual add

group-wise norm
PlatonicLinear FFN
dropout / optional LayerScale / optional DropPath
residual add
```

In equations:

$$
h \leftarrow h + \mathrm{DropPath}(
\gamma_1\odot \mathrm{PlatonicConv}(\mathrm{Norm}(h),r,b)
),
$$

$$
h \leftarrow h + \mathrm{DropPath}(
\gamma_2\odot
W_2\,\phi(W_1\mathrm{Norm}(h))
).
$$

For the QM9 config:

```text
dropout = 0.0
drop_path_rate = 0.0
layer_scale_init_value = null
activation = gelu
ffn_dim_factor = 4
```

so DropPath and LayerScale are effectively off, and the FFN hidden total width
is

$$
d_{\mathrm{ffn}} = 4d = 4608.
$$

For tetrahedron this is \((G,C_{\mathrm{ffn}})=(12,384)\). For trivial it is
\((1,4608)\).

### 8.1 Group-Wise Normalization

Normalization reshapes

$$
h \in \mathbb{R}^{G C}
\quad\to\quad
h \in \mathbb{R}^{G\times C},
$$

then applies LayerNorm or RMSNorm over the \(C\) channel dimension using shared
normalization parameters for all group elements. This preserves the group-axis
equivariance.

QM9 uses the default `norm_type="layernorm"`.

## 9. PlatonicConv

`PlatonicConv` is the main token interaction layer. In the active QM9 setup,
it is a fully connected graph attention layer with group-equivariant RoPE.

### 9.1 Projection Shapes

`PlatonicConv` first applies group-constrained projections:

$$
Q_{\mathrm{raw}} = \mathrm{PlatonicLinear}_Q(h),
$$

$$
V_{\mathrm{raw}} = \mathrm{PlatonicLinear}_V(h).
$$

If no RoPE is used or `use_key=true`, it also learns keys:

$$
K_{\mathrm{raw}} = \mathrm{PlatonicLinear}_K(h).
$$

But in the active QM9 baseline:

```text
rope_sigma = 4.0
use_key = false
```

so keys are all ones:

$$
K_{\mathrm{raw}} = \mathbf{1}.
$$

The projections are reshaped as

$$
Q,K,V\in\mathbb{R}^{M\times G\times H_g\times d_h},
$$

where

$$
H_g = \frac{\texttt{num_heads}}{G},
\qquad
d_h = \frac{d/G}{H_g}=\frac{d}{\texttt{num_heads}}.
$$

For both tetra and trivial baseline:

$$
d_h = 1152/72 = 16.
$$

The difference is the split:

| Baseline | \(G\) | \(H_g\) | Shape of \(Q,K,V\) |
| --- | ---: | ---: | --- |
| tetrahedron | 12 | 6 | \((M,12,6,16)\) |
| trivial | 1 | 72 | \((M,1,72,16)\) |

### 9.2 Group-Equivariant RoPE

`PlatonicRoPE` stores base frequencies per base head and feature pair:

$$
\omega_{hf}\in\mathbb{R}^3,
\qquad
h=1,\dots,H_g,
\qquad
f=1,\dots,d_h/2.
$$

For the `"spiral"` initializer, these frequencies are deterministic spiral
directions with magnitudes increasing up to `rope_sigma`. In the active config:

```text
learned_freqs = true
freq_init = "spiral"
rope_sigma = 4.0
```

so they are initialized by the spiral rule and then trained.

The group acts by rotating the frequencies:

$$
\omega_{g h f} = R_g \omega_{h f}.
$$

The phase for node \(i\), group element \(g\), head \(h\), and pair \(f\) is

$$
\theta_{i g h f} = r_i^\top \omega_{g h f}.
$$

Each pair of head features is then rotated:

$$
\mathrm{Rot}_\theta(a,b)
=
(a\cos\theta-b\sin\theta,\;a\sin\theta+b\cos\theta).
$$

The code applies RoPE to \(Q\) and \(K\):

$$
\tilde Q_{i g h} =
\mathrm{RoPE}(Q_{i g h},r_i,g,h),
$$

$$
\tilde K_{i g h} =
\mathrm{RoPE}(K_{i g h},r_i,g,h).
$$

With `use_key=false`, this means

$$
\tilde K_{i g h} =
\mathrm{RoPE}(\mathbf{1},r_i,g,h),
$$

so the key side is positional rather than token-content based.

### 9.3 RoPE On Values

The active QM9 config uses:

```text
rope_on_values = true
```

so values are also rotated:

$$
\tilde V_{i g h} =
\mathrm{RoPE}(V_{i g h},r_i,g,h).
$$

After attention aggregates values, the output is inverse-rotated at the query
position:

$$
O_{i g h}^{\mathrm{final}}
=
\mathrm{RoPE}^{-1}(O_{i g h}^{\mathrm{attn}},r_i,g,h).
$$

This is the GTA-style "rotate values, then transport them back" path. It is
important for the vector readout behavior because the hidden values carry
coordinate-dependent orientation information but are returned to the query
frame before the equivariant output projection.

### 9.4 Softmax Attention Mode

The active QM9 config uses:

```text
attention = true
```

In graph mode, the attention is fully connected within each molecule. For every
source node \(i\), destination node \(j\), group element \(g\), and group-head
\(h\), the score is

$$
\ell_{ijgh}
=
\frac{
\langle \tilde Q_{igh},\tilde K_{jgh}\rangle
}{\sqrt{d_h}}.
$$

The softmax is over destination nodes \(j\) inside the same molecule, separately
for each source and group-head:

$$
\alpha_{ijgh}
=
\frac{\exp(\ell_{ijgh})}
{\sum_{j':b_{j'}=b_i}\exp(\ell_{ij'gh})}.
$$

The attention output is

$$
O_{igh}
=
\sum_{j:b_j=b_i}
\alpha_{ijgh}\tilde V_{jgh}.
$$

Then, if `rope_on_values=true`, \(O_{igh}\) is inverse-RoPE-rotated at \(r_i\),
flattened back to \((M,d)\), and passed through an equivariant output projection:

$$
\mathrm{out}_i = \mathrm{PlatonicLinear}_{\mathrm{out}}(O_i).
$$

### 9.5 Flash And Scatter Backends

There are two graph softmax attention implementations:

| Backend | Meaning |
| --- | --- |
| `scatter` | explicit fully connected within-graph edges and scatter softmax/sum |
| `flash` | FlashAttention varlen implementation, treating \(G\times H_g\) as heads |

Both implement the same attention semantics. The flash backend casts Q/K/V to
bf16 inside the kernel and returns to the original dtype, so small numerical
differences from scatter are expected.

### 9.6 Linear Attention Mode

If `attention=false`, `PlatonicConv` uses a kernelized linear-attention-like
aggregation. This is not the active QM9 generation baseline, but it is part of
the architecture.

It forms a per-graph kernel:

$$
A_b =
\frac{1}{Z_b}
\sum_{j:b_j=b}
\tilde K_j\otimes \tilde V_j,
$$

where \(Z_b\) is either `avg_num_nodes` or the actual node count when
`mean_aggregation=true`. Then

$$
O_i = \tilde Q_i A_{b_i}.
$$

If `rope_on_values=true`, the result is inverse-rotated at the query position.

## 10. How Geometry Enters

There is no Matterformer-style explicit distance RBF geometry bias in the
Platonic QM9 baseline. Geometry enters through these mechanisms:

1. EDM centers positions per molecule before noising.
2. Training applies random 3D rotation augmentation.
3. `pos` is passed into every `PlatonicConv`.
4. `PlatonicRoPE` converts `pos` into coordinate-dependent phases.
5. The group rotates RoPE frequencies, producing group-indexed orientation
   channels.
6. With `rope_on_values=true`, values are rotated by position and transported
   back with inverse RoPE.
7. The final vector readout turns group-indexed scalar channels into a 3D vector
   by rotating per-group vector predictions back through the group matrices and
   averaging.

This means geometry is not an additive pairwise bias. It is part of the
attention kernel and value transport.

### 10.1 Translation Handling

RoPE uses absolute positions as phases, so the model is not intrinsically
translation-invariant. The QM9 diffusion path subtracts per-graph means from
coordinates before noising, so the model is trained in a centered gauge:

$$
\sum_{i:b_i=b} r_i = 0.
$$

This is analogous to giving the model centered coordinates rather than asking it
to learn translation invariance from arbitrary origins.

### 10.2 Rotation Handling

For tetrahedron, the architecture exactly handles the finite set of tetrahedral
rotations through the group axis. For arbitrary rotations outside that group,
the architecture is not exactly equivariant. The training loop applies random
\(\mathrm{SO}(3)\) rotations, which encourages broader rotational behavior.

For trivial, there is no nontrivial exact rotation symmetry in the architecture.
It sees coordinates through ordinary identity-group RoPE and must rely on data
augmentation and learning.

## 11. Pooling And Readout

QM9 generation is node-level:

```text
scalar_task_level = "node"
vector_task_level = "node"
```

so the trunk output is not pooled across atoms. Each atom gets its own scalar
and vector output.

### 11.1 Scalar Readout

With `ffn_readout=true`, scalar readout is:

```text
PlatonicLinear(d -> d)
GELU
PlatonicLinear(d -> G * output_dim)
readout_scalars: mean over group axis
```

Let the readout produce grouped scalar channels

$$
z_{igc}\in\mathbb{R}^{G\times C_{\mathrm{out}}}.
$$

The final scalar output is the group average:

$$
y_{ic}^{\mathrm{scalar}} =
\frac{1}{G}\sum_{g=1}^G z_{igc}.
$$

For tetrahedron this average makes scalar outputs invariant to tetrahedral group
actions. For trivial, it is just the identity because \(G=1\).

### 11.2 Vector Readout

With `ffn_readout=true`, vector readout is:

```text
PlatonicLinear(d -> d)
GELU
PlatonicLinear(d -> G * output_dim_vec * 3)
readout_vectors: rotate each group component and average
```

The grouped vector channels are reshaped as

$$
z_{igv}\in\mathbb{R}^3,
\qquad
g=1,\dots,G.
$$

Then

$$
y_{iv}^{\mathrm{vector}}
=
\frac{1}{G}\sum_{g=1}^G R_g z_{igv}.
$$

For QM9:

$$
v=1,
\qquad
y_i^{\mathrm{vector}}\in\mathbb{R}^3.
$$

This vector readout is the main coordinate correction head. There is no
per-token MLP directly emitting xyz from hidden features independent of the
group structure; the readout is explicitly tied to the group matrices.

For the trivial group \(R_0=I\), this collapses to an ordinary vector readout:

$$
y_i^{\mathrm{vector}} = z_{i0}.
$$

## 12. Main Tetrahedral Baseline

The main baseline is the config default:

```text
solid_name = "tetrahedron"
hidden_dim = 1152
num_layers = 14
num_heads = 72
attention = true
rope_sigma = 4.0
learned_freqs = true
freq_init = "spiral"
use_key = false
rope_on_values = true
ape_sigma = null
```

### 12.1 Shape Summary

For a ragged batch with \(M\) atoms:

```text
scalars_in:    (M, 7)
lifted input:  (M, 12, 7) -> (M, 84)
hidden:        (M, 12, 96) -> (M, 1152)
Q/K/V:         (M, 12, 6, 16)
scalar output: (M, 6)
vector output: (M, 1, 3)
```

### 12.2 What Tetra Gives

Tetra introduces 12 orientation channels per atom. These are not separate atoms
or graph nodes. They are group-indexed channels inside each atom representation.

The main consequences are:

1. Exact equivariance to the 12 tetrahedral rotations.
2. Strong weight sharing through `PlatonicLinear`.
3. Group-rotated RoPE frequencies, so each group channel sees coordinates in a
   rotated frequency frame.
4. A vector readout that averages rotated group components to produce a
   covariant 3D coordinate correction.
5. Much lower parameter count than trivial at the same total hidden width.

### 12.3 Parameter Count

For the active QM9 architecture, instantiated with `attention_backend="scatter"`
and no compile wrapper:

```text
tetrahedron 1152/14/72: 17,289,129 trainable parameters
```

This is small relative to the total hidden width because most large maps are
group-constrained and carry roughly \(1/G = 1/12\) of the corresponding dense
parameters.

### 12.4 Intuition

The tetra model is not simply a small standard transformer. It uses the same
total hidden width as the trivial model, but it organizes that width as 12
structured orientation channels. The inductive bias says: "features should
transform predictably under tetrahedral rotations, and vector outputs should be
assembled from group-oriented components."

That bias is useful when the target is geometric and vector-valued, but it also
restricts the function class compared with a completely unconstrained dense
transformer of the same total width.

## 13. Trivial Platonic Baseline

The trivial baseline changes only the group:

```text
solid_name = "trivial"
G = 1
R_0 = I
```

Everything else can remain the same:

```text
hidden_dim = 1152
num_layers = 14
num_heads = 72
attention = true
rope_sigma = 4.0
learned_freqs = true
freq_init = "spiral"
use_key = false
rope_on_values = true
ape_sigma = null
```

### 13.1 Shape Summary

For a ragged batch with \(M\) atoms:

```text
scalars_in:    (M, 7)
lifted input:  (M, 1, 7) -> (M, 7)
hidden:        (M, 1, 1152) -> (M, 1152)
Q/K/V:         (M, 1, 72, 16)
scalar output: (M, 6)
vector output: (M, 1, 3)
```

### 13.2 What Trivial Removes

The trivial group removes:

1. Nontrivial group-axis permutations.
2. Tetrahedral weight sharing.
3. Exact nontrivial rotation equivariance.
4. Group averaging over multiple rotated vector components.
5. Group-rotated frequency banks.

What remains is still the Platonic implementation style:

1. `PlatonicLinear` is present, but with \(G=1\) it is an ordinary linear layer.
2. `PlatonicRoPE` is present, but with only the identity group.
3. `use_key=false` still makes keys all-ones before RoPE.
4. `rope_on_values=true` still rotates values and inverse-rotates outputs.
5. Scalar and vector readouts are still called through `to_scalars_vectors`, but
   group averaging over one element is the identity.

### 13.3 Parameter Count

For the active QM9 architecture:

```text
trivial 1152/14/72: 207,266,697 trainable parameters
```

This is about 12 times larger than the tetra model because there is no group
weight sharing.

### 13.4 Intuition

The trivial model is close to a conventional node transformer with coordinate
RoPE:

```text
scalar features + c_noise
  -> dense hidden tokens
  -> learned Q and V
  -> all-ones positional RoPE keys
  -> RoPE-rotated values with inverse value transport
  -> scalar and vector MLP readouts
```

Its geometry still enters strongly through RoPE and value transport, but it has
no nontrivial discrete rotation symmetry built into the weights. It is more
flexible but has a weaker geometric prior.

## 14. Tetra Versus Trivial

| Aspect | Tetra baseline | Trivial baseline |
| --- | --- | --- |
| `solid_name` | `"tetrahedron"` | `"trivial"` |
| group order | \(G=12\) | \(G=1\) |
| exact equivariance | 12 tetra rotations | identity only |
| total hidden width | 1152 | 1152 |
| channels per group | 96 | 1152 |
| total heads | 72 | 72 |
| heads per group | 6 | 72 |
| head dim | 16 | 16 |
| large linear parameter scaling | roughly \(d^2/12\) | roughly \(d^2\) |
| actual params | 17.3M | 207.3M |
| keys with active config | all-ones + RoPE | all-ones + RoPE |
| values with active config | RoPE + inverse RoPE | RoPE + inverse RoPE |
| vector readout | average of 12 rotated components | identity readout over one component |

The two baselines share the same EDM wrapper, same scalar noise concatenation,
same fully connected within-molecule attention semantics, same RoPE-on-values
choice, same width/depth/head-dim, and same node-level scalar/vector denoising
objective. The meaningful difference is the group: tetra has a structured
orientation axis with equivariant weight tying; trivial collapses that machinery
to dense unconstrained channels.

## 15. Why `use_key=false` Matters In Both Baselines

With active RoPE and `use_key=false`, `PlatonicConv` does not learn token-content
keys. Instead,

$$
K_{\mathrm{raw}} = \mathbf{1},
$$

and RoPE turns that constant tensor into a positional key:

$$
\tilde K_{i g h} =
\mathrm{RoPE}(\mathbf{1},r_i,g,h).
$$

Queries and values are still learned from token features:

$$
Q_i=\mathrm{PlatonicLinear}_Q(h_i),
\qquad
V_i=\mathrm{PlatonicLinear}_V(h_i).
$$

So attention is not purely positional. The score

$$
\ell_{ijgh}
=
\langle \mathrm{RoPE}(Q_i,r_i),\mathrm{RoPE}(\mathbf{1},r_j)\rangle
$$

depends on the source token content through \(Q_i\), and on destination
coordinates through the positional key.

This is one of the main differences from standard learned-key MHA.

## 16. Why `rope_on_values=true` Matters

With `rope_on_values=false`, values are aggregated in the raw learned feature
frame:

$$
O_i=\sum_j\alpha_{ij}V_j.
$$

With `rope_on_values=true`, the value path becomes:

$$
\tilde V_j=\mathrm{RoPE}(V_j,r_j),
$$

$$
O_i^{\mathrm{attn}}=\sum_j\alpha_{ij}\tilde V_j,
$$

$$
O_i=\mathrm{RoPE}^{-1}(O_i^{\mathrm{attn}},r_i).
$$

Intuitively, the value coming from atom \(j\) is expressed with position
dependent phase information at \(j\), transported through attention, and then
brought back to the query position \(i\). This is why the trivial baseline is
not just a vanilla transformer even though \(G=1\).

## 17. Dense Mode Versus Graph Mode

QM9 generation uses:

```text
dense_mode = false
```

so the model runs graph mode with ragged tensors and `batch` indices.

If `dense_mode=true`, `to_dense_and_mask` packs ragged atoms into padded tensors,
and `PlatonicConv` uses dense attention or dense linear attention with a boolean
mask. This is not the active QM9 baseline.

Graph mode attention is fully connected within each molecule. There is no bond
graph or cutoff graph in the active QM9 generation path.

## 18. Comparison To Matterformer Concepts

This table maps the Platonic mechanisms to the Matterformer mechanisms discussed
in the neighboring architecture document.

| Concept | Platonic QM9 | Matterformer QM9 |
| --- | --- | --- |
| input layout | ragged \((M,\cdot)\) plus `batch` | padded \((B,N,\cdot)\) plus `pad_mask` |
| sigma conditioning | concatenate \(c_{\mathrm{noise}}\) to scalar input | concat, AdaLN, both, or none |
| coordinate embedding | APE optional, off in baseline | Fourier/RFF/learnable RFF optional |
| main geometry path | group-equivariant RoPE in attention | RoPE, explicit geometry bias, simplicial geometry |
| group/equivariance | finite group axis, exact for chosen group | no equivalent group axis |
| active keys | all-ones positional RoPE keys | configurable learned or all-ones in Matterformer RoPE MHA |
| active values | RoPE values plus inverse unrotation | configurable in Matterformer RoPE MHA |
| coordinate readout | Platonic vector readout from group components | direct MLP head or relative-vector head |
| coordinate delta mean | not hard mean-subtracted in backbone | hard mean-subtracted in both Matterformer coord heads |

## 19. Practical Takeaways

### 19.1 Tetra Is More Structured, Not Wider

The tetra model's total width is 1152, but internally it is \(12\times96\).
Because most weights are group-constrained, it is parameter-efficient and
geometrically biased. It should be read as "wide representation with strong
weight sharing", not as a 207M dense transformer.

### 19.2 Trivial Is The Dense Control

The trivial model keeps the same external code path but sets \(G=1\). This makes
it a useful control for asking: how much of the result comes from the Platonic
group structure, and how much comes from RoPE/value transport/EDM training?

### 19.3 Geometry Enters Without Distances

The active Platonic baselines do not explicitly compute pair distances, RBF
features, bond edges, or triangle angles. Geometry enters through coordinates in
RoPE and through the group vector readout. This is very different from
Matterformer's geometry-bias and simplicial-geometry paths.

### 19.4 Vector Readout Is A Real Architectural Difference

The coordinate correction in Platonic is not a generic MLP from hidden features
to xyz, except in the degenerate \(G=1\) sense. For tetra, the model predicts
group-indexed vector components and then averages them after rotating by the 12
tetrahedral matrices. That readout is part of the equivariance story.

### 19.5 Tetra Equivariance Is Finite-Group Equivariance

The tetra model is exactly equivariant to tetrahedral rotations, not all
\(\mathrm{SO}(3)\). The training augmentation supplies arbitrary rotations, so
empirical behavior can be more broadly rotationally robust, but the formal
guarantee is for the selected finite group.

### 19.6 One-Sentence Summary

The main tetra Platonic QM9 model is a finite-group equivariant transformer whose
hidden states are group-indexed scalar fields, whose attention uses
coordinate-dependent group-RoPE with positional all-ones keys and transported
values, and whose coordinate denoising vector is read out by averaging rotated
group components; the trivial baseline collapses the same machinery to the
identity group, keeping RoPE/value transport but removing nontrivial group
equivariance and weight sharing.
