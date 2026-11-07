
# Variational Bayesian Active Site Alignment

VB-ASA is a type of probabilistic point cloud alignment specially designed
for superimposing the alpha and beta carbons of two protein backbones as
part of a structure-based active site search tool.

## Usage

Building requires a C++20 compiler, CMake 3.24 or newer, and
[Eigen](https://eigen.tuxfamily.org).

Build and install the **vbasa** python package using
[uv](https://docs.astral.sh/uv):
```bash
uv build
```
or to run the tests:
```bash
uv run --group test pytest
```
and then use it to fit models:
```python
(Xa, Xb, Ya, Yb) = load_point_cloud_data()
(target_seq, source_seq) = load_sequences()
model = vbasa.model(
    Xa, Xb, Ya, Yb, target_seq, source_seq,
    sub_matrix=vbasa.matrices.blosum62(),
)
elbo = model.optimize(iterations=1000)
print(model.Z)
print(elbo)
```

Model fits may also be performed from multiple initial rotations to ensure
convergence to the correct posterior mode:
```python
model = vbasa.multistart(
    Xa, Xb, Ya, Yb, target_seq, source_seq,
    sub_matrix=vbasa.matrices.blosum62(),
)
```
Target pairs need not all have a partner in the source. The last column
of `model.Z` holds the probability that each target pair is unmatched,
and explained instead by a uniform density over the target's bounding box.
Its prior probability is set by `prior_unmatched` (default 0.1) in
`vbasa.Model` and `vbasa.model`; passing zero forces every target pair
to be assigned to some source pair.

## Licensing

The **vbasa** sources are released under the
[MIT License](https://opensource.org/licenses/MIT). See the
[LICENSE.md](LICENSE.md) file for the complete license terms.

