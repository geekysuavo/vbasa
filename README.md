
# Variational Bayesian Active Site Alignment

VB-ASA is a type of probabilistic point cloud alignment specially designed
for superimposing the alpha and beta carbons of two protein backbones as
part of a structure-based active site search tool.

## Usage

Build and install the **vbasa** python package:
```bash
cd vbasa
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl
```
and then use it to fit models:
```python
(Xa, Xb, Ya, Yb) = load_point_cloud_data()
model = vbasa.model(Xa, Xb, Ya, Yb, prior_precision=1e-6)
elbo = model.fit(iterations=1000)
print(model.Z)
```

## Licensing

The **vbasa** sources are released under the
[MIT License](https://opensource.org/licenses/MIT). See the
[LICENSE.md](LICENSE.md) file for the complete license terms.

