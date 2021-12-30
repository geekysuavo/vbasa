
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_module = Pybind11Extension(
    name="vbasa",
    sources=["src/vbasa.cc"],
    include_dirs=["/usr/local/include/eigen3"],
)

setup(
    name="vbasa",
    version="0.0.1",

    description="Variational Bayesian Active Site Alignment",
    url="http://github.com/geekysuavo/vbasa",

    author="Bradley Worley",
    author_email="geekysuavo@gmail.com",

    license="MIT",

    cmdclass={"build_ext": build_ext},
    ext_modules=[ext_module],
)
