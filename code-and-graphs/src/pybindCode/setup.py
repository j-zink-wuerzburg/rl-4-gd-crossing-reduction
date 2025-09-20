from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "graph_utils",
        ["graph_utils.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]
#
setup(
    name="graph_utils",
    ext_modules=ext_modules,
    zip_safe=False,
)