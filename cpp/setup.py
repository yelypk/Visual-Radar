from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = Path(__file__).parent.resolve()
SRC = str(HERE / "_ncc.cpp")   

ext_modules = [
    Pybind11Extension(
        "visual_radar._ncc",
        [SRC],
        cxx_std=17,
    ),
]

setup(
    name="visual-radar-ncc",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=[],  
    py_modules=[],
    zip_safe=False,
)
