from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "skat_rl._skat_cpp",
        [
            "src/skat_rl/cpp_engine/bindings.cpp",
            "src/skat_rl/cpp_engine/fast_skat.cpp",
        ],
        cxx_std=17,
    ),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
