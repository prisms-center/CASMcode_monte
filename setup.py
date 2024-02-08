from skbuild import setup

setup(
    name="libcasm-monte",
    version="2.0a1",
    packages=[
        "libcasm",
        "libcasm.monte",
        "libcasm.monte.events",
        "libcasm.monte.calculators",
        "libcasm.monte.calculators.complete_semigrand_canonical_py",
        "libcasm.monte.implementations",
        "libcasm.monte.implementations.ising_cpp",
        "libcasm.monte.methods",
        "libcasm.monte.models",
        "libcasm.monte.models.ising_cpp",
        "libcasm.monte.models.ising_py",
    ],
    package_dir={"": "python"},
    cmake_install_dir="python/libcasm",
    include_package_data=False,
)
