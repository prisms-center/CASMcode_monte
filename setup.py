from skbuild import setup

setup(
    name="libcasm-monte",
    version="2.2.0",
    packages=[
        "libcasm",
        "libcasm.monte",
        "libcasm.monte.events",
        "libcasm.monte.methods",
        "libcasm.monte.ising_cpp",
        "libcasm.monte.ising_cpp.semigrand_canonical",
        "libcasm.monte.ising_py",
        "libcasm.monte.ising_py.semigrand_canonical",
        "libcasm.monte.sampling",
    ],
    package_dir={"": "python"},
    cmake_install_dir="python/libcasm",
    include_package_data=False,
)
