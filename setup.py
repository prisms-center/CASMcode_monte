from skbuild import setup

setup(
    name="libcasm-monte",
    version="2.0a1",
    packages=[
        "libcasm",
        "libcasm.monte",
    ],
    package_dir={"": "python"},
    cmake_install_dir="python/libcasm",
    include_package_data=False,
)