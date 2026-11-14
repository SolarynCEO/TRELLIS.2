from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
IS_WINDOWS = platform.system() == "Windows"
EIGEN_INCLUDE_DIR = os.path.join(ROOT, "third_party", "eigen")
EIGEN_DENSE_HEADER = os.path.join(EIGEN_INCLUDE_DIR, "Eigen", "Dense")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True
    else:
        raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET}")

if not os.path.isfile(EIGEN_DENSE_HEADER):
    raise RuntimeError(
        "Missing Eigen headers at o-voxel/third_party/eigen. "
        "Initialize submodules before building: "
        "git submodule update --init --recursive"
    )

if IS_HIP:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    nvcc_flags = [f"--offload-arch={arch}" for arch in archs]
else:
    nvcc_flags = []

if IS_WINDOWS:
    cxx_flags = ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/Zc:__cplusplus"]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-Xcompiler=/std:c++17",
        "-Xcompiler=/EHsc",
        "-Xcompiler=/permissive-",
        "-Xcompiler=/Zc:__cplusplus",
    ] + nvcc_flags
    if not IS_HIP:
        nvcc_flags += ["-allow-unsupported-compiler"]
else:
    cxx_flags = ["-O3", "-std=c++17"]
    nvcc_flags = ["-O3", "-std=c++17"] + nvcc_flags

setup(
    name="o_voxel",
    packages=[
        'o_voxel',
        'o_voxel.convert',
        'o_voxel.io',
    ],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                # Convert functions
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                ## Serialization functions
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                # IO functions
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                # Rasterization functions
                "src/rasterize/rasterize.cu",
                
                # main
                "src/ext.cpp",
            ],
            include_dirs=[
                EIGEN_INCLUDE_DIR,
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
