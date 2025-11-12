import os
import glob
import torch
from platform import system
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

SYSTEM = system()

bsp_root_path = os.getcwd() + "/bspline_surface/Cpp/"
bsp_src_path = bsp_root_path + "src/"
bsp_sources = glob.glob(bsp_src_path + "*.cpp")
bsp_include_dirs = [bsp_root_path + "include"]

bsp_extra_compile_args = [
    "-O3",
    "-DCMAKE_BUILD_TYPE=Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DTORCH_USE_CUDA_DSA",
]

if SYSTEM == "Darwin":
    bsp_extra_compile_args.append("-std=c++17")
elif SYSTEM == "Linux":
    bsp_extra_compile_args.append("-std=c++17")

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    arch_str = f"{cc[0]}.{cc[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

    bsp_sources += glob.glob(bsp_src_path + "*.cu")

    extra_compile_args = {
        "cxx": bsp_extra_compile_args
        + [
            "-DUSE_CUDA",
            "-DTORCH_USE_CUDA_DSA",
        ],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
            "-std=c++17",
            "-DTORCH_USE_CUDA_DSA",
        ],
    }

    bsp_module = CUDAExtension(
        name="bsp_cpp",
        sources=bsp_sources,
        include_dirs=bsp_include_dirs,
        extra_compile_args=extra_compile_args,
    )

else:
    bsp_module = CppExtension(
        name="bsp_cpp",
        sources=bsp_sources,
        include_dirs=bsp_include_dirs,
        extra_compile_args=bsp_extra_compile_args,
    )

setup(
    name="BSP-CPP",
    version="1.0.0",
    author="Changhao Li",
    packages=find_packages(),
    ext_modules=[bsp_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
