import os
import glob
import multiprocessing
from setuptools import setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
    CUDA_HOME,
    IS_HIP_EXTENSION,
)

from setuptools.command.install import install
import shutil

name = "flashfast"
current_directory = os.path.abspath(os.path.dirname(__file__))


class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        num_cores = multiprocessing.cpu_count()

        for ext in self.extensions:
            ext.extra_compile_args = ["-j14"]  # 使用-j选项设置线程数
        super().build_extensions()


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print("Running custom install command...")

        # try:
        #     shutil.rmtree('build')
        #     print('Deleted build directory.')
        # except Exception as e:
        #     print(f"Error deleting build directory: {e}")

        # try:
        #     shutil.rmtree(f'{name}.egg-info')
        #     print(f'Deleted {name}.egg-info directory.')
        # except Exception as e:
        #     print(f"Error deleting {name}.egg-info directory: {e}")


ext_modules = []


def build_for_cuda():
    cutlass_include_dir = os.path.join(
        current_directory, "3rdparty", "flash-attention", "csrc", "cutlass", "include"
    )
    cutlass_tools_include_dir = os.path.join(
        current_directory,
        "3rdparty",
        "flash-attention",
        "csrc",
        "cutlass",
        "tools",
        "util",
        "include",
    )
    flash_attention_include_dir = os.path.join(
        current_directory, "3rdparty", "flash-attention", "csrc", "flash_attn", "src"
    )
    flash_attention_sources = glob.glob(
        os.path.join(
            current_directory,
            "3rdparty",
            "flash-attention",
            "csrc",
            "flash_attn",
            "src",
            "flash_fwd*.cu",
        )
    )

    sources = [
        os.path.join("flashfast", "pybind.cpp"),
        # os.path.join("flashfast", "kernels", "nvidia", "flash_attn", f"flash_api.cpp"),
        # os.path.join("flashfast", "kernels", "nvidia", "flash_attn", f"flash_attn.cpp"),

        os.path.join("flashfast", "ops", "unittest", f"ut.cpp"),
        os.path.join("flashfast", "ops", "nvidia", "attn", f"attention.cu"),
    ] #+ flash_attention_sources

    ext_modules.append(
        CUDAExtension(
            name,
            sources=sources,
            include_dirs=[
                cutlass_include_dir,
                cutlass_tools_include_dir,
                flash_attention_include_dir,
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-line-info",
                    "-Xptxas=-v",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                ],
            },
        ),
    )


def build_for_rocm():
    # Rename cpp file in flash_attn_rocm to hip.
    def rename_cpp_to_hip(cpp_files):
        for entry in cpp_files:
            file_name = os.path.splitext(entry)[0] + ".hip"
            if not os.path.exists(file_name):
                shutil.copy(entry, file_name)

    fa_sources = glob.glob(
        os.path.join(
            current_directory,
            "3rdparty",
            "flash-attention-rocm",
            "csrc",
            "flash_attn_rocm",
            "src",
            "*.cpp",
        )
    )
    rename_cpp_to_hip(fa_sources)

    # Switch AMD compilers
    os.environ["CC"] = "hipcc"
    os.environ["CXX"] = "hipcc"

    # Dependencies include
    flash_attn_rocm_include_dir = os.path.join(
        current_directory, "3rdparty", "flash-attention-rocm", "csrc", "flash_attn_rocm"
    )
    flash_attn_rocm_src_include_dir = os.path.join(
        current_directory,
        "3rdparty",
        "flash-attention-rocm",
        "csrc",
        "flash_attn_rocm",
        "src",
    )
    composable_kernel_src_include_dir = os.path.join(
        current_directory,
        "3rdparty",
        "flash-attention-rocm",
        "csrc",
        "flash_attn_rocm",
        "composable_kernel",
        "include",
    )
    composable_kernel_lib_include_dir = os.path.join(
        current_directory,
        "3rdparty",
        "flash-attention-rocm",
        "csrc",
        "flash_attn_rocm",
        "composable_kernel",
        "library",
        "include",
    )
    flashfast_amd_include_dir = os.path.join(
        current_directory, "flashfast", "kernels", "amd", "flash_attn_rocm"
    )

    flash_attention_sources = glob.glob(
        os.path.join(
            current_directory,
            "3rdparty",
            "flash-attention-rocm",
            "csrc",
            "flash_attn_rocm",
            "src",
            "*.hip",
        )
    )
    sources = [
        os.path.join("flashfast", "pybind.cpp"),
        os.path.join("flashfast", "kernels", "amd", "flash_attn_rocm", f"flash_api.hip"),
        os.path.join("flashfast", "ops", "unittest", f"ut.cpp"),
        os.path.join("flashfast", "ops", "amd", "attn", f"attention.hip"),
        os.path.join("flashfast", "ops", "amd", "norm", f"norm.hip"),
        os.path.join("flashfast", "ops", "amd", "embedding", f"embedding.hip"),
        os.path.join("flashfast", "ops", "amd", "linear", f"gemm.hip"),
        os.path.join("flashfast", "ops", "amd", "element", f"residual.hip"),
        os.path.join("flashfast", "layers", f"attn_layer.cpp"),
        os.path.join("flashfast", "layers", f"attn_layer_long.cpp"),
        os.path.join("flashfast", "layers", f"ffn_layer.cpp"),
        os.path.join("flashfast", "layers", f"ffn_layer_long.cpp"),
    ] + flash_attention_sources

    ext_modules.append(
        CppExtension(
            name,
            sources=sources,
            include_dirs=[
                flash_attn_rocm_include_dir,
                flash_attn_rocm_src_include_dir,
                composable_kernel_src_include_dir,
                composable_kernel_lib_include_dir,
                flashfast_amd_include_dir,
            ],
            # todo [Hard code] : Defualt gfx90a for MI210
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fopenmp",
                    "-lgomp",
                    "-DNDEBUG",
                    "-std=c++20",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "--offload-arch=gfx90a",
                ]
            },
        ),
    )


BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        build_for_rocm()
    else:
        build_for_cuda()
else:
    if BUILD_TARGET == "cuda":
        build_for_cuda()
    elif BUILD_TARGET == "rocm":
        build_for_rocm()


setup(
    name=name,
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension,
        "install": CustomInstallCommand,
    },
)

