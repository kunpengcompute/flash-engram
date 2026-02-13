from setuptools import setup, Extension, find_packages
import pybind11

# 编译参数
compile_args = [
    '-O3',
    '-march=armv8.2-a+sve+sve2',
    '-flto',
    '-fopenmp',
    '-Wall',                  
    '-Wextra',
]

# 链接参数
link_args = [
    '-flto',
    '-fopenmp',
]

ext_modules = [
    Extension(
        'flash_engram.core_engram',
        ['flash_engram/core_engram.cpp'],
        include_dirs=[pybind11.get_include(), 'flash_engram'],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name='flash_engram',
    version='0.0.1',
    author='boostkit',
    packages=find_packages(), 
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
)
