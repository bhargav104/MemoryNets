from setuptools import setup, find_packages

setup(
    name="torch_ac",
    version="1.0.1",
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.13.0",
        "torch>=1.0.0"
    ]
)