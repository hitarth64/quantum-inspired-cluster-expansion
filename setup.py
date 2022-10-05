import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quantum-inspired-cluster-expansion",
    version="0.0.1",
    author="Hitarth Choubisa",
    author_email="hitarth.choubisa@mail.utoronto.ca",
    description="Quantum-inspired Cluster Expansion: formulating chemical space search as QUBOs and Ising models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hitarth64/quantum-inspired-cluster-expansion",
    package_data={'QuantumInspiredClusterExpansion':['dataset/*.json','dataset/*.csv']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=open("requirements.txt", "r").readlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD 3-Clause Clear License",
        "Operating System :: OS Independent",
    ],
)
