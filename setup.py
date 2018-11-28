import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OpenMLResultExtractor",
    version="0.1.0",
    author="Arlind Kadra",
    author_email="arlindkadra@gmail.com",
    description="A tool that can extract results from OpenML, given"
                "different restrictions on tasks, flows or both"
                "combined",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArlindKadra/Benchmark",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
)
