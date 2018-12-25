import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="omlextractor",
    version="0.1.0",
    author="Arlind Kadra",
    author_email="arlindkadra@gmail.com",
    description="A tool that can extract results from OpenML, given"
                "different restrictions on tasks, flows or both"
                "combined",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArlindKadra/ResultExtractor/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
)
