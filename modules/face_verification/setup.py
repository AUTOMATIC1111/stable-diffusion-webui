from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yk_face",
    version="0.3.3",
    description="Python SDK for the YouFace API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Youverse",
    author_email="tech@youverse.id",
    url="https://github.com/dev-yoonik/YK-Face-Python",
    license='MIT',
    packages=["yk_face"],
    install_requires=[
        'yk-face-api-model>=3.0.2,<4',
        'yk-utils>=1.3.1,<2'
    ],
    extras_require={
      "tests": ['pytest'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)