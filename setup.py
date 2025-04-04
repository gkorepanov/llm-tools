from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-tools",
    version="0.1.1",
    description="Random tools for working with LLMs",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/gkorepanov/llm-tools",
    author="George Korepanov",
    author_email="gkorepanov.gk@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "iso639-lang~=2.2.2",
        "tenacity~=8.2.3",
        "funcy==2.0",
        "langchain~=0.3.22",
        "litellm~=1.65.3",
    ],
)
