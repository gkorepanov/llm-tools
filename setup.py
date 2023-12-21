from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

requirements = []
dependency_links = []
with open(here / "requirements.txt", encoding="utf-8") as f:
    for line in f:
        if "git+" in line:
            dependency_links.append(line.strip())
        else:
            requirements.append(line.strip())


setup(
    name="llm-tools",
    version="0.1.0",
    description="Random tools for working with LLMs",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/gkorepanov/llm-tools",
    author="George Korepanov",
    author_email="gkorepanov.gk@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    dependency_links=dependency_links,
)