from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

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
    install_requires=[
        "openai",
        "iso639-lang",
        "tenacity",
        "funcy",
        "langchain @ git+https://github.com/FlowerWrong/langchain.git@support-multi-openai-api-keys",
    ],
)
