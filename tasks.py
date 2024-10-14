import glob
import os
import shutil
import tarfile

from invoke import task

import pvgridder


@task
def build(c):
    shutil.rmtree("dist", ignore_errors=True)
    c.run("python -m build --sdist --wheel .")


@task
def html(c):
    c.run("sphinx-build -b html doc/source doc/build")


@task
def tag(c):
    c.run(f"git tag v{pvgridder.__version__}")
    c.run("git push --tags")


@task
def clean(c, bytecode=False):
    patterns = [
        "build",
        "dist",
        "pvgridder.egg-info",
        "doc/build",
        "doc/source/examples",
    ]

    if bytecode:
        patterns += glob.glob("**/*.pyc", recursive=True)
        patterns += glob.glob("**/__pycache__", recursive=True)

    for pattern in patterns:
        if os.path.isfile(pattern):
            os.remove(pattern)
        else:
            shutil.rmtree(pattern, ignore_errors=True)


@task
def ruff(c):
    c.run("ruff check --fix pvgridder")
    c.run("ruff format --target-version py38 --line-length 88 pvgridder")


@task
def tar(c):
    patterns = [
        "__pycache__",
    ]

    def filter(filename):
        for pattern in patterns:
            if filename.name.endswith(pattern):
                return None

        return filename

    with tarfile.open("pvgridder.tar.gz", "w:gz") as tf:
        tf.add("pvgridder", arcname="pvgridder/pvgridder", filter=filter)
        tf.add("pyproject.toml", arcname="pvgridder/pyproject.toml")
