"""
Custom setup.py to exclude pygame-dependent files from the wheel.

viewer.py, controls.py, and __main__.py require pygame which is not
available in the Scope deployment environment. They are only needed
for local interactive use.
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


# Files that require pygame and should not be packaged in the wheel,
# or that are packaging artifacts (setup.py itself).
_EXCLUDE_MODULES = {"viewer", "controls", "__main__", "setup"}


class BuildPy(_build_py):
    """Custom build_py that excludes pygame-dependent modules."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, path)
            for (pkg, mod, path) in modules
            if mod not in _EXCLUDE_MODULES
        ]


setup(
    cmdclass={"build_py": BuildPy},
)
