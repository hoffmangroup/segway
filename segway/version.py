from importlib.metadata import version

# Allow raising a PackageNotFoundError if somehow segway was not
# installed
__version__ = version("segway")
