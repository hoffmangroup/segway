from pkg_resources import get_distribution

# Allow raising a DistributionNotFound error if somehow segway was not
# installed
__version__ = get_distribution(__name__.split('.')[0]).version
