#!/usr/bin/env python
"""
install.py

General installation script for the segway package. Interacts with
the user to configure the environment to download and install all
dependencies and the segway package.

This script is designed to be downloaded and run independently, and
will guide the process of downloading and installing all other source
code.

(c) 2009: Orion Buske <orion.buske@gmail.com>

"""
import platform

####################### BEGIN COMMON CODE HEADER #####################

import os
import pkg_resources
import sys

from distutils.spawn import find_executable
from distutils.version import LooseVersion
from urllib import urlretrieve
from site import addsitedir
from string import Template
from subprocess import call, PIPE, Popen
from tempfile import gettempdir

assert sys.version_info >= (2, 4)

MIN_HDF5_VERSION = "1.8"
MIN_NUMPY_VERSION = "1.2"

HDF5_URL = "http://www.hdfgroup.org/ftp/HDF5/prev-releases/hdf5-1.8.2/src/hdf5-1.8.2.tar.gz"
EZ_SETUP_URL = "http://peak.telecommunity.com/dist/ez_setup.py"

EASY_INSTALL_PROMPT = "May I install %s and dependencies?"
HDF5_INSTALL_MESSAGE = "\nHDF5 is very large and installation usually takes 5-10 minutes. Please be patient.\nIt is common to see many warnings during compilation."

# Template for cfg file contents
CFG_FILE_TEMPLATE = """
[install]
prefix = $prefix
exec_prefix = $prefix
install_platlib = $platlib
install_purelib = $platlib
install_scripts = $scripts

[easy_install]
install_dir = $platlib
script_dir = $scripts
"""

# One command per line
HDF5_INSTALL_SCRIPT = """
cd $tmpdir
if [ ! -e $file ]; then wget $url -O $file; fi
if [ ! -d $filebase ]; then tar -xzf $file; fi
cd $filebase
./configure --prefix=$dir
make
make install
cd ..
"""

####################### END COMMON CODE HEADER #####################


MIN_HDF5_VERSION = "1.8"
MIN_NUMPY_VERSION = "1.2"
MIN_DRMAA_VERSION = "0.4a3"

LSF_DRMAA_URL = "http://softlayer.dl.sourceforge.net/project/lsf-drmaa\
/lsf_drmaa/1.0.3/lsf_drmaa-1.0.3.tar.gz"

GMTK_VERSION_FILENAME = "gmtk.version"
GMTK_USER = "segway"
GMTK_URL = "http://noble.gs.washington.edu/proj/segway/gmtk/gmtk-20091016.tar.gz"

SEGWAY_USER = "comp"
SEGWAY_URL = "http://noble.gs.washington.edu/proj/segway/src/segway-0.1.18.tar.gz"


LSF_DRMAA_INSTALL_SCRIPT = """
cd $tmpdir
wget $url -O $file
if [ ! -d $filebase ]; then tar -xzf $file; fi
cd $filebase
./configure --prefix=$dir
make
make install
cd ..
rm -rf $filebase
rm -f $file
"""

# also has keywords: user, password, OPTFLAGS, ARCH_GCC, SSELEVEL
GMTK_INSTALL_SCRIPT = """
cd $tmpdir
wget --user=$user --password=$password $url -O $file
if [ ! -d $filebase ]; then tar -xzf $file; fi
cd $filebase

mkdir -p "$dir/bin"
mkdir tksrc/bin 2>/dev/null || true

if [ "$(find . -name "*.a" -print0)" ]; \
then find . -name "*.a" -print0 | xargs -0 rm; \
fi

make clean "OPTFLAGS=$OPTFLAGS"
make linux "OPTFLAGS=$OPTFLAGS"
make depend "OPTFLAGS=$OPTFLAGS"

make "OPTFLAGS=$OPTFLAGS"
make -C featureFileIO install "OPTFLAGS=$OPTFLAGS"\
    "INSTALL=install" "install_prefix=$dir"
make -C tksrc install "OPTFLAGS=$OPTFLAGS" "install_prefix=$dir"

mkdir -p "$dir/etc"
declare -p OPTFLAGS > "$dir/etc/gmtk.build-options"
echo "$version" > "$dir/etc/gmtk.version"
cd ..
rm -rf $filebase
rm -f $file
"""

SEGWAY_INSTALL_SCRIPT = """
cd $tmpdir
wget --user=$user --password=$password $url -O $file
if [ ! -d $filebase ]; then tar -xzf $file; fi
cd $filebase
$python setup.py install
cd ..
rm -rf $filebase
rm -f $file
"""

####################### BEGIN COMMON CODE BODY #####################

############################ CLASSES ##########################
class DependencyError(Exception):
    pass

class InstallationError(Exception):
    pass

class InteractiveShell(object):
    """Class to manage running scripts through an interactive shell

    Executes each command in a separate shell
    """

    def __init__(self, env={}):
        self.env = dict(os.environ)
        self.env.update(env)
        self.old_cwd = os.getcwd()

    def execute(self, command, verbose=False):
        """Execute the given string command and return the retcode."""

        if verbose:
            print >>sys.stderr, ">> %s" % command

        # Trap calls to `cd`
        if command.strip().startswith("cd"):
            try:
                dest = command.strip().split()[1]
                if dest == "-":
                    dest = self.old_cwd
            except IndexError:
                dest = os.expanduser("~")  # Defaults to $HOME

            # Change directory
            self.old_cwd = os.getcwd()  # Save for `cd -`
            os.chdir(dest)
            return 0
        else:
            return call(str(command), shell=True,
                        env=self.env, cwd=os.getcwd())

    def run_script(self, script, verbose=False):
        """Runs each string command in a list, stopping if an error occurs

        verbose: if true, each command is first echo'd to stderr

        """
        for line in script:
            if len(line) == 0:
                continue

            retcode = self.execute(line, verbose=verbose)
            if retcode != 0:
                raise OSError("Command failed: %s" % line)

    def run_block(self, block, verbose=False):
        """Run each line of a multi-line string as a separate command"""
        script = block.strip().split("\n")
        self.run_script(script, verbose=verbose)


class ShellManager(object):
    """Class to manage details of shells.

    Provides an interface for writing to shell rc files and
    setting and testing environment variables.

    Attributes:
    file: The shell rc file, or None if none known
    has_file: True if there is a known shell rc file
    name: The string name of the shell (e.g. "bash", "csh", etc.)
      or "" if unknown
    """
    def __init__(self, shell=None):
        """Creates a ShellManager given a shell string.

        If shell is None, tries to use environment variable SHELL.
        self.name: basically same as shell, but if shell is None, self.name=""
        """
        self.name = shell
        if shell is None:
            try:
                shell = os.path.basename(os.environ["SHELL"])
                self.name = shell
            except KeyError:
                self.name = ""

        if self.name.endswith("csh"):
            self._env_format = "setenv %s %s"
        elif self.name.endswith("sh"):
            self._env_format = "export %s=%s"
        else:
            self._env_format = "set %s to %s"  # Console output

        # What RC file are we using, or are we printing to the terminal?
        if self.name.endswith("sh"):
            self.file = os.path.join("~", ".%src" % shell)
        else:
            self.file = None  # Print to terminal
            if shell is None:
                print >>sys.stderr, "SHELL variable missing or confusing."
            else:
                print >>sys.stderr, "Unrecognized shell: %s" % shell

        if self.file is None:
            print >>sys.stderr, ("Shell-specific commands will be printed"
                                 " to the terminal")

        self._out = None  # Output file not yet open
        self.has_file = (self.file is not None)

    def set_rc_env(self, variable, value):
        """Write the given variable and value the the shell rc file (or stdout)
        """
        if self._out is None:
            if self.file is None:
                self._out = sys.stdout
            else:
                self._out = open(os.path.expanduser(self.file), "a")

        cmd = self._env_format % (variable, value)
        print >>self._out, "\n%s  # Added by install script" % cmd

    def add_to_rc_env(self, variable, value):
        """Prepend the value to the variable in the shell rc file (or stdout)
        """
        full_value = "%s:$%s" % (value, variable)
        self.set_rc_env(variable, full_value)

    def set_env(self, variable, value):
        """Set the environment variable to the given value.
        """
        os.environ[variable] = value

    def add_to_env(self, variable, value):
        """Prepend value to the value of the environment variable.
        """
        if variable in os.environ:
            value = "%s:%s" % (value, os.environ[variable])

        self.set_env(variable, value)

    def in_env(self, variable, value):
        """Checks if value is in environment variable.

        Returns true if variable found and value is one of the ':'-
        delimited values in it.
        """
        try:
            env = os.environ[variable]
            env_values = env.strip().split(":")
            return value in env_values
        except KeyError:
            return False

    def close(self):
        """Close the open rc file, if one exists
        """
        if self._out not in [None, sys.stdout, sys.stderr]:
            try:
                self._out.close()
            except ValueError:
                pass

######################## UTIL FUNCTIONS ####################
def setup_arch_home():
    query = "\nWhere should platform-specific files be installed?"
    default_arch_home = get_default_arch_home()
    arch_home = prompt_user(query, default_arch_home)
    make_dir(arch_home)
    return arch_home

def get_default_arch_home():
    if "ARCHHOME" in os.environ:
        return os.environ["ARCHHOME"]
    elif "ARCH" in os.environ:
        arch = os.environ["ARCH"]
    else:
        (sysname, nodename, release, version, machine) = os.uname()
        arch = "-".join([sysname, machine])

    return os.path.expanduser("~/arch/%s" % arch)

def setup_python_home(arch_home=None):
    if arch_home is None:
        arch_home = get_default_arch_home()

    query = "\nWhere should new Python packages be installed?"
    default_python_home = get_default_python_home(arch_home)
    python_home = fix_path(prompt_user(query, default_python_home))
    make_dir(python_home)
    addsitedir(python_home)  # Load already-installed packages/eggs
    reload(pkg_resources)  # Update working set
    return python_home, default_python_home

def get_default_python_home(arch_home):
    python_version = sys.version[:3]
    return os.path.join(arch_home, "lib", "python%s" % python_version)

def setup_script_home(arch_home=None):
    if arch_home is None:
        arch_home = get_default_arch_home()

    query = "\nWhere should new scripts and executables be installed?"
    default_script_home = os.path.join(arch_home, "bin")
    script_home = fix_path(prompt_user(query, default_script_home))
    make_dir(script_home)
    return script_home, default_script_home

def setup_hdf5_installation(shell, arch_home):
    if "HDF5_DIR" in os.environ:
        hdf5_bin_dir = os.path.join(os.environ["HDF5_DIR"], "bin")
        if not shell.in_env("PATH", hdf5_bin_dir):
            # Add hdf5 bin dir to path for now to use h52gif for version
            shell.add_to_env("PATH", hdf5_bin_dir)
            sys.path.append(hdf5_bin_dir)

    hdf5_dir = prompt_install_hdf5(arch_home)
    if hdf5_dir:
        print >>sys.stderr, ("\nPyTables uses the environment variable"
                             " HDF5_DIR to locate HDF5.")
        prompt_set_env(shell, "HDF5_DIR", hdf5_dir)
        bin_path = os.path.join(hdf5_dir, "bin")
        include_path = os.path.join(hdf5_dir, "include")
        lib_path = os.path.join(hdf5_dir, "lib")
        prompt_add_to_env(shell, "PATH", bin_path)
        prompt_add_to_env(shell, "C_INCLUDE_PATH", include_path)
        prompt_add_to_env(shell, "LIBRARY_PATH", lib_path)
        prompt_add_to_env(shell, "LD_LIBRARY_PATH", lib_path)
    return hdf5_dir

def prompt_add_to_env(shell, variable, value):
    if shell.in_env(variable, value):
        print >>sys.stderr, "\nFound %s already in your %s!" % \
            (value, variable)
    else:
        shell.add_to_env(variable, value)
        query = "\nMay I edit your %s to add %s to your %s?" % \
            (shell.file, value, variable)
        permission = prompt_yes_no(query)
        if permission:
            shell.add_to_rc_env(variable, value)

def prompt_set_env(shell, variable, value):
    if variable in os.environ and os.environ[variable] == value:
        print >>sys.stderr, "\nYour %s was already set to %s!" % \
            (variable, value)
    else:
        shell.set_env(variable, value)
        query = "\nMay I edit your %s to set your %s to %s?" % \
            (shell.file, variable, value)
        permission = prompt_yes_no(query)
        if permission:
            shell.set_rc_env(variable, value)

def fix_path(path):
    # Put path in standard form
    return os.path.abspath(os.path.expanduser(path))

def make_dir(dirname, verbose=True):
    """Make directory if it does not exist"""
    absdir = fix_path(dirname)
    if not os.path.isdir(absdir):
        os.makedirs(absdir)
        if verbose:
            print >>sys.stderr, "Created directory: %s" % dirname

def substitute_template(template, fields, safe=False, *args, **kwargs):
    if safe:
        return Template(template).safe_substitute(fields, *args, **kwargs)
    else:
        return Template(template).substitute(fields, *args, **kwargs)

def check_executable_in_path(bin):
    """Checks if an executable of the given name is in the user's path.

    If found, returns the path.
    If not found (and the user chooses to continue anyway, returns None)
    """
    path = find_executable(bin)
    if not path:
        print >>sys.stderr, "Required executable: %s not found in path." % bin
        query = "Would you like to continue the installation anyway?"
        continue_inst = prompt_yes_no(query)
        if not continue_inst:
            die("\n============ Installation Aborted ============")

    return path

def has_lsf():
    return "LSF_ENVDIR" in os.environ

def has_sge():
    return "SGE_ROOT" in os.environ

def can_find_library(libname):
    """Returns a boolean indicating if the given library could be found"""
    try:
        from ctypes import CDLL
        CDLL(libname)
        return True
    except OSError:
        return None

########################## CFG FILE ##########################
def prompt_create_cfg(arch_home, python_home, default_python_home,
                      script_home, default_script_home,
                      cfg_file="~/.pydistutils.cfg"):
    cfg_path = fix_path(cfg_file)
    if os.path.isfile(cfg_path):
        print >>sys.stderr, ("\nFound your %s! (hopefully the configuration"
                             " matches)") % cfg_file
    else:
        query = ("\nMay I create %s? It will be used by distutils"
                 " to install new Python modules into this directory"
                 " (and subdirectories) automatically.") % cfg_file
        permission = prompt_yes_no(query)
        if permission:
            write_pydistutils_cfg(cfg_path, arch_home,
                                  python_home, default_python_home,
                                  script_home, default_script_home)

def write_pydistutils_cfg(cfg_file, arch_home,
                          python_home, default_python_home,
                          script_home, default_script_home):
    """Write a pydistutils.cfg file
    """
    fields = {}
    fields["prefix"] = fix_path(arch_home)

    if python_home == default_python_home:
        platlib = "$platbase/lib/python$py_version_short"
    else:
        platlib = python_home
    fields["platlib"] = platlib

    if script_home == default_script_home:
        scripts = "$platbase/bin"
    else:
        scripts = script_home
    fields["scripts"] = scripts

    cfg_file_contents = substitute_template(CFG_FILE_TEMPLATE, fields)

    ofp = open(cfg_file, "w")
    try:
        print >>ofp, cfg_file_contents
    finally:
        ofp.close()

########################### GET VERSION ########################
def get_hdf5_version():
    """Returns HDF5 version as string or None if not found or installed

    Only works if h5repack is installed and in current user path
    """
    try:
        cmd = Popen(["h5repack", "-V"], stdout=PIPE, stderr=PIPE)
        res = cmd.stdout.readlines()[0].strip()
        if "Version" in res:
            # HDF5 Found!
            return res.split("Version ")[1]
        else:
            return None
    except (OSError, IndexError):
        return None

def get_numpy_version():
    """Returns Numpy version as a string or None if not found or installed
    """
    try:
        import numpy
        return numpy.__version__
    except (AttributeError, ImportError):
        return None

def get_setuptools_version():
    """Returns setuptools version as a string or None if not found or installed
    """
    try:
        import setuptools
        return setuptools.__version__
    except (AttributeError, ImportError):
        return None

def parse_download_url(url):
    """Returns a dict of URL components"""
    dir, filename = os.path.split(url)
    # Remove extensions
    filebase = filename
    if filebase.endswith(".gz"):
        filebase = filebase[:-3]
    if filebase.endswith(".tar"):
        filebase = filebase[:-4]
    if filebase.endswith(".tgz"):
        filebase = filebase[:-4]

    assert "-" in filebase
    filebase_tokens = filebase.split("-")
    version = filebase_tokens[-1]
    program = "-".join(filebase_tokens[:-1])
    return {"dirname": dir,
            "file": filename,
            "filebase": filebase,
            "version": version,
            "program": program}

def str2version(ver):  # string to version object
    if ver.startswith("$Revision:"):
        ver = ver.split()[1]  # Get revision number
    return LooseVersion(ver)

##################### SPECIFIC PROGRAM INSTALLERS ################
def prompt_install_hdf5(arch_home):
    return _installer("HDF5", install_hdf5, get_hdf5_version,
                      install_message=HDF5_INSTALL_MESSAGE, arch_home=arch_home,
                      url=HDF5_URL)

def prompt_install_numpy(min_version=MIN_NUMPY_VERSION):
    return _installer("Numpy", install_numpy, get_numpy_version,
                      min_version=min_version,
                      install_prompt=EASY_INSTALL_PROMPT)

def install_hdf5(arch_home, *args, **kwargs):
    hdf5_dir = prompt_install_path("HDF5", arch_home)
    make_dir(hdf5_dir)
    install_dir = install_script("HDF5", hdf5_dir, HDF5_INSTALL_SCRIPT,
                                 url=HDF5_URL)
    return install_dir

def install_numpy(min_version=MIN_NUMPY_VERSION, *args, **kwargs):
    # Unset LDFLAGS when installing numpy as kludgy solution to
    #   http://projects.scipy.org/numpy/ticket/182
    env_old = None
    if "LDFLAGS" in os.environ:
        env_old = os.environ["LDFLAGS"]
        del os.environ["LDFLAGS"]

    try:
        return easy_install("numpy", min_version=min_version)
    finally:
        if env_old is not None:
            # Make sure variable didn't return, and then replace variable
            assert "LDFLAGS" not in os.environ
            os.environ["LDFLAGS"] = env_old

#################### GENERIC INSTALL METHODS ###################
def _installer(progname, install_func, version_func=None,
               install_message=None, *args, **kwargs):
    """Program installation wrapper method.

    Input:
    progname: string name of program
    install_func: function to call with *args, **kwargs to install progname
    version_func: function to call with *args, **kwargs to find progname version
      If version_func is none, installation will be promped no matter what.
    install_message: helpful message to print if user chooses to install program

    Checks if program is installed in a succificent version,
    if not, prompts user and installs program.

    Returns result of installation if installation attempted,
    else returns False
    """
    found = _check_install(progname, version_func, *args, **kwargs)
    if not found:
        permission = prompt_install(progname, *args, **kwargs)
        if permission:
            if install_message: print str(install_message)
            success = _abort_skip_install(install_func, *args, **kwargs)
            if success:
                print >>sys.stderr, "%s successfully installed." % progname
            else:
                print >>sys.stderr, "%s not installed." % progname
            return success
    return False

def _abort_skip_install(func, *args, **kwargs):
    """Calls func with args. If it fails, prompts user to abort or skip.

    On success, returns what the function would.
    On failure and skip, returns None
    On failure and abort, program terminates
    """
    try:
        return func(*args, **kwargs)
    except Exception, e:
        e_str = str(e)
        if e_str:
            print >>sys.stderr, "Error: %s" % e_str  # print any error message

        query = ("\nWould you like to try to continue the installation"
                 " without this program?")
        default = "n"
        permission = prompt_yes_no(query, default)
        if permission:
            return None
        else:
            die("\n============== Installation aborted =================")

def _check_install(progname, version_func, min_version=None, *args, **kwargs):
    """Returns True if program found with at least min_version, False otherwise

    version_func should either:
    - be a function that, when called, returns the version of
      the installation as a tuple, or True if installed,
      or None if not found/unavailable.
    - or None, in which False will be returned immediately

    If version_func returns True, installation accepted regardless of
    min_version

    """
    if version_func is None:
        return False
    else:
        print >>sys.stdout, "\nSearching for %s..." % progname,
        sys.stdout.flush()
        version = version_func()

    if version is not None:
        print >>sys.stderr, "found!"
        if min_version is None or version is True:
            # "True" testing necessary to distinguish from tuple
            return True
        elif str2version(min_version) > str2version(version):
            print >>sys.stderr, ("Found version: %s. Version %s or above"
                                 " required.") % (version, min_version)
        else:
            return True

    else:
        print >>sys.stderr, "not found."

    return False

def easy_install(progname, min_version=None):
    """Easy-installs the given program (can specifiy minimum version).
    """
    if get_setuptools_version() == None:
        query = ("Unable to find setuptools. This is necessary to install"
                 " this program and many of its dependencies. May I"
                 " download and install %s?")
        permission = prompt_install("setuptools", install_prompt=query)
        if permission:
            try:
                urlretrieve(EZ_SETUP_URL)
                from ez_setup import use_setuptools
                use_setuptools(delay=0)
            except:
                print >>sys.stderr, ("Error occurred while trying to"
                                     " install setuptools")
                raise InstallationError()
        else:
            raise InstallationError()

    cmd = ["easy_install", progname.lower()]
    if min_version is not None:  # Add version requirement
        cmd.append('"%s>=%s"' % (progname, min_version))

    if os.path.isdir(progname):
        print >>sys.stderr, ("\nWarning: installation may fail because"
                             " there is a subdirectory named %s at your"
                             " current path.") % progname

    print >>sys.stderr, ">> %s" % "".join(cmd)
    code = call(cmd, stdout=None, stderr=None)

    if code != 0:
        print >>sys.stderr, "Error occured installing %s" % progname
        raise InstallationError()
    else:
        return True

def install_script(progname, prog_dir, script, safe=False, env=[], **kwargs):
    """Tries to install the specified program, given a script, url

    progname: string name of program being installed

    prog_dir: directory program is to be installed in

    env: environment variables to be set during script execution

    script: multi-line string, where each line is a command to run in the
    shell in order to install the program. Lines should be independent and
    should use local variables not defined in the same line.
    Variables in this script will be substituted with keywords in kwargs,
    as well as:
    - dir: the program installation directory
    - file: the downloaded file (if url specified)
    - filebase: the basename of the downloaded file (if url specified)
    - version: the downloaded file url (if url specified and in std form)
    - python: sys.executable (should be the python command used to call
      this program)
    - tmpdir: a temporary directory for downloading files to

    Returns installation directory if installation is successful
    (or True if unknown), and None otherwise.

    """
    # Set fields for template substitution
    fields = {}
    fields["dir"] = prog_dir
    fields["python"] = sys.executable
    fields["tmpdir"] = gettempdir()
    if "url" in kwargs:
        url_fields = parse_download_url(kwargs["url"])
        fields.update(url_fields)

    # Add in kwargs (overwriting if collision)
    fields.update(kwargs)

    # Make dir absolute (even if specified as kwarg)
    fields["dir"] = fix_path(fields["dir"])

    script = substitute_template(script, fields, safe=safe)

    # Setup shell
    shell = InteractiveShell(env=env)
    shell.run_block(script, verbose=True)

    return prog_dir

########################## PROGRAM TESTING ######################
def prompt_test_packages(python_home, *args, **kwargs):
    """Run each dependency's unit tests and if they fail, prompt reinstall

    XXX: implement this for more than pytables (but numpy always fails)

    """
    # Start by making sure everything is up to date loaded into sys.path
    if python_home:
        addsitedir(python_home)

    print >>sys.stderr, "\n"
    try:
        prompt_test_pytables(*args, **kwargs)
    except InstallationError:
        die("\n=========== Some tests failed! =============="
            "\nYour installation may be incomplete and might not work.")

def prompt_test_pytables(*args, **kwargs):
    query = ("May I test the PyTables installation? This should provide"
             " a reasonable test of the HDF5 and NumPy installations as well.")
    permission = prompt_yes_no(query)
    if permission:
        try:
            import tables
            tables.test()
            print >>sys.stderr, "Test seemed to have passed."
        except:
            print >>sys.stderr, ("There seems to be an error with the"
                                 " PyTables installation!")
            raise InstallationError()

########################## USER INTERACTION #######################
def prompt_install(progname, install_prompt=None, url=None,
                   version=None, default="Y", *args, **kwargs):
    if version is not None:
        info = "%s %s" % (progname, version)
    elif url is not None:
        url_info = parse_download_url(url)
        info = "%s %s" % (progname, url_info["version"])
    else:
        info = "%s" % progname

    if install_prompt is None:
        install_prompt = "May I download and install %s?"

    query = install_prompt % info
    return prompt_yes_no(query, default=default)

def prompt_path(query, default):
    path = prompt_user(query, default)
    return fix_path(path)

def prompt_install_path(progname, default):
    query = "Where should %s be installed?" % progname
    path = prompt_user(query, default)
    return fix_path(path)

def prompt_yes_no(query, default="Y"):
    """Prompt user with query, given default and return boolean response

    Returns True if the user responds y/Y/Yes, and False if n/N/No
    """
    # Loop until we get a valid response
    while True:
        # Query user and get response
        print >>sys.stderr, "%s (Y/n) [%s] " % (query, default),
        sys.stdin.flush()
        response = raw_input().strip().lower()
        if len(response) == 0:
            response = default.strip().lower()

        if response.startswith("y"):
            return True
        elif response.startswith("n"):
            return False
        else:
            print >>sys.stderr, "Please enter yes or no."

def prompt_user(query, default=None, choices=None):
    """Prompt user with query, given default answer and optional choices."""

    if choices is None:
        prompt = str(query)
    else:
        try:
            # Uniquify and convert to strings
            str_choices = list(str(choice) for choice in set(choices))
            assert(len(str_choices) > 1)
            lower_choices = list(choice.lower() for choice in str_choices)
            prompt = "%s (%s)" % (query, " / ".join(choices))
        except (AssertionError, TypeError):
            die("Invalid choice list: %s" % choices)

    # Loop until we get a valid response
    while True:
        # Query user and get response
        if default is None:
            msg = str(prompt)
        else:
            msg = "%s [%s] " % (prompt, default)

        print >>sys.stderr, msg,
        sys.stdin.flush()
        response = raw_input().strip()

        if len(response) == 0:  # User didn't enter a response
            if default is None:
                print >>sys.stderr, "Response required."
            else:
                return default
        elif choices is None:
            return response
        else:
            # Ensure the user picked from the set of choices
            matches = []
            for choice in lower_choices:
                if choice.startswith(response) or response.startswith(choice):
                    matches.append(choice)

            matched = len(matches)
            if matched == 0:
                print >>sys.stderr, "Invalid answer: %s" % response
                print >>sys.stderr, "Please select one of: (%s)" % \
                    ",".join(str_choices)
            elif matched == 1:
                return matches[0]
            else:
                print >>sys.stderr, ("Response matched multiple choices."
                                     " Please be more specific.")

def die(message):
    print >>sys.stderr, str(message)
    sys.exit(1)

####################### END COMMON CODE BODY #####################


############################## MAIN #########################
def main(args=sys.argv[1:]):
    # Set up shell details
    try:
        shell_name = os.path.basename(os.environ["SHELL"])
    except KeyError:
        shell_name = None
    shell = ShellManager(shell_name)

    try:
        arch_home = setup_arch_home()
        prompt_set_env(shell, "ARCHHOME", fix_path(arch_home))

        python_home, default_python_home = setup_python_home(arch_home)
        # Add python_home to PYTHONPATH
        prompt_add_to_env(shell, "PYTHONPATH", python_home)

        script_home, default_script_home = setup_script_home(arch_home)
        # Add script_home to PATH
        prompt_add_to_env(shell, "PATH", script_home)

        # Maybe create pydistutils.cfg
        prompt_create_cfg(arch_home, python_home, default_python_home,
                          script_home, default_script_home)

        setup_hdf5_installation(shell, arch_home)

        prompt_install_numpy()

        prompt_install_gmtk(arch_home)
        #check_executable_in_path("gmtkViterbi")

        setup_drmaa_installation(shell, arch_home)

        prompt_install_segway()

        # Test package installations
        prompt_test_packages(python_home)

        print >>sys.stderr, "\n============ Installation complete! ==========="

    finally:  # Clean up
        shell.close()

########################### GET VERSION ########################
def get_segway_version():
    """Returns segway version as a string or None if not found or installed

    Temporarily removes '.' from sys.path during installation to prevent
    finding segway in current directory (but uninstalled)

    Since segway __version__ is currently a revision number, get the full
    number from pkg_resources
    """
    dir = os.getcwd()
    index = None
    if dir in sys.path:
        index = sys.path.index(dir)
        del sys.path[index]

    try:
        try:
            ref = pkg_resources.Requirement.parse("segway")
            data = pkg_resources.working_set.find(ref)
            return data.version
        except (AttributeError, ImportError):
            return None
    finally:
        if index is not None:
            sys.path.insert(index, dir)

def is_lsf_drmaa_installed():
    """Returns True if library found, None otherwise."""
    return can_find_library("libdrmaa.so")

def get_drmaa_version():
    """Returns drmaa-python version as a string or None if not found or
    installed
    """
    try:
        import drmaa
        return drmaa.__version__
    except (AttributeError, ImportError):
        return None

def get_gmtk_version():
    """Returns the version in the gmtk version file or None if not found"""
    if "ARCHHOME" in os.environ:
        gmtkdir = fix_path(os.environ["ARCHHOME"])
    else:
        cmd = Popen(["which", "gmtkViterbi"], stdout=PIPE, stderr=PIPE)
        if cmd.poll() == 0:
            stdout = cmd.communicate()[0].strip()
            bindir = stdout[0]
            gmtkdir = os.path.dirname(os.path.dirname(bindir))
        else:
            return None

    version_filename = os.path.join(gmtkdir, "etc", GMTK_VERSION_FILENAME)
    if os.path.isfile(version_filename):
        ifp = open(version_filename)
        version = ifp.readline().strip()
        if len(version) > 0:
            return version

    return None

##################### SPECIFIC PROGRAM INSTALLERS ################
def setup_drmaa_installation(shell, arch_home):
    return _abort_skip_install(prompt_drmaa_installation, shell, arch_home)

def prompt_drmaa_installation(shell, arch_home):
    print >>sys.stderr, "\nSearching for LSF or SGE...",
    lsf_found = has_lsf()
    sge_found = has_sge()
    if not (lsf_found or sge_found):
        print >>sys.stderr, "not found."
        raise InstallationError("Segway can only be run where there is \
a cluster management system.\nI was unable to find either an LSF or SGE \
system.\nPlease try reinstalling on a system with one of these installed.")

    print >>sys.stderr, "found!"
    if lsf_found and not sge_found:
        # Need to download and install FedStage lsf-drmaa
        lsf_drmaa_dir = prompt_install_lsf_drmaa(arch_home)
        if lsf_drmaa_dir:
            prompt_add_to_env(shell, "DRMAA_LIBRARY_PATH", lsf_drmaa_dir)

    return prompt_install_drmaa()

def prompt_install_lsf_drmaa(arch_home):
    return _installer("FedStage DRMAA for LSF", install_lsf_drmaa,
                      is_lsf_drmaa_installed, url=LSF_DRMAA_URL,
                      arch_home=arch_home)

def prompt_install_drmaa():
    return _installer("drmaa-python", install_drmaa, get_drmaa_version,
                      install_prompt=EASY_INSTALL_PROMPT)

def prompt_install_gmtk(arch_home):
    return _installer("gmtk", install_gmtk, get_gmtk_version,
                      arch_home=arch_home)

def prompt_install_segway():
    return _installer("segway", install_segway, get_segway_version,
                      install_prompt=EASY_INSTALL_PROMPT)

def install_lsf_drmaa(arch_home, *args, **kwargs):
    progname = "FedStage DRMAA for LSF"
    drmaa_dir = prompt_install_path(progname, arch_home)
    install_dir = install_script(progname, drmaa_dir,
                                 LSF_DRMAA_INSTALL_SCRIPT,
                                 url=LSF_DRMAA_URL)
    return install_dir

def install_drmaa(min_version=MIN_DRMAA_VERSION, *args, **kwargs):
    return easy_install("drmaa", min_version=min_version)

def install_gmtk(arch_home, *args, **kwargs):
    query = "\nALERT: GMTK source code is password protected.\
\n[Username: %s] Password: " % GMTK_USER
    password = prompt_user(query)
    optflags = get_gmtk_optflags()
    env = {"OPTFLAGS": optflags}
    return install_script("GMTK", arch_home, GMTK_INSTALL_SCRIPT,
                          url=GMTK_URL, user=GMTK_USER, password=password,
                          env=env, safe=True)

def install_segway(*args, **kwargs):
    query = "\nALERT: Segway source code is password protected.\
\n[Username: %s] Password: " % SEGWAY_USER
    password = prompt_user(query)
    return install_script("segway", ".", SEGWAY_INSTALL_SCRIPT,
                          url=SEGWAY_URL, user=SEGWAY_USER, password=password)

######################### OTHER FUNCTIONS ########################
def get_gmtk_optflags():
    if "OPTFLAGS" in os.environ:
        return os.environ["OPTFLAGS"]

    if "ARCH" in os.environ:
        arch = os.environ["ARCH"]
    else:
        arch = "-".join([platform.system(), platform.machine()])

    arch_gcc = None
    sselevel = None
    if arch in ["Linux-i386", "Linux-x86_64"] or arch.startswith("CYGWIN"):
        # Determine SSELEVEL
        sselevel = 2  # default
        cmd = Popen('grep "^flags" /proc/cpuinfo | cut -d : -f 2',
                    shell=True, stdout=PIPE)
        stdout, stderr = cmd.communicate()
        for line in stdout.split("\n"):
            tokens = line.strip().split()
            if "bpni" in tokens:
                sselevel = 3

        # Determine ARCH_GCC
        cmd = Popen('grep "^model name" /proc/cpuinfo | cut -d : -f 2',
                    shell=True, stdout=PIPE)
        cpu_model, stderr = cmd.communicate()
        if "Opteron" in cpu_model:
            arch_gcc = "opteron"
        elif "Pentium" in cpu_model or "Xeon" in cpu_model:
            if arch == "Linux-i386" or arch.startswith("CYGWIN"):
                if sselevel == 3:
                    arch_gcc = "prescott"
                else:
                    arch_gcc = "pentium4"
            elif arch == "Linux-x86_64":
                arch_gcc = "nocona"

    optflags = "-g -O3 -D_TABLE"
    if arch_gcc:
        optflags += " -march=%s" % arch_gcc
    if sselevel:
        optflags += " -mfpmath=sse -msse%d" % sselevel

    return optflags

if __name__ == "__main__":
    sys.exit(main())
