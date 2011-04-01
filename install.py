#!/usr/bin/env python
"""
General installation script for the segway package. Interacts with
the user to configure the environment to download and install all
dependencies and the segway package.

This script is designed to be downloaded and run independently, and
will guide the process of downloading and installing all other source
code.

(c) 2009: Orion Buske <stasis {at} uw {dot} edu>

"""
import platform

PKG_VERSION = "1.0.0"
GMTK_VERSION = "20091016"

####################### BEGIN COMMON CODE HEADER #####################

import os
import sys

from distutils.version import LooseVersion
from shutil import rmtree
from site import addsitedir
from string import Template
from subprocess import call, PIPE, Popen
from tempfile import mkdtemp
from urllib import urlretrieve

assert sys.version_info >= (2, 4)

MIN_HDF5_VERSION = "1.8"
MIN_NUMPY_VERSION = "1.2"
PYTABLES_VERSION = ">2.0.4,<2.2a0"

HDF5_URL = "ftp://ftp.hdfgroup.org/HDF5/prev-releases/hdf5-1.8.4-patch1/" \
    "src/hdf5-1.8.4-patch1.tar.gz"
PYTABLES_LINKS = ["http://www.pytables.org/download/pytables-2.1.2/"]
EZ_SETUP_URL = "http://peak.telecommunity.com/dist/ez_setup.py"

# Template for cfg file contents
CFG_FILE_TEMPLATE = """
[install]
prefix = $prefix
install-lib = $platlib
install-scripts = $scripts

[easy_install]
prefix = $prefix
install-dir = $platlib
script-dir = $scripts
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
"""

####################### END COMMON CODE HEADER #####################


MIN_HDF5_VERSION = "1.8"
MIN_NUMPY_VERSION = "1.2"
MIN_DRMAA_VERSION = "0.4a3"

LSF_DRMAA_URL = "http://downloads.sourceforge.net/project/lsf-drmaa/lsf_drmaa/1.0.4/lsf_drmaa-1.0.4.tar.gz"

GMTK_USER = "segway"
GMTK_VERSION_FILENAME = "gmtk.version"
GMTK_URL = "http://noble.gs.washington.edu/proj/segway/gmtk/" \
    "gmtk-%s.tar.gz" % GMTK_VERSION

SEGWAY_USER = "segway"
SEGWAY_URL = "http://noble.gs.washington.edu/proj/segway/src/" \
    "segway-%s.tar.gz" % PKG_VERSION

# also has keywords: user, password, OPTFLAGS, ARCH_GCC, SSELEVEL
GMTK_INSTALL_SCRIPT = """
cd $tmpdir
wget --user=$user --password=$password $url -O $file
if [[ ! -d $filebase ]]; then tar -xzf $file; fi
cd $filebase

mkdir -p "$dir/bin"
mkdir tksrc/bin 2>/dev/null || true

if [[ "$(find . -name "*.a" -print0)" ]]; \
then find . -name "*.a" -print0 | xargs -0 rm; \
fi

make clean "OPTFLAGS=$OPTFLAGS"
make linux "OPTFLAGS=$OPTFLAGS"
make depend "OPTFLAGS=$OPTFLAGS"

make "OPTFLAGS=$OPTFLAGS"
make -C featureFileIO install "OPTFLAGS=$OPTFLAGS" \
    "INSTALL=install" "install_prefix=$dir"
make -C tksrc install "OPTFLAGS=$OPTFLAGS" "install_prefix=$dir"

mkdir -p "$dir/etc"
declare -p OPTFLAGS > "$dir/etc/gmtk.build-options"
echo "$version" > "$dir/etc/gmtk.version"
"""

LSF_DRMAA_INSTALL_SCRIPT = """
cd $tmpdir
wget $url -O $file
if [[ ! -d $filebase ]]; then tar -xzf $file; fi
cd $filebase
./configure --prefix=$dir
make
make install
"""

SEGWAY_INSTALL_SCRIPT = """
cd $tmpdir
wget --user=$user --password=$password $url -O $file
if [[ ! -d $filebase ]]; then tar -xzf $file; fi
cd $filebase
$python setup.py install
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

    def execute(self, command, shell="/bin/bash", verbose=False):
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
            return call(str(command), executable=shell, shell=True,
                        env=self.env, cwd=os.getcwd())

    def run_script(self, script, verbose=False):
        """Runs each string command in a list, stopping if an error occurs

        verbose: if true, each command is first echo'd to stderr

        Calls to `cd` are trapped so cwd state persists between commands,
        however, the cwd state before the script is restored after.

        """
        try:
            dir = os.getcwd()
        except OSError:
            dir = None

        try:
            for line in script:
                if len(line) == 0:
                    continue

                retcode = self.execute(line, verbose=verbose)
                if retcode != 0:
                    raise OSError("Command failed: %s" % line)
        finally:
            if dir:
                os.chdir(dir)

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
    name: The string name of the shell (e.g. "bash", "csh", etc.)
      or "" if unknown
    """
    def __init__(self, shell=None):
        """Creates a ShellManager given a shell string.

        If shell is None, tries to use environment variable SHELL.
        """
        if shell is None:
            try:
                shell = os.path.basename(os.environ["SHELL"])
            except KeyError:
                shell = ""


        if shell.endswith("csh"):
            env_format = "setenv %s %s"
        elif shell.endswith("sh"):
            env_format = "export %s=%s"
        else:
            env_format = "set %s to %s"  # Console output

        # What RC file are we using, or are we printing to the terminal?
        if shell.endswith("sh"):
            file = os.path.join("~", ".%src" % shell)
        else:
            file = None  # Print to terminal
            if shell:
                print >>sys.stderr, "Unrecognized shell: %s" % shell
            else:
                print >>sys.stderr, "SHELL variable missing or confusing."

        if file is None:
            print >>sys.stderr, ("Shell-specific commands will be printed"
                                 " to the terminal")
        else:
            prompt = """
In the course of this installation, a number of environment variables
must be permanently set so the installed programs will work. Your shell
was found to be: %s (from the value of $SHELL), so these settings
should probably be saved in: %s.

May I save settings to this file (if not, required settings will just
be printed to the terminal)?""" % (shell, file)
            permission = prompt_yes_no(prompt)
            if not permission:
                print >>sys.stderr, "Okay, but this is not recommended."
                file = None

        self.shell = shell
        self.file = file
        self._env_format = env_format
        self._out = None  # Output file not yet open
        self.file_open = False


    def write_var(self, variable, value):
        """Write the given variable and value to the shell rc file"""
        if self._out is None:
            if self.file is None:
                self._out = sys.stdout
            else:
                self._out = open(os.path.expanduser(self.file), "a")
                self.file_open = True
                self._out.write("\n### Below added by install script ###\n")

        cmd = self._env_format % (variable, value)
        self._out.write("%s\n" % cmd)

    def save_var(self, variable, value):
        """Write the given variable and value to the shell rc file"""
        if self.is_var(variable, value):
            print >>sys.stderr, "\nYour %s is already %s!" % (variable, value)
        else:
            self.write_var(variable, value)
            self.set_var(variable, value)

    def save_to_var(self, variable, value):
        """Prepend the value to the variable in the shell rc file"""
        if self.in_var(variable, value):
            print >>sys.stderr, "\nYour %s already includes %s!" % \
                (variable, value)
        else:
            full_value = "%s:$%s" % (value, variable)
            self.write_var(variable, full_value)
            self.add_to_var(variable, value)

    def set_var(self, variable, value):
        """Set the environment variable to the given value"""
        os.environ[variable] = value

    def add_to_var(self, variable, value):
        """Prepend value to the value of the environment variable."""
        if variable in os.environ:
            value = "%s:%s" % (value, os.environ[variable])

        self.set_var(variable, value)

    def in_var(self, variable, value):
        """Checks if value is in a list-based environment variable.

        Returns true if variable found and value is one of the ':'-
        delimited values in it.

        """
        try:
            env = os.environ[variable]
            env_values = env.strip().split(":")
            return value in env_values
        except KeyError:
            return False

    def is_var(self, variable, value):
        """Returns True if environment variable has given value."""
        try:
            return os.environ[variable] == value
        except KeyError:
            return False

    def close(self):
        """Close the open rc file, if one exists
        """
        if self.file_open:
            self._out.write("### Above added by install script ###\n")
            self._out.close()
            self._out = None
            self.file_open = False


class Environment(object):
    def __init__(self):
        self.shell = ShellManager()

        self.setup_arch_home()
        self.setup_python_home()
        self.setup_script_home()
        self.setup_cfg()

    def close(self):
        lines = ["INSTALLATION COMPLETE"]
        if self.shell.file:
            lines.append("Source your %s to update your environment"
                         % self.shell.file)

        print >>sys.stderr, "\n%s\n" % string_sign(lines)
        self.shell.close()

    def check_spaces(self, path):
        if " " in path:
            print >>sys.stderr, ("Warning: spaces in paths are not recommended"
                                 " as they are not supported by some"
                                 " dependencies")

    def refresh_packages(self):
        """Refresh list of packages/eggs that can be imported"""
        print >>sys.stderr, "Updating list of packages/eggs in %s" % \
            self.python_home
        addsitedir(fix_path(self.python_home))

    def has_lsf(self):
        return "LSF_ENVDIR" in os.environ

    def has_sge(self):
        return "SGE_ROOT" in os.environ

    ##### HOME SETUP ######
    def initialize(self):
        """Default initialization (arch, python, script homes; cfg file)"""

    def setup_arch_home(self, default=None):
        if default is None:
            default = self.get_default_arch_home()

        query = "\nWhere should platform-specific files be installed?"
        self.arch_home = prompt_user(query, default)

        arch_home_path = fix_path(self.arch_home)
        self.check_spaces(arch_home_path)

        make_dir(arch_home_path)
        self.shell.save_var("ARCHHOME", arch_home_path)
        return arch_home_path

    def setup_python_home(self, default=None):
        if default is None:
            default = self.get_default_python_home()

        query = "\nWhere should new Python packages be installed?"
        self.python_home = prompt_user(query, default)

        python_home_path = fix_path(self.python_home)
        self.check_spaces(python_home_path)

        make_dir(python_home_path)
        addsitedir(python_home_path)  # Load already-installed packages/eggs
        self.shell.save_to_var("PYTHONPATH", python_home_path)
        return python_home_path

    def setup_script_home(self, default=None):
        if default is None:
            default = self.get_default_script_home()

        query = "\nWhere should new scripts and executables be installed?"
        self.script_home = prompt_user(query, default)

        script_home_path = fix_path(self.script_home)
        self.check_spaces(script_home_path)

        make_dir(script_home_path)
        self.shell.save_to_var("PATH", script_home_path)
        return script_home_path

    def get_default_arch_home(self, root="~"):
        if "ARCHHOME" in os.environ:
            return os.environ["ARCHHOME"]
        elif "ARCH" in os.environ:
            arch = os.environ["ARCH"]
        else:
            (sysname, nodename, release, version, machine) = os.uname()
            arch = "-".join([sysname, machine])

        arch = os.path.expanduser("%s/arch/%s" % (root, arch))
        arch = arch.replace(" ", "_")  # Spaces cause issues
        return arch

    def get_default_python_home(self, root=None):
        if root is None:
            root = self.arch_home

        dir = os.path.join(root, "lib", "python%s" % sys.version[:3])
        # If there is a python installation here,
        # use the site-packages subdirectory instead
        alternate_dir = os.path.join(dir, "site-packages")
        if os.path.samefile(sys.prefix, fix_path(root)) or \
                os.path.isdir(fix_path(alternate_dir)):
            return alternate_dir
        else:
            return dir

    def get_default_script_home(self, root=None):
        if root is None:
            root = self.arch_home
        return os.path.join(root, "bin")

    ##### CFG FILE #####
    def setup_cfg(self, cfg_file="~/.pydistutils.cfg"):
        """Prompt user whether or not to create a cfg file"""
        cfg_path = fix_path(cfg_file)
        if os.path.isfile(cfg_path):
            print >>sys.stderr, ("\nFound your %s! (hopefully the configuration"
                                 " matches)" % cfg_file)
        else:
            query = """
May I create %s?
It will be used by distutils to install new Python modules
into this directory (and subdirectories) automatically.""" % cfg_file
            permission = prompt_yes_no(query)
            if permission:
                self._write_pydistutils_cfg(cfg_path)

    def _write_pydistutils_cfg(self, cfg_path):
        """Write a pydistutils.cfg file based upon homes set up"""
        arch_home = fix_path(self.arch_home)
        python_home = fix_path(self.python_home)
        script_home = fix_path(self.script_home)

        fields = {}
        fields["prefix"] = arch_home

        platlib = python_home
        fields["platlib"] = platlib

        scripts = script_home
        fields["scripts"] = scripts

        cfg_file_contents = substitute_template(CFG_FILE_TEMPLATE, fields)

        ofp = open(cfg_path, "w")
        try:
            ofp.write(cfg_file_contents)
        finally:
            ofp.close()


class Installer(object):
    """Base class for creating program installers

    Attributes:
      name: Name of program
      min_version: None if not relevant
      install_prompt: should contain one %s for the name of the program
      url: None if not relevant

    run(*args, **kwargs): runs the installer, calling (potentially overridden)
        methods in the following order:
      start_install(): returns True or False
        if True, installation continues
        if False, installation halts and returns False
      check_version(): return True or False
        if True
          installation stops and returns True
        else
          installation continues
        calls get_version() by default
      get_version(): returns True, False, None, or version information
        if True, program is considered installed
        if False or None, program is not considered installed
        else, version information should be supplied as a string or tuple
      prompt_install(): same as start_install()
        calls get_install_version() by default
      announce_install(): opportunity to print any final message before install
      install(*args, **kwargs): called with arguments passed to run()
        if error occurs, cleanup(False) is called and False is returned
        else, cleanup(True) is called and True is returned

    """

    min_version = None
    install_prompt = "May I download and install %s?"
    url = None

    def start_install(self):
        return True

    def prompt_install_path(self, default):
        query = "Where should %s be installed?" % self.name
        path = prompt_user(query, default)
        return fix_path(path)

    def prompt_install(self):
        assert "%s" in self.install_prompt

        version = self.get_install_version()
        if version:
            info = "%s %s" % (self.name, version)
        else:
            info = self.name

        query = self.install_prompt % info
        return prompt_yes_no(query, default="Y")

    def parse_url(self):
        """Returns a dict of URL components"""
        components = {}
        if not self.url:
            return components

        dir, filename = os.path.split(self.url)

        # Remove extensions
        filebase = filename
        if filebase.endswith(".gz"):
            filebase = filebase[:-3]
        if filebase.endswith(".tar"):
            filebase = filebase[:-4]
        if filebase.endswith(".tgz"):
            filebase = filebase[:-4]

        filebase_tokens = filebase.split("-", 1)
        # Try to extract version from filename
        # Let version contain '-'s, but not progname
        #(Assumes form: <progname>-<version>.<ext>)
        if "-" in filebase:
            components["version"] = "-".join(filebase_tokens[1:])
            components["program"] = filebase_tokens[0]
        else:
            components["program"] = filebase

        components["dirname"] = dir
        components["file"] = filename
        components["filebase"] = filebase
        return components

    def get_install_version(self):
        """Returns the version being installed"""
        if self.url is None:
            return self.min_version
        else:
            url_info = self.parse_url()
            return url_info.get("version", None)

    def check_version(self):
        """Checks version and returns True if version is adequate"""
        print >>sys.stderr, ("\nSearching for an installation of %s..."
                             % self.name),

        version = self.get_version()
        if not version:
            print >>sys.stderr, "not found."
            return False
        else:
            print >>sys.stderr, "found."
            if version is True or self.min_version is None:
                # "True" testing necessary to distinguish from tuple or string
                return True
            elif str2version(self.min_version) > str2version(version):
                print >>sys.stderr, ("Found version: %s. Version %s or above"
                                     " required.") % (version,
                                                      self.min_version)
            else:
                return True

    def announce_install(self):
        print >>sys.stderr, "\n===== Installing %s =====" % self.name

    def cleanup(self, success):
        if success:
            print >>sys.stderr, "%s successfully installed." % self.name
        else:
            print >>sys.stderr, "%s not installed." % self.name
            query = ("\nWould you like to try to continue the installation"
                     " without this program?")
            permission = prompt_yes_no(query, default="n")
            if not permission:
                die("\n============== Installation aborted =================")

    def run(self, *args, **kwargs):
        """Program installation wrapper method.

        Checks if program is installed in a succificent version,
        if not, prompts user and installs program.

        Returns result of installation if installation attempted,
        else returns False

        """
        permission = self.start_install()
        if not permission:
            return False

        if self.check_version():
            return True

        permission = self.prompt_install()
        if not permission:
            return False

        self.announce_install()

        success = True
        try:
            self.install(*args, **kwargs)
        except Exception, e:
            success = False
            if str(e):  # print any error message
                print >>sys.stderr, "===== ERROR: %s =====" % e

        self.cleanup(success)

        return success


class EasyInstaller(Installer):
    """An installer that uses easy_install

    New attributes:
      version_requirment: None or a string that specifies version requirement
        for easy_install. These requirements will be appended to the default
        requirement of '>=min_version'. Thus, '!=1.2' would be a reasonable
        value.
      pkg_name: Name of package to easy_install. Defaults to self.name.lower().
      links: List of URLs to also search for package source. Uses
        easy_install's '-f' option.

    """
    install_prompt = "May I install %s (or later) and dependencies?"
    version_requirement = None
    pkg_name = None  # Replaced by self.name.lower() in __init__
    links = []

    def __init__(self):
        if self.pkg_name is None:
            self.pkg_name = self.name.lower()

    def install(self):
        """Easy-installs the program

        uses:
          self.name
          self.pkg_name
          self.get_install_version()
          self.version_requirement
          self.links

        if self.url is set, easy_install is run on that url instead of the
          program name and version requirement

        """
        # Make sure easy_install (setuptools) is installed
        try:
            import setuptools
        except ImportError:
            raise InstallationError("Setuptools necessary for easy_install")

        version = self.get_install_version()

        cmd = ["easy_install"]

        if self.links:
            for link in self.links:
                cmd.append("--find-links=%s" % link)

        if self.url is None:
            requirements = []
            if version:
                requirements.append(">=%s" % version)
            if self.version_requirement:
                requirements.append(self.version_requirement)

            cmd.append("%s%s" % (self.pkg_name, ",".join(requirements)))
        else:
            cmd.append(self.url)

        if os.path.isdir(self.pkg_name):
            print >>sys.stderr, ("\nWarning: installation may fail because"
                                 " there is a subdirectory named %s at your"
                                 " current path.") % self.pkg_name

        print >>sys.stderr, ">> %s" % " ".join(cmd)
        code = call(cmd, stdout=None, stderr=None)

        if code != 0:
            print >>sys.stderr, "Error occured installing %s" % self.name
            raise InstallationError()

    def get_version(self):
        """Return the package version, assuming normal Python conventions"""
        try:
            return __import__(self.pkg_name).__version__
        except (AttributeError, ImportError):
            return None

    def get_egg_version(self):
        """Generic version finder for installed eggs

        Temporarily removed '.' from sys.path to prevent getting confused by
        a version of the module in the current directory (but uninstalled).

        Useful for if the <module>.__version__ is a revision number
        or does not exist.

        XXX: Doesn't always get the *latest* version.

        """
        try:
            dir = os.getcwd()
        except OSError:
            dir = None

        index = None
        if dir and dir in sys.path:
            index = sys.path.index(dir)
            del sys.path[index]

        try:
            try:
                import pkg_resources
                ref = pkg_resources.Requirement.parse(self.pkg_name)
                return pkg_resources.working_set.find(ref).version
            except (AttributeError, ImportError):
                return None
        finally:
            if index is not None:
                sys.path.insert(index, dir)


class ScriptInstaller(Installer):
    def script_install(self, script=None, dir=os.getcwd(), safe=False,
                       env=[], **kwargs):
        """Tries to install the specified program, given a script, url

        dir: the directory the script will be run from

        env: environment variables to be set during script execution

        safe: should safe_substitution or regular substitution be used
              (safe_substitution allows $vars to not be specified)

        script: multi-line string, where each line is a command to run in the
          shell in order to install the program. Lines should be independent
          and should use local variables not defined in the same line.
          Defaults to self.install_script
          Variables in this script will be substituted with keywords in kwargs,
          as well as:
          - dir: the program installation directory
          - file: the downloaded file (if url specified)
          - filebase: the basename of the downloaded file (if url specified)
          - version: the downloaded file url (if url specified and in std form)
          - python: sys.executable (should be the python command used to call
            this program)
          - tmpdir: a temporary directory for the execution of the script
            (it will be deleted upon completion, unless tmpdir was specified
             in kwargs directly)

        """
        if script is None:
            script = self.install_script

        # Set fields for template substitution
        fields = {}
        fields["dir"] = dir
        fields["python"] = sys.executable

        if self.url:
            fields["url"] = self.url
            url_fields = self.parse_url()
            fields.update(url_fields)

        # Add in kwargs (overwriting if collision)
        fields.update(kwargs)

        # Make dir absolute (even if specified as kwarg)
        fields["dir"] = fix_path(fields["dir"])

        createtempdir = ("tmpdir" not in fields)
        if createtempdir:
            # Make a tempdir if none was specified
            fields["tmpdir"] = mkdtemp()

        try:
            script = substitute_template(script, fields, safe=safe)

            # Setup shell
            shell = InteractiveShell(env=env)
            shell.run_block(script, verbose=True)
        finally:
            if createtempdir:
                rmtree(fields["tmpdir"])


class SetuptoolsInstaller(Installer):
    name = "setuptools"
    url = EZ_SETUP_URL
    install_prompt = """Unable to find setuptools. \
It is used to download and install many of this program's prerequisites \
and handle versioning. May I download and install %s?"""

    def get_version(self):
        try:
            return __import__(self.name.lower()).__version__
        except (AttributeError, ImportError):
            return None

    def download(self, save_dir=os.curdir):
        # Download ez_setup.py
        url_components = self.parse_url()
        filename = url_components["file"]
        save_path = fix_path(os.path.join(save_dir, filename))
        if not os.path.isfile(save_path):  # Avoid duplicate downloads
            urlretrieve(self.url, save_path)

        return save_path

    def install(self):
        old_sys_path = sys.path
        try:
            save_path = self.download()
            sys.path.insert(0, save_path)
            # Run ez_setup.py to install setuptools
            from ez_setup import main
            try:
                main([])
            except SystemExit, e:
                if e.code != 0:
                    raise InstallationError("Setuptools installation failed.")

            # Load previously-installed packages/eggs in new python dir
            addsitedir(os.path.dirname(save_path))
        finally:
            sys.path = old_sys_path
            if save_path and os.path.exists(save_path):
                os.unlink(save_path)
            pyc_path = "%sc" % save_path
            if os.path.exists(pyc_path):
                os.unlink(pyc_path)

class Hdf5Installer(ScriptInstaller):
    name = "HDF5"
    min_version = MIN_HDF5_VERSION
    url = HDF5_URL
    install_script = HDF5_INSTALL_SCRIPT

    def __init__(self, env):
        self.env = env
        super(self.__class__, self).__init__()

    def start_install(self):
        if "HDF5_DIR" in os.environ:
            hdf5_bin_dir = os.path.join(os.environ["HDF5_DIR"], "bin")
            if not self.env.shell.in_var("PATH", hdf5_bin_dir):
                # Add hdf5 bin dir to path for now to use h5repack for version
                self.env.shell.add_to_var("PATH", hdf5_bin_dir)
                sys.path.insert(0, hdf5_bin_dir)

        return True

    def announce_install(self):
        lines = ["HDF5 is very large and installation"
                 " usually takes 5-10 minutes.",
                 "Please be patient.",
                 "It is common to see many warnings during compilation."]
        print >>sys.stderr, "\n%s\n" % string_sign(lines)

    def get_version(self):
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

    def install(self):
        hdf5_dir = self.prompt_install_path(self.env.arch_home)
        make_dir(hdf5_dir)
        self.script_install(dir=hdf5_dir)
        # Save dir for adding to path at cleanup
        self.hdf5_dir = hdf5_dir

    def cleanup(self, success):
        if success:
            if self.hdf5_dir:
                hdf5_dir = self.hdf5_dir
                print >>sys.stderr, ("\nPyTables uses the environment variable"
                                     " HDF5_DIR to locate HDF5.")
                self.env.shell.save_var("HDF5_DIR", hdf5_dir)
                bin_path = os.path.join(hdf5_dir, "bin")
                include_path = os.path.join(hdf5_dir, "include")
                lib_path = os.path.join(hdf5_dir, "lib")
                self.env.shell.save_to_var("PATH", bin_path)
                self.env.shell.save_to_var("C_INCLUDE_PATH", include_path)
                self.env.shell.save_to_var("LIBRARY_PATH", lib_path)
                self.env.shell.save_to_var("LD_LIBRARY_PATH", lib_path)
            else:
                die("Unknown error installing HDF5")

        super(self.__class__, self).cleanup(success)


class NumpyInstaller(EasyInstaller):
    name = "NumPy"
    min_version = MIN_NUMPY_VERSION

    def install(self):
        # Unset LDFLAGS when installing numpy as kludgy solution to
        #   http://projects.scipy.org/numpy/ticket/182
        env_old = None
        if "LDFLAGS" in os.environ:
            env_old = os.environ["LDFLAGS"]
            del os.environ["LDFLAGS"]

        try:
            return super(self.__class__, self).install()
        finally:
            if env_old is not None:
                # Make sure variable didn't return, and then replace variable
                assert "LDFLAGS" not in os.environ
                os.environ["LDFLAGS"] = env_old

class PytablesInstaller(EasyInstaller):
    name = "PyTables"
    pkg_name = "tables"
    get_version = EasyInstaller.get_egg_version
    links = PYTABLES_LINKS
    version_requirement = PYTABLES_VERSION


class Tester(object):
    """Skeleton for package tester

    The following fields and methods should be specified:
      name
      query
      test()

    """
    def prompt_test(self):
        permission = prompt_yes_no(self.query)
        if permission:
            try:
                self.test()
                print >>sys.stderr, "Test passed."
            except Exception, e:
                print >>sys.stderr, "Error: %r" % e
                print >>sys.stderr, ("There seems to be an error with the"
                                     " installation of %s!" % self.name)
                raise InstallationError()

class PytablesTester(Tester):
    name = "PyTables"
    query = """
May I test the PyTables installation? This should also provide
a reasonable test of the HDF5 and NumPy installations."""

    def test(self):
        import tables
        tables.test()

class TestSuite(object):
    """Run suite of Tester objects"""

    def run(self, testers):
        """Run each tester, dying if any fail (short-circuits)"""
        for tester in testers:
            if isinstance(tester, Tester):
                try:
                    tester.prompt_test()
                except InstallationError:
                    die("""
===== Test failed! =====
Your installation may be incomplete and might not work.""")

            else:
                raise TypeError("Expected instance of Tester class")

######################## UTIL FUNCTIONS ####################
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


def str2version(ver):  # string to version object
    # If setuptools installed, use its versioning; else, use distutils'
    try:
        import pkg_resources
        return pkg_resources.parse_version(ver)
    except (ImportError, NameError):
        return LooseVersion(ver)

def can_find_library(libname):
    """Returns a boolean indicating if the given library could be found"""
    try:
        from ctypes import CDLL
        CDLL(libname)
        return True
    except OSError:
        return None

def string_sign(lines, width=70):
    if isinstance(lines, basestring):
        lines = [lines]

    inner_width = width - 4  # 2 edge chars and 2 spaces

    header = "+%s+" % ("-" * (width - 2))
    sign = [header]
    for line in lines:
        sign.append("| %s |" % line.center(inner_width))

    sign.append(header)
    return "\n".join(sign)

########################## USER INTERACTION #######################
def prompt_path(query, default):
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
    message = ["INSTALLATION FAILED",
               "Source your ~/.*rc before retrying to avoid re-installing",
               "dependencies that were successfully installed"
               " during this run."]
    print >>sys.stderr, "\n%s\n" % string_sign(message)
    sys.exit(1)

####################### END COMMON CODE BODY #####################


class GmtkInstaller(ScriptInstaller):
    name = "GMTK"
    min_version = GMTK_VERSION
    url = GMTK_URL
    install_script = GMTK_INSTALL_SCRIPT

    def __init__(self, env):
        self.env = env
        super(self.__class__, self).__init__()

    def get_version_file(self, gmtkdir):
        return os.path.join(gmtkdir, "etc", GMTK_VERSION_FILENAME)

    def get_version(self):
        """Returns the version in the gmtk version file or None if not found"""
        version_file = self.get_version_file(self.env.arch_home)
        if not os.path.isfile(version_file):
            cmd = Popen(["which", "gmtkViterbi"], stdout=PIPE, stderr=PIPE)
            if cmd.poll() == 0:
                stdout = cmd.communicate()[0].strip()
                bindir = stdout[0]
                gmtkdir = os.path.dirname(os.path.dirname(bindir))
                version_file = self.get_version_file(gmtkdir)
            else:
                return None

        if version_file and os.path.isfile(version_file):
            ifp = open(version_file)
            try:
                version = ifp.readline().strip()
            finally:
                ifp.close()

            if len(version) > 0:
                return version

        return None

    def install(self):
        query = "\nALERT: GMTK source code is password protected.\
    \n[Username: %s] Password: " % GMTK_USER
        password = prompt_user(query)

        optflags = self.get_optflags()
        env = {"OPTFLAGS": optflags}
        self.script_install(dir=self.env.arch_home, safe=True, env=env,
                            user=GMTK_USER, password=password)

    def get_optflags(self):
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


class LsfDrmaaInstaller(ScriptInstaller):
    name = "FedStage DRMAA for LSF"
    url = LSF_DRMAA_URL
    install_script = LSF_DRMAA_INSTALL_SCRIPT

    def __init__(self, env):
        self.env = env
        super(self.__class__, self).__init__()

    def get_version(self):
        """Returns True if library found, None otherwise."""
        return can_find_library("libdrmaa.so")

    def install(self):
        drmaa_dir = self.prompt_install_path(self.env.arch_home)
        self.script_install(dir=drmaa_dir)
        self.drmaa_dir = drmaa_dir

    # XXX: save_to_var should be renamed save_var_path
    # but this is not a PATH in that sense
    def cleanup(self, success):
        if success and self.drmaa_dir:
            # Save environment variable for drmaa-python
            filename = os.path.join(self.drmaa_dir, "libdrmaa.so")
            self.env.shell.save_var("DRMAA_LIBRARY_PATH", filename)

        super(self.__class__, self).cleanup(success)


class DrmaaInstaller(EasyInstaller):
    name = "drmaa"
    min_version = MIN_DRMAA_VERSION
    get_version = EasyInstaller.get_egg_version

    def __init__(self, env):
        self.env = env
        super(self.__class__, self).__init__()

    def start_install(self):
        print >>sys.stderr, "\nSearching for LSF or SGE...",
        lsf_found = self.env.has_lsf()
        sge_found = self.env.has_sge()
        if not (lsf_found or sge_found):
            print >>sys.stderr, "not found."
            print >> sys.stderr, """
Segway can only be run where there is a cluster management system.
I was unable to find either an LSF or SGE system.
Please try reinstalling on a system with one of these installed.
"""
            return False

        print >>sys.stderr, "found!"
        if lsf_found and not sge_found:
            # Need to download and install FedStage lsf-drmaa
            installer = LsfDrmaaInstaller(self.env)
            success = installer.run()
            if not success:
                print >>sys.stderr, "%s is required for %s" % \
                    (installer.name, self.name)
                return False

        return True


class SegwayInstaller(ScriptInstaller, EasyInstaller):
    name = "Segway"
    min_version = PKG_VERSION
    url = SEGWAY_URL
    install_script = SEGWAY_INSTALL_SCRIPT
    get_version = EasyInstaller.get_egg_version

    def install(self):
        query = "\nALERT: Segway source code is password protected.\
\n[Username: %s] Password: " % SEGWAY_USER
        password = prompt_user(query)
        self.script_install(user=SEGWAY_USER, password=password)


############################## MAIN #########################
def main(args=sys.argv[1:]):
    env = Environment()
    env.initialize()

    installers = [SetuptoolsInstaller(),
                  Hdf5Installer(env),
                  NumpyInstaller(),
                  DrmaaInstaller(env),
                  GmtkInstaller(env),
                  PytablesInstaller(),
                  SegwayInstaller()]

    for installer in installers:
        installer.run()
        # Next step may need just-installed eggs, so update site list
        env.refresh_packages()

    # DONE: Test package installations?
    TestSuite().run([PytablesTester()])

    env.close()


if __name__ == "__main__":
    sys.exit(main())
