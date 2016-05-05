======================
Contributing to Segway
======================

To best way to help improve Segway is to report any issues in the `issue tracker`_ or submit contributions through `pull requests`_.

Reporting Issues
----------------

All issues should be placed in the `issue tracker`_. Use the search function first to check if your issue has been posted already. If an issue (or a similar issue) to yours has been posted please add any additional information you see might be necessary to resolve the problem.

We ask that when you post your issue that you include the following information:

1. System specifications (including what cluster system you may be submitting jobs to)
2. Your current version of Segway
3. *All* output from error that occurred
4. If possible, steps to reproduce


Submitting Contributions
------------------------

The best way to submit contributions to Segway is through the pull request feature on Bitbucket. To perform a pull request on Bitbucket, first fork (or sync your repo if you have already forked) your own copy of Segway on Bitbucket. Commit any set of changes to your own Segway repository on Bitbucket and then submit a pull request from the web interface. If the pull request is intended to address an existing issue, please add the issue number in your pull request message or subject along with a helpful description of the changes involved. We request that changes submitted have clear commit messages since these are often referred to when addressing bugs in the future. See `A Note About Commit Messages`_ for a quick overview of what would ideally be expected.

Coding Style
------------

We ask that all code submissions follow the `PEP 8`_ python coding guidelines and to only support Python 2.7 (and above if possible). Python 3 migration is a long term goal of Segway and by simply only supporting Python 2.7 the process should be far easier in the future.
In addition we recommend that you use the `flake8`_ tool to check your
contributions with the added option of "--max-complexity=10".

.. _issue tracker: https://bitbucket.org/hoffmanlab/segway/issues/
.. _pull requests: https://bitbucket.org/hoffmanlab/segway/pull-requests
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _A Note About Commit Messages: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
.. _flake8: https://pypi.python.org/pypi/flake8


============
Contributors
============

This list below contains a list of all contributors to Segway. The contributions
have been taken from our commit logs but if you feel you've been left out
please let us know!

Original Author
---------------
- Michael Hoffman ( michael.hoffman at utoronto.ca )

Additional Contributions
------------------------
- Orion Buske
- Paul Ellenbogen ( epaul9 )
- Jay Hesselberth ( jay.hesselberth at gmail.com )
- Max Libbrecht ( maxwl at cs.washington.edu )
- Eric Roberts ( eroberts at uhnresearch.ca )
- Adam Shaw ( ajshaw at uchicago.edu )
- Avinash Sahu ( avinash )
- Xing Zeng ( tamaki.sakura at hotmail.com )
