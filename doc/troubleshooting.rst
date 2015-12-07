===============
Troubleshooting
===============

When Segway reports an error, it usually means that dispatched GMTK jobs
failed somehow. Look in ``log/jobs.tab`` to see all the jobs and whether
they reported an error in the form of nonzero exit status in the last column.
If the job had a "75" exit status (or an "EX_TEMPFAIL") the job simply ran out
of memory. If this was the last job with this error, increase the allowed
memory in your ``-mem-usage`` option. Otherwise if the job had a non-zero exit
status and it wasn't due to out of memory issues, look in 
``output/e/``\ *instance*\ ``/``\ *jobname* to find the cause of the underlying 
error. See :ref:`task-output` for more output information.

Are your bundle jobs failing? This might be because an accumulator
file (written by individual job) is corrupted or truncated. This can
happen when you run out of disk space.

If it is not immediately apparent why a job is failing, it is probably
useful to look in ``log/details.sh`` to find the command line that Segway
uses to call GMTK. Try running that to see if it gives you any clues.
You may want to switch the GMTK option -verbosity 0 to -verbosity 30
to get more information.

An error like

  ERROR: discrete observed random variable 'presence_dnase', frame 0, line 23, specifies a feature element 14:14 that is out of discrete range ([23:45] inclusive) of observation matrix

probably indicates that you are incorrectly mixing and matching
``train.tab`` files and ``segway.str`` files from different training
runs.

If you are unable to reslove your issue on your own, consider inquiring on the mailing list <segway-users@uw.edu> listed on the :ref:`support` page.
