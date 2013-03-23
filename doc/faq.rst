============================================
Frequently Asked Questions
============================================

What do these memory progression errors mean?
---------------------------------------------

See :doc:`troubleshooting` and :ref:`task-output`.

How do I make segments longer?
------------------------------

There are several ways to do this:

- Soft weight constraints:

  - :option:`--segtransition-weight-scale` will increase the strength
    of the soft length prior for longer segments

- Hard weight constraints

  - The :option:`--seg-table` option will allow you to specify a hard
    minimum segment length, as described :ref:`here
    <hard-length-constraints>`.
  - Downsampling resolution

See :ref:`segment-duration-model` for model related methods.

How can I make Segway go faster?
--------------------------------

- Train on smaller portion of the genome. Use the
  :option:`--include-coords` option and supply a BED file.
- Splitting up into smaller subsequences by reducing --split-sequences
  can also help.
- Decrease Resolution (Not currently implemented segway)

