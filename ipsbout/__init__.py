"""
IPS-BOUT (`ipsbout`)

This package provides IPS components for running BOUT++/Hermes-3 simulations
and related pre/post-processing steps.

Key design choices (not always "standard" IPS patterns)
------------------------------------------------------

- **Services logging**: components should log via ``self.services.info(...)``
  rather than module-level loggers. IPS services should not be used in
  ``__init__``; any logging is done in ``init(...)``.

- **Worker inheritance**: BOUT++-based workers inherit from
  :class:`ipsbout.bout_worker.bout_worker` so they can share task launching,
  restart tarball handling, and processor-count selection.

- **State file handling**: some workers optionally use ``INPUT_STATE_FILES``
  and ``OUTPUT_STATE_FILES`` (in addition to the usual ``STATE_FILES``) to
  distinguish *which state files are staged in* vs *which are updated* during
  a step. See ``examples/proto_mpex/ipsbout.config`` for a concrete workflow.

"""
