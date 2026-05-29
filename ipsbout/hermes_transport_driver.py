from ipsframework import Component


class hermes_transport_driver(Component):
    """Driver for a simple transport workflow.

    If a `GRIDGEN` port is present, it is run before the `TRANSPORT` port.

    Notes
    -----
    - This driver currently calls workers with a timestamp of `0.0` rather
      than forwarding its own `timestamp` argument. This matches historical
      usage in this repository but may not be appropriate for multi-step
      time-dependent workflows.
    """

    def __init__(self, services, config):
        """Construct the driver."""
        super().__init__(services, config)

    def restart(self):
        """IPS restart hook (unused)."""
        pass

    def step(self, timestamp=0.0):
        """Run GRIDGEN (if present) and then the TRANSPORT worker."""
        try:
            worker_comp = self.services.get_port("GRIDGEN")
            self.services.info("Generating grid")
            self.services.call(worker_comp, "step", 0.0)
            self.services.info("Finished generating grid")
        except KeyError:
            self.services.info("Skipping grid generation")

        self.services.info("Running transport step")
        worker = self.services.get_port("TRANSPORT")
        self.services.call(worker, "step", 0.0)
        self.services.info("Finished")
