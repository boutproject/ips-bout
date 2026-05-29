from ipsframework import Component


class hermes_transport_driver(Component):
    """

    If GRIDGEN port is available then a grid will be generated



    """

    def __init__(self, services, config):
        super().__init__(services, config)

    def restart(self):
        pass

    def step(self, timestamp=0.0):
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
