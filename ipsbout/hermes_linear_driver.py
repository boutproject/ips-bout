from ipsframework import Component


class hermes_linear_driver(Component):
    """
    Workflow for linear device with electron heating.
    """

    def init(self, timeStamp=0.0):
        """
        Runs GRIDGEN
        """
        gridgen = self.services.get_port("GRIDGEN")
        self.services.call(gridgen, "init", timeStamp)
        self.services.call(gridgen, "step", timeStamp)
        self.services.call(gridgen, "finalize", timeStamp)

    def step(self, timeStamp=0.0):
        """
        Runs ELECTRON_HEATING to update the heating profile,
        then HERMES to update the plasma profiles.
        """
        for name in ["ELECTRON_HEATING", "HERMES"]:
            component = self.services.get_port(name)
            self.services.call(component, "init", timeStamp)
            self.services.call(component, "step", timeStamp)
            self.services.call(component, "finalize", timeStamp)
