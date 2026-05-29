from ipsframework import Component

class hermes_linear_driver(Component):
    """
    Workflow for linear device with electron heating.
    """

    def init(self, timeStamp=0.0):
        """
        Runs GRIDGEN and updates the state

        To copy a file into the state, list it in both
        INPUT_FILES and STATE_FILES.
        """
        gridgen = self.services.get_port("GRIDGEN")
        self.services.call(gridgen, "init", timeStamp)
        self.services.call(gridgen, "step", timeStamp)
        self.services.call(gridgen, "finalize", timeStamp)

        # Stage any initial state files
        # This can be used to copy an initial restart file
        self.services.update_state()

    def step(self, timeStamp=0.0):
        """
        Runs ELECTRON_HEATING to update the heating profile,
        then HERMES to update the plasma profiles.
        """
        for name in ["ELECTRON_HEATING", "HERMES"]:
            try:
                component = self.services.get_port(name)
            except KeyError:
                self.services.info(f"Port '{name}' not found. Skipping")
                continue
            self.services.info(f"Running port '{name}'")
            self.services.call(component, "init", timeStamp)
            self.services.call(component, "step", timeStamp)
            self.services.call(component, "finalize", timeStamp)
