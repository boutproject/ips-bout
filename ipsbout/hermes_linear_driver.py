from ipsframework import Component

class hermes_linear_driver(Component):
    """
    Workflow for linear device with electron heating.
    """
    def init(self, timeStamp=0.0):
        """
        Runs GRIDGEN
        """
        gridgen = self.services.get_port('GRIDGEN')
        self.services.call(gridgen, 'init', timeStamp)
        self.services.call(gridgen, 'step', timeStamp)
        self.services.call(gridgen, 'finalize', timeStamp)

    def step(self, timeStamp=0.0):
        """
        Runs ELECTRON_HEATING
        """
        heating = self.services.get_port('ELECTRON_HEATING')
        self.services.call(heating, 'init', timeStamp)
        self.services.call(heating, 'step', timeStamp)
        self.services.call(heating, 'finalize', timeStamp)

