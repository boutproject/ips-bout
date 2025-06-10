from ipsframework import Component
import logging
logger = logging.getLogger(__name__)

class hermes_transport_driver(Component):
    """

    If GRIDGEN port is available then a grid will be generated

    
    
    """
    def __init__(self, services, config):
        super().__init__(services, config)
        logging.basicConfig(level=logging.INFO)
        logger.info(f'Created {self.__class__}')

    def restart(self):
        pass

    def step(self, timestamp=0.0):
        try:
            worker_comp = self.services.get_port('GRIDGEN')
            logger.info('Generating grid')
            self.services.call(worker_comp, 'step', 0.0)
            logger.info('Finished generating grid')
        except KeyError:
            logger.info("Skipping grid generation")

        worker = self.services.get_port('TRANSPORT')
        self.services.call(worker, 'step', 0.0)
        logger.info("Finished")
