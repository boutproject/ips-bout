from ipsframework import Component
import logging
logger = logging.getLogger(__name__)

class transport_driver(Component):
    def __init__(self, services, config):
        super().__init__(services, config)
        logging.basicConfig(level=logging.INFO)
        logger.info(f'Created {self.__class__}')

    def step(self, timestamp=0.0):
        logger.info('Generating grid')
        worker_comp = self.services.get_port('GRIDGEN')
        self.services.call(worker_comp, 'step', 0.0)
        logger.info('Finished generating grid')
