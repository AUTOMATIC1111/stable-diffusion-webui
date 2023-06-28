from scripts.extLogging import logger

import modules.scripts as scripts


class Script(scripts.Script):
    def postprocess(self, p, processed, *args):
        logger.info(p)
        logger.info(args)
