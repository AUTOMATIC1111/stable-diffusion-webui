version_flag = 'v1.1.440'

from scripts.logging import logger

logger.info(f"ControlNet {version_flag}")
# A smart trick to know if user has updated as well as if user has restarted terminal.
# Note that in "controlnet.py" we do NOT use "importlib.reload" to reload this "controlnet_version.py"
# This means if user did not completely restart terminal, the "version_flag" will be the previous version.
# Then, if we get a screenshot from user, we will know that if that user has restarted the terminal.
# And we will also know what version the user is using so that bug track becomes easier.
