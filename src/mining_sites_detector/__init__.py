# read version from installed package
from importlib.metadata import version
import logging

package_name = __name__
__version__ = version(package_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="train_session.logs",
                    filemode="a"
                    )

logger.info(f"{package_name} version {__version__} initialized.")
