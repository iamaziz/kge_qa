import logging
import sys

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


# hide traceback in assertions errors: https://stackoverflow.com/a/27674608/2839786
sys.tracebacklimit = 0


# disable all logging calls with levels less severe than or equal to CRITICAL
# logging.disable(logging.CRITICAL)
