from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys

try:
    n = 2/0
    logging.info("The execution is started")

except Exception as e:
    raise CustomException(e,sys)

if __name__ == "__main__":
    logging.info("Run the logging function")