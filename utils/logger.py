import logging
FORMATTER = '%(asctime)s - %(message)s'
LOGFILE = 'panolatefusion.log'

def create_logger(logger_name, logfile=LOGFILE, verbose=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if verbose else logging.DEBUG)
    file_handler = logging.FileHandler(logfile, mode='w').setFormatter(FORMATTER)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
