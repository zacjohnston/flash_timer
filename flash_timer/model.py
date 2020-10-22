from . import flash_timer


class Model:
    """A single FLASH model
    """
    def __init__(self,
                 filepath):
        """

        parameters
        ----------
        filepath: str
            path to log file
        """
        self.filepath = filepath
        self.table = flash_timer.extract_table(filepath=filepath)
