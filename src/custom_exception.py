class CustomException(Exception):
    """A custom exception class. Used to identify error with our scripts."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"