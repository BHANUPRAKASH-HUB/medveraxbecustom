import sys
import traceback

class MedVeraxException(Exception):
    """
    Base custom exception for MedVerax project
    """

    def __init__(self, message: str, error: Exception = None):
        super().__init__(message)
        self.error = error
        self.traceback = traceback.format_exc()

    def __str__(self):
        return (
            f"\n🚨 MedVeraxException\n"
            f"Message: {self.args[0]}\n"
            f"Original Error: {self.error}\n"
            f"Traceback:\n{self.traceback}"
        )
