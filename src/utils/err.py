class BaseErr(BaseException):
    def __init__(self, message, code=400):
        self.message = message
        self.code = code
        super().__init__(self.message)

# GPT
# 410 - 419
class NoPromptErr(BaseErr):
    def __init__(self, message="Prompt is required"):
        super().__init__(message, code=410)
class UnknownModelErr(BaseErr):
    def __init__(self, message="Unknown model specified"):
        super().__init__(message, code=411)

# UPLOAD
# 420 - 439
class NoFilePartErr(BaseErr):
    def __init__(self, message="No file part"):
        super().__init__(message, code=420)

class NoPathSpecifiedErr(BaseErr):
    def __init__(self, message="No path specified"):
        super().__init__(message, code=421)

class InvalidPathErr(BaseErr):
    def __init__(self, message="Invalid path"):
        super().__init__(message, code=422)

class PathDoesNotExistErr(BaseErr):
    def __init__(self, message="Path does not exist"):
        super().__init__(message, code=423)

class NoSelectedFileErr(BaseErr):
    def __init__(self, message="No selected file"):
        super().__init__(message, code=424)

class PathAlreadyExistsErr(BaseErr):
    def __init__(self, message="Path already exists"):
        super().__init__(message, code=425)