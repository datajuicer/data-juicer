import importlib.util
import inspect
import os

from ..base_op import OPERATORS, Mapper

OP_NAME = "python_file_mapper"


@OPERATORS.register_module(OP_NAME)
class PythonFileMapper(Mapper):
    """Executes a Python function defined in a file on input data.

    This operator loads a specified Python function from a given file and applies it to the
    input data. The function must take exactly one argument and return a dictionary. The
    operator can process data either sample by sample or in batches, depending on the
    `batched` parameter. If the file path is not provided, the operator acts as an identity
    function, returning the input sample unchanged. The function is loaded dynamically, and
    its name and file path are configurable. Important notes:
    - The file must be a valid Python file (`.py`).
    - The function must be callable and accept exactly one argument.
    - The function's return value must be a dictionary."""

    def __init__(self, file_path: str = "", function_name: str = "process_single", batched: bool = False, **kwargs):
        """
        Initialization method.

        :param file_path: The path to the Python file containing the function
            to be executed.
        :param function_name: The name of the function defined in the file
            to be executed.
        :param batched: A boolean indicating whether to process input data in
            batches.
        :param kwargs: Additional keyword arguments passed to the parent class.
        """
        self._batched_op = bool(batched)
        super().__init__(**kwargs)

        self.file_path = file_path
        self.function_name = function_name
        if not file_path:
            self.func = lambda sample: sample
        else:
            self.func = self._load_function()

    def _load_function(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

        if not self.file_path.endswith(".py"):
            raise ValueError(f"The file '{self.file_path}' is not a Python file.")

        # Load the module from the file
        module_name = os.path.splitext(os.path.basename(self.file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Fetch the specified function from the module
        if not hasattr(module, self.function_name):
            raise ValueError(f"Function '{self.function_name}' not found in '{self.file_path}'.")  # noqa: E501

        func = getattr(module, self.function_name, None)

        if not callable(func):
            raise ValueError(f"The attribute '{self.function_name}' is not callable.")

        # Check that the function has exactly one argument
        argspec = inspect.getfullargspec(func)
        if len(argspec.args) != 1:
            raise ValueError(f"The function '{self.function_name}' must take exactly one argument")  # noqa: E501

        return func

    def process_single(self, sample):
        """Invoke the loaded function with the provided sample."""
        result = self.func(sample)

        if not isinstance(result, dict):
            raise ValueError(f"Function must return a dictionary, got {type(result).__name__} instead.")  # noqa: E501

        return result

    def process_batched(self, samples):
        """Invoke the loaded function with the provided samples."""
        result = self.func(samples)

        if not isinstance(result, dict):
            raise ValueError(f"Function must return a dictionary, got {type(result).__name__} instead.")  # noqa: E501

        return result
