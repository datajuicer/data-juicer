import ast

from ..base_op import OPERATORS, Mapper

OP_NAME = "python_lambda_mapper"


@OPERATORS.register_module(OP_NAME)
class PythonLambdaMapper(Mapper):
    """Mapper for applying a Python lambda function to data samples.

    This operator allows users to define a custom transformation using a Python lambda
    function. The lambda function is applied to each sample, and the result must be a
    dictionary. If the `batched` parameter is set to True, the lambda function will process
    a batch of samples at once. If no lambda function is provided, the identity function is
    used, which returns the input sample unchanged. The operator validates the lambda
    function to ensure it has exactly one argument and compiles it safely."""

    def __init__(self, lambda_str: str = "", batched: bool = False, **kwargs):
        """
        Initialization method.

        :param lambda_str: A string representation of the lambda function to be
            executed on data samples. If empty, the identity function is used.
        :param batched: A boolean indicating whether to process input data in
            batches.
        :param kwargs: Additional keyword arguments passed to the parent class.
        """
        self._batched_op = bool(batched)
        super().__init__(**kwargs)

        # Parse and validate the lambda function
        if not lambda_str:
            self.lambda_func = lambda sample: sample
        else:
            self.lambda_func = self._create_lambda(lambda_str)

    def _create_lambda(self, lambda_str: str):
        # Parse input string into an AST and check for a valid lambda function
        try:
            node = ast.parse(lambda_str, mode="eval")

            # Check if the body of the expression is a lambda
            if not isinstance(node.body, ast.Lambda):
                raise ValueError("Input string must be a valid lambda function.")

            # Check that the lambda has exactly one argument
            if len(node.body.args.args) != 1:
                raise ValueError("Lambda function must have exactly one argument.")

            # Compile the AST to code
            compiled_code = compile(node, "<string>", "eval")
            # Safely evaluate the compiled code allowing built-in functions
            func = eval(compiled_code, {"__builtins__": __builtins__})
            return func
        except Exception as e:
            raise ValueError(f"Invalid lambda function: {e}")

    def process_single(self, sample):
        # Process the input through the lambda function and return the result
        result = self.lambda_func(sample)

        # Check if the result is a valid
        if not isinstance(result, dict):
            raise ValueError(f"Lambda function must return a dictionary, " f"got {type(result).__name__} instead.")

        return result

    def process_batched(self, samples):
        # Process the input through the lambda function and return the result
        result = self.lambda_func(samples)

        # Check if the result is a valid
        if not isinstance(result, dict):
            raise ValueError(f"Lambda function must return a dictionary, " f"got {type(result).__name__} instead.")

        return result
