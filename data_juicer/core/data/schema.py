from dataclasses import dataclass
from typing import Any, Dict, List

import pyarrow as pa
from datasets import Array2D, Array3D, ClassLabel, Features, Sequence, Value


@dataclass
class Schema:
    """Dataset schema representation.

    Attributes:
        column_types: Mapping of column names to their types
        columns: List of column names in order
    """

    column_types: Dict[str, Any]
    columns: List[str]

    @classmethod
    def from_hf_features(cls, features: Features):
        # Convert features to schema dict
        column_types = {}
        for name, feature in features.items():
            # Map HF feature types to Python types
            column_types[name] = Schema.map_hf_type_to_python(feature)

        # Get column names
        columns = list(features.keys())

        return Schema(column_types=column_types, columns=columns)

    @classmethod
    def from_ray_schema(cls, schema):
        # convert schema to proper list and dict
        column_types = {k: Schema.map_ray_type_to_python(v) for k, v in zip(schema.names, schema.types)}
        return Schema(column_types=column_types, columns=list(column_types.keys()))

    @classmethod
    def map_hf_type_to_python(cls, feature):
        """Map HuggingFace feature type to Python type.

        Recursively maps nested types (e.g., List[str], Dict[str, int]).

        Examples:
            Value('string') -> str
            Sequence(Value('int32')) -> List[int]
            Dict({'text': Value('string')}) -> Dict[str, Any]

        Args:
            feature: HuggingFace feature type

        Returns:
            Corresponding Python type
        """
        type_mapping = {
            "string": str,
            "int32": int,
            "int64": int,
            "float32": float,
            "float64": float,
            "bool": bool,
            "binary": bytes,
        }
        if isinstance(feature, Value):
            # Map Value types
            return type_mapping.get(feature.dtype, Any)

        elif isinstance(feature, Sequence):
            return List[Schema.map_hf_type_to_python(feature.feature)]
        elif isinstance(feature, (Array2D, Array3D)):
            # Handle sequences/lists
            return List[type_mapping.get(feature.dtype, Any)]

        # Dictionary types - check if it's a dictionary feature
        elif isinstance(feature, dict) or str(type(feature)).endswith("Dict"):
            return Schema.from_hf_features(feature)

        elif isinstance(feature, ClassLabel):
            # Handle class labels
            return int

        else:
            # Default to Any for unknown types
            return Any

    @classmethod
    def map_ray_type_to_python(cls, ray_type: pa.DataType):
        """Map Ray/Arrow data type to Python type.

        Args:
            ray_type: PyArrow DataType

        Returns:
            Corresponding Python type
        """

        # String types
        if pa.types.is_string(ray_type):
            return str
        if pa.types.is_binary(ray_type) or pa.types.is_fixed_size_binary(ray_type):
            return bytes

        # Numeric types
        if pa.types.is_integer(ray_type):
            return int
        if pa.types.is_floating(ray_type):
            return float

        # Boolean
        if pa.types.is_boolean(ray_type):
            return bool

        # List/Array types
        if pa.types.is_list(ray_type):
            return List[Schema.map_ray_type_to_python(ray_type.value_type)]

        # Dictionary/Struct types
        if pa.types.is_struct(ray_type):
            names = ray_type.names
            types = [Schema.map_ray_type_to_python(t.type) for t in ray_type.fields]
            return Schema(column_types=dict(zip(names, types)), columns=names)

        if pa.types.is_map(ray_type):
            return dict

        # Fallback
        return Any

    def __post_init__(self):
        """Validate schema after initialization"""
        # Ensure all columns are in column_types
        if not all(col in self.column_types for col in self.columns):
            missing = set(self.columns) - set(self.column_types.keys())
            raise ValueError(f"Missing type definitions for columns: {missing}")

    def __str__(self) -> str:
        """Return formatted string representation of schema"""
        lines = ["Dataset Schema:"]
        lines.append("-" * 40)
        for col in self.columns:
            lines.append(f"{col}: {self.column_types[col]}")
        return "\n".join(lines)
