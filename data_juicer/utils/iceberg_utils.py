"""Shared validation and catalog loading for existing-table writers."""

import ntpath
from copy import deepcopy
from urllib.parse import urlparse


def validate_iceberg_artifact(export_path, shard_size=0, encrypt=False):
    """Validate the local artifact anchor and unsupported file options."""
    if not isinstance(export_path, str) or not export_path.strip():
        raise ValueError("Iceberg export_path must be a non-empty local artifact path, not a table URI")
    drive, _ = ntpath.splitdrive(export_path)
    if urlparse(export_path).scheme and not (len(drive) == 2 and drive[1] == ":"):
        raise ValueError("Iceberg export_path must be a non-empty local artifact path, not a table URI")
    if shard_size > 0:
        raise ValueError("Iceberg export does not support export_shard_size > 0")
    if encrypt:
        raise ValueError("Iceberg export does not support local-file encryption")


def _validate_catalog_name(catalog_kwargs):
    """Require an explicit catalog name for writes."""
    if not isinstance(catalog_kwargs, dict):
        raise ValueError("Iceberg export requires catalog_kwargs as a dict")
    name = catalog_kwargs.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Iceberg catalog name must be an explicit non-empty string")


def iceberg_export_config(export_path, shard_size, encrypt, kwargs):
    """Validate write configuration without resolving or logging credentials."""
    validate_iceberg_artifact(export_path, shard_size, encrypt)
    if set(kwargs) - {"table_identifier", "catalog_kwargs", "mode"}:
        raise ValueError("Iceberg export only accepts table_identifier, catalog_kwargs and mode")
    identifier = kwargs.get("table_identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Iceberg export requires a non-empty table_identifier")
    catalog_kwargs = kwargs.get("catalog_kwargs")
    _validate_catalog_name(catalog_kwargs)
    mode = kwargs.get("mode", "overwrite")
    if mode not in ("append", "overwrite"):
        raise ValueError("Iceberg export mode must be append or overwrite")
    try:
        import pyiceberg.catalog  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Iceberg writing requires py-data-juicer[iceberg]") from e
    return {"table_identifier": identifier, "catalog_kwargs": deepcopy(catalog_kwargs), "mode": mode}


def load_iceberg_table(table_identifier, catalog_kwargs):
    """Load an existing table without substituting a file writer."""
    _validate_catalog_name(catalog_kwargs)
    from pyiceberg.catalog import load_catalog
    from pyiceberg.exceptions import NoSuchTableError

    properties = deepcopy(catalog_kwargs)
    name = properties.pop("name")
    try:
        return load_catalog(name, **properties).load_table(table_identifier)
    except NoSuchTableError as e:
        raise ValueError(f"Iceberg target table does not exist: {table_identifier}") from e


def validate_iceberg_schema(schema, table):
    """Reject incompatible fields and any automatic schema evolution."""
    from pyiceberg.io.pyarrow import _check_pyarrow_schema_compatible

    _check_pyarrow_schema_compatible(table.schema(), schema, format_version=table.metadata.format_version)
    update = table.update_schema().union_by_name(schema)
    if update._apply() != table.schema():
        raise ValueError("Iceberg export does not support automatic schema evolution")


def _align_iceberg_schema(schema, table_schema):
    """Restore top-level required flags that Arrow conversions drop."""
    import pyarrow as pa

    required = {field.name for field in table_schema.fields if field.required}
    fields = [field.with_nullable(False) if field.name in required else field for field in schema]
    return pa.schema(fields, metadata=schema.metadata)


def align_iceberg_schema(schema, table):
    return _align_iceberg_schema(schema, table.schema())


def align_iceberg_nullability(arrow, table):
    """Mark required columns non-nullable after checking they contain no nulls."""
    schema = align_iceberg_schema(arrow.schema, table)
    if schema.equals(arrow.schema):
        return arrow
    for field in arrow.schema:
        if field.nullable and not schema.field(field.name).nullable and arrow.column(field.name).null_count:
            raise ValueError(f"Iceberg required column {field.name} contains null values")
    return arrow.cast(schema)


def iceberg_arrow_schema(table):
    """Return the table's exact Arrow schema for Ray worker batches."""
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return schema_to_pyarrow(table.schema())


def cast_iceberg_batch(batch, schema, table_schema, format_version, excluded_fields):
    """Validate and cast one Ray Arrow batch to the target schema."""
    from pyiceberg.io.pyarrow import _check_pyarrow_schema_compatible

    excluded_fields = set(excluded_fields)
    batch = batch.select([name for name in batch.column_names if name not in excluded_fields])
    aligned_schema = _align_iceberg_schema(batch.schema, table_schema)
    _check_pyarrow_schema_compatible(table_schema, aligned_schema, format_version=format_version)
    if set(batch.column_names) != set(schema.names) or len(batch.column_names) != len(schema.names):
        raise ValueError(f"Iceberg schema fields do not match: {batch.column_names} != {schema.names}")
    batch = batch.select(schema.names)
    for field in schema:
        if not field.nullable and batch.column(field.name).null_count:
            raise ValueError(f"Iceberg required column {field.name} contains null values")
    return batch.cast(schema)
