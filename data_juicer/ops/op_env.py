import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from loguru import logger
from packaging.requirements import Requirement as PackageRequirement
from packaging.specifiers import SpecifierSet


def parse_single_requirement(req_str: str):
    # parse a single requirement specifier
    req_str = req_str.strip()

    # handle editable & local package
    editable = False
    if req_str.startswith("-e "):
        editable = True
        req_str = req_str[3:].strip()
    # a local package
    if os.path.isdir(req_str) or os.path.isfile(req_str):
        return Requirement(is_local=True, path=req_str, is_editable=editable)

    # a direct git package
    if req_str.startswith("git+") or req_str.startswith("git@"):
        return Requirement(url=req_str)

    # standard package, use packaging to parse
    try:
        req = PackageRequirement(req_str)
        return Requirement(
            name=req.name,
            version=req.specifier,
            extras=list(req.extras),
            markers=req.marker,
            url=req.url,
        )
    except Exception:
        logger.error(f"Failed to parse requirement from the requirement string: {req_str}")
        return None


def parse_requirements_list(req_list: List[str]):
    # parse the detailed requirements info from a list of pip package specifiers
    return [parse_single_requirement(req_str) for req_str in req_list]


def parse_requirements_file(req_file: str):
    # parse the detailed requirements info from a requirements file
    req_list = []
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            req_list.append(line)
    return parse_requirements_list(req_list)


@dataclass
class Requirement:
    """
    A requirement for an operator.
    """

    name: Optional[str] = None  # the name of the package
    version: Optional[Union[SpecifierSet, str]] = None  # the version specifier of the package
    extras: List[str] = None  # the extra optional dependencies to install of this package
    markers: Optional[str] = None  # the environment markers
    url: Optional[str] = None  # the URL of the package
    is_editable: bool = False  # whether the package is editable
    is_local: bool = False  # whether the package is a local package
    path: Optional[str] = None  # the path to the local package

    def __post_init__(self):
        # convert the string version to a SpecifierSet
        if isinstance(self.version, str):
            self.version = SpecifierSet(self.version)

    def __str__(self):
        # special cases: editable & local package
        if self.is_local and self.path:
            if self.is_editable:
                return f"-e {self.path}"
            else:
                return f"{self.path}"

        # general requirement specifier
        result = ""
        if self.name:
            result = f"{self.name}"
            # only consider to add the extra parts when there is a package name
            if self.extras:
                extras_str = ",".join(self.extras)
                result += f"[{extras_str}]"

        # parse two forms: name-based and URL-based
        if self.url:
            # URL-based specifier
            if self.name:
                result += f" @ {self.url}"
            else:
                result += f"{self.url}"
        else:
            # Name-based specifier
            if self.version:
                result += f"{self.version}"

        # 添加环境标记，如果有
        if self.markers:
            result += f" ; {self.markers}"

        return result


class OPEnvSpec:
    """
    Specification of the environment dependencies for an operator.
    """

    def __init__(
        self,
        op_name: str,
        pip_pkgs: Optional[Union[List[str], str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        backend: str = "uv",
        extra_env_params: Optional[Dict] = None,
        parsed_requirements: Optional[Dict[str, Requirement]] = None,
    ):
        """
        Initialize an OPEnvSpec instance.

        :param op_name: Name of the operator
        :param pip_pkgs: Pip packages to install, default is None. Could be a list or a str path to the requirements file
        :param env_vars: Dictionary of environment variables, default is None
        :param working_dir: Path to the working directory, default is None
        :param backend: Package management backend, default is "uv". Should be one of ["pip", "uv"].
        :param extra_env_params: Additional parameters dictionary passed to the ray runtime environment, default is None
        :param parsed_requirements: a resolved version of requirements. It's a dict of req_name-resolved_info, where
            the parsed package info includes version/url/...
        """
        self.op_name = op_name
        self.pip_pkgs = pip_pkgs
        self.env_vars = env_vars
        self.working_dir = working_dir
        self.backend = backend
        assert self.backend in ["pip", "uv"], "Backend should be one of ['pip', 'uv']"
        self.extra_env_params = extra_env_params or {}
        if parsed_requirements:
            self.parsed_requirements = parsed_requirements
        elif self.pip_pkgs:
            if isinstance(self.pip_pkgs, str):
                parsed_res = parse_requirements_file(self.pip_pkgs)
                self.parsed_requirements = {req.name: req for req in parsed_res}
                # update with the parsed pip package list
                self.pip_pkgs = [str(req) for req in parsed_res]
            else:
                self.parsed_requirements = {req.name: req for req in parse_requirements_list(self.pip_pkgs)}

    def to_dict(self):
        """
        Convert the OPEnvSpec instance to a dictionary.

        :return: Dictionary representation of the OPEnvSpec instance
        """
        runtime_env_dict = {}
        if self.pip_pkgs:
            runtime_env_dict[self.backend] = self.pip_pkgs
        if self.env_vars:
            runtime_env_dict["env_vars"] = self.env_vars
        if self.working_dir:
            runtime_env_dict["working_dir"] = self.working_dir
        runtime_env_dict.update(self.extra_env_params)
        return runtime_env_dict


def op_requirements_to_op_env_spec(op_name: str, requirements: Optional[Union[List[str], str]] = None) -> OPEnvSpec:
    if requirements is None:
        return OPEnvSpec(op_name)
    elif isinstance(requirements, str) or isinstance(requirements, list):
        return OPEnvSpec(op_name, pip_pkgs=requirements)
    else:
        raise ValueError(
            f"Invalid type of specified requirements: {type(requirements)} for op {op_name}. "
            f"Expected str or List[str]."
        )
