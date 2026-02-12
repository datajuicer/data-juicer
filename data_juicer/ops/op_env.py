import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
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
    parsed = []
    failed = []
    for s in req_list:
        r = parse_single_requirement(s)
        if r is None:
            failed.append(s)
            continue
        parsed.append(r)
    if failed:
        logger.error(f"Failed to parse {len(failed)} requirement(s), ignored: {failed}")
    return parsed


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

        # Add environment markers, if any
        if self.markers:
            result += f" ; {self.markers}"

        return result


class OPEnvSpec:
    """
    Specification of the environment dependencies for an operator.
    """

    def __init__(
        self,
        pip_pkgs: Optional[Union[List[str], str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        backend: str = "uv",
        extra_env_params: Optional[Dict] = None,
        parsed_requirements: Optional[Dict[str, Requirement]] = None,
    ):
        """
        Initialize an OPEnvSpec instance.

        :param pip_pkgs: Pip packages to install, default is None. Could be a list or a str path to the requirement file
        :param env_vars: Dictionary of environment variables, default is None
        :param working_dir: Path to the working directory, default is None
        :param backend: Package management backend, default is "uv". Should be one of ["pip", "uv"].
        :param extra_env_params: Additional parameters dictionary passed to the ray runtime environment, default is None
        :param parsed_requirements: a resolved version of requirements. It's a dict of req_name-resolved_info, where
            the parsed package info includes version/url/...
        """
        self.pip_pkgs = pip_pkgs
        self.env_vars = env_vars
        self.working_dir = working_dir
        self.backend = backend
        assert self.backend in ["pip", "uv"], "Backend should be one of ['pip', 'uv']"
        self.extra_env_params = extra_env_params or {}
        if parsed_requirements:
            self.parsed_requirements = parsed_requirements
            # update pip_pkgs with the parsed pip package list
            self.pip_pkgs = [str(req) for req in self.parsed_requirements.values()]
        elif self.pip_pkgs:
            if isinstance(self.pip_pkgs, str):
                parsed_res = parse_requirements_file(self.pip_pkgs)
                self.parsed_requirements = {req.name if req.name else req.url: req for req in parsed_res}
                # update with the parsed pip package list
                self.pip_pkgs = [str(req) for req in parsed_res]
            else:
                self.parsed_requirements = {
                    req.name if req.name else req.url: req for req in parse_requirements_list(self.pip_pkgs)
                }
        else:
            self.pip_pkgs = []

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

    def get_hash(self):
        op_env_spec_dict = self.to_dict()
        serialized_spec = json.dumps(op_env_spec_dict, sort_keys=True)
        return hashlib.sha1(serialized_spec.encode("utf-8")).hexdigest()

    def get_requirement_name_list(self):
        return sorted(self.parsed_requirements.keys())


def op_requirements_to_op_env_spec(
    op_name: str,
    requirements: Optional[Union[List[str], str]] = None,
    auto_recommended_requirements: Optional[List[str]] = None,
) -> OPEnvSpec:
    if requirements is None:
        if auto_recommended_requirements:
            logger.info(
                f"No requirements are specified for op {op_name}. Use auto recommended requirements instead: {auto_recommended_requirements}"
            )
            return OPEnvSpec(pip_pkgs=auto_recommended_requirements)
        else:
            return OPEnvSpec()
    elif isinstance(requirements, str) or isinstance(requirements, list):
        if auto_recommended_requirements is None:
            auto_recommended_requirements = []
        specified_spec = OPEnvSpec(pip_pkgs=requirements)
        recommended_reqs = {
            req.name if req.name else req.url: req for req in parse_requirements_list(auto_recommended_requirements)
        }
        new_recommended_reqs = [
            str(recommended_reqs[req_key])
            for req_key in recommended_reqs
            if req_key not in specified_spec.parsed_requirements
        ]
        if len(new_recommended_reqs) > 0:
            logger.info(
                f"Adding {len(new_recommended_reqs)} recommended requirements to op {op_name}: {new_recommended_reqs}"
            )
        return OPEnvSpec(pip_pkgs=specified_spec.pip_pkgs + new_recommended_reqs)
    else:
        raise ValueError(
            f"Invalid type of specified requirements: {type(requirements)} for op {op_name}. "
            f"Expected str or List[str]."
        )


class ConflictResolveStrategy(Enum):
    # Strategies to resolve the dependency conflicts.
    # It decides how to handle the conflicts when merging OP environments.
    # SPLIT: Log an error and keep the two specs split when there is a conflict.
    # OVERWRITE: Overwrite the existing dependency with one from the later OP.
    # LATEST: Use the latest version of all specified dependency versions.
    SPLIT = "split"
    OVERWRITE = "overwrite"
    LATEST = "latest"


class OPEnvManager:
    """
    OPEnvManager is a class that manages the environment dependencies for operators,
    including recording OP dependencies, resolving dependency conflicts, merging OP environments, and so on.
    """

    def __init__(
        self,
        min_common_dep_num_to_combine: Optional[int] = -1,
        conflict_resolve_strategy: Union[ConflictResolveStrategy, str] = ConflictResolveStrategy.SPLIT,
    ):
        """
        Initialize OPEnvManager instance.

        :param min_common_dep_num_to_combine: The minimum number of common dependencies required to
                                              determine whether to merge two operation environment specifications.
                                              If set to -1, it means no combination of operation environments.
        :param conflict_resolve_strategy: Strategy for resolving dependency conflicts, default is SPLIT strategy.
                                          SPLIT: Keep the two specs split when there is a conflict.
                                          OVERWRITE: Overwrite the existing dependency with one from the later OP.
                                          LATEST: Use the latest version of all specified dependency versions.
        """
        self.min_common_dep_num_to_combine = min_common_dep_num_to_combine
        if self.min_common_dep_num_to_combine == -1:
            logger.warning("min_common_dep_num_to_combine is set to -1, which means no combination on OP Environments.")
        elif self.min_common_dep_num_to_combine < 0:
            raise ValueError("min_common_dep_num_to_combine should be >= 0 or == -1.")
        else:
            logger.info(
                f"Try to combine OP Environments with at least "
                f"{self.min_common_dep_num_to_combine} common dependencies "
            )
        if isinstance(conflict_resolve_strategy, str):
            self.conflict_resolve_strategy = ConflictResolveStrategy(conflict_resolve_strategy)
        else:
            self.conflict_resolve_strategy = conflict_resolve_strategy

        # OP name -> OPEnvSpec with two isolated lists
        self.op2hash = {}
        self.hash2ops = defaultdict(list)
        self.hash2specs = {}

    def print_the_current_states(self):
        """
        Get the current states of OPEnvManager, including:
        - number of recorded OPs
        - number of used env specs
        - what OPs share the same env spec

        :return: A dictionary containing the current states of OPEnvManager
        """
        num_unique_specs = len(self.hash2ops)

        logger.info("The current states of OPEnvManager:")
        logger.info(f"\t- Total number of unique environment specs: {num_unique_specs}")
        logger.info("\t- OP-spec relations: (OP name -> Packages)")
        mappings = {}
        for hash_val in self.hash2ops:
            op_key = ", ".join(self.hash2ops[hash_val])
            pkgs = self.hash2specs[hash_val].pip_pkgs
            mappings[op_key] = pkgs
            logger.info(f"\t\t- [{op_key}]: {pkgs}")
        return mappings

    def record_op_env_spec(self, op_name: str, op_env_spec: OPEnvSpec):
        """
        Record the OP environment specification for an operator.

        :param op_name: Name of the operator
        :param op_env_spec: OP environment specification
        """
        env_spec_hash = self.merge_op_env_specs(op_env_spec)
        self.op2hash[op_name] = env_spec_hash
        self.hash2ops[env_spec_hash].append(op_name)

    def merge_op_env_specs(self, new_env_spec: OPEnvSpec):
        """
        Merge the OP environment specification for an operator with existing OP environment specification.

        :param new_env_spec: OP environment specification
        """
        new_hash = new_env_spec.get_hash()
        if new_hash in self.hash2specs:
            # this env spec is existing, do nothing
            return new_hash

        # check if there are any existing env specs can be combined with the new one
        for _, curr_hash in self.op2hash.items():
            curr_env_spec = self.hash2specs[curr_hash]
            if self.can_combine_op_env_specs(curr_env_spec, new_env_spec):
                # combine the two specs
                combined_spec = self.try_to_combine_op_env_specs(curr_env_spec, new_env_spec)
                if combined_spec is None:
                    # combine failed
                    continue
                combined_hash = combined_spec.get_hash()
                if combined_hash != curr_hash:
                    # use a new env spec
                    self.hash2specs[combined_hash] = combined_spec
                    # update existing OPs that use the current env spec
                    for op in self.hash2ops[curr_hash]:
                        self.op2hash[op] = combined_hash
                    self.hash2ops[combined_hash] = self.hash2ops.pop(curr_hash)
                return combined_hash
        # no existing env specs can be combined
        self.hash2specs[new_hash] = new_env_spec
        return new_hash

    def can_combine_op_env_specs(self, first_env_spec: OPEnvSpec, second_env_spec: OPEnvSpec) -> bool:
        """
        Check if two OP environment specifications can be combined.

        :param first_env_spec: Existing OP environment specification
        :param second_env_spec: New OP environment specification
        :return: True if the two specifications can be combined, False otherwise
        """
        if self.min_common_dep_num_to_combine == -1:
            # no combination
            return False
        # check the number of common deps
        first_env_req_set = set(first_env_spec.get_requirement_name_list())
        second_env_req_set = set(second_env_spec.get_requirement_name_list())
        if len(first_env_req_set & second_env_req_set) < self.min_common_dep_num_to_combine:
            return False
        # check if they share the same working dir
        if (
            first_env_spec.working_dir
            and second_env_spec.working_dir
            and first_env_spec.working_dir != second_env_spec.working_dir
        ):
            return False
        return True

    def try_to_combine_op_env_specs(self, first_env_spec: OPEnvSpec, second_env_spec: OPEnvSpec):
        """
        Try to combine the OP environment specification for an operator with existing OP environment specification.

        :param first_env_spec: Name of the operator
        :param second_env_spec: OP environment specification
        :return: True if the two specifications can be combined, False otherwise
        """
        first_parsed_reqs = first_env_spec.parsed_requirements
        second_parsed_reqs = second_env_spec.parsed_requirements
        combined_req_names = set(first_parsed_reqs.keys()) | set(second_parsed_reqs.keys())
        new_parsed_reqs = {}
        for req_name in combined_req_names:
            if req_name in first_parsed_reqs and req_name in second_parsed_reqs:
                # resolve conflict
                first_req = first_parsed_reqs[req_name]
                second_req = second_parsed_reqs[req_name]
                combined_req = self._resolve_with_strategy(first_req, second_req)
                if combined_req is None:
                    # conflict cannot be resolved
                    return None
                new_parsed_reqs[req_name] = combined_req
            elif req_name in first_parsed_reqs:
                new_parsed_reqs[req_name] = first_parsed_reqs[req_name]
            else:
                new_parsed_reqs[req_name] = second_parsed_reqs[req_name]

        # combine other attributes
        if first_env_spec.env_vars and second_env_spec.env_vars:
            combined_env_vars = first_env_spec.env_vars.copy()
            combined_env_vars.update(second_env_spec.env_vars)
        elif first_env_spec.env_vars:
            combined_env_vars = first_env_spec.env_vars
        else:
            combined_env_vars = second_env_spec.env_vars
        combined_working_dir = first_env_spec.working_dir or second_env_spec.working_dir
        combined_backend = first_env_spec.backend if first_env_spec.backend == second_env_spec.backend else "uv"
        if first_env_spec.extra_env_params and second_env_spec.extra_env_params:
            combined_extra_env_params = first_env_spec.extra_env_params.copy()
            combined_extra_env_params.update(second_env_spec.extra_env_params)
        elif first_env_spec.extra_env_params:
            combined_extra_env_params = first_env_spec.extra_env_params
        else:
            combined_extra_env_params = second_env_spec.extra_env_params

        # create a new combined OPEnvSpec
        return OPEnvSpec(
            env_vars=combined_env_vars,
            working_dir=combined_working_dir,
            backend=combined_backend,
            extra_env_params=combined_extra_env_params,
            parsed_requirements=new_parsed_reqs,
        )

    def get_op_env_spec(self, op_name: str) -> OPEnvSpec:
        """
        Get the OP environment specification for an operator.

        :param op_name: Name of the operator
        :return: OP environment specification
        """

        if op_name not in self.op2hash:
            raise ValueError(f"OP {op_name} is not recorded in OPEnvManager")
        return self.hash2specs[self.op2hash[op_name]]

    def _resolve_with_strategy(self, first_req: Requirement, second_req: Requirement):
        # check if there are conflicts. Only consider the version conflicts for now.
        version1 = first_req.version
        version2 = second_req.version
        # quick check
        if version1 is None:
            return second_req
        if version2 is None:
            return first_req
        if version1 == version2:
            # no conflict
            return first_req

        from dep_logic.specifiers import (
            ArbitrarySpecifier,
            EmptySpecifier,
            UnionSpecifier,
            parse_version_specifier,
        )

        p1 = parse_version_specifier(str(version1))
        p2 = parse_version_specifier(str(version2))
        combined = p1 & p2
        # combine other fields
        if first_req.extras and second_req.extras:
            combined_extras = list(set(first_req.extras) | set(second_req.extras))
        elif first_req.extras:
            combined_extras = first_req.extras
        else:
            combined_extras = second_req.extras
        if first_req.markers and second_req.markers:
            from dep_logic.markers import parse_marker

            marker1 = parse_marker(first_req.markers)
            marker2 = parse_marker(second_req.markers)
            combined_markers = str(marker1 & marker2)
        elif first_req.markers:
            combined_markers = first_req.markers
        else:
            combined_markers = second_req.markers
        if not isinstance(combined, EmptySpecifier):
            # there is no conflict
            return Requirement(
                name=first_req.name or second_req.name,
                version=SpecifierSet(str(combined)),
                extras=combined_extras,
                markers=combined_markers,
                url=first_req.url or second_req.url,
                is_editable=first_req.is_editable or second_req.is_editable,
                is_local=first_req.is_local or second_req.is_local,
                path=first_req.path or second_req.path,
            )

        # both specifiers are arbitrary equality, must be split
        if isinstance(p1, ArbitrarySpecifier) and isinstance(p2, ArbitrarySpecifier):
            return None
        # and if any of them is arbitrary equality, just use it
        elif isinstance(p1, ArbitrarySpecifier):
            return first_req
        elif isinstance(p2, ArbitrarySpecifier):
            return second_req

        # there are conflicts, resolve them with specified strategy
        if self.conflict_resolve_strategy == ConflictResolveStrategy.SPLIT:
            # split
            return None
        elif self.conflict_resolve_strategy == ConflictResolveStrategy.OVERWRITE:
            # overwrite
            return second_req
        else:
            # latest
            # None means +inf
            include_latest = False
            latest_version = None
            # use union to combine them to find the latest version
            combined = p1 | p2
            if isinstance(combined, UnionSpecifier):
                max_str = []
                for r in combined.ranges:
                    if r.max:
                        if r.include_max:
                            max_str.append(f"<={r.max}")
                        else:
                            max_str.append(f"<{r.max}")
                    else:
                        max_str = []
                        break
                if len(max_str) > 0:
                    max_spec = parse_version_specifier("||".join(max_str))
                    include_latest = max_spec.include_max
                    latest_version = max_spec.max
            else:
                include_latest = combined.include_max
                latest_version = combined.max
            if latest_version:
                latest_version = (
                    SpecifierSet(f"=={latest_version}") if include_latest else SpecifierSet(f"<{latest_version}")
                )
            elif latest_version is None:
                logger.warning(
                    f"Dependency conflict for {first_req.name or second_req.name}, "
                    f"fallback to unpinned version under LATEST strategy: "
                    f"{first_req} vs {second_req}"
                )
            return Requirement(
                name=first_req.name or second_req.name,
                version=latest_version,
                extras=combined_extras,
                markers=combined_markers,
                url=first_req.url or second_req.url,
                is_editable=first_req.is_editable or second_req.is_editable,
                is_local=first_req.is_local or second_req.is_local,
                path=first_req.path or second_req.path,
            )


def analyze_lazy_loaded_requirements_for_code_file(code_file: str) -> List[str]:
    with open(code_file, "r") as fin:
        code_content = fin.read()
    return analyze_lazy_loaded_requirements(code_content)


def analyze_lazy_loaded_requirements(code_content: str) -> List[str]:
    import ast

    reqs = []
    tree = ast.parse(code_content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if callee == "LazyLoader":
                # calling LazyLoader(module_name, package_name, package_url, ...)
                args = [ast.literal_eval(ast.unparse(arg)) for arg in node.args]
                kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
                target_args = ["module_name", "package_name", "package_url"]
                existing_args = args[: min(len(target_args), len(args))]
                parsed_args = dict(zip(target_args[: len(existing_args)], existing_args))
                # find missing kwargs
                for i in range(len(existing_args), len(target_args)):
                    if target_args[i] in kwargs:
                        parsed_args[target_args[i]] = ast.literal_eval(kwargs[target_args[i]])
                req = Requirement()
                if "package_name" in parsed_args:
                    req.name = parsed_args["package_name"]
                else:
                    req.name = parsed_args["module_name"]
                if "package_url" in parsed_args:
                    req.url = parsed_args["package_url"]
                reqs.append(str(req))
            elif callee == "LazyLoader.check_packages":
                args = [ast.unparse(arg) for arg in node.args]
                kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
                if len(args) > 0:
                    parsed_args = args[0]
                else:
                    parsed_args = kwargs.get("package_specs", None)
                if parsed_args:
                    req_list = ast.literal_eval(parsed_args)
                    reqs.extend(req_list)
            # ignore other situations
    return reqs
