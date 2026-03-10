import unittest
import os
from pathlib import Path

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RequirementTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import Requirement
        self.Requirement = Requirement

    def test_basic_requirement(self):
        req = self.Requirement(name="numpy", version=">=1.20.0")
        self.assertEqual(str(req), "numpy>=1.20.0")

    def test_requirement_with_extras(self):
        req = self.Requirement(name="scipy", version=">=1.7.0", extras=["io"])
        self.assertEqual(str(req), "scipy[io]>=1.7.0")

    def test_requirement_with_markers(self):
        req = self.Requirement(name="torch", version=">=1.8.0", markers="python_version>='3.6'")
        self.assertEqual(str(req), "torch>=1.8.0 ; python_version>='3.6'")

    def test_url_requirement(self):
        req = self.Requirement(name="mypkg", url="https://github.com/user/repo.git")
        self.assertEqual(str(req), "mypkg @ https://github.com/user/repo.git")

    def test_url_requirement_wo_name(self):
        req = self.Requirement(url="https://github.com/user/repo.git")
        self.assertEqual(str(req), "https://github.com/user/repo.git")

    def test_local_package_requirement(self):
        req = self.Requirement(is_local=True, path="/path/to/local/pkg", is_editable=False)
        self.assertEqual(str(req), "/path/to/local/pkg")

    def test_editable_local_package_requirement(self):
        req = self.Requirement(is_local=True, path="/path/to/local/pkg", is_editable=True)
        self.assertEqual(str(req), "-e /path/to/local/pkg")


class OPEnvSpecTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvSpec
        self.OPEnvSpec = OPEnvSpec

        self.work_dir = 'tmp/test_op_env_spec/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_init_with_pip_packages_list(self):
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "pandas>=1.3.0"])
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas>=1.3.0"])
        self.assertEqual(spec.backend, "uv")

    def test_init_with_pip_packages_string(self):
        # create a temp requirements.txt file
        req_file = Path(self.work_dir) / "requirements.txt"
        with open(req_file, "w") as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\n")
        
        spec = self.OPEnvSpec(pip_pkgs=str(req_file))
        self.assertEqual(len(spec.pip_pkgs), 2)
        self.assertIn("numpy>=1.20.0", spec.pip_pkgs)
        self.assertIn("pandas>=1.3.0", spec.pip_pkgs)

    def test_init_with_env_vars(self):
        env_vars = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}
        spec = self.OPEnvSpec(env_vars=env_vars)
        self.assertEqual(spec.env_vars, env_vars)

    def test_to_dict_with_pip_packages(self):
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], backend="pip")
        expected = {"pip": ["numpy>=1.20.0"]}
        self.assertEqual(spec.to_dict(), expected)

    def test_to_dict_with_env_vars(self):
        env_vars = {"CUDA_VISIBLE_DEVICES": "0"}
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], env_vars=env_vars, working_dir=self.work_dir)
        expected = {"uv": ["numpy>=1.20.0"], "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}, "working_dir": self.work_dir}
        self.assertEqual(spec.to_dict(), expected)

    def test_backend_validation(self):
        with self.assertRaises(AssertionError):
            self.OPEnvSpec(backend="invalid_backend")

    def test_parsed_requirements(self):
        from data_juicer.ops.op_env import Requirement
        req1 = Requirement(name="numpy", version=">=1.20.0")
        req2 = Requirement(name="pandas", version=">=1.3.0")
        spec = self.OPEnvSpec(parsed_requirements={"numpy": req1, "pandas": req2})
        self.assertEqual(spec.parsed_requirements, {
            "numpy": req1,
            "pandas": req2
        })
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas>=1.3.0"])

    def test_hash(self):
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "pandas>=1.3.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0", "pandas>=1.3.0"])
        spec3 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "pandas>=1.3.0"])
        self.assertNotEqual(spec1.get_hash(), spec2.get_hash())
        self.assertEqual(spec1.get_hash(), spec3.get_hash())

    def test_get_requirements_name_list(self):
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        self.assertEqual(spec1.get_requirement_name_list(), ["numpy"])
        from data_juicer.ops.op_env import Requirement
        req1 = Requirement(name="numpy", version=">=1.20.0")
        req2 = Requirement(name="pandas", version=">=1.3.0")
        spec2 = self.OPEnvSpec(parsed_requirements={"numpy": req1, "pandas": req2})
        self.assertEqual(spec2.get_requirement_name_list(), ["numpy", "pandas"])

class ParseSingleRequirementTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import parse_single_requirement
        self.parse_single_requirement = parse_single_requirement

        self.work_dir = 'tmp/test_parse_single_requirement/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_parse_basic_requirement(self):
        req = self.parse_single_requirement("numpy>=1.20.0")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "numpy")
        self.assertEqual(str(req.version), ">=1.20.0")

    def test_parse_requirement_with_extras(self):
        req = self.parse_single_requirement("scipy[io]>=1.5.0")
        self.assertIsNotNone(req)
        self.assertEqual(req.name, "scipy")
        self.assertEqual(req.extras, ["io"])

    def test_parse_editable_package(self):
        path_to_pkg = os.path.join(self.work_dir, "pkg")
        os.makedirs(path_to_pkg, exist_ok=True)
        req = self.parse_single_requirement(f"-e {path_to_pkg}")
        self.assertIsNotNone(req)
        self.assertTrue(req.is_editable)
        self.assertTrue(req.is_local)
        self.assertEqual(req.path, path_to_pkg)

    def test_parse_git_package(self):
        req = self.parse_single_requirement("git+https://github.com/user/repo.git")
        self.assertIsNotNone(req)
        self.assertEqual(req.url, "git+https://github.com/user/repo.git")


class ParseRequirementsListTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import parse_requirements_list
        self.parse_requirements_list = parse_requirements_list

    def test_parse_requirements_list(self):
        req_list = ["numpy>=1.20.0", "pandas>=1.3.0"]
        parsed_list = self.parse_requirements_list(req_list)
        self.assertEqual(len(parsed_list), 2)
        self.assertEqual(parsed_list[0].name, "numpy")
        self.assertEqual(parsed_list[1].name, "pandas")


class OpRequirementsToOpEnvSpecTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import op_requirements_to_op_env_spec
        self.op_requirements_to_op_env_spec = op_requirements_to_op_env_spec

        self.work_dir = 'tmp/test_op_requirements_to_op_env_spec/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_empty_requirements(self):
        spec = self.op_requirements_to_op_env_spec("test_op")
        self.assertEqual(len(spec.pip_pkgs), 0)

    def test_list_requirements(self):
        spec = self.op_requirements_to_op_env_spec("test_op", ["numpy>=1.20.0"])
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0"])

    def test_string_requirements(self):
        # create a temp requirements.txt file
        req_file = os.path.join(self.work_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("# comment will be ignored\nnumpy>=1.20.0\npandas>=1.3.0\n")

        spec = self.op_requirements_to_op_env_spec("test_op", req_file)
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas>=1.3.0"])

    def test_invalid_requirements_type(self):
        with self.assertRaises(ValueError):
            self.op_requirements_to_op_env_spec("test_op", 123)

    def test_recommended_requirements(self):
        spec = self.op_requirements_to_op_env_spec("test_op", ["numpy>=1.20.0"], ["pandas"])
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas"])

    def test_recommended_requirements_wo_requirements(self):
        spec = self.op_requirements_to_op_env_spec("test_op", None, ["pandas"])
        self.assertEqual(spec.pip_pkgs, ["pandas"])

    def test_recommended_requirements_overlapped(self):
        spec = self.op_requirements_to_op_env_spec("test_op", ["numpy>=1.20.0"], ["numpy", "pandas"])
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas"])


class OPEnvManagerTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import OPEnvManager, OPEnvSpec, ConflictResolveStrategy
        self.OPEnvManager = OPEnvManager
        self.OPEnvSpec = OPEnvSpec
        self.ConflictResolveStrategy = ConflictResolveStrategy

        self.work_dir = 'tmp/test_op_env_manager/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_init_with_default_values(self):
        manager = self.OPEnvManager()
        self.assertEqual(manager.min_common_dep_num_to_combine, -1)
        self.assertEqual(manager.conflict_resolve_strategy, self.ConflictResolveStrategy.SPLIT)

    def test_init_with_custom_values(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=2,
                                    conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST)
        self.assertEqual(manager.min_common_dep_num_to_combine, 2)
        self.assertEqual(manager.conflict_resolve_strategy, self.ConflictResolveStrategy.LATEST)

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValueError):
            self.OPEnvManager(min_common_dep_num_to_combine=-100)

    def test_record_and_get_op_env_spec(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        op_name = "test_op"
        
        manager.record_op_env_spec(op_name, spec)
        retrieved_spec = manager.get_op_env_spec(op_name)
        
        self.assertEqual(retrieved_spec, spec)
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_record_nonexistent_op(self):
        manager = self.OPEnvManager()
        with self.assertRaises(ValueError):
            manager.get_op_env_spec("nonexistent_op")

    def test_can_combine_op_env_specs_false_min_common(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=2)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["pandas>=1.3.0"])
        
        result = manager.can_combine_op_env_specs(spec1, spec2)
        self.assertFalse(result)

    def test_can_combine_op_env_specs_true_min_common(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "pandas>=1.3.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0", "scipy>=1.7.0"])
        
        result = manager.can_combine_op_env_specs(spec1, spec2)
        self.assertTrue(result)

    def test_can_combine_op_env_specs_different_working_dirs(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], working_dir="/path1")
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0"], working_dir="/path2")
        
        result = manager.can_combine_op_env_specs(spec1, spec2)
        self.assertFalse(result)

    def test_merge_op_env_specs_same_spec(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        
        manager.record_op_env_spec("op1", spec)
        manager.record_op_env_spec("op2", spec)
        
        self.assertEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        self.assertEqual(len(manager.hash2specs), 1)
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_merge_op_env_specs_combine_allowed(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], env_vars={"A": "1"})
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0"], env_vars={"B": "2"})
        
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        
        # Since we allow combining (common deps exist), there should be a combined spec
        self.assertEqual(len(manager.hash2ops), 1)   # Only one combined hash
        combined_spec = next(iter(manager.hash2specs.values()))
        # Check that both ops reference the same combined spec
        self.assertEqual(len(manager.hash2ops[manager.op2hash["op1"]]), 2)  # Both ops use the same hash
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_merge_op_env_specs_combine_not_allowed(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=-1)  # No combination
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["pandas>=1.3.0"])
        
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        
        # Since combination is disabled, there should be 2 separate specs
        self.assertEqual(len(manager.hash2specs), 2)
        self.assertEqual(len(manager.hash2ops), 2)
        self.assertNotEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 2)

    def test_merge_op_env_specs_combine_general(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0", "fire"])
        spec2 = self.OPEnvSpec(pip_pkgs=["pandas>=1.3.0", "numpy<2.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # Since we allow combining (common deps exist), there should be a combined spec
        self.assertEqual(len(manager.hash2ops), 1)  # Only one combined hash
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # Check that both ops reference the same combined spec
        self.assertEqual(len(manager.hash2ops[manager.op2hash["op1"]]), 2)  # Both ops use the same hash
        self.assertEqual(sorted(combined_spec.pip_pkgs), sorted(["numpy<2.0,>=1.20.0", "pandas>=1.3.0", "fire"]))
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_merge_op_env_specs_combine_with_env_vars(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], env_vars={"A": "1"})
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # Since we allow combining (common deps exist), there should be a combined spec
        self.assertEqual(len(manager.hash2ops), 1)  # Only one combined hash
        combined_spec = manager.hash2specs[manager.op2hash["op2"]]
        # Check that both ops reference the same combined spec
        self.assertEqual(len(manager.hash2ops[manager.op2hash["op1"]]), 2)  # Both ops use the same hash

        self.assertEqual(combined_spec.env_vars, {"A": "1"})
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_merge_op_env_specs_combine_with_extra_env_params(self):
        manager = self.OPEnvManager(min_common_dep_num_to_combine=1)
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"], extra_env_params={"A": "1"})
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.21.0"])
        spec3 = self.OPEnvSpec(pip_pkgs=["numpy<2.0"], extra_env_params={"B": "2"})

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        manager.record_op_env_spec("op3", spec3)

        # Since we allow combining (common deps exist), there should be a combined spec
        self.assertEqual(len(manager.hash2ops), 1)  # Only one combined hash
        combined_spec = manager.hash2specs[manager.op2hash["op3"]]
        # Check that both ops reference the same combined spec
        self.assertEqual(len(manager.hash2ops[manager.op2hash["op1"]]), 3)  # Both ops use the same hash

        self.assertEqual(combined_spec.extra_env_params, {"A": "1", "B": "2"})
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_conflict_resolution_split_strategy(self):
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.SPLIT
        )
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])  # This would cause conflict
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy<1.19.0"])  # This conflicts with above
        
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        
        # With SPLIT strategy, specs should remain separate due to conflict
        self.assertNotEqual(manager.op2hash["op1"], manager.op2hash["op2"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 2)

    def test_conflict_resolution_latest_strategy(self):
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy==1.25.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0,<1.22.0"])
        
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)
        
        # With LATEST strategy, the latest version in the spec should be used.
        # In this case the latest version is 1.25.0.
        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2specs), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy==1.25.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_conflict_resolution_latest_strategy_not_include_max(self):
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy<1.25.0,>1.23.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0,<1.22.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # With LATEST strategy, the latest version in the spec should be used.
        # In this case the latest version is the one less than but closest to 1.25.0,
        # which can be described as <.
        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2specs), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy<1.25.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_conflict_resolution_latest_strategy_without_max(self):
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>1.23.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0,<1.22.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # With LATEST strategy, the latest version in the spec should be used.
        # In this case the latest version is the one less than but closest to 1.25.0,
        # which can be described as <.
        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2specs), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_conflict_resolution_latest_strategy_allow_union(self):
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy>=1.22.0,<2.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0,<1.22.0"])

        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # With LATEST strategy, the latest version in the spec should be used.
        # In this case these two ranges can be union perfectly.
        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2specs), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy<2.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

    def test_conflict_resolution_with_arbitrary_equal(self):
        spec1 = self.OPEnvSpec(pip_pkgs=["numpy===2.0"])
        spec2 = self.OPEnvSpec(pip_pkgs=["numpy>=1.19.0,<1.22.0"])
        spec3 = self.OPEnvSpec(pip_pkgs=["numpy===1.23.0"])

        # the first one is arbitrary
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op2", spec2)

        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op1"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2ops), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy===2.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

        # the second one is arbitrary
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        manager.record_op_env_spec("op2", spec2)
        manager.record_op_env_spec("op3", spec3)

        # The exact behavior depends on the implementation of conflict resolution
        combined_spec = manager.hash2specs[manager.op2hash["op2"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2ops), 1)
        self.assertEqual(combined_spec.pip_pkgs, ["numpy===1.23.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 1)

        # both are arbitrary, should be split
        manager = self.OPEnvManager(
            min_common_dep_num_to_combine=1,
            conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST
        )
        manager.record_op_env_spec("op1", spec1)
        manager.record_op_env_spec("op3", spec3)

        # The exact behavior depends on the implementation of conflict resolution
        op1_spec = manager.hash2specs[manager.op2hash["op1"]]
        op3_spec = manager.hash2specs[manager.op2hash["op3"]]
        # We expect at least one combined spec to exist
        self.assertGreaterEqual(len(manager.hash2ops), 2)
        self.assertEqual(op1_spec.pip_pkgs, ["numpy===2.0"])
        self.assertEqual(op3_spec.pip_pkgs, ["numpy===1.23.0"])
        states = manager.print_the_current_states()
        self.assertEqual(len(states), 2)

    def test_op_to_hash_mapping(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        op_name = "test_op"
        
        manager.record_op_env_spec(op_name, spec)
        
        self.assertIn(op_name, manager.op2hash)
        self.assertIn(manager.op2hash[op_name], manager.hash2specs)
        self.assertIn(op_name, manager.hash2ops[manager.op2hash[op_name]])

    def test_multiple_ops_same_spec(self):
        manager = self.OPEnvManager()
        spec = self.OPEnvSpec(pip_pkgs=["numpy>=1.20.0"])
        op1, op2 = "test_op1", "test_op2"
        
        manager.record_op_env_spec(op1, spec)
        manager.record_op_env_spec(op2, spec)
        
        # Both ops should have the same hash
        self.assertEqual(manager.op2hash[op1], manager.op2hash[op2])
        # Both ops should be listed under the same hash
        self.assertIn(op1, manager.hash2ops[manager.op2hash[op1]])
        self.assertIn(op2, manager.hash2ops[manager.op2hash[op1]])

    def test_resolve_with_strategy(self):
        from data_juicer.ops.op_env import Requirement, ConflictResolveStrategy
        manager = self.OPEnvManager(conflict_resolve_strategy=self.ConflictResolveStrategy.SPLIT)
        
        # Test basic conflict resolution with SPLIT strategy
        req1 = Requirement(name="numpy", version=">=1.20.0")
        req2 = Requirement(name="numpy", version="<1.19.0")
        
        result = manager._resolve_with_strategy(req1, req2)
        self.assertIsNone(result, "SPLIT strategy should return None for conflicting versions")
        
        # Test no conflict case
        req3 = Requirement(name="numpy", version=">=1.20.0")
        req4 = Requirement(name="numpy", version=">=1.19.0")
        
        result = manager._resolve_with_strategy(req3, req4)
        self.assertIsNotNone(result, "Should return merged requirement when no conflict exists")
        self.assertEqual(result.name, "numpy")
        
        # Test OVERWRITE strategy
        manager_overwrite = self.OPEnvManager(conflict_resolve_strategy=self.ConflictResolveStrategy.OVERWRITE)
        result = manager_overwrite._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req2, "OVERWRITE strategy should return the second requirement")
        
        # Test LATEST strategy
        manager_latest = self.OPEnvManager(conflict_resolve_strategy=self.ConflictResolveStrategy.LATEST)
        req5 = Requirement(name="numpy", version=">=1.20.0")
        req6 = Requirement(name="numpy", version=">=1.19.0,<1.22.0")
        
        result = manager_latest._resolve_with_strategy(req5, req6)
        # With these specific versions, it should find a compatible range
        self.assertIsNotNone(result, "LATEST strategy should resolve compatible ranges")
        
        # Test with extras and markers
        req7 = Requirement(name="scipy", version=">=1.7.0", extras=["io"], markers="python_version>='3.6'")
        req8 = Requirement(name="scipy", version=">=1.8.0", extras=["optimize"])
        
        result = manager_overwrite._resolve_with_strategy(req7, req8)
        self.assertEqual(sorted(result.extras), ["io", "optimize"], "Extras should come from second requirement with OVERWRITE")
        self.assertEqual(result.markers, "python_version>='3.6'", "Markers should be combined")

        # Test with extras and markers
        req7 = Requirement(name="scipy", version=">=1.7.0", extras=["io"], markers="python_version>='3.6'")
        req8 = Requirement(name="scipy", version=">=1.8.0", markers="python_version<'3.8'")

        result = manager_overwrite._resolve_with_strategy(req7, req8)
        self.assertEqual(sorted(result.extras), ["io"],
                         "Extras should come from second requirement with OVERWRITE")
        self.assertEqual(result.markers, 'python_version >= "3.6" and python_version < "3.8"', "Markers should be combined")

        # Test with no version infos
        req1 = Requirement(name="numpy")
        req2 = Requirement(name="numpy", version="<1.19.0")

        result = manager._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req2, "Should return the second requirement when no version info is provided")

        req1 = Requirement(name="numpy", version="<1.19.0")
        req2 = Requirement(name="numpy")

        result = manager._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req1, "Should return the second requirement when no version info is provided")

        # Test with same version infos
        req1 = Requirement(name="numpy", version="<1.19.0")
        req2 = Requirement(name="numpy", version="<1.19.0")

        result = manager._resolve_with_strategy(req1, req2)
        self.assertEqual(result, req2, "Should return the any requirement when their versions are the same")


class AnalyzeLazyLoadedRequirementsTest(DataJuicerTestCaseBase):
    
    def setUp(self):
        super().setUp()
        from data_juicer.ops.op_env import analyze_lazy_loaded_requirements, analyze_lazy_loaded_requirements_for_code_file
        self.analyze_lazy_loaded_requirements = analyze_lazy_loaded_requirements
        self.analyze_lazy_loaded_requirements_for_code_file = analyze_lazy_loaded_requirements_for_code_file

        self.work_dir = 'tmp/test_analyze_lazy_loaded_requirements/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_analyze_lazy_loaded_requirements_with_LazyLoader_calls(self):
        code_content = '''
from data_juicer.utils.lazy_loader import LazyLoader

# Define some lazy-loaded packages
transformers = LazyLoader('transformers', package_name='transformers')
torch = LazyLoader('torch', package_name='torch', 
                   package_url='https://github.com/pytorch/pytorch.git')

# Another usage
numpy = LazyLoader('numpy', 'numpy')
'''
        expected_reqs = [
            'transformers',
            'torch @ https://github.com/pytorch/pytorch.git',
            'numpy'
        ]
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(sorted(result), sorted(expected_reqs))

    def test_analyze_lazy_loaded_requirements_with_check_packages_calls(self):
        code_content = '''
from data_juicer.utils.lazy_loader import LazyLoader

# Using check_packages
LazyLoader.check_packages(['numpy>=1.20.0', 'pandas>=1.3.0'])
LazyLoader.check_packages(package_specs=['torch>=1.8.0'])

# Mixed usage
LazyLoader.check_packages(['scipy'])
'''
        expected_reqs = ['numpy>=1.20.0', 'pandas>=1.3.0', 'torch>=1.8.0', 'scipy']
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(sorted(result), sorted(expected_reqs))

    def test_analyze_lazy_loaded_requirements_mixed_usage(self):
        code_content = '''
from data_juicer.utils.lazy_loader import LazyLoader

# Mixed usage of LazyLoader and check_packages
transformers = LazyLoader('transformers', package_name='transformers')
LazyLoader.check_packages(['numpy>=1.20.0'])
torch = LazyLoader('torch',
                   package_url='https://github.com/pytorch/pytorch.git')
LazyLoader.check_packages(package_specs=['pandas>=1.3.0'])
'''
        expected_reqs = [
            'transformers',
            'numpy>=1.20.0',
            'torch @ https://github.com/pytorch/pytorch.git',
            'pandas>=1.3.0'
        ]
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(sorted(result), sorted(expected_reqs))

    def test_analyze_lazy_loaded_requirements_for_code_file(self):
        # Create a temporary Python file
        code_file = os.path.join(self.work_dir, 'temp_test_code.py')
        with open(code_file, 'w') as f:
            f.write('''
from data_juicer.utils.lazy_loader import LazyLoader

# Test code for file analysis
transformers = LazyLoader('transformers', package_name='transformers')
LazyLoader.check_packages(['numpy>=1.20.0'])
''')
        
        expected_reqs = ['transformers', 'numpy>=1.20.0']
        result = self.analyze_lazy_loaded_requirements_for_code_file(code_file)
        self.assertEqual(sorted(result), sorted(expected_reqs))

    def test_analyze_lazy_loaded_requirements_empty_code(self):
        code_content = '''
# Just some comments
a = 1
b = 2
'''
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(result, [])

    def test_analyze_lazy_loaded_requirements_no_LazyLoader_calls(self):
        code_content = '''
import numpy as np
import pandas as pd

def func():
    return "no LazyLoader calls here"
'''
        result = self.analyze_lazy_loaded_requirements(code_content)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
