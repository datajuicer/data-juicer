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
        spec = self.OPEnvSpec(op_name="test_op", pip_pkgs=["numpy>=1.20.0", "pandas>=1.3.0"])
        self.assertEqual(spec.op_name, "test_op")
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas>=1.3.0"])
        self.assertEqual(spec.backend, "uv")

    def test_init_with_pip_packages_string(self):
        # create a temp requirements.txt file
        req_file = Path(self.work_dir) / "requirements.txt"
        with open(req_file, "w") as f:
            f.write("numpy>=1.20.0\npandas>=1.3.0\n")
        
        spec = self.OPEnvSpec(op_name="test_op", pip_pkgs=str(req_file))
        self.assertEqual(spec.op_name, "test_op")
        self.assertEqual(len(spec.pip_pkgs), 2)
        self.assertIn("numpy>=1.20.0", spec.pip_pkgs)
        self.assertIn("pandas>=1.3.0", spec.pip_pkgs)

    def test_init_with_env_vars(self):
        env_vars = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}
        spec = self.OPEnvSpec(op_name="test_op", env_vars=env_vars)
        self.assertEqual(spec.env_vars, env_vars)

    def test_to_dict_with_pip_packages(self):
        spec = self.OPEnvSpec(op_name="test_op", pip_pkgs=["numpy>=1.20.0"], backend="pip")
        expected = {"pip": ["numpy>=1.20.0"]}
        self.assertEqual(spec.to_dict(), expected)

    def test_to_dict_with_env_vars(self):
        env_vars = {"CUDA_VISIBLE_DEVICES": "0"}
        spec = self.OPEnvSpec(op_name="test_op", pip_pkgs=["numpy>=1.20.0"], env_vars=env_vars)
        expected = {"uv": ["numpy>=1.20.0"], "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}
        self.assertEqual(spec.to_dict(), expected)

    def test_backend_validation(self):
        with self.assertRaises(AssertionError):
            self.OPEnvSpec(op_name="test_op", backend="invalid_backend")


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
        self.assertEqual(spec.op_name, "test_op")
        self.assertIsNone(spec.pip_pkgs)

    def test_list_requirements(self):
        spec = self.op_requirements_to_op_env_spec("test_op", ["numpy>=1.20.0"])
        self.assertEqual(spec.op_name, "test_op")
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0"])

    def test_string_requirements(self):
        # create a temp requirements.txt file
        req_file = os.path.join(self.work_dir, "requirements.txt")
        with open(req_file, "w") as f:
            f.write("# comment will be ignored\nnumpy>=1.20.0\npandas>=1.3.0\n")

        spec = self.op_requirements_to_op_env_spec("test_op", req_file)
        self.assertEqual(spec.op_name, "test_op")
        self.assertEqual(spec.pip_pkgs, ["numpy>=1.20.0", "pandas>=1.3.0"])

    def test_invalid_requirements_type(self):
        with self.assertRaises(ValueError):
            self.op_requirements_to_op_env_spec("test_op", 123)


if __name__ == '__main__':
    unittest.main()
