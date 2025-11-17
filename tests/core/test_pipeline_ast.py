import unittest
from data_juicer.core.pipeline_ast import PipelineAST, OpType

class TestPipelineAST(unittest.TestCase):
    def setUp(self):
        self.ast = PipelineAST()
        
    def test_build_from_config(self):
        config = {
            'process': [
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'clean_copyright_mapper': {}},
                {'alphanumeric_filter': {'min_ratio': 0.25, 'max_ratio': 1.0}}
            ]
        }
        
        self.ast.build_from_config(config)
        
        # Test root node
        self.assertIsNotNone(self.ast.root)
        self.assertEqual(self.ast.root.name, "root")
        self.assertEqual(self.ast.root.op_type, OpType.MAPPER)
        
        # Test operation chain
        chain = self.ast.get_operation_chain()
        self.assertEqual(len(chain), 3)
        
        # Test first operation
        self.assertEqual(chain[0].name, "language_id_score_filter")
        self.assertEqual(chain[0].op_type, OpType.FILTER)
        self.assertEqual(chain[0].config, {'lang': 'zh', 'min_score': 0.8})
        
        # Test second operation
        self.assertEqual(chain[1].name, "clean_copyright_mapper")
        self.assertEqual(chain[1].op_type, OpType.MAPPER)
        self.assertEqual(chain[1].config, {})
        
        # Test third operation
        self.assertEqual(chain[2].name, "alphanumeric_filter")
        self.assertEqual(chain[2].op_type, OpType.FILTER)
        self.assertEqual(chain[2].config, {'min_ratio': 0.25, 'max_ratio': 1.0})
    
    def test_visualize(self):
        config = {
            'process': [
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'clean_copyright_mapper': {}}
            ]
        }
        
        self.ast.build_from_config(config)
        visualization = self.ast.visualize()
        
        expected = """Pipeline:
└── root (mapper)
    └── clean_copyright_mapper (mapper)
        └── language_id_score_filter (filter)
"""
        self.assertEqual(visualization, expected)
    
    def test_to_dict(self):
        config = {
            'process': [
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'clean_copyright_mapper': {}}
            ]
        }
        
        self.ast.build_from_config(config)
        ast_dict = self.ast.to_dict()
        
        expected = {
            'name': 'root',
            'type': 'mapper',
            'config': {},
            'children': [{
                'name': 'language_id_score_filter',
                'type': 'filter',
                'config': {'lang': 'zh', 'min_score': 0.8},
                'children': [{
                    'name': 'clean_copyright_mapper',
                    'type': 'mapper',
                    'config': {},
                    'children': []
                }]
            }]
        }
        self.assertEqual(ast_dict, expected)
    
    def test_empty_config(self):
        config = {'process': []}
        self.ast.build_from_config(config)
        self.assertIsNone(self.ast.root)
        self.assertEqual(self.ast.visualize(), "Empty pipeline")
    
    def test_invalid_config(self):
        config = {}
        with self.assertRaises(ValueError):
            self.ast.build_from_config(config)
    
    def test_validate_dependencies(self):
        config = {
            'process': [
                {'clean_copyright_mapper': {}},
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'document_deduplicator': {}},
                {'text_length_filter': {'min_len': 10, 'max_len': 1000}}
            ]
        }
        
        self.ast.build_from_config(config)
        invalid_deps = self.ast._validate_dependencies()
        self.assertEqual(len(invalid_deps), 0)  # This pipeline should be valid
        
        # Test invalid pipeline
        invalid_config = {
            'process': [
                {'document_deduplicator': {}},  # Deduplicator before mapper
                {'clean_copyright_mapper': {}}
            ]
        }
        
        self.ast.build_from_config(invalid_config)
        invalid_deps = self.ast._validate_dependencies()
        self.assertGreater(len(invalid_deps), 0)  # Should have invalid dependencies
    
    def test_optimize_operation_order(self):
        config = {
            'process': [
                {'text_length_filter': {'min_len': 10, 'max_len': 1000}},
                {'clean_copyright_mapper': {}},
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'document_deduplicator': {}}
            ]
        }
        
        self.ast.build_from_config(config)
        self.ast._optimize_operation_order()
        
        chain = self.ast.get_operation_chain()
        self.assertEqual(len(chain), 4)
        
        # Check order: Mapper -> Filter -> Deduplicator
        self.assertEqual(chain[0].name, "clean_copyright_mapper")
        self.assertEqual(chain[0].op_type, OpType.MAPPER)
        
        self.assertEqual(chain[1].name, "text_length_filter")
        self.assertEqual(chain[1].op_type, OpType.FILTER)
        
        self.assertEqual(chain[2].name, "language_id_score_filter")
        self.assertEqual(chain[2].op_type, OpType.FILTER)
        
        self.assertEqual(chain[3].name, "document_deduplicator")
        self.assertEqual(chain[3].op_type, OpType.DEDUPLICATOR)
    
    def test_merge_compatible_operations(self):
        config = {
            'process': [
                {'text_length_filter': {'min_len': 10, 'max_len': 1000}},
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'clean_copyright_mapper': {}},
                {'text_clean_mapper': {}}
            ]
        }
        
        self.ast.build_from_config(config)
        self.ast._merge_compatible_operations()
        
        chain = self.ast.get_operation_chain()
        self.assertEqual(len(chain), 3)  # Two filters should be merged
        
        # Check merged operations
        self.assertEqual(chain[0].name, "merged_text_length_filter_language_id_score_filter")
        self.assertEqual(chain[0].op_type, OpType.FILTER)
        self.assertEqual(chain[0].config, {
            'min_len': 10,
            'max_len': 1000,
            'lang': 'zh',
            'min_score': 0.8
        })
        
        self.assertEqual(chain[1].name, "merged_clean_copyright_mapper_text_clean_mapper")
        self.assertEqual(chain[1].op_type, OpType.MAPPER)
    
    def test_full_optimization(self):
        config = {
            'process': [
                {'text_length_filter': {'min_len': 10, 'max_len': 1000}},
                {'language_id_score_filter': {'lang': 'zh', 'min_score': 0.8}},
                {'clean_copyright_mapper': {}},
                {'text_clean_mapper': {}},
                {'document_deduplicator': {}}
            ]
        }
        
        self.ast.build_from_config(config)
        self.ast.optimize()
        
        chain = self.ast.get_operation_chain()
        self.assertEqual(len(chain), 3)  # Two filters and two mappers should be merged
        
        # Check final order and merged operations
        self.assertEqual(chain[0].name, "merged_clean_copyright_mapper_text_clean_mapper")
        self.assertEqual(chain[0].op_type, OpType.MAPPER)
        
        self.assertEqual(chain[1].name, "merged_text_length_filter_language_id_score_filter")
        self.assertEqual(chain[1].op_type, OpType.FILTER)
        
        self.assertEqual(chain[2].name, "document_deduplicator")
        self.assertEqual(chain[2].op_type, OpType.DEDUPLICATOR)

if __name__ == '__main__':
    unittest.main() 