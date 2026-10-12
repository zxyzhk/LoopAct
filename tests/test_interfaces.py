"""Dependency-free import routing, namespace and serialization-contract checks."""
import ast
import importlib
import json
from pathlib import Path
import sys
import tomllib
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ['policy', 'action_proposal_planner', 'visual_refinement_network', 'visual_observation_encoder']
PROPERTIES = {'LoopAct': {'action_proposal_planner': 'action_head', 'visual_refinement_network': 'feedback_action', 'action_context_cache': '_action_context_cache'}, 'ActionProposalPlanner': {'action_context_encoder': 'action_encoder'}, 'VisualRefinementNetwork': {'cross_modal_fusion_block': 'action_decoder_observe'}, 'CrossModalFusionBlock': {'visual_observation_encoder': 'obs_encoder'}}


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class InterfaceTests(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for p in ROOT.rglob('*.py'):
            with self.subTest(file=str(p.relative_to(ROOT))):
                compile(p.read_text(encoding='utf-8'), str(p), 'exec')

    def test_package_discovery(self):
        from setuptools import find_packages
        packages = find_packages(where=str(ROOT))
        self.assertIn('loopact.model', packages)
        metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertEqual(metadata['project']['name'], 'loopact')
        self.assertIn('loopact*', metadata['tool']['setuptools']['packages']['find']['include'])

    def test_lightweight_package_import(self):
        import loopact
        self.assertEqual(loopact.__version__, '0.1.0')
        self.assertIn('LoopAct', loopact.__all__)
        with self.assertRaises(AttributeError):
            getattr(loopact, 'nonexistent_component')

    def test_package_exports_resolve_to_implementation_classes(self):
        import loopact
        modules = {}
        expected = {}
        for name, (filename, attribute) in loopact._EXPORTS.items():
            key = 'loopact.model.' + filename
            modules.setdefault(key, module(key))
            value = type(attribute, (), {})
            setattr(modules[key], attribute, value)
            expected[name] = value
        with patch.dict(sys.modules, modules):
            for name, value in expected.items():
                self.assertIs(getattr(loopact, name), value)

    def test_fetcher_baseline_and_feedback_routing(self):
        from utils.module_fetcher import ModuleFetcher
        baseline_model, baseline_head, new_model, new_head, refiner = [object() for _ in range(5)]
        loaded = {
            'gr00t.model.gr00t_n1': module('old.model', GR00T_N1_5=baseline_model),
            'gr00t.model.action_head.flow_matching_action_head': module('old.head', FlowmatchingActionHead=baseline_head),
            'loopact.model.policy': module('new.model', LoopAct=new_model),
            'loopact.model.action_proposal_planner': module('new.head', ActionProposalPlanner=new_head),
            'loopact.model.visual_refinement_network': module('new.refiner', VisualRefinementNetwork=refiner),
        }
        def load(name):
            return loaded.get(name, module(name))
        with patch('utils.module_fetcher.importlib.import_module', side_effect=load):
            for mode, model, head in [('baseline', baseline_model, baseline_head), ('feedback', new_model, new_head)]:
                fetcher = ModuleFetcher(mode)
                self.assertIs(fetcher.Model, model)
                self.assertIs(fetcher.GR00T_N1_5, model)
                self.assertIs(fetcher.ActionPlanner, head)
                self.assertIs(fetcher.FlowmatchingActionHead, head)
                self.assertIs(fetcher.VisualRefinementNetwork, refiner if mode == 'feedback' else None)
                self.assertIs(fetcher.FeedbackAction, refiner if mode == 'feedback' else None)

    def test_accessors_do_not_register_duplicate_parameters(self):
        for filename in MODEL_FILES:
            tree = ast.parse((ROOT / f'loopact/model/{filename}.py').read_text(encoding='utf-8'))
            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                expected = PROPERTIES.get(cls.name, {})
                if not expected:
                    continue
                functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in expected]
                namespace = {}
                exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<properties>', 'exec'), namespace)
                shell = type(cls.name, (), {k: v for k, v in namespace.items() if k in expected})()
                for public, storage in expected.items():
                    value = object()
                    setattr(shell, storage, value)
                    before = dict(vars(shell))
                    self.assertIs(getattr(shell, public), value)
                    self.assertEqual(vars(shell), before)
                    with self.assertRaises(AttributeError):
                        setattr(shell, public, object())

    def test_checkpoint_storage_contract(self):
        expected = {
            'policy': {'action_head', 'feedback_action'},
            'action_proposal_planner': {'action_encoder'},
            'visual_refinement_network': {'obs_encoder', 'action_decoder_observe'},
        }
        for filename, names in expected.items():
            tree = ast.parse((ROOT / f'loopact/model/{filename}.py').read_text(encoding='utf-8'))
            assigned = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == 'self'}
            self.assertTrue(names <= assigned, (filename, names - assigned))
            for public in ['action_proposal_planner', 'visual_refinement_network', 'action_context_encoder', 'visual_observation_encoder', 'cross_modal_fusion_block']:
                self.assertNotIn(public, assigned)

    def test_model_components_are_defined(self):
        expected = {'LoopAct', 'LoopActConfig', 'ActionProposalPlanner',
                    'ActionProposalPlannerConfig', 'ActionContextEncoder',
                    'VisualRefinementNetwork', 'CrossModalFusionBlock',
                    'VisualObservationEncoder'}
        defined = set()
        for filename in MODEL_FILES:
            tree = ast.parse((ROOT / f'loopact/model/{filename}.py').read_text(encoding='utf-8'))
            defined.update(n.name for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertTrue(expected <= defined)


if __name__ == '__main__':
    unittest.main()
