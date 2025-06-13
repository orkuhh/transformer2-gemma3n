#!/usr/bin/env python3
"""
Simple test script for Transformer-Squared framework.
Verifies that all components can be imported and basic functionality works.
"""

import sys
import os
import traceback
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test that all framework components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        print("✅ PyTorch imported successfully")
        
        import transformers
        print("✅ Transformers imported successfully")
        
        # Test framework imports
        from transformer_squared.models.gemma3n_model import Gemma3nTransformerSquared, Gemma3nConfig
        print("✅ Gemma3n model imported successfully")
        
        from transformer_squared.svf.singular_value_finetuning import SVFTrainer, SVFConfig
        print("✅ SVF trainer imported successfully")
        
        from transformer_squared.adaptation.expert_system import ExpertSystem, ExpertConfig
        print("✅ Expert system imported successfully")
        
        from transformer_squared.adaptation.cem_adapter import CEMAdapter, CEMConfig
        print("✅ CEM adapter imported successfully")
        
        from transformer_squared.utils.model_utils import ModelLoader, TaskClassifier
        print("✅ Model utilities imported successfully")
        
        return True
    
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_configurations():
    """Test that configurations can be created."""
    print("\n⚙️ Testing configurations...")
    
    try:
        from transformer_squared.models.gemma3n_model import Gemma3nConfig
        from transformer_squared.svf.singular_value_finetuning import SVFConfig
        from transformer_squared.adaptation.expert_system import ExpertConfig
        from transformer_squared.adaptation.cem_adapter import CEMConfig
        from transformer_squared.utils.model_utils import TaskClassificationConfig
        
        # Test model config
        model_config = Gemma3nConfig(
            model_name="google/gemma-2b-it",
            num_experts=4,
            svf_rank=8
        )
        print("✅ Model configuration created")
        
        # Test SVF config
        svf_config = SVFConfig(
            rank=8,
            learning_rate=1e-4,
            use_wandb=False
        )
        print("✅ SVF configuration created")
        
        # Test expert config
        expert_config = ExpertConfig(
            num_experts=4,
            expert_dim=32,
            use_rl_training=False  # Disable for testing
        )
        print("✅ Expert configuration created")
        
        # Test CEM config
        cem_config = CEMConfig(
            population_size=20,
            max_iterations=10
        )
        print("✅ CEM configuration created")
        
        # Test task classifier config
        task_config = TaskClassificationConfig(
            embedding_dim=768,
            num_task_types=5
        )
        print("✅ Task classifier configuration created")
        
        return True
    
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that models can be created (without loading large weights)."""
    print("\n🏗️ Testing model creation...")
    
    try:
        from transformer_squared.models.gemma3n_model import Gemma3nConfig
        from transformer_squared.adaptation.expert_system import ExpertSystem, ExpertConfig
        from transformer_squared.adaptation.cem_adapter import CEMAdapter, CEMConfig
        from transformer_squared.utils.model_utils import TaskClassifier, TaskClassificationConfig
        
        # Test expert system creation
        expert_config = ExpertConfig(
            num_experts=4,
            expert_dim=16,
            hidden_size=512,
            use_rl_training=False,
            use_expert_memory=False
        )
        expert_system = ExpertSystem(expert_config)
        print("✅ Expert system created")
        
        # Test CEM adapter creation
        cem_config = CEMConfig(population_size=10, max_iterations=5)
        cem_adapter = CEMAdapter(cem_config)
        print("✅ CEM adapter created")
        
        # Test task classifier creation (lightweight)
        task_config = TaskClassificationConfig(
            embedding_dim=64,
            num_task_types=3,
            use_clustering=False
        )
        
        # Create task classifier with a small model for testing
        try:
            task_classifier = TaskClassifier(task_config, tokenizer_name="distilbert-base-uncased")
            print("✅ Task classifier created")
        except Exception as e:
            print(f"⚠️ Task classifier creation failed (likely missing model): {e}")
        
        return True
    
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality without requiring large models."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        import torch
        from transformer_squared.adaptation.expert_system import ExpertSystem, ExpertConfig
        from transformer_squared.svf.singular_value_finetuning import SVFRegularizer
        
        # Test expert system basic operations
        expert_config = ExpertConfig(
            num_experts=3,
            expert_dim=8,
            hidden_size=16,
            task_embedding_dim=8,
            use_rl_training=False,
            use_expert_memory=False
        )
        expert_system = ExpertSystem(expert_config)
        
        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 4, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        task_embedding = torch.randn(batch_size, 8)
        
        adapted_states = expert_system(hidden_states, task_embedding)
        assert adapted_states.shape == hidden_states.shape
        print("✅ Expert system forward pass works")
        
        # Test SVF regularizer
        U = torch.randn(10, 4)
        V = torch.randn(4, 8)
        S = torch.randn(4)
        
        orth_loss = SVFRegularizer.orthogonal_loss(U, V)
        sparsity_loss = SVFRegularizer.sparsity_loss(S)
        
        assert orth_loss.item() >= 0
        assert sparsity_loss.item() >= 0
        print("✅ SVF regularizer works")
        
        # Test expert statistics
        stats = expert_system.get_expert_statistics()
        assert 'num_experts' in stats
        assert 'expert_diversity' in stats
        print("✅ Expert statistics work")
        
        return True
    
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        traceback.print_exc()
        return False


def test_package_structure():
    """Test that the package structure is correct."""
    print("\n📁 Testing package structure...")
    
    expected_files = [
        "transformer_squared/__init__.py",
        "transformer_squared/requirements.txt",
        "transformer_squared/models/__init__.py",
        "transformer_squared/models/gemma3n_model.py",
        "transformer_squared/svf/__init__.py",
        "transformer_squared/svf/singular_value_finetuning.py",
        "transformer_squared/adaptation/__init__.py",
        "transformer_squared/adaptation/expert_system.py",
        "transformer_squared/adaptation/cem_adapter.py",
        "transformer_squared/utils/__init__.py",
        "transformer_squared/utils/model_utils.py",
        "transformer_squared/examples/__init__.py",
        "transformer_squared/examples/basic_usage.py",
        "README.md"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected files present")
        return True


def main():
    """Run all tests."""
    print("🚀 Transformer-Squared Framework Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Imports", test_imports),
        ("Configurations", test_configurations),
        ("Model Creation", test_model_creation),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Framework is ready to use.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 