import pytest
import torch
import json
from dataclasses import asdict
from argparse import Namespace
from unittest.mock import patch

# Import implementations
from src.model.model import CoExVQA
from src.model.config.presets import get_base_config



@pytest.mark.slow
def test_save_checkpoint(sample_model, sample_args, tmp_path):
    """
    Test comprehensive checkpoint saving functionality.

    Verifies:
    - Directory structure is created correctly
    - checkpoint.pt contains all required fields
    - args.json is created and formatted correctly
    - Backbone is saved in HuggingFace format
    - Model config is saved properly
    """
    checkpoint_dir = tmp_path / "checkpoint"

    # Save checkpoint with all parameters
    sample_model.save_checkpoint(
        str(checkpoint_dir),
        epoch=5,
        args=sample_args
    )

    # Verify directory structure
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "checkpoint.pt").exists()
    assert (checkpoint_dir / "args.json").exists()
    assert (checkpoint_dir / "backbone").exists()

    # Verify checkpoint.pt contents
    checkpoint = torch.load(checkpoint_dir / "checkpoint.pt", map_location='cpu')
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "epoch" in checkpoint
    assert "args" in checkpoint
    assert checkpoint["epoch"] == 5
    assert checkpoint["args"]["learning_rate"] == 1e-4
    assert checkpoint["args"]["batch_size"] == 8

    # Verify args.json
    with open(checkpoint_dir / "args.json", 'r') as f:
        args_dict = json.load(f)
    assert args_dict["learning_rate"] == 1e-4
    assert args_dict["batch_size"] == 8
    assert args_dict["num_epochs"] == 10

    # Verify backbone HuggingFace format
    backbone_dir = checkpoint_dir / "backbone"
    assert (backbone_dir / "config.json").exists()
    assert (backbone_dir / "model.safetensors").exists() or (backbone_dir / "pytorch_model.bin").exists()
    assert (backbone_dir / "preprocessor_config.json").exists()

    # Verify config was saved correctly
    assert checkpoint["config"]["d_model"] == 768
    assert checkpoint["config"]["backbone_model"] == "google/pix2struct-docvqa-base"
    assert checkpoint["config"]["backbone_freeze_mode"] == "frozen"


@pytest.mark.slow
def test_from_pretrained(sample_model, sample_args, tmp_path):
    """
    Test comprehensive checkpoint loading functionality.

    Verifies:
    - Model can be loaded from saved checkpoint
    - All weights are restored correctly
    - Config is preserved
    - Backbone is loaded from saved directory
    - Args are loaded correctly
    - Multiple save/load cycles work correctly
    """
    checkpoint_dir = tmp_path / "checkpoint"

    # Save the original model
    sample_model.save_checkpoint(
        str(checkpoint_dir),
        epoch=10,
        args=sample_args
    )

    # Get original state for comparison
    original_state = sample_model.state_dict()
    original_config = sample_model.config

    # Load the model
    loaded_model = CoExVQA.from_pretrained(str(checkpoint_dir))

    # Verify model was loaded successfully
    assert loaded_model is not None
    assert isinstance(loaded_model, CoExVQA)

    # Verify all weights match exactly
    loaded_state = loaded_model.state_dict()
    assert original_state.keys() == loaded_state.keys()

    for key in original_state.keys():
        assert torch.allclose(original_state[key], loaded_state[key]), \
            f"Weights mismatch for {key}"

    # Verify config is preserved
    assert loaded_model.config.d_model == original_config.d_model
    assert loaded_model.config.backbone_model == original_config.backbone_model
    assert loaded_model.config.backbone_freeze_mode == original_config.backbone_freeze_mode
    assert loaded_model.config.max_patches == original_config.max_patches


@pytest.mark.slow
def test_from_pretrained_missing_args_raises_error(sample_model, tmp_path):
    """
    Test that from_pretrained raises ValueError when args are missing.

    Verifies error handling when args are not provided and not found in checkpoint.
    """
    checkpoint_dir = tmp_path / "checkpoint"

    # Save without args
    sample_model.save_checkpoint(str(checkpoint_dir))

    # Manually remove args from checkpoint and args.json
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    if "args" in checkpoint:
        del checkpoint["args"]
    torch.save(checkpoint, checkpoint_file)

    args_file = checkpoint_dir / "args.json"
    if args_file.exists():
        args_file.unlink()

    # Should raise ValueError
    with pytest.raises(ValueError, match="args not provided"):
        CoExVQA.from_pretrained(str(checkpoint_dir))


@pytest.mark.slow
def test_from_pretrained_missing_checkpoint_raises_error(tmp_path):
    """
    Test that from_pretrained raises FileNotFoundError for nonexistent checkpoint.

    Verifies error handling for missing checkpoint files.
    """
    nonexistent_dir = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        CoExVQA.from_pretrained(str(nonexistent_dir))


@pytest.mark.slow
def test_fusion_modes_serialization(tmp_path):
    """
    Test that fusion_modes are correctly serialized and deserialized.

    This test verifies the fix for the bug where fusion_modes were not
    correctly saved in args.json due to malformed type annotation (List [str]
    instead of List[str]).

    Verifies:
    - fusion_modes are correctly saved in checkpoint config
    - fusion_modes are correctly loaded from checkpoint
    - Both answer and question projector configs preserve fusion_modes
    """

    # Get the BASE preset config which has fusion_modes = ["film", "cross_attention"]
    config = get_base_config()

    # Verify the config has the expected fusion_modes
    assert config.answer_projector_config.fusion_modes == ["film", "cross_attention"], \
        "BASE preset should have fusion_modes=['film', 'cross_attention']"
    assert config.question_projector_config.fusion_modes == ["film", "cross_attention"], \
        "BASE preset should have fusion_modes=['film', 'cross_attention']"

    # Test that asdict serialization works correctly
    config_dict = asdict(config)

    # Check that fusion_modes are present in the serialized config
    assert "answer_projector_config" in config_dict
    assert "fusion_modes" in config_dict["answer_projector_config"]
    assert config_dict["answer_projector_config"]["fusion_modes"] == ["film", "cross_attention"], \
        "Answer projector fusion_modes not correctly serialized"

    assert "question_projector_config" in config_dict
    assert "fusion_modes" in config_dict["question_projector_config"]
    assert config_dict["question_projector_config"]["fusion_modes"] == ["film", "cross_attention"], \
        "Question projector fusion_modes not correctly serialized"

    # Create a model with the BASE preset
    model = CoExVQA(config=config)

    # Create sample args for saving
    sample_args = Namespace(
        learning_rate=1e-4,
        batch_size=8,
    )

    # Save the model
    checkpoint_dir = tmp_path / "fusion_test"
    model.save_checkpoint(str(checkpoint_dir), args=sample_args)

    # Load the checkpoint and verify fusion_modes are preserved
    checkpoint = torch.load(checkpoint_dir / "checkpoint.pt", map_location='cpu')

    assert "config" in checkpoint
    assert "answer_projector_config" in checkpoint["config"]
    assert "fusion_modes" in checkpoint["config"]["answer_projector_config"]
    assert checkpoint["config"]["answer_projector_config"]["fusion_modes"] == ["film", "cross_attention"], \
        "Answer projector fusion_modes not correctly saved in checkpoint"

    assert "question_projector_config" in checkpoint["config"]
    assert "fusion_modes" in checkpoint["config"]["question_projector_config"]
    assert checkpoint["config"]["question_projector_config"]["fusion_modes"] == ["film", "cross_attention"], \
        "Question projector fusion_modes not correctly saved in checkpoint"

    # Load the model from checkpoint
    loaded_model = CoExVQA.from_pretrained(str(checkpoint_dir))

    # Verify fusion_modes are correctly restored
    assert loaded_model.config.answer_projector_config.fusion_modes == ["film", "cross_attention"], \
        "Answer projector fusion_modes not correctly loaded from checkpoint"
    assert loaded_model.config.question_projector_config.fusion_modes == ["film", "cross_attention"], \
        "Question projector fusion_modes not correctly loaded from checkpoint"


@pytest.mark.slow
def test_preset_config_values_preserved_in_checkpoint(tmp_path):
    """
    Test that model architecture args are filtered out when saving.

    Since checkpoint["config"] contains the complete model configuration,
    model architecture args (a_*, q_*, gating_*, d_model) are filtered out
    from checkpoint["args"] to avoid confusion and keep args.json clean.

    Verifies:
    - Model architecture args are NOT saved in checkpoint["args"]
    - Training hyperparameters ARE saved in checkpoint["args"]
    - Config contains the actual model architecture from the preset
    - Loaded model uses config from checkpoint
    """

    # Get the BASE preset config which has init_w=0.65, init_h=0.22
    config = get_base_config()

    # Verify the preset has the expected values (different from argparse defaults)
    assert config.answer_projector_config.init_w == 0.65, \
        "BASE preset should have init_w=0.65"
    assert config.answer_projector_config.init_h == 0.22, \
        "BASE preset should have init_h=0.22"

    # Create a model with the BASE preset
    model = CoExVQA(config=config)

    # Simulate argparse args with both training hyperparameters and model architecture args
    sample_args = Namespace(
        learning_rate=1e-4,  # Training hyperparameter - should be saved
        batch_size=8,         # Training hyperparameter - should be saved
        a_init_w=0.07,        # Model architecture arg - should be filtered out
        a_init_h=0.05,        # Model architecture arg - should be filtered out
        config="base",        # Training arg - should be saved
    )

    # Save the model with args
    checkpoint_dir = tmp_path / "preset_test"
    model.save_checkpoint(str(checkpoint_dir), args=sample_args)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_dir / "checkpoint.pt", map_location='cpu')

    # Model architecture args should be filtered out
    assert "a_init_w" not in checkpoint["args"], \
        "Model architecture args should be filtered out from saved args"
    assert "a_init_h" not in checkpoint["args"], \
        "Model architecture args should be filtered out from saved args"

    # Training hyperparameters should be saved
    assert checkpoint["args"]["learning_rate"] == 1e-4, \
        "Training hyperparameters should be saved"
    assert checkpoint["args"]["batch_size"] == 8, \
        "Training hyperparameters should be saved"

    # Config should have the actual model values from the preset
    assert checkpoint["config"]["answer_projector_config"]["init_w"] == 0.65, \
        "Config should have preset value (0.65)"
    assert checkpoint["config"]["answer_projector_config"]["init_h"] == 0.22, \
        "Config should have preset value (0.22)"

    # Load the model from checkpoint
    loaded_model = CoExVQA.from_pretrained(str(checkpoint_dir))

    # Verify the loaded model uses config from checkpoint
    assert loaded_model.config.answer_projector_config.init_w == 0.65, \
        "Loaded model should use config from checkpoint (0.65)"
    assert loaded_model.config.answer_projector_config.init_h == 0.22, \
        "Loaded model should use config from checkpoint (0.22)"


@pytest.mark.slow
def test_from_hf(sample_model, sample_args, tmp_path):
    """
    Test that from_hf() downloads and loads the model correctly.

    Uses unittest.mock to patch snapshot_download so no network
    call is made. Verifies that from_hf delegates to from_pretrained
    and returns a valid CoExVQA instance with the correct weights.
    """
    checkpoint_dir = tmp_path / "hf_checkpoint"

    # Save a real checkpoint to act as the "downloaded" snapshot
    sample_model.save_checkpoint(str(checkpoint_dir), args=sample_args)
    original_state = sample_model.state_dict()

    dummy_font = tmp_path / "Arial.TTF"
    dummy_font.write_bytes(b"")

    with patch(
        "src.model.model.snapshot_download",
        return_value=str(checkpoint_dir),
    ) as mock_download, patch(
        "src.model.model.hf_hub_download",
        return_value=str(dummy_font),
    ):
        loaded_model = CoExVQA.from_hf(
            repo_id="indrehus/CoExVQA",
            cache_dir=str(tmp_path / "cache"),
        )

    # Verify snapshot_download was called with the correct arguments
    mock_download.assert_called_once_with(
        repo_id="indrehus/CoExVQA",
        cache_dir=str(tmp_path / "cache"),
    )

    # Verify the returned object is a valid model
    assert isinstance(loaded_model, CoExVQA)

    # Verify weights were restored correctly
    loaded_state = loaded_model.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], loaded_state[key]), \
            f"Weight mismatch for {key}"
