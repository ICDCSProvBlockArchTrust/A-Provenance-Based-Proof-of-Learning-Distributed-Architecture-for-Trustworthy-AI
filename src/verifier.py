import argparse
import hashlib
import json
import os
import random
import re

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from component_registry import (
    create_criterion,
    create_dataloader,
    create_model,
    create_optimizer,
)

TRUSTED_MODEL_REGISTRY_DIR = "./trusted_models"

# --- Helper Functions ---


def convert_str_to_type(s: str):
    """Converts a string from provenance to its original type."""
    s = s.strip()
    if s == "None":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.startswith("[") and s.endswith("]"):
        try:
            return json.loads(s.replace("'", '"'))
        except json.JSONDecodeError:
            return s
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def unflatten_dict(flat_dict: dict) -> dict:
    """Converts a flattened dictionary back to a nested one."""
    unflattened = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = unflattened
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return unflattened


def parse_prov_graph_to_config(prov_graph_path: str) -> dict:
    """Parses a prov4ml graph to reconstruct the experiment config and verification steps."""
    print("--- Parsing Provenance Graph for Verification Steps ---")
    if not os.path.exists(prov_graph_path):
        raise FileNotFoundError(f"Provenance graph not found at: {prov_graph_path}")
    with open(prov_graph_path, "r") as f:
        prov_data = json.load(f)

    flat_config = {}
    for key, entity_data in prov_data.get("entity", {}).items():
        if isinstance(entity_data, dict) and "prov-ml:parameter_value" in entity_data:
            flat_config[key] = entity_data["prov-ml:parameter_value"]
    config = unflatten_dict({k: convert_str_to_type(v) for k, v in flat_config.items()})

    checkpoints, indices_params, data_hashes, model_hashes = {}, {}, {}, {}
    cp_pattern = re.compile(r"checkpoint_epoch_(\d+)_step_(\d+)")
    idx_param_pattern = re.compile(r"checkpoint\.epoch_(\d+)_step_(\d+)\.indices")
    hash_param_pattern = re.compile(r"checkpoint\.epoch_(\d+)_step_(\d+)\.data_hash")
    m_hash_param_pattern = re.compile(r"checkpoint\.epoch_(\d+)_step_(\d+)\.model_hash")

    for key, entity_data in prov_data.get("entity", {}).items():
        if "prov-ml:artifact_path" in entity_data:
            path = entity_data["prov-ml:artifact_path"]
            match_cp = cp_pattern.search(key)
            if match_cp:
                checkpoints[(int(match_cp.group(1)), int(match_cp.group(2)))] = path
            if "initial_weights.pt" in key:
                config.setdefault("artifacts", {})["initial_weights"] = path
        if "prov-ml:parameter_value" in entity_data:
            value = entity_data["prov-ml:parameter_value"]
            match_idx = idx_param_pattern.match(key)
            if match_idx:
                indices_params[(int(match_idx.group(1)), int(match_idx.group(2)))] = (
                    json.loads(value)
                )
            match_hash = hash_param_pattern.match(key)
            if match_hash:
                data_hashes[(int(match_hash.group(1)), int(match_hash.group(2)))] = (
                    value
                )
            match_m_hash = m_hash_param_pattern.match(key)
            if match_m_hash:
                model_hashes[
                    (int(match_m_hash.group(1)), int(match_m_hash.group(2)))
                ] = value

    verification_steps = []
    for key in sorted(checkpoints.keys()):
        if (
            key not in indices_params
            or key not in data_hashes
            or key not in model_hashes
        ):
            raise RuntimeError(
                f"Checkpoint {key} is missing an index, data_hash, or model_hash parameter."
            )
        verification_steps.append(
            {
                "epoch": key[0],
                "step": key[1],
                "model_path": checkpoints[key],
                "indices": indices_params[key],
                "data_hash": data_hashes[key],
                "model_hash": model_hashes[key],
            }
        )
    config["verification_steps"] = verification_steps
    print(f"Found {len(verification_steps)} intra-epoch verification steps.")
    return config


def set_seed(seed: int = 42):
    """Sets all random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_l2_distance(state_dict1: dict, state_dict2: dict) -> float:
    """Calculates the L2 distance between two model state dictionaries to check for equality."""
    total_diff_sq = 0.0
    for key in state_dict1.keys():
        p1 = state_dict1[key].to(torch.float32)
        p2 = state_dict2[key].to(torch.float32)
        total_diff_sq += torch.sum((p1 - p2) ** 2).item()
    return np.sqrt(total_diff_sq)


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper to ensure dataset provides indices, mirroring the training script."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)


def perform_cross_run_uniqueness_check(verified_states: list, comparison_dir: str):
    """
    Checks if the verified sequence of checkpoints is a subsequence of any
    past run found in the comparison directory.
    """
    print("\n--- Performing Final Cross-Run Uniqueness Check ---")
    if not os.path.isdir(comparison_dir):
        print(
            f"Warning: Comparison directory not found at '{comparison_dir}'. Skipping check."
        )
        return

    threshold = 1e-6
    num_verified = len(verified_states)

    # Iterate through each past run in the comparison directory
    for run_id in os.listdir(comparison_dir):
        run_path = os.path.join(comparison_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        checkpoint_dir = os.path.join(run_path, "checkpoints")
        if not os.path.isdir(checkpoint_dir):
            continue

        print(f"Comparing against past run: {run_id}")

        try:
            existing_files = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
                key=lambda f: tuple(map(int, re.findall(r"\d+", f))),
            )
            existing_states = [
                torch.load(os.path.join(checkpoint_dir, f))["model_state_dict"]
                for f in existing_files
            ]
        except (ValueError, FileNotFoundError):
            print(f"  - Could not load or sort checkpoints for run {run_id}. Skipping.")
            continue

        # Use a sliding window to check for a subsequence match
        if len(existing_states) < num_verified:
            continue
        for i in range(len(existing_states) - num_verified + 1):
            is_match = True
            for j in range(num_verified):
                distance = calculate_l2_distance(
                    verified_states[j], existing_states[i + j]
                )
                if distance >= threshold:
                    is_match = False
                    break
            if is_match:
                raise AssertionError(
                    f"UNIQUENESS CHECK FAILED: This run is an exact duplicate of a segment from run ID '{run_id}'."
                )

    print("Uniqueness Check PASSED: This run is not a duplicate of any existing run.")


def perform_external_verification(config: dict, verification_id: str):
    """
    Verifies the integrity of the model loaded from a previous run. It compares
    the hash of the file specified in `pretrained_model_path` against the hash of
    the corresponding file in the trusted model registry.
    """
    print("\n--- Performing External Continuity Verification on Loaded Model ---")
    print(
        f"Verifying that the loaded model originates from trusted run ID: {verification_id}"
    )

    # 1. Get the path to the PRE-TRAINED model that this run claims to have loaded.
    loaded_model_path_from_prov = config.get("model_config", {}).get(
        "pretrained_model_path"
    )
    if not loaded_model_path_from_prov:
        raise AssertionError(
            "VERIFICATION FAILED: 'model_config.pretrained_model_path' not found in "
            "provenance, which is required for a continuity check."
        )

    if not os.path.exists(loaded_model_path_from_prov):
        raise AssertionError(
            f"VERIFICATION FAILED: The loaded model file specified in provenance "
            f"does not exist at '{loaded_model_path_from_prov}'."
        )

    # 2. Extract the base filename of the loaded model.
    loaded_model_filename = os.path.basename(loaded_model_path_from_prov)

    # 3. Construct the path to the corresponding file in the trusted registry.
    #  TRUSTED_REGISTRY_DIR / {verification_id} / {filename}
    trusted_run_dir = os.path.join(TRUSTED_MODEL_REGISTRY_DIR, str(verification_id))
    trusted_model_path = os.path.join(trusted_run_dir, loaded_model_filename)

    print(f"  - Locating trusted model at: {trusted_model_path}")
    if not os.path.exists(trusted_model_path):
        raise AssertionError(
            f"VERIFICATION FAILED: Trusted source model '{loaded_model_filename}' not "
            f"found in registry under run ID '{verification_id}'. Expected it at '{trusted_model_path}'."
        )

    # 4. Hash the model file that this run claims to have loaded.
    run_loaded_model_hash = calculate_sha256(loaded_model_path_from_prov)
    print(f"  - Hash of the model this run loaded: {run_loaded_model_hash}")

    # 5. Hash the trusted source model file from the registry.
    trusted_model_hash = calculate_sha256(trusted_model_path)
    print(f"  - Hash of the trusted source model:  {trusted_model_hash}")

    # 6. Compare the hashes. They must be identical.
    if run_loaded_model_hash != trusted_model_hash:
        raise AssertionError(
            "External verification FAILED: The hash of the loaded model does not match "
            "the hash of the trusted source model."
        )

    print(
        "External verification PASSED: The run correctly loaded the specified trusted model."
    )
    print("---------------------------------------------")
    return True

    # get UID interview the BC, if not verified reject
    # if pass get the specified checkpoint from trusted dir
    # check hash if not the same reject


def calculate_sha256(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


# -- MAIN --

def main(prov_graph_path: str, check_uniqueness_dir: str = None):

    # check prov hash file from bc

    # 1. Reconstruct the experiment from the provenance graph
    config = parse_prov_graph_to_config(prov_graph_path)
    print(config)
    set_seed(42)#config["seed"])
    device = "mps"# config["device"]

    # Re-create components that don't depend on the model first
    full_dataset, _ = create_dataloader(config, train=True)
    indexed_dataset = IndexedDataset(full_dataset)
    criterion = create_criterion(config)
    task_type = config.get("training_config", {}).get("task_type")

    # 2. Load initial state and perform continuity checks
    initial_model_id = config.get("continuity", {}).get("initial_model_id")

    if initial_model_id:
        print("\n--- Verifying Continuity and Re-creating Initial State ---")

        # Step A: Verify that the source pretrained model file is trusted.
        perform_external_verification(config, str(initial_model_id))

        # Step B: Verify that the `initial_weights.pt` file is a correct derivative of the trusted source model.

        # B.1: Create the model
        model = create_model(config).to(device)

        # B.2: Replicate the initial state creation by loading the trusted source
        pretrained_path = config["model_config"]["pretrained_model_path"]
        model.load_state_dict(
            torch.load(pretrained_path, map_location=device), strict=False
        )
        recreated_initial_state = model.state_dict()

        # B.3: Load the initial weights that were actually saved by the original run.
        initial_weights_path_from_prov = config["artifacts"]["initial_weights"]
        saved_initial_state = torch.load(initial_weights_path_from_prov)

        # B.4: The recreated state and the saved state must be identical.
        distance = calculate_l2_distance(recreated_initial_state, saved_initial_state)
        assert distance < 1e-6, (
            "Initial State Verification FAILED: The provided 'initial_weights.pt' does not match "
            "the state derived from applying modifications to the trusted source model."
        )
        print(
            "Initial State Verification PASSED: 'initial_weights.pt' is a verified derivative of the source."
        )

        # B.5: Ensure the model is loaded with the verified initial state for the replay.
        model.load_state_dict(saved_initial_state)

    else:
        print("\nNo continuity ID found. Loading initial state directly for replay.")
        model = create_model(config).to(device)
        initial_weights_path = config["artifacts"]["initial_weights"]
        model.load_state_dict(torch.load(initial_weights_path))

    # Create the optimizer after the model is fully loaded and on the correct device
    optimizer = create_optimizer(model, config)

    # 3. Perform preliminary checks (e.g., dataset hash)
    if "sha256_hash" in config["data_config"] and hasattr(full_dataset, "filename"):
        filepath = os.path.join(full_dataset.root, full_dataset.filename)
        if os.path.exists(filepath):
            current_hash = calculate_sha256(filepath)
            assert (
                current_hash == config["data_config"]["sha256_hash"]
            ), "Dataset SHA256 hash mismatch!"
            print("Dataset SHA256 hash verified successfully.")

    verified_checkpoint_states = []

    # 4. Main Incremental Verification Loop
    # Iterate through each training segment defined in the provenance.
    for step_info in config["verification_steps"]:
        epoch, step = step_info["epoch"], step_info["step"]
        print(f"\n--- Verifying Epoch {epoch}, Step {step} ---")

        subset = torch.utils.data.Subset(indexed_dataset, step_info["indices"])
        verifier_loader = torch.utils.data.DataLoader(
            subset, batch_size=config["data_config"]["batch_size"], shuffle=False
        )

        replayed_tensors = []
        model.train()
        for data in tqdm(verifier_loader, desc=f"Replaying E{epoch}-S{step}"):
            optimizer.zero_grad()
            if task_type == "reconstruction":
                inputs, _, _ = data
                replayed_tensors.append(inputs.cpu())
                inputs = inputs.to(device)
                loss = criterion(model(inputs), inputs)
            elif task_type == "classification":
                inputs, labels, _ = data
                replayed_tensors.append(inputs.cpu())
                inputs, labels = inputs.to(device), labels.to(device)
                loss = criterion(model(inputs), F.one_hot(labels, 10).float())
            loss.backward()
            optimizer.step()

        # Check 1: Data Integrity. Does the replayed data match the original data?
        replayed_hash = hashlib.sha256(
            torch.cat(replayed_tensors, dim=0).numpy().tobytes()
        ).hexdigest()
        assert (
            replayed_hash == step_info["data_hash"]
        ), f"Data hash mismatch for E{epoch}-S{step}!"
        print("Data hash check PASSED.")

        # Check 2: Checkpoint Integrity. Has the saved checkpoint file been tampered with?
        model_path = step_info["model_path"]
        on_disk_model_hash = calculate_sha256(model_path)
        assert (
            on_disk_model_hash == step_info["model_hash"]
        ), f"Checkpoint file hash mismatch for E{epoch}-S{step}! File may be tampered."
        print("Checkpoint file integrity check PASSED.")

        # Check 3: Reproducibility. Does the replayed model state match the original?
        original_weights = torch.load(step_info["model_path"])["model_state_dict"]
        distance = calculate_l2_distance(model.state_dict(), original_weights)
        assert distance < 1e-6, f"Reproducibility FAILED at E{epoch}-S{step}!"

        # If successful, load the original state to prevent minor floating point errors from accumulating.
        # Ensures each segment is verified independently from the correct starting point.
        print(f"Success Distance: {distance}")
        print(f" Loading state from E{epoch}-S{step} checkpoint.")
        model.load_state_dict(original_weights)
        verified_checkpoint_states.append(original_weights)

    print("\n--- Incremental Reproducibility Verification Complete ---")

    # 5. Final Uniqueness Check
    if check_uniqueness_dir:  # dir_name/id/checkpoints/files.pt
        perform_cross_run_uniqueness_check(
            verified_checkpoint_states, check_uniqueness_dir
        )
    else:
        print("\nSkipping final uniqueness check (no directory provided).")

    # BC upload PID, result of verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifies training reproducibility and uniqueness."
    )
    parser.add_argument(
        "prov_graph_path", type=str, help="Path to the provgraph...json file."
    )
    parser.add_argument(
        "--check-uniqueness-in",
        type=str,
        default=None,
        help="Optional. Path to a directory of past runs to check for duplicates.",
    )
    args = parser.parse_args()
    main(args.prov_graph_path, args.check_uniqueness_in)
