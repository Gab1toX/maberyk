from __future__ import annotations
import torch

OLD_OBSERVATION_SIZE = 115
NEW_OBSERVATION_SIZE = 163
ACTION_SIZE = 5
SOURCE_PATH = "agent_state.pt"
TARGET_PATH = "agent_state_migrated.pt"

def expand_rows(state_dict, key, new_size):
    """Expande la dimension 0 (filas) de un tensor."""
    old_weight = state_dict[key]
    new_weight = torch.zeros(new_size, *old_weight.shape[1:], dtype=old_weight.dtype)
    new_weight[:old_weight.shape[0]] = old_weight
    state_dict[key] = new_weight
    return old_weight.shape, new_weight.shape

def expand_cols(state_dict, key, new_size):
    """Expande la dimension 1 (columnas) de un tensor."""
    old_weight = state_dict[key]
    new_weight = torch.zeros(old_weight.shape[0], new_size, dtype=old_weight.dtype)
    new_weight[:, :old_weight.shape[1]] = old_weight
    state_dict[key] = new_weight
    return old_weight.shape, new_weight.shape

def expand_bias(state_dict, key, new_size):
    """Expande un bias vector."""
    old_bias = state_dict[key]
    new_bias = torch.zeros(new_size, dtype=old_bias.dtype)
    new_bias[:old_bias.shape[0]] = old_bias
    state_dict[key] = new_bias
    return old_bias.shape, new_bias.shape

def main():
    checkpoint = torch.load(SOURCE_PATH, map_location="cpu")
    policy = checkpoint["policy_state_dict"]
    curiosity = checkpoint["curiosity_state_dict"]

    # Policy — primera capa entrada
    expand_cols(policy, "model.0.weight", NEW_OBSERVATION_SIZE)
    print("policy model.0.weight OK")

    # Predictor — primera capa entrada
    expand_cols(curiosity, "next_observation_predictor.model.0.weight", NEW_OBSERVATION_SIZE + ACTION_SIZE)
    print("predictor model.0.weight OK")

    # Predictor — ultima capa salida (predice next_observation)
    expand_rows(curiosity, "next_observation_predictor.model.4.weight", NEW_OBSERVATION_SIZE)
    expand_bias(curiosity, "next_observation_predictor.model.4.bias", NEW_OBSERVATION_SIZE)
    print("predictor model.4.weight OK")

    # Error network — primera capa entrada (obs + next_obs)
    expand_cols(curiosity, "prediction_error_network.model.0.weight", NEW_OBSERVATION_SIZE * 2)
    print("error_network model.0.weight OK")

    checkpoint["observation_size"] = NEW_OBSERVATION_SIZE
    checkpoint["policy_state_dict"] = policy
    checkpoint["curiosity_state_dict"] = curiosity

    torch.save(checkpoint, TARGET_PATH)
    print(f"\nMigrado correctamente: {SOURCE_PATH} -> {TARGET_PATH}")
    print(f"observation_size: {OLD_OBSERVATION_SIZE} -> {NEW_OBSERVATION_SIZE}")

if __name__ == "__main__":
    main()