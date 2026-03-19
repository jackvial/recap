# Binned return's example
import numpy as np
import torch
import torch.nn.functional as F

NUM_BINS = 5

def value_targets_from_success(success: bool, episode_len: int, max_episode_len: int, c_fail: int):
    assert NUM_BINS <= episode_len
    assert max_episode_len >= episode_len
    if success:
        # t = 0..episode_len-1, target is negative remaining steps normalized to [-1, 0]
        returns = -np.arange(episode_len - 1, -1, -1, dtype=np.float32) / max_episode_len
    else:
        # failed episodes get a large negative value, then clamp into training range if desired
        returns = np.full(episode_len, -c_fail / max_episode_len, dtype=np.float32)

    # paper says values are predicted in (-1, 0)
    returns = np.clip(returns, -1.0, 0.0)

    bin_edges = np.linspace(-1.0, 0.0, NUM_BINS + 1)
    
    # Categorical label that will used as the target for cross entropy loss
    # we are learning to predict bin ids not returns. the predicted bin
    # will be used to recover an approximated return-to-go at inference
    target_bin_ids = np.clip(np.digitize(returns, bin_edges) - 1, 0, NUM_BINS - 1)

    return target_bin_ids, returns


if __name__ == "__main__":
    target_bin_ids, returns = value_targets_from_success(success=True, episode_len=10, max_episode_len=11, c_fail=10)
    print("returns: ", returns)
    print("target_bin_ids: ", target_bin_ids)
    
    # Place holder for the predicted output from the model
    logits = torch.randn(len(target_bin_ids), NUM_BINS)
    print("logits: ", logits)
    target_bin_ids = torch.tensor(target_bin_ids, dtype=torch.long)
    print("target_bin_ids: ", target_bin_ids)

    loss = F.cross_entropy(logits, target_bin_ids)
    print("loss: ", loss)
    
    ### Inference ###
    # At inference time we recover a non-categorical scalar return to go by
    predicted_probs = F.softmax(logits)
    print("predicted_probs: ", predicted_probs)
    
    # We want to recover an approximation of the normalized return to go at inference
    # of length NUM_BINS. One option is to take the last value of each bin, this way the terminal value will be 0.
    vb_maybe = returns.reshape(NUM_BINS, -1)[:, -1]
    print("vb_maybe: ", vb_maybe)
    
    # but we can also just approximate it with.
    vb = torch.linspace(-1, 0, NUM_BINS)
    print("vb: ", vb)
    
    # Finally we take the expect value value 
    
    v_ref = (predicted_probs * vb)
    print("v_ref: ", v_ref)