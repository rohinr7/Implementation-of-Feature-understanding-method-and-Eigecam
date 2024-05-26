import numpy as np

def expand_flat_values_to_activation_shape(values, W_layer, H_layer):
    if False:
        # Initial implementation in original FEM paper
        expanded = np.expand_dims(values, axis=1)
        expanded = np.kron(expanded, np.ones((W_layer, 1, H_layer)))
        expanded = np.transpose(expanded, axes=[0, 2, 1])
    else:
        # Simplified implementation
        expanded = values.reshape((1,1,-1)) * np.ones((W_layer, H_layer, len(values)))
    return expanded
###############################################################################
# Input:
#   feature_map: [W, H, D] Tensor OR [B, W, H, D] Tensor.
#   sigma: Scalar, optional, default: 2
# Output:
#   [W, H, D] tensor OR [B, W, H, D] Tensor
#   compute the binarization part of FEM, tresholding every [W, H] slice of feature_map by
#   k sigma tresholding.
# Description:
#   feature_map is the activation map of the layer that you want FEM applied to
# Example:
#   import visualize
#   import numpy as np
#   # Pretend that this tensor comes from a CNN vvvvvvvvvvvvvv
#   b = visualize.binary_map_of_feature_map(np.rand(64, 64, 16))
###############################################################################
def compute_binary_maps(feature_map, sigma=None):

    batch_size, W_layer, H_layer, N_channels = feature_map.shape
    thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))

    if sigma is None:
        feature_sigma = 2
    else:
        feature_sigma = sigma
        
    for B in range(batch_size):
        # Get the activation value of the current sample
        activation = feature_map[B, :, :, :]
                
        # Calculate its mean and its std per channel
        mean_activation_per_channel = activation.mean(axis=(0,1))
        std_activation_per_channel = activation.std(axis=(0,1))
        assert len(mean_activation_per_channel) == N_channels
        assert len(std_activation_per_channel) == N_channels
        
        # Transform the mean in the same shape than the activation maps
        mean_activation_expanded = expand_flat_values_to_activation_shape(mean_activation_per_channel, W_layer,H_layer)
        
        # Transform the std in the same shape than the activation maps
        std_activation_expanded = expand_flat_values_to_activation_shape(std_activation_per_channel, W_layer,H_layer)

        # Build the binary map
        thresholded_tensor[B, :, :, :] = 1.0 * (activation > (mean_activation_expanded + feature_sigma * std_activation_expanded))
        
    return thresholded_tensor

###############################################################################
# Input:
#   binary_feature_map: [W, H, D] Tensor OR [B, W, H, D] Tensor.
#   original_feature_map: [W, H, D] Tensor OR [B, W, H, D] Tensor.
# Output:
#   [W, H] tensor OR [B, W, H] Tensor
# Description:
#   compute the FEM heatmap from the activation map of a layer and it's binarization.
# Example:
#   import numpy as np
#   # Pretend that this  vvvvvvvvvvvvv   tensor comes from a CNN
#   activation_map = np.rand(64, 64, 16)
#   b = binary_map_of_feature_map(activation_map)
#   FEM = calculate_weighted_feature_frame(b, activation_map)
###############################################################################
def aggregate_binary_maps(binary_feature_map, orginal_feature_map):
    
    # This weigths the binary map based on original feature map
    batch_size, W_layer, H_layer, N_channels = orginal_feature_map.shape
    
    orginal_feature_map = orginal_feature_map[0]
    binary_feature_map = binary_feature_map[0]

    # Get the weights
    channel_weights = np.mean(orginal_feature_map, axis=(0,1))  # Take means for each channel-values
    if False:
        # Original paper implementation
        expanded_weights = np.kron(np.ones(
            (binary_feature_map.shape[0], binary_feature_map.shape[1], 1)), channel_weights)
    else:
        # Simplified version
        expanded_weights = expand_flat_values_to_activation_shape(channel_weights, W_layer,H_layer)
    
    # Apply the weights on each binary feature map
    expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)
    
    # Aggregate the feature map of each channel
    feat_map = np.sum(expanded_feat_map, axis=2)
    
    # Normalize the feature map
    if np.max(feat_map) == 0:
        return feat_map
    feat_map = feat_map / np.max(feat_map)
    return feat_map

###############################################################################
# Input:
#   img_array: the "batch"
#   model: our model
#   last_conv_layer_name: name of the final conv layer
# Output:
#   [W, H] tensor OR [B, W, H] Tensor
# Description:
#   Compute FEM from the activation map of a layer,
#   binary_map_of_feature_map + calculate_weighted_feature_frame
###############################################################################

def compute_fem(feature_map):
    binary_feature_map = compute_binary_maps(feature_map)
    saliency = aggregate_binary_maps(binary_feature_map, feature_map)
    return saliency