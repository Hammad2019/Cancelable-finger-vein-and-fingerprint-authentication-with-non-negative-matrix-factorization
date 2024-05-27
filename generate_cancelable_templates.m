% Function to generate cancelable templates using SHA-256 hash
function cancelable_templates = generate_cancelable_templates(W)
    [num_samples, num_components] = size(W);
    cancelable_templates = cell(size(W));
    
    for i = 1:num_samples
        for j = 1:num_components
            value_str = num2str(W(i, j));
            hash_value = generate_sha256_hash(value_str);
            cancelable_templates{i, j} = hash_value;
        end
    end
end
