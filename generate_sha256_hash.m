% Function to generate SHA-256 hash
function hash_value = generate_sha256_hash(input_str)
    hash_object = java.security.MessageDigest.getInstance('SHA-256');
    hash_object.update(uint8(input_str), 0, length(input_str));
    hash_bytes = typecast(hash_object.digest(), 'uint8');
    hash_value = sprintf('%02x', hash_bytes);
end
