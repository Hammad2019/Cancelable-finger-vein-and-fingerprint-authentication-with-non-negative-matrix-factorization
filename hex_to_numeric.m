% Function to convert hex to decimal numeric value
function numeric_value = hex_to_numeric(hex_value)
    hex_chars = '0123456789ABCDEF';
    hex_value = upper(hex_value);
    numeric_value = 0;
    
    for i = 1:length(hex_value)
        numeric_value = numeric_value * 16 + strfind(hex_chars, hex_value(i)) - 1;
    end
end