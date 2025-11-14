def transform_file(input_file, output_file):
    transformed_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                # Remove the first number
                parts = parts[1:]
                # Join all but the last number with spaces, then append comma + last number
                new_line = ' '.join(parts[:-1]) + ',' + parts[-1]
                transformed_lines.append(new_line)

    # Write to output file
    with open(output_file, 'w') as f:
        for line in transformed_lines:
            f.write(line + '\n')

# Example usage:
input_file = 'indata.txt'   # your input file
output_file = 'test_data_11.csv' # desired output file
transform_file(input_file, output_file)