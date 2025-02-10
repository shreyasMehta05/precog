def fix_vector_file(input_path, output_path):
    """
    Fix FastText vector file by handling multi-word tokens and ensuring proper formatting
    """
    print(f"Fixing vector file from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        # Read header
        header = fin.readline().strip()
        vocab_size, dim = map(int, header.split())
        print(f"Vocabulary size: {vocab_size}, Dimensions: {dim}")
        fout.write(f"{vocab_size} {dim}\n")
        
        line_count = 0
        for line in fin:
            parts = line.strip().split()
            
            # Find where the vector starts (first numerical value)
            vector_start = 0
            for i, part in enumerate(parts):
                try:
                    float(part)
                    vector_start = i
                    break
                except ValueError:
                    continue
            
            if vector_start > 0:  # If we found the vector part
                word = '_'.join(parts[:vector_start])  # Join multi-word tokens with underscore
                vector = parts[vector_start:]
                
                # Verify vector length
                if len(vector) == dim:
                    try:
                        # Verify all vector components are valid floats
                        [float(x) for x in vector]
                        fout.write(f"{word} {' '.join(vector)}\n")
                        line_count += 1
                        
                        if line_count % 10000 == 0:
                            print(f"Processed {line_count} lines...")
                    except ValueError:
                        print(f"Skipping line with invalid vector values: {word}")
                        continue
            
    print(f"Fixed file saved to {output_path}")
    print(f"Total vectors processed: {line_count}")

# Usage
input_file = "data/english_filtered.vec"
output_file = "data/english_filtered_fixed.vec"
fix_vector_file(input_file, output_file)