import re
import numpy as np

def read_losses(filename):
    """
    Reads loss metrics from a specified file.

    Args:
        filename (str): The path to the input file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Matrices for l_0 and l_1 losses.
    """
    with open(filename, 'r') as f:
        file_content = f.read()

    data = {}
    labels = ['l_0', 'l_1']

    for label in labels:
        pattern = re.compile(f"{label}\\s*:\\s*\\((.*?)\\)")
        matches = pattern.findall(file_content)

        extracted_arrays = [np.fromstring(m.replace(' ', ''), sep=',') for m in matches]

        if extracted_arrays and all(len(arr) == len(extracted_arrays[0]) for arr in extracted_arrays):
            data[label] = np.vstack(extracted_arrays)
        else:
            data[label] = extracted_arrays 
            
    l0, l1 = data.get('l_0'), data.get('l_1')
    return l0, l1