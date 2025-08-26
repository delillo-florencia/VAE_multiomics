def txt_to_list(filepath):
    """
    Reads a text file where each line contains one isoform name
    and returns a list of isoform names.
    
    Parameters
    ----------
    filepath : str
        Path to the text file
    
    Returns
    -------
    list of str
        List of isoform names
    """
    with open(filepath, 'r') as f:
        isoform_list = [line.strip() for line in f if line.strip()]
    return isoform_list
