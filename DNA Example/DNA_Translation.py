# -*- coding: utf-8 -*-
import difflib

#Dictionary of Amino Acid Sequence
AA_Table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def readSequence(fileName):
    """
    Reads a text file and returns a string without 
    \n and \r

    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.

    Returns
    -------
    seq : TYPE
        DESCRIPTION.

    """
    with open(fileName, "r") as f:
        seq = f.read()
    seq = seq.replace("\n", "")
    seq = seq.replace("\r","")
    return seq

def writeSequence(fileName, p_sequence):
    with open(fileName, "w+") as f:
        f.write(p_sequence)

def findStart (sequence):
    """
    Searches for the ATG codon to start the protein translation

    Parameters
    ----------
    sequence : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    start_pos = 0
    for a in sequence:
        if (sequence[start_pos:(start_pos + 3)] == "ATG"):
            return start_pos
        else:
            start_pos += 1
    return -1

def findEnd (sequence):
    """
    Returns a tuple including the starting point and ending point
    not including the last codon
    """
    start_pos = findStart(sequence)
    end_pos = start_pos + 3
    for a in range(start_pos, len(sequence), 3):
        if (sequence[a:(a+3)] == "TAA" or 
            sequence[a:(a+3)] == "TAG" or 
            sequence[a:(a+3)] == "TGA"):
            # end_pos - 3 because the end codon should not be included
            return (start_pos, (end_pos-3))
        else:
            end_pos += 3
    return (start_pos, -1)

def translate(sequence):

    """
    This function takes in a string and returns an amino acid seqence.
    The function will not run if the input string is not
    divisible by 3.

    Parameters
    ----------
    sequence : TYPE
        DESCRIPTION.

    Returns
    -------
    protein : TYPE
        DESCRIPTION.
    """
    start_pos, end_pos = findEnd(sequence)
    protein = ""
    if ((end_pos-start_pos) % 3 == 0):
        for i in range(start_pos, end_pos, 3):
            codon = sequence[i:(i+3)]
            protein += AA_Table[codon]
    return protein

def checkDiff(file1, file2):
    for line in difflib.unified_diff(file1, file2):
        print(line)

def main():
    input_file = input("Input File Name (including .txt):")
    
    DNA_Sequence = readSequence(input_file)
    protein_Sequence = translate(DNA_Sequence)
    
    output_file = input("Output File Name (including .txt):")
    writeSequence(output_file, protein_Sequence)
    
    check_file = input("Check File Name (including .txt):")
    checkDiff(open(check_file).readlines(), open(output_file).readlines())
    
main()