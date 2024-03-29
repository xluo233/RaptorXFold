import os
import sys
import numpy as np
import pickle
from .SubstitutionMatrices import gonnet
from .SequenceUtils import AALetter2OrderOf1LetterCode, \
    AA1LetterOrder23LetterOrder, DetectMultiHIS


# This script reads an .hhm file and save it as a python dict().
# To use the position-specfic frequency matrix, please use the keyword PSFM.
# To use the position-specific scoring matrix, please use the keyword PSSM.
# PSFM and PSSM are derived from the HMM block, so there is no need to
#   directly use the keys containing 'hmm'.
# PSFM and PSSM columns are arranged by the alphabetical order
#   of amino acids in their 1-letter code.
# the following keys are also available: name, sequence, NEFF, DateCreated

M_M, M_I, M_D, I_M, I_I, D_M, D_D, _NEFF, I_NEFF, D_NEFF = 0, 1, 2, 3, 4, \
    5, 6, 7, 8, 9

# HMMNull is the background score of amino acids,
#   in the alphabetical order of 1-letter code
HMMNull = np.array([3706, 5728, 4211, 4064, 4839, 3729, 4763, 4308, 4069,
                    3323, 5509, 4640, 4464, 4937, 4285, 4423, 3815, 3783,
                    6325, 4665, 0], dtype=np.float32)
gonnet = np.array(gonnet, np.float32)


# a function reading the HMM block in .hhm,. tgt and .tpl files.
# The HMM block is generated by HHpred/HHblits package
# for the tgt and tpl file, the number of lines for the header is 6;
# for the profileHMM file, the header has 4 lines
def ReadHHM(lines, start_position, length, one_protein, numLines4Header=6):

    i = start_position
    one_protein['HMMHeader'] = lines[i: i+numLines4Header]
    i += numLines4Header

    # the columns of hmm1 are in the alphabetical order of amino acids in
    # 1-letter code, different from the above PSP and PSM matrices
    one_protein['hmm1'] = np.zeros((length, 20), np.float32)
    one_protein['hmm2'] = np.zeros((length, 10), np.float32)
    one_protein['hmm1_prob'] = np.zeros((length, 20), np.float32)
    one_protein['hmm1_score'] = np.zeros((length, 20), np.float32)
    seqStr = ''

    for j in range(length):
        # this line is for emission score. The amino acids are ordered from
        # left to right alphabetically by their 1-letter code
        fields = lines[i+j*3+0].replace("*", "99999").split()

        assert len(fields) == 23
        one_protein['hmm1'][j] = np.array(
            [-np.int32(num) for num in fields[2: -1]])/1000.
        aa = fields[0]
        seqStr += aa

        # the first 7 columns of this line is for state transition
        one_protein['hmm2'][j][0:7] = [np.exp(-np.int32(num)/1000.0*0.6931)
                                       for num in lines[i + j*3 + 1].replace(
                                           "*", "99999").split()[0:7]]

        # the last 3 columns of this line is for Neff of
        #       Match, Insertion and Deletion
        one_protein['hmm2'][j][7:10] = [np.int32(num)/1000.0 for num in
                                        lines[i + j*3 + 1].split()[7:10]]

        # _NEFF is for match, I_NEFF for insertion and D_NEFF for deletion.
        rm = 0.1
        one_protein['hmm2'][j][M_M] = (
            one_protein['hmm2'][j][_NEFF] * one_protein['hmm2'][j][M_M] +
            rm * 0.6)/(rm + one_protein['hmm2'][j][_NEFF])
        one_protein['hmm2'][j][M_I] = (
            one_protein['hmm2'][j][_NEFF] * one_protein['hmm2'][j][M_I] +
            rm * 0.2)/(rm + one_protein['hmm2'][j][_NEFF])
        one_protein['hmm2'][j][M_D] = (
            one_protein['hmm2'][j][_NEFF] * one_protein['hmm2'][j][M_D] +
            rm * 0.2)/(rm + one_protein['hmm2'][j][_NEFF])
        ri = 0.1

        one_protein['hmm2'][j][I_I] = (
            one_protein['hmm2'][j][I_NEFF] * one_protein['hmm2'][j][I_I] +
            ri * 0.75)/(ri + one_protein['hmm2'][j][I_NEFF])
        one_protein['hmm2'][j][I_M] = (
            one_protein['hmm2'][j][I_NEFF] * one_protein['hmm2'][j][I_M] +
            ri * 0.25)/(ri + one_protein['hmm2'][j][I_NEFF])
        rd = 0.1

        one_protein['hmm2'][j][D_D] = (
            one_protein['hmm2'][j][D_NEFF] * one_protein['hmm2'][j][D_D] +
            rd * 0.75)/(rd + one_protein['hmm2'][j][D_NEFF])
        one_protein['hmm2'][j][D_M] = (
            one_protein['hmm2'][j][D_NEFF] * one_protein['hmm2'][j][D_M] +
            rd * 0.25)/(rd + one_protein['hmm2'][j][D_NEFF])

        one_protein['hmm1_prob'][j] = pow(2.0, one_protein['hmm1'][j])
        wssum = sum(one_protein['hmm1_prob'][j])

        # renormalize to make wssum = 1
        if wssum > 0:
            one_protein['hmm1_prob'][j] /= wssum
        else:
            one_protein['hmm1_prob'][j, AALetter2OrderOf1LetterCode[aa]] = 1.

        # add pseudo count
        g = np.zeros((20), np.float32)
        for ll in range(20):
            orderIn3LetterCode_j = AA1LetterOrder23LetterOrder[ll]
            for k in range(20):
                orderIn3LetterCode_k = AA1LetterOrder23LetterOrder[k]
                g[ll] += one_protein['hmm1_prob'][j, k] * \
                    gonnet[orderIn3LetterCode_k, orderIn3LetterCode_j]
            g[ll] *= pow(2.0, -1.0 * HMMNull[ll] / 1000.0)
        g = g / sum(g)

        ws_tmp_neff = one_protein['hmm2'][j][_NEFF] - 1
        one_protein['hmm1'][j] = (ws_tmp_neff * one_protein['hmm1_prob'][j] +
                                  g*10) / (ws_tmp_neff+10)

        # recalculate the emission score and probability
        # after pseudo count is added
        one_protein['hmm1_prob'][j] = one_protein['hmm1'][j]
        one_protein['hmm1'][j] = np.log2(one_protein['hmm1_prob'][j])
        one_protein['hmm1_score'][j] = one_protein['hmm1'][j] + \
            HMMNull[:20]/1000.0

        # PSFM: position-specific frequency matrix,
        # PSSM: position-specific scoring matrix
        one_protein['PSFM'] = one_protein['hmm1_prob']
        one_protein['PSSM'] = one_protein['hmm1_score']

    if len(seqStr) != len(one_protein['sequence']):
        print('ERROR: inconsistent sequence length in HMM section and '
              'orignal sequence for protein: ', one_protein['name'])
        exit(1)

    comparison = [(aa == 'X' or bb == 'X' or aa == bb) for aa, bb in
                  zip(seqStr, one_protein['sequence'])]

    if not all(comparison):
        print('ERROR: inconsistent sequence between HMM section and '
              'orignal sequence for protein: ', one_protein['name'])
        print(' original seq: ', one_protein['sequence'])
        print(' HMM seq: ', seqStr)
        exit(1)

    return i + 3 * length, one_protein


# this function reads a profile HMM file generated by HHpred/HHblits package
def load_hhm(hhmfile):
    if hhmfile.endswith('.hhm.pkl'):
        with open(hhmfile, 'rb') as fh:
            return pickle.load(fh, encoding="latin1")

    with open(hhmfile, 'r') as fh:
        content = [r.strip() for r in list(fh)]
    if not bool(content):
        print('ERROR: empty profileHMM file: ', hhmfile)
        exit(1)
    if not content[0].startswith('HHsearch '):
        print('ERROR: this file may not be a profileHMM file generated by '
              'HHpred/HHblits: ', hhmfile)
        exit(1)
    if len(content) < 10:
        print('ERROR: this profileHMM file is too short: ', hhmfile)
        exit(1)
    requiredSections = ['name', 'length', 'sequence', 'NEFF',  'hmm1', 'hmm2',
                        'hmm1_prob', 'hmm1_score',
                        'PSFM', 'PSSM', 'DateCreated']
    protein = {}

    # get sequence name
    if not content[1].startswith('NAME '):
        print('ERROR: the protein name shall appear at the second line '
              'of profileHMM file: ', hhmfile)
        exit(1)
    fields = content[1].split()
    if len(fields) < 2:
        print('ERROR: incorrect name format in profileHMM file: ', hhmfile)
        exit(1)
    protein['name'] = fields[1]
    protein['annotation'] = ' '.join(fields[2:])

    i = 2
    while i < len(content):
        row = content[i]
        if len(row) < 1:
            i += 1
            continue

        if row.startswith('DATE '):
            protein['DateCreated'] = row[6:]
            i += 1
            continue

        if row.startswith('LENG '):
            protein['length'] = np.int32(row.split()[1])
            i += 1
            continue

        if row.startswith('FILT '):
            protein['filterinfo'] = row[len('FILT '):]
            i += 1
            continue

        if row.startswith('NEFF '):
            protein['NEFF'] = np.float32(row.split()[1])
            i += 1
            continue

        if row.startswith('>ss_dssp '):
            # read native secondary structure
            start = i+1
            end = i+1
            while not content[end].startswith('>'):
                end += 1
            protein['nativeSS8'] = ''.join(content[start:end]).replace(
                'C', 'L').replace('-', 'L')
            if len(protein['nativeSS8']) != protein['length']:
                print('ERROR: inconsistent sequence length and native '
                      'SS length in hmmfile: ', hhmfile)
                exit(1)
            i = end
            continue

        if row.startswith('>ss_pred'):
            # read predicted secondary structure
            start = i+1
            end = i+1
            while not content[end].startswith('>'):
                end += 1
            protein['SSEseq'] = ''.join(content[start:end]).replace('C', 'L')
            if len(protein['SSEseq']) != protein['length']:
                print('ERROR: inconsistent sequence length and '
                      'predicted SS length in hmmfile: ', hhmfile)
                exit(1)
            i = end
            continue

        if row.startswith('>ss_conf'):
            # read predicted secondary structure confidence score
            start = i+1
            end = i+1
            while not content[end].startswith('>'):
                end += 1

            SSEconfStr = ''.join(content[start:end])
            protein['SSEconf'] = [np.int16(score) for score in SSEconfStr]

            if len(protein['SSEconf']) != protein['length']:
                print('ERROR: inconsistent sequence length and predicted SS '
                      'confidence sequence length in hhmfile: ', hhmfile)
                exit(1)

            i = end
            continue

        if row.startswith('>' + protein['name']):
            # read the primary sequence in the following lines
            start = i+1
            end = i+1
            while not content[end].startswith('>') and \
                    (not content[end].startswith('#')):
                end += 1

            # at this point, content[end] shall start with >
            protein['sequence'] = ''.join(content[start:end])
            if len(protein['sequence']) != protein['length']:
                print('ERROR: inconsistent sequence length in hmmfile: ',
                      hhmfile)
                exit(1)
            i = end
            protein['HISflag'] = DetectMultiHIS(protein['sequence'])
            continue

        if len(row) == 1 and row[0] == '#' and \
                content[i+1].startswith('NULL') and \
                content[i+2].startswith('HMM'):
            nullfields = content[i+1].split()[1:]
            HMMNull[:20] = [np.float32(f) for f in nullfields]

            i, protein = ReadHHM(
                content, i+1, protein['length'], protein, numLines4Header=4)
            continue

        i += 1

    # double check to see some required sections are read in
    for section in requiredSections:
        if section not in protein:
            print('ERROR: one section for ', section,
                  ' is missing in the hmm file: ', hhmfile)
            print('ERROR: it is also possible that the hmm file '
                  'has a format incompatible with this script.')
            exit(1)

    protein['requiredSections'] = requiredSections

    return protein


# for test only
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('python LoadHHM.py hhm_file')
        print(' the input file shall end with .hhm')
        exit(1)

    file = sys.argv[1]

    if file.endswith('.hhm') or file.endswith('.hhm.pkl'):
        protein = load_hhm(file)
    else:
        print('ERROR: the input file shall have suffix .hhm')
        exit(1)

    savefile = os.path.basename(file) + '.pkl'
    with open(savefile, 'wb') as fh:
        pickle.dump(protein, fh, protocol=pickle.HIGHEST_PROTOCOL)
