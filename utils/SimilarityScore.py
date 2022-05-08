import copy
import numpy as np

from . import SequenceUtils
from .SubstitutionMatrices import Ori_BLOSUM_62, Ori_BLOSUM_45, Ori_BLOSUM_80,\
    Ori_HDSM, Ori_CC50, Ori_Singleton, WS_Singleton

# In this script, the rows and cols of all the mutation matrices are arranged
# in the alphabetical order of amino acids in their 3-letter code.
# That is, A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, Z


# BLOSUM Score
Ori_BLOSUM_62 = np.array(Ori_BLOSUM_62, dtype=np.int32)
newBLOSUM62 = copy.deepcopy(Ori_BLOSUM_62)
newBLOSUM62[20, :] = 0
newBLOSUM62[:, 20] = 0

Ori_BLOSUM_45 = np.array(Ori_BLOSUM_45, dtype=np.int32)
newBLOSUM45 = copy.deepcopy(Ori_BLOSUM_45)
newBLOSUM45[20, :] = 0
newBLOSUM45[:, 20] = 0

Ori_BLOSUM_80 = np.array(Ori_BLOSUM_80, dtype=np.int32)
newBLOSUM80 = copy.deepcopy(Ori_BLOSUM_80)
newBLOSUM80[20, :] = 0
newBLOSUM80[:, 20] = 0


# both a and b are 1-letter code
def BLOSUM62(a, b):
    if ord(a) < ord('A') or ord(a) > ord('Z'):
        print("a non standard amino acid: " + a)
        return 0    # a non-standard amino acid, neither reward nor penalize

    orderIn3LetterCode_a = SequenceUtils.AALetter2OrderOf3LetterCode[a]

    if ord(b) < ord('A') or ord(b) > ord('Z'):
        print("a non standard amino acid: " + b)
        return 0    # a non-standard amino acid

    orderIn3LetterCode_b = SequenceUtils.AALetter2OrderOf3LetterCode[b]

    return Ori_BLOSUM_62[orderIn3LetterCode_a, orderIn3LetterCode_b]


def BLOSUM45(a, b):
    if ord(a) < ord('A') or ord(a) > ord('Z'):
        print("a non standard amino acid: " + a)
        return 0    # a non-standard amino acid, neither reward nor penalize

    orderIn3LetterCode_a = SequenceUtils.AALetter2OrderOf3LetterCode[a]

    if ord(b) < ord('A') or ord(b) > ord('Z'):
        print("a non standard amino acid: " + b)
        return 0   # a non-standard amino acid

    orderIn3LetterCode_b = SequenceUtils.AALetter2OrderOf3LetterCode[b]

    return Ori_BLOSUM_45[orderIn3LetterCode_a, orderIn3LetterCode_b]


def BLOSUM80(a, b):
    if ord(a) < ord('A') or ord(a) > ord('Z'):
        print("a non standard amino acid: " + a)
        return 0    # a non-standard amino acid, neither reward nor penalize

    orderIn3LetterCode_a = SequenceUtils.AALetter2OrderOf3LetterCode[a]

    if ord(b) < ord('A') or ord(b) > ord('Z'):
        print("a non standard amino acid: " + b)
        return 0    # a non-standard amino acid

    orderIn3LetterCode_b = SequenceUtils.AALetter2OrderOf3LetterCode[b]

    return Ori_BLOSUM_80[orderIn3LetterCode_a, orderIn3LetterCode_b]


# Profile Score
# score derived from sequence profile (profile HMM built by HHpred or HHblits)
# calculate profile score by PSFM * PSSM, for two directions:
# seq-template and template-seq
def MutationOf2Pos5(tPos, sPos, temp, seq):
    m = np.inner(seq['PSFM'][sPos], temp['PSSM'][tPos])
    m += np.inner(temp['PSFM'][tPos], seq['PSSM'][sPos])

    return m


# calculate profile score by sequence PSFM * template PSSM
def MutationOf2Pos5_ST(tPos, sPos, temp, seq):
    m = np.inner(seq['PSFM'][sPos], temp['PSSM'][tPos])

    return m


# similarity score between sequence and profile: two-way
def MutationOf2Pos6(tPos, sPos, temp, seq):
    tAA = temp['sequence'][tPos]    # the template residue at tPos
    sAA = seq['sequence'][sPos]     # the sequence residue at sPos

    # x and y are the order in 1-letter code
    x = SequenceUtils.AALetter2OrderOf1LetterCode[tAA]
    y = SequenceUtils.AALetter2OrderOf1LetterCode[sAA]

    # here we silently skip the non-standard amino acids
    # in the sequence or template
    m = 0
    # score between the template profile and sequence residue
    if y >= 0 and y < 20:
        m += temp['PSSM'][tPos][y]
    # score between the sequence profile and template residue
    if x < 20 and x >= 0:
        m += seq['PSSM'][sPos][x]

    return m


# similarity score between the primary sequence and the profile,
# one-way: primary used for target and profile for template
def MutationOf2Pos6_ST(tPos, sPos, temp, seq):
    sAA = seq['sequence'][sPos]   # the sequence residue at sPos
    y = SequenceUtils.AALetter2OrderOf1LetterCode[sAA]

    # here we silently skip the non-standard amino acids
    # in the sequence or template
    # score between the template profile and sequence residue
    if y >= 0 and y < 20:
        m = temp['PSSM'][tPos][y]
    else:
        m = 0.
    return m


# below are two amino acid scoring matrices for very weak similarity.
# One is derived from structure alignment (HDSM)
# and the other from amino acid properties (cc50)
Ori_HDSM_core = np.array(Ori_HDSM, dtype=np.float32)
Ori_HDSM = np.full((21, 21), -3.5, dtype=np.float32)
Ori_HDSM[:20, :20] = Ori_HDSM_core


def HDSM(a, b):
    orderIn3LetterCode_a = SequenceUtils.AALetter2OrderOf3LetterCode[a]
    orderIn3LetterCode_b = SequenceUtils.AALetter2OrderOf3LetterCode[b]
    return Ori_HDSM[orderIn3LetterCode_a, orderIn3LetterCode_b]


newHDSM = np.zeros((21, 21), dtype=np.float32)
newHDSM[:20, :20] = Ori_HDSM_core


Ori_CC50_core = np.array(Ori_CC50, dtype=np.float32) - 0.5
Ori_CC50 = np.full((21, 21), -1, dtype=np.float32)
Ori_CC50[:20, :20] = Ori_CC50_core


def CC50(a, b):
    orderIn3LetterCode_a = SequenceUtils.AALetter2OrderOf3LetterCode[a]
    orderIn3LetterCode_b = SequenceUtils.AALetter2OrderOf3LetterCode[b]
    return Ori_CC50[orderIn3LetterCode_a, orderIn3LetterCode_b]


newCC50 = np.zeros((21, 21), dtype=np.float32)
newCC50[:20, :20] = Ori_CC50_core


# Secondary Structure Mutation Score
# note that in the tgt file, secondary structure is arranged in
#   the order of helix, sheet and loop.
# it is important to make them consistent

SSMutation = [
    [0.941183,  -2.32536,  -0.87487],   # HELIX
    [-2.11462,   1.41307,  -0.401386],  # SHEET
    [-0.760861, -0.540041,  0.269711]]  # LOOP
#     HELIX      SHEET       LOOP

SSMutation = np.array(SSMutation, dtype=np.float32)

# 8-state secondary structure mutation score
# from template to the predicted SS8 of the target
SS8Mutation = [
    [0.9823,  -0.1944, -1.905,  -0.5508, -1.051,   -1.163],    # H(I)
    [-0.07923, 1.139,  -0.7431,  0.2849,  0.07965, -0.1479],   # G
    [-1.868,  -0.7317,  1.274,  -0.7433, -0.2456,  -0.07621],  # E(B)
    [-0.4469,  0.2968, -0.8554,  0.9231,  0.2446,  -0.1803],   # T
    [-1.064,   0.0251, -0.3282,  0.3049,  0.6813,   0.2468],   # S
    [-1.327,  -0.3154, -0.2324, -0.2839,  0.1512,   0.3150]]   # L
#     H(I)       G       E(B)      T        S         L

SS8Mutation = np.array(SS8Mutation, dtype=np.float32)

HELIX = 0
SHEET = 1
LOOP = 2

SS8Letter2SS3Code = {'H': HELIX, 'G': LOOP, 'I': LOOP, 'E': SHEET,
                     'B': LOOP,  'T': LOOP, 'S': LOOP, 'L': LOOP}


# calculate secondary structure mutation score
def SSMutationScore_3State(tPos, sPos, temp, seq):
    if tPos < 0 or tPos >= temp['length']:
        print("SSMutationScore_3State: out of range by tPos %d"
              " in template: %s" % (tPos, temp['name']))
        exit(-1)

    if sPos < 0 or sPos >= seq['length']:
        print("SSOf2Pos1: out of range by sPos %d"
              " in sequence: %s" % (sPos, seq['name']))
        exit(-1)

    ss_type = temp['SS_str'][tPos]

    if ss_type not in SS8Letter2SS3Code:
        print("SSMutationScore_3State: unknown secondary structure type "
              "at position %d in template: %s" % (tPos, temp['name']))
        return 0

    ss_type = SS8Letter2SS3Code[ss_type]

    score = np.dot(SSMutation[ss_type], seq['SS3'][sPos])
    return score


SS82SS6 = {'H': 0, 'G': 1, 'I': 0, 'E': 2, 'B': 2,
           'T': 3, 'S': 4, 'L': 5, 'C': 5}
SS82SS6_n = {0: 0, 1: 1, 2: 0, 3: 2, 4: 2,
             5: 3, 6: 4, 7: 5, 8: 5}


# it is better not to use this function
def SSMutationScore_6State(tPos, sPos, temp, seq):
    if tPos < 0 or tPos >= temp['length']:
        print("SSOf2Pos2: out of range by tPos %d"
              " in template: %s" % (tPos, temp['name']))
        exit(-1)

    if sPos < 0 or sPos >= seq['length']:
        print("SSOf2Pos2: out of range by sPos %d"
              " in sequence: %s" % (sPos, seq['name']))
        exit(-1)

    tSS6 = SS82SS6[temp['SS_str'][tPos]]

    score = 0.0
    for j in range(8):
        # sum over the secondary structure mutation score
        # by predicted probability
        sSS6 = SS82SS6_n[j]
        score += SS8Mutation[tSS6][sSS6] * seq['SS8'][sPos][j]

    return score


# Solvent Accessibility Mutation score
# in the tgt file, the predicted solvent accessibility is arranged
#   in the order of buried, medium and exposed
# in the tpl file, the native solvent accessibility is arranged
#   in the order of buried, medium and exposed

ACCMutation = [
    [0.760885,   -0.0701501,  -0.903965],   # buried
    [-0.0798508,  0.218512,   -0.0623839],  # medium
    [-1.14008,   -0.099655,    0.233613]]   # exposed
#     buried       medium      exposed

ACCMutation = np.array(ACCMutation, dtype=np.float32)


def ACCMutationScore_3State(tPos, sPos, temp, seq):
    if tPos < 0 or tPos >= temp['length']:
        print("ACC_New_Score2: out of range by tPos %d"
              " in template: %s" % (tPos, temp['name']))
        exit(1)
    if sPos < 0 or sPos >= seq['length']:
        print("ACC_New_Score2: out of range by sPos %d"
              " in sequence: %s" % (sPos, seq['name']))
        exit(1)

    tACC = temp['ACC'][tPos]
    if tACC > 2 or tACC < 0:
        print("ACC_New_Score2: unknown solvent accessibility status "
              "at position %d of template: %s" % (tPos, temp['name']))
        exit(1)

    sACC = seq['ACC'][sPos]

    if sACC > 2 or sACC < 0:
        print("ACC_New_Score2: Unknown solvent accessibility status "
              "at position %d of sequence: %s " % (tPos, seq['name']))
        exit(1)

    score = np.dot(ACCMutation[tACC], seq['ACC_prob'][sPos])

    return score


Ori_Singleton = np.array(Ori_Singleton, dtype=np.int32)
newSingleton = np.zeros((21, 9), dtype=np.int32)
newSingleton[:20, ] = Ori_Singleton

WS_Singleton = np.array(WS_Singleton, dtype=np.float32)
newWSSingleton = np.zeros((21, 9), dtype=np.float32)
newWSSingleton[:20, ] = WS_Singleton


# [singleton], sequence profile of the target is used here
def SingletonScore_ProfileBased(tPos, sPos, temp, seq):
    if tPos < 0 or tPos >= temp['length']:
        print("ContactCapacityOf2Pos_old: out of range by tPos %d"
              " in template: %s" % (tPos, temp['name']))
        exit(1)

    if sPos < 0 or sPos >= seq['length']:
        print("ContactCapacityOf2Pos_old: out of range by sPos %d"
              " in sequence: %s" % (sPos, seq['name']))
        exit(1)

    ss = temp['SS_str'][tPos]

    if ss not in SS8Letter2SS3Code:
        print("Unknown secondary structure type at position %d"
              " of template: %s" % (tPos, temp['name']))
        exit(1)
    ac = temp['ACC'][tPos]
    if ac < 0 or ac > 2:
        print("Unknown solvent accessibility type at position %d"
              " of template: %s" % (tPos, temp['name']))
        exit(1)

    ss = SS8Letter2SS3Code[ss]
    score = np.dot(seq['PSFM'][sPos], Ori_Singleton[:, ss*3+ac])
    return score


# [ws_singleton], sequence profile of the target not used here,
# calculate the fitness score of one amino acid in a specific enviroment
# described by a combination of secondary structure and ACC
def SingletonScore_WS(tPos, sPos, temp, seq):
    if tPos < 0 or tPos >= temp['length']:
        print("ContactCapacityOf2Pos_WS: out of range by tPos: %d"
              " in template: %s" % (tPos, temp['name']))
        exit(1)

    if sPos < 0 or sPos >= seq['length']:
        print("ContactCapacityOf2Pos_WS: out of range by sPos: %d"
              " in sequence: %s" % (sPos, seq['name']))
        exit(1)

    ss = temp['SS_str'][tPos]
    if ss not in SS8Letter2SS3Code:
        print("Unknown secondary structure type at position %d"
              " of template: %s" % (tPos, temp['name']))
        exit(1)

    ac = temp['ACC'][tPos]
    if ac < 0 or ac > 2:
        print("Unknown solvent accessibility type at position %d"
              " of template: %s" % (tPos, temp['name']))
        exit(1)

    ss = SS8Letter2SS3Code[ss]

    res = seq['sequence'][sPos]

    j = SequenceUtils.AALetter2OrderOf3LetterCode[res]

    if j >= 20 or j < 0:
        return 0

    return WS_Singleton[j][ss*3+ac]
