#!/usr/bin/env python3
# encoding: utf-8

import os
import pickle
import numpy as np
from scipy.spatial import distance_matrix
from . import Utils
from . import SequenceUtils
from . import SimilarityScore


InvalidDistance = -1
NoCBDegree = np.float16(-540)
InvalidDegree = np.float32(-720)
ValueOfSelf = 0

OriTypeMapping = {'Ca1Cb1Cb2Ca2': 'Ca1-Cb1-Cb2-Ca2',
                  'N1Ca1Cb1Cb2': 'N1-Ca1-Cb1-Cb2',
                  'Ca1Cb1Cb2': 'Ca1-Cb1-Cb2'}


# this function returns an alignment, which is a tuple.
# The first entry in the tuple is the template sequence in alignment;
# the second entry is the query sequence in alignment including gaps.
# if returnNames is True, this function also returns a tuple
# (templateName, queryName)

# tpl is the template object and tgt is the query object
# tgt and tpl are Python dict() and shall have the following keys:
#   length, sequence, name, PSSM, PSFM and optionally SS

# the template sequence in alignment (after removing gaps)
#   shall be a substring of the tpl sequence
# the target sequence in alignment (after removing gaps)
#   shall be exactly same as the tgt sequence

# read one alignment. templateFirst=True indicates template sequence
# is placed before query sequence
# seqTemplatePair specifies the pair of query seq and template names
# queryName specifies the query sequence name
# a user just needs to specify one of templateFirst,
# seqTemplatePair and queryName
def ReadAlignment(alnfile, templateFirst=True, seqTemplatePair=None,
                  queryName=None, returnNames=False):
    with open(alnfile, 'r') as fh:
        content = [line.strip() for line in list(fh)]
        content = [line for line in content if (not line.startswith('#')
                                                and len(line) > 1)]

    if len(content) < 4:
        print("ERROR: the pairwise alignment file %s shall "
              "have at least 4 lines" % alnfile)
        exit(1)

    line = content[0]
    # the first non-empty line shall starst with > followed by a protein name
    if line[0] != '>':
        print('ERROR: incorrect format in the fasta file at line: ', line)
        exit(1)

    firstName = line[1:].split()[0]
    firstSeq = ""

    # read the sequence for this first protein.
    # the sequence could be in several lines
    i = 1
    line = content[i]
    while not line.startswith('>'):
        firstSeq += line
        i += 1
        line = content[i]

    # get the name of the 2nd protein
    secondName = line[1:].split()[0]

    # get the sequence of the 2nd protein
    secondSeq = ''.join(content[i+1:])

    if len(firstSeq) != len(secondSeq):
        print("ERROR: inconsistent query and template sequence length "
              "(including gaps) in alignment file", alnfile)
        exit(1)

    if seqTemplatePair is None:
        if queryName is not None:
            if firstName == queryName:
                if returnNames:
                    return (secondSeq, firstSeq), (secondName, firstName)
                return (secondSeq, firstSeq)
            elif secondName == queryName:
                if returnNames:
                    return (firstSeq, secondSeq), (firstName, secondName)
                return (firstSeq, secondSeq)
            else:
                print('ERROR: targetName does not match any names '
                      'in alignment file: ', queryName, alnfile)
                exit(1)

        elif templateFirst:
            if returnNames:
                return (firstSeq, secondSeq), (firstName, secondName)
            return (firstSeq, secondSeq)
        else:
            if returnNames:
                return (secondSeq, firstSeq), (secondName, firstName)
            return (secondSeq, firstSeq)
    else:
        queryName = seqTemplatePair[0]
        templateName = seqTemplatePair[1]
        if queryName in firstName and templateName in secondName:
            return (secondSeq, firstSeq), (templateName, queryName)
        else:
            return (firstSeq, secondSeq), (templateName, queryName)


# calculate the distance matrix
# coordinates is a list of None and Dict().
# Each Dict has 3D coordinates for some atoms.
# The 3D coordinate of one atom is represented as Vector.
# The results are saved in a dict(), in which each value is a matrix of float16
def CalcDistMatrix(coordinates, apts=['CbCb', 'CaCa', 'NO', 'CaCg', 'CgCg']):
    distMatrix = dict()

    for apt in apts:
        if len(apt) == 2:
            atom1, atom2 = apt[0].upper(), apt[1].upper()
        elif len(apt) == 4:
            atom1, atom2 = apt[:2].upper(), apt[2:].upper()
        else:
            print('ERROR: unsupported atom pair types: %s' % apt)
            exit(1)

        X = [list(c) for c in coordinates[atom1]]
        Xvalid = [0 if not any(c) else
                  1 for c in coordinates[atom1]]
        Y = [list(c) for c in coordinates[atom2]]
        Yvalid = [0 if not any(c) else
                  1 for c in coordinates[atom2]]

        dist = distance_matrix(X, Y).astype(np.float16)
        XYvalid = np.outer(Xvalid, Yvalid)
        np.putmask(dist, XYvalid == 0, InvalidDistance)

        # set the self distance to 0
        if atom1 == atom2:
            np.fill_diagonal(dist, 0)

        distMatrix[apt] = dist

    return distMatrix


# calculate the inter-residue orientation using N, Ca, and Cb atoms
# coordinates is the dict containing N, Ca and Cb atom 3D coordinates
# the orientation angle is represented as Radian instead of Degree
def CalcOriMatrix(coordinates,
                  apts=['Ca1-Cb1-Cb2-Ca2', 'N1-Ca1-Cb1-Cb2',
                        'Ca1-Cb1-Cb2']):
    nan_fill = {3: -1, 4: -4}
    res_num = len(coordinates['CA'])
    oriMatrix = dict()
    for oriType in apts:
        points_map = None
        id_atom = {'1': [], '2': []}
        for _atom in oriType.split('-'):
            _atom_type = _atom[:-1].upper()
            id_atom[_atom[-1]].append(_atom_type)
            _data = np.repeat(coordinates[_atom_type],
                              res_num, axis=0).reshape((res_num, res_num, 3))
            if _atom[-1] == '2':  # j
                _data = np.transpose(_data, (1, 0, 2))
            if points_map is None:
                points_map = _data[:, :, None, :]
            else:
                points_map = np.concatenate((points_map, _data[:, :, None, :]),
                                            axis=2)
    if len(oriType.split('-')) == 3:     # angle
        data_map = Utils.calc_angle_map(points_map, -1)
    elif len(oriType.split('-')) == 4:     # dihedral
        data_map = Utils.calc_dihedral_map(points_map, -4)

    # mask the no residue and CB sites
    res_tags = coordinates['_res_tag']
    CB_tags = coordinates['_CB_tag']

    idx = np.where(res_tags == 0)[0].tolist()   # no residue
    if len(idx) > 0:
        data_map[np.array(idx), :] = nan_fill[len(oriType.split('-'))]
        data_map[:, np.array(idx)] = nan_fill[len(oriType.split('-'))]
    # row
    if ("CA" in id_atom['1']) and ("CB" in id_atom['1']):
        idx_row = np.where(CB_tags == 0)[0].tolist()
        if len(idx_row) > 0:
            data_map[np.array(idx_row), :] = nan_fill[
                len(oriType.split('-'))]
    # col
    if ("CA" in id_atom['2']) and ("CB" in id_atom['2']):
        idx_col = np.where(CB_tags == 0)[0].tolist()
        if len(idx_col) > 0:
            data_map[:, np.array(idx_col)] = nan_fill[
                len(oriType.split('-'))]
    # save
    oriMatrix[oriType] = data_map.astype(np.float16)[:, :, None]

    return oriMatrix


# create matrix for query sequence by copying from templateMatrix
# based upon seq2template mapping
# templateMatrix: python dict() for a set of template matrices
# seq2templateMapping is a tuple of two entries.
# Each is a list of residue indices.
# The first is for seq and the 2nd for template.
# These two lists shall have same length
# for a sequence matrix, if it does not have corresponding entry in template,
# then its value is set to an invalid value
# matrix type could be Orientation or Distance
def CopyTemplateMatrix(seqLen, feature_type,
                       seq2templateMapping, templateMatrices,
                       sub_matrix_index=None):
    if sub_matrix_index is not None:
        (start_i, end_i) = sub_matrix_index
    else:
        start_i = 1
        end_i = seqLen
    seqIndices, templateIndices = seq2templateMapping
    assert len(seqIndices) == len(templateIndices)
    assert len(seqIndices) <= seqLen

    seqMatrix_x = [e for e in seqIndices for _ in
                   range(len(seqIndices))]
    seqMatrix_y = seqIndices * len(seqIndices)

    tempMatrix_x = [e for e in templateIndices for _ in
                    range(len(templateIndices))]
    tempMatrix_y = templateIndices * len(templateIndices)

    seqDistMatrices = dict()
    seqOriMatrices = dict()

    tempDistMatrices, tempOriMatrices = templateMatrices

    for k, v in tempDistMatrices.items():
        if k not in feature_type:
            continue
        seqMatrix = np.full((seqLen, seqLen), InvalidDistance,
                            dtype=np.float16)
        seqMatrix[(seqMatrix_x, seqMatrix_y)] = v[(tempMatrix_x, tempMatrix_y)]
        seqDistMatrices[k] = seqMatrix[start_i-1:end_i, start_i-1:end_i]

    for k, v in tempOriMatrices.items():
        if k not in feature_type:
            continue
        seqMatrix = np.full((seqLen, seqLen), InvalidDegree,
                            dtype=np.float16)
        seqMatrix[(seqMatrix_x, seqMatrix_y)] = v[(tempMatrix_x, tempMatrix_y)]
        seqOriMatrices[k] = \
            np.deg2rad(seqMatrix[start_i-1:end_i, start_i-1:end_i])

    return (seqDistMatrices, seqOriMatrices)


# score an alignment and also copy coordinates from template
def ScoreAlignment(
        alignment, tpl, tgt, feature_type, tplAtomCoordinates=None,
        tplMatrices=None, sub_matrix_index=None, debug=False):
    seq_len = tgt['length']
    if sub_matrix_index is not None:
        (start_i, end_i) = sub_matrix_index
    else:
        start_i = 1
        end_i = seq_len
    # check consistency between alignment, tpl and tgt
    if len(alignment[0]) != len(alignment[1]):
        print('ERROR: the query and template (including gaps) '
              'in alignment have inconsistent length')
        exit(1)

    template = alignment[0].replace('-', '')
    query = alignment[1].replace('-', '')

    # template and query shall be substrings of tpl and tgt, respectively
    tpl_start = tpl['sequence'].find(template)
    if tpl_start == -1:
        print('ERROR: the template sequence in alignment is not '
              'a substring of the sequence in template', tpl['name'])
        print('TPLstring:', tpl['sequence'])
        print('TPLinAli :', template)
        exit(1)

    tgt_start = tgt['sequence'].find(query)
    if tgt_start == -1:
        print('ERROR: the query sequence in alignment is not a substring '
              'of the sequence in query')
        exit(1)

    # wrong if tgt_start is not 0, here we require that the query sequence
    # in alignment is exactly same as that in tgt
    assert (tgt_start == 0)

    # a flag vector indicating insertion in query, indicated by a flag 1
    # Note: InsertX is only related to template and alignment
    insertX = np.zeros((seq_len, 1), dtype=np.uint8)

    # effective coordinates copied from aligned template positions
    if tplAtomCoordinates is not None:
        copiedCoordinates = [None] * seq_len
    else:
        copiedCoordinates = None

    # for debug only
    if debug:
        XYresidues = -np.ones((seq_len, 4), dtype=np.int16)
    else:
        XYresidues = None

    tgtAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tgt['sequence'])
    tplAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tpl['sequence'])

    seq2templateMapping = [[], []]

    sequentialFeatures = dict()

    if 'SeqSimlarity' in feature_type['node']:
        sequentialFeatures['SeqSimlarity'] = np.zeros(
            (seq_len, 4), dtype=np.float16)
    if 'ProfileSimlarity' in feature_type['node']:
        sequentialFeatures['ProfileSimlarity'] = np.zeros(
            (seq_len, 6), dtype=np.float16)
    if 'SS3' in feature_type['node']:
        sequentialFeatures['SS3'] = np.zeros(
            (seq_len, 1), dtype=np.float16)
    if 'SS8' in feature_type['node']:
        sequentialFeatures['SS8'] = np.zeros(
            (seq_len, 1), dtype=np.float16)
    if 'ACC' in feature_type['node']:
        sequentialFeatures['ACC'] = np.zeros(
            (seq_len, 1), dtype=np.float16)
    if 'Env' in feature_type['node']:
        sequentialFeatures['Env'] = np.zeros(
            (seq_len, 2), dtype=np.float16)

    # index for tgt and tpl, respectively
    tgt_pos = tgt_start
    tpl_pos = tpl_start

    # alignment[0] represents the template sequence with gaps
    for al_pos in range(len(alignment[0])):
        if alignment[0][al_pos] == '-' and alignment[1][al_pos] == '-':
            print('WARNING: there shall not be two gaps '
                  'at any aligned positions')
            exit(1)

        # there is a gap in template, i.e., an insertion in query
        if alignment[0][al_pos] == '-':
            # need to generate some flag features for insertion in query
            insertX[tgt_pos] = 1
            if debug:
                XYresidues[tgt_pos][0] = tgt_pos
                XYresidues[tgt_pos][1] = ord(tgt['sequence'][tgt_pos]) - \
                    ord('A')

            tgt_pos += 1
            continue

        # if there is a gap in query, just skip it
        if alignment[1][al_pos] == '-':
            tpl_pos += 1
            # no need to generate flag features for insertion in template
            continue

        # match here
        if debug:
            XYresidues[tgt_pos][0] = tgt_pos
            XYresidues[tgt_pos][1] = ord(tgt['sequence'][tgt_pos]) - \
                ord('A')
            XYresidues[tgt_pos][2] = tpl_pos
            XYresidues[tgt_pos][3] = ord(tpl['sequence'][tpl_pos]) - \
                ord('A')

        seq2templateMapping[0].append(tgt_pos)
        seq2templateMapping[1].append(tpl_pos)

        tAA = tplAAIndex[tpl_pos]
        sAA = tgtAAIndex[tgt_pos]

        if 'SeqSimlarity' in feature_type['node']:
            seq_Id = int(tAA == sAA)
            blosum80 = SimilarityScore.newBLOSUM80[tAA, sAA]
            blosum62 = SimilarityScore.newBLOSUM62[tAA, sAA]
            blosum45 = SimilarityScore.newBLOSUM45[tAA, sAA]
            sequentialFeatures['SeqSimlarity'][tgt_pos] = np.array(
                [seq_Id, blosum80, blosum62, blosum45], dtype=np.float32)

        if 'ProfileSimlarity' in feature_type['node']:
            cc = SimilarityScore.newCC50[tAA, sAA]
            hdsm = SimilarityScore.newHDSM[tAA, sAA]
            x, y = tpl_pos, tgt_pos
            spScore = SimilarityScore.MutationOf2Pos6(x, y, tpl, tgt)
            spScore_ST = SimilarityScore.MutationOf2Pos6_ST(
                x, y, tpl, tgt)
            pmScore = SimilarityScore.MutationOf2Pos5(x, y, tpl, tgt)
            pmScore_ST = SimilarityScore.MutationOf2Pos5_ST(
                x, y, tpl, tgt)
            sequentialFeatures['ProfileSimlarity'][tgt_pos] = np.array(
                [spScore, spScore_ST, pmScore, pmScore_ST, cc, hdsm],
                dtype=np.float32)

        if 'SS3' in feature_type['node']:
            sequentialFeatures['SS3'][tgt_pos] = \
                SimilarityScore.SSMutationScore_3State(x, y, tpl, tgt)

        if 'SS8' in feature_type['node']:
            sequentialFeatures['SS8'][tgt_pos] = \
                SimilarityScore.SSMutationScore_6State(x, y, tpl, tgt)

        if 'ACC' in feature_type['node']:
            sequentialFeatures['ACC'][tgt_pos] = \
                SimilarityScore.ACCMutationScore_3State(x, y, tpl, tgt)

        if 'Env' in feature_type['node']:
            envScore = SimilarityScore.SingletonScore_ProfileBased(
                x, y, tpl, tgt)
            wsEnvScore = SimilarityScore.SingletonScore_WS(
                x, y, tpl, tgt)
            sequentialFeatures['Env'][tgt_pos] = np.array(
                [envScore, wsEnvScore], dtype=np.float32)

        # copy 3D coordinates from template.
        # It is possible that the template coordinate = None
        if tplAtomCoordinates is not None:
            copiedCoordinates[tgt_pos] = tplAtomCoordinates[tpl_pos]

        tpl_pos += 1
        tgt_pos += 1

    if 'GapPosition' in feature_type['node']:
        sequentialFeatures['GapPosition'] = insertX.astype(np.float32)

    # sub_matrix_index
    for feature in feature_type['node']:
        sequentialFeatures[feature] = sequentialFeatures[
            feature][start_i-1:end_i]

    if tplMatrices is not None:
        copiedMatrix = CopyTemplateMatrix(seq_len, feature_type['edge'],
                                          seq2templateMapping, tplMatrices,
                                          sub_matrix_index)
        return sequentialFeatures, copiedMatrix, XYresidues

    return sequentialFeatures, copiedCoordinates[start_i-1:end_i], XYresidues


def GenSeqSimlarity(Seq2TplMapping, tpl, tgt):
    SeqSimlarity = np.zeros((tgt['length'], 4), dtype=np.float32)
    tgtAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tgt['sequence'])
    tplAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tpl['sequence'])

    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        tAA = tplAAIndex[tpl_pos]
        sAA = tgtAAIndex[tgt_pos]
        seq_Id = int(tAA == sAA)
        blosum80 = SimilarityScore.newBLOSUM80[tAA, sAA]
        blosum62 = SimilarityScore.newBLOSUM62[tAA, sAA]
        blosum45 = SimilarityScore.newBLOSUM45[tAA, sAA]
        SeqSimlarity[tgt_pos] = np.array(
            [seq_Id, blosum80, blosum62, blosum45], dtype=np.float32)
    return SeqSimlarity


# Returns: A numpy array of shape (L, 6)
def GenProfileSimlarity(Seq2TplMapping, tpl, tgt):
    ProfileSimlarity = np.zeros((tgt['length'], 6), dtype=np.float32)
    tgtAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tgt['sequence'])
    tplAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tpl['sequence'])

    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        tAA = tplAAIndex[tpl_pos]
        sAA = tgtAAIndex[tgt_pos]
        cc = SimilarityScore.newCC50[tAA, sAA]
        hdsm = SimilarityScore.newHDSM[tAA, sAA]
        spScore = SimilarityScore.MutationOf2Pos6(
            tpl_pos, tgt_pos, tpl, tgt)
        spScore_ST = SimilarityScore.MutationOf2Pos6_ST(
            tpl_pos, tgt_pos, tpl, tgt)
        pmScore = SimilarityScore.MutationOf2Pos5(tpl_pos, tgt_pos, tpl, tgt)
        pmScore_ST = SimilarityScore.MutationOf2Pos5_ST(
            tpl_pos, tgt_pos, tpl, tgt)
        ProfileSimlarity[tgt_pos] = np.array(
            [spScore, spScore_ST, pmScore, pmScore_ST, cc, hdsm],
            dtype=np.float32)
    return ProfileSimlarity


# Returns: A numpy array of shape (L, 1)
def GenSS3(Seq2TplMapping, tpl, tgt):
    SS3 = np.zeros((tgt['length'], 1), dtype=np.float32)
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        SS3[tgt_pos] = SimilarityScore.SSMutationScore_3State(
            tpl_pos, tgt_pos, tpl, tgt)
    return SS3


# Returns: A numpy array of shape (L, 1)
def GenSS8(Seq2TplMapping, tpl, tgt):
    SS8 = np.zeros((tgt['length'], 1), dtype=np.float32)
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        SS8[tgt_pos] = SimilarityScore.SSMutationScore_6State(
            tpl_pos, tgt_pos, tpl, tgt)
    return SS8


# Returns: A numpy array of shape (L, 1)
def GenACC(Seq2TplMapping, tpl, tgt):
    ACC = np.zeros((tgt['length'], 1), dtype=np.float32)
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        ACC[tgt_pos] = SimilarityScore.ACCMutationScore_3State(
            tpl_pos, tgt_pos, tpl, tgt)
    return ACC


# Returns: A numpy array of shape (L, 2)
#   default value: -1
def GenPhi(Seq2TplMapping, tpl, tgt):
    Phi = np.zeros((tgt['length'], 2), dtype=np.float32) - 1
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        Phi[tgt_pos] = np.array(
            [np.sin(tpl['Phi'][tpl_pos]), np.cos(tpl['Phi'][tpl_pos])])
    return Phi


# Returns: A numpy array of shape (L, 2)
#   default value: -1
def GenPsi(Seq2TplMapping, tpl, tgt):
    Psi = np.zeros((tgt['length'], 2), dtype=np.float32) - 1
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        Psi[tgt_pos] = np.array(
            [np.sin(tpl['Psi'][tpl_pos]), np.cos(tpl['Psi'][tpl_pos])])
    return Psi


# Returns: A numpy array of shape (L, 22)
def GenTemplateAAType(Seq2TplMapping, tpl, tgt, n_alphabet=22,
                      RESIDUE_TYPE="ACDEFGHIKLMNPQRSTVWY-XBZUOJ"):
    template_sequence = tgt['length'] * ['-']
    for tgt_pos, tpl_pos in zip(Seq2TplMapping[0], Seq2TplMapping[1]):
        template_sequence[tgt_pos] = tpl['sequence'][tpl_pos]
    Template_index = np.array(
        [RESIDUE_TYPE.index(_) for _ in template_sequence
         if _ in RESIDUE_TYPE])
    Template_index[Template_index >= n_alphabet] = n_alphabet - 1
    return np.eye(n_alphabet)[Template_index]


# Generates features for those query positions with effective
# aligned template positions
def GenFeature4Alignment(alignment, tpl, tgt, feature_type,
                         tplAtomCoordinates=None, tplMatrices=None,
                         sub_matrix_index=None):
    sequentialFeatures, copied, XYresidues = ScoreAlignment(
        alignment, tpl, tgt, feature_type,
        tplAtomCoordinates=tplAtomCoordinates,
        tplMatrices=tplMatrices, sub_matrix_index=sub_matrix_index)

    if tplMatrices is not None:
        copiedDistMatrix, copiedOriMatrix = copied
        return sequentialFeatures, copiedDistMatrix, copiedOriMatrix

    if tplAtomCoordinates is not None:
        copiedDistMatrix = CalcDistMatrix(copied)
        copiedOriMatrix = CalcOriMatrix(copied)
        return sequentialFeatures, copiedDistMatrix, copiedOriMatrix

    return sequentialFeatures


# Generates features for those query positions with effective
# alignment template positions
# for sequence in MSA, consider them as different sequence
def GenFeature4MSA(alignment, tpl, tgt, a2m, feature_type,
                   tplAtomCoordinates=None, tplMatrices=None,
                   sub_matrix_index=None):
    sequentialFeatures, copied, XYresidues = ScoreAlignment(
        alignment, tpl, tgt, a2m, feature_type,
        tplAtomCoordinates=tplAtomCoordinates,
        tplMatrices=tplMatrices, sub_matrix_index=sub_matrix_index)

    if tplMatrices is not None:
        copiedDistMatrix, copiedOriMatrix = copied
        return sequentialFeatures, copiedDistMatrix, copiedOriMatrix

    if tplAtomCoordinates is not None:
        copiedDistMatrix = CalcDistMatrix(copied)
        copiedOriMatrix = CalcOriMatrix(copied)
        return sequentialFeatures, copiedDistMatrix, copiedOriMatrix

    return sequentialFeatures


def GenerateAlignmentFeatures(seqTemplatePair, queryData, feature_type,
                              aliDir=None, tplDir=None, sub_matrix_index=None):
    template, query = seqTemplatePair.split('-')
    alnfile = template + '-' + query + '.fasta'
    alnfile = os.path.join(aliDir, alnfile)

    if not os.path.isfile(alnfile):
        print("%s alignment file not found" % alnfile)
    alignment = ReadAlignment(alnfile, seqTemplatePair)

    tplFile = os.path.join(tplDir, template + '.tpl.pkl')
    if not os.path.isfile(tplFile):
        print('%s template file not found in folder %s' % (template, tplDir))
        exit(1)

    with open(tplFile, 'rb') as fh:
        tpl = pickle.load(fh, encoding='latin1')

    templateMatrices = (tpl['atomDistMatrix'], tpl['atomOrientationMatrix'])
    sequentialFeatures, distMatrix, oriMatrix = GenFeature4Alignment(
        alignment, tpl, queryData, feature_type, tplMatrices=templateMatrices,
        sub_matrix_index=sub_matrix_index)

    feature = dict()
    feature['SimilarityScore'] = sequentialFeatures
    feature['tplDistMatrix'] = distMatrix
    feature['tplOriMatrix'] = oriMatrix
    feature['template'] = tpl['name']
    feature['alignment'] = alignment

    return feature


# Calculate the MSA SimilarityScore
def ScoreMSASimilarity(
        alignment, tpl, tgt, MSA, sub_matrix_index=None):
    seq_len = tgt['length']
    depth_MSA = len(MSA)
    if sub_matrix_index is not None:
        (start_i, end_i) = sub_matrix_index
    else:
        start_i = 1
        end_i = seq_len
    # check consistency between alignment, tpl and tgt
    if len(alignment[0]) != len(alignment[1]):
        print('ERROR: the query and template (including gaps) '
              'in alignment have inconsistent length')
        exit(1)

    template = alignment[0].replace('-', '')
    query = alignment[1].replace('-', '')

    # template and query shall be substrings of tpl and tgt, respectively
    tpl_start = tpl['sequence'].find(template)
    if tpl_start == -1:
        print('ERROR: the template sequence in alignment is not '
              'a substring of the sequence in template', tpl['name'])
        print('TPLstring:', tpl['sequence'])
        print('TPLinAli :', template)
        exit(1)

    tgt_start = tgt['sequence'].find(query)
    if tgt_start == -1:
        print('ERROR: the query sequence in alignment is not a substring '
              'of the sequence in query')
        exit(1)

    # wrong if tgt_start is not 0, here we require that the query sequence
    # in alignment is exactly same as that in tgt
    assert (tgt_start == 0)

    # a flag vector indicating insertion in query, indicated by a flag 1
    # Note: InsertX is only related to template and alignment
    insertX = np.zeros((seq_len, 1), dtype=np.uint8)

    tplAAIndex = SequenceUtils.Seq2OrderOf3LetterCode(tpl['sequence'])
    msaAAIndex = []
    for seq in MSA:
        msaAAIndex.append(SequenceUtils.Seq2OrderOf3LetterCode(seq))

    MSASeqSimlarity = np.zeros(
        (depth_MSA, seq_len, 4), dtype=np.float16)

    # index for tgt and tpl, respectively
    tgt_pos = tgt_start
    tpl_pos = tpl_start

    # alignment[0] represents the template sequence with gaps
    for al_pos in range(len(alignment[0])):
        if alignment[0][al_pos] == '-' and alignment[1][al_pos] == '-':
            print('WARNING: there shall not be two gaps '
                  'at any aligned positions')
            exit(1)

        # there is a gap in template, i.e., an insertion in query
        if alignment[0][al_pos] == '-':
            # need to generate some flag features for insertion in query
            insertX[tgt_pos] = 1
            tgt_pos += 1
            continue

        # if there is a gap in query, just skip it
        if alignment[1][al_pos] == '-':
            tpl_pos += 1
            # no need to generate flag features for insertion in template
            continue

        # match here
        tAA = tplAAIndex[tpl_pos]
        for ba in range(depth_MSA):
            sAA = msaAAIndex[ba][tgt_pos]
            seq_Id = int(tAA == sAA)
            blosum80 = SimilarityScore.newBLOSUM80[tAA, sAA]
            blosum62 = SimilarityScore.newBLOSUM62[tAA, sAA]
            blosum45 = SimilarityScore.newBLOSUM45[tAA, sAA]
            MSASeqSimlarity[ba, tgt_pos] = np.array(
                [seq_Id, blosum80, blosum62, blosum45], dtype=np.float32)
        tpl_pos += 1
        tgt_pos += 1

    # sub_matrix_index
    MSASeqSimlarity = MSASeqSimlarity[:, start_i-1:end_i]

    return MSASeqSimlarity


def GenerateMSASimilarityFeatures(
        seqTemplatePair, queryData, MSA, aliDir=None, tplDir=None,
        sub_matrix_index=None):

    template, query = seqTemplatePair.split('-')
    alnfile = template + '-' + query + '.fasta'
    alnfile = os.path.join(aliDir, alnfile)

    if not os.path.isfile(alnfile):
        print("%s alignment file not found" % alnfile)
    alignment = ReadAlignment(alnfile, seqTemplatePair)

    tplFile = os.path.join(tplDir, template + '.tpl.pkl')
    if not os.path.isfile(tplFile):
        print('%s template file not found in folder %s' % (template, tplDir))
        exit(1)

    with open(tplFile, 'rb') as fh:
        tpl = pickle.load(fh, encoding='latin1')

    sequentialFeatures = ScoreMSASimilarity(
        alignment, tpl, queryData, MSA, sub_matrix_index=sub_matrix_index)

    return sequentialFeatures


def GetSeq2TplMapping(alignment, tpl, tgt):
    seq2templateMapping = [[], []]
    template = alignment[0].replace('-', '')
    query = alignment[1].replace('-', '')
    GapPosition = np.zeros((tgt['length'], 1), dtype=np.uint8)

    # template and query shall be substrings of tpl and tgt, respectively
    tpl_start = tpl['sequence'].find(template)
    if tpl_start == -1:
        print('ERROR: the template sequence in alignment is not '
              'a substring of the sequence in template', tpl['name'])
        print('TPLstring:', tpl['sequence'])
        print('TPLinAli :', template)
        exit(1)

    tgt_start = tgt['sequence'].find(query)
    if tgt_start == -1:
        print('ERROR: the query sequence in alignment is not a substring '
              'of the sequence in query', tgt['name'])
        exit(1)

    # wrong if tgt_start is not 0, here we require that the query sequence
    # in alignment is exactly same as that in tgt
    assert (tgt_start == 0)

    tgt_pos = tgt_start
    tpl_pos = tpl_start

    for al_pos in range(len(alignment[0])):
        if alignment[0][al_pos] == '-' and alignment[1][al_pos] == '-':
            print('WARNING: there shall not be two gaps '
                  'at any aligned positions')
            continue

        # there is a gap in template, i.e., an insertion in query
        if alignment[0][al_pos] == '-':
            GapPosition[tgt_pos] = 1
            tgt_pos += 1
            continue

        # if there is a gap in query, just skip it
        if alignment[1][al_pos] == '-':
            tpl_pos += 1
            # no need to generate flag features for insertion in template
            continue

        # match here
        seq2templateMapping[0].append(tgt_pos)
        seq2templateMapping[1].append(tpl_pos)
        tpl_pos += 1
        tgt_pos += 1
    return seq2templateMapping, GapPosition


def GenerateMSAFeatures(seqTemplatePair, queryData, msaData, feature_type,
                        aliDir=None, tplDir=None, sub_matrix_index=None):
    template, query = seqTemplatePair.split('-')
    alnfile = template + '-' + query + '.fasta'
    alnfile = os.path.join(aliDir, alnfile)

    if not os.path.isfile(alnfile):
        print("%s alignment file not found" % alnfile)
    alignment = ReadAlignment(alnfile, seqTemplatePair)

    tplFile = os.path.join(tplDir, template + '.tpl.pkl')
    if not os.path.isfile(tplFile):
        print('%s template file not found in folder %s' % (template, tplDir))
        exit(1)
    with open(tplFile, 'rb') as fh:
        tpl = pickle.load(fh, encoding='latin1')

    templateMatrices = (tpl['atomDistMatrix'], tpl['atomOrientationMatrix'])

    sequentialFeatures, distMatrix, oriMatrix = GenFeature4MSA(
        alignment, tpl, queryData, msaData, feature_type,
        tplMatrices=templateMatrices, sub_matrix_index=sub_matrix_index)

    feature = dict()
    feature['SimilarityScore'] = sequentialFeatures
    feature['tplDistMatrix'] = distMatrix
    feature['tplOriMatrix'] = oriMatrix
    feature['template'] = tpl['name']
    feature['alignment'] = alignment

    return feature
