
import os
import numpy as np
import torch
from torch.nn import functional as F
from .LoadTPLTGT import load_tgt, load_tpl
from . import AlignmentUtils
import Bio.PDB
import warnings

# parse raw pdb
warnings.filterwarnings("ignore")
pdb_parser = Bio.PDB.PDBParser(QUIET=True)


# Constants
InvalidDistance = -1
NoCBDegree = np.float16(-540)
InvalidDegree = np.float16(-720)

RESIDUE_TYPES = "ARNDCQEGHILKMFPSTWYVX-BZUOJ"
UNK_RESIDUE_INDEX = 20 # unknown residue (X and BZUOJ)
GAP_RESIDUE_INDEX = 21 # gap

def pair_pred_transform(pred, label=None, symmetric=False):
    '''
    Apply permute, symmetric, softmax (and/or align) on the pred.
    Agrs:
        pred (array): the logits of model, (C, H, W)
        label (array): the label map, (H, W)
        symmetric (bool): symmetric the pred
    '''
    # (C, H, W) -> (H, W, C)
    pred = pred.permute(1, 2, 0)
    # align the pred by label
    if label is not None:
        pred = pred[:label.shape[0], :label.shape[1], ]
    if symmetric:
        pred = (pred + pred.transpose(0, 1)) * 0.5  # symmetric
    # softmax
    pred = F.softmax(pred, dim=-1)
    return pred

def MSA_encoding(MSA, n_alphabet=21, encoding_type='index', ):
    '''
    Encoding the MSA by onehot or index.

    Agrs:
        MSA (list): the MSA, i.e. sequence list.
        n_alphabet (int): the size for encoding (>=21).
        encoding_type (str): the encoding type: 'onehot' or 'index'.
    '''
    assert n_alphabet>=21, "encoding size must >= 21"

    # index
    MSA_index = np.array([[RESIDUE_TYPES.index(_) for _ in line if _ in RESIDUE_TYPES] for line in MSA])
    MSA_index[MSA_index>GAP_RESIDUE_INDEX] = UNK_RESIDUE_INDEX

    # onehot
    if encoding_type=='onehot': return np.eye(n_alphabet)[MSA_index]

    return MSA_index

def calc_seq_weight(MSA_matrix, encoding_type='index', n_alphabet=21, sim_cutoff=0.8):
    '''
    Calc the seq weight by simlarity.

    Agrs:
        MSA_matrix (K, L, D): the onehot MSA.
        encoding_type (str): 'onehot' or 'index'
        n_alphabet (int): the size for encoding if encoding_type is 'index'.
        sim_cutoff (float): the similarity cutoff.
    '''
    if encoding_type == 'index': MSA_matrix = np.eye(n_alphabet)[MSA_matrix]  # convert to onehot
    K, L, D = MSA_matrix.shape
    msa_onehot = np.reshape(MSA_matrix, [K, -1])
    sim_map = np.matmul(msa_onehot, np.transpose(msa_onehot))
    greater_map = np.greater(sim_map, L * sim_cutoff)
    sim_num = np.sum(greater_map, axis=1)*1.0
    seq_weight = 1.0 * np.reciprocal(sim_num)

    return seq_weight

def TemplateFeature1d(TGT_file, TPL_files, ALN_files, FeatureType):
    tgt = load_tgt(TGT_file)
    template_feat1d = []

    for tpl_file, aln_file in zip(TPL_files, ALN_files):
        feature1d = dict()
        alignment = AlignmentUtils.ReadAlignment(aln_file)
        tpl = load_tpl(tpl_file)
        mapping, GapPosition = AlignmentUtils.GetSeq2TplMapping(
            alignment, tpl, tgt)
        if 'GapPosition' in FeatureType:        # shape: (L, 1)
            feature1d['GapPosition'] = GapPosition
        if 'SeqSimlarity' in FeatureType:       # shape: (L, 4)
            feature1d['SeqSimlarity'] = AlignmentUtils.GenSeqSimlarity(
                mapping, tpl, tgt)
        if 'ProfileSimlarity' in FeatureType:   # shape: (L, 6)
            feature1d['ProfileSimlarity'] = AlignmentUtils.GenProfileSimlarity(
                mapping, tpl, tgt)
        if 'SS3' in FeatureType:                # shape: (L, 1)
            feature1d['SS3'] = AlignmentUtils.GenSS3(mapping, tpl, tgt)
        if 'SS8' in FeatureType:                # shape: (L, 1)
            feature1d['SS8'] = AlignmentUtils.GenSS8(mapping, tpl, tgt)
        if 'ACC' in FeatureType:                # shape: (L, 1)
            feature1d['ACC'] = AlignmentUtils.GenACC(mapping, tpl, tgt)
        if 'TemplateAAType' in FeatureType:     # shape: (L, 22)
            feature1d['TemplateAAType'] = AlignmentUtils.GenTemplateAAType(
                mapping, tpl, tgt)
        if 'Phi' in FeatureType:                # shape: (L, 2)
            feature1d['Phi'] = AlignmentUtils.GenPhi(mapping, tpl, tgt)
        if 'Psi' in FeatureType:                # shape: (L, 2)
            feature1d['Psi'] = AlignmentUtils.GenPsi(mapping, tpl, tgt)

        # concatenate all template 1d feature
        template_feat1d.append(
            np.concatenate([feature1d[_] for _ in feature1d], axis=-1))

    return np.stack(template_feat1d)

def TemplateFeature2d(TGT_file, TPL_files, ALN_files, FeatureType):
    tgt = load_tgt(TGT_file)
    template_feat2d = []
    BasicFeatureType = list(set([ftype.split('_')[0] for
                                 ftype in FeatureType]))

    for tpl_file, aln_file in zip(TPL_files, ALN_files):
        feature2d = dict()
        alignment = AlignmentUtils.ReadAlignment(aln_file)
        tpl = load_tpl(tpl_file)
        mapping, GapPosition = AlignmentUtils.GetSeq2TplMapping(
            alignment, tpl, tgt)

        templateMatrices = (tpl['atomDistMatrix'],
                            tpl['atomOrientationMatrix'])

        copiedDistMatrix, copiedOriMatrix = AlignmentUtils.CopyTemplateMatrix(
            tgt['length'], BasicFeatureType, mapping, templateMatrices)

        for ftype in FeatureType:
            if ftype == 'TemplateAAType':   # shape: (L, L, 44)
                template_aatype = AlignmentUtils.GenTemplateAAType(
                    mapping, tpl, tgt)
                TemplateAAType = np.concatenate(
                    (np.tile(template_aatype[None, :, :],
                             [tgt['length'], 1, 1]),
                     np.tile(template_aatype[:, None, :],
                             [1, tgt['length'], 1])), axis=-1)
                feature2d[ftype] = TemplateAAType

            else:
                # CaCa, CbCb, NO, Ca1Cb1Cb2, N1Ca1Cb1Cb2, Ca1Cb1Cb2Ca2
                # shape: (L, L, 1)
                if ftype in BasicFeatureType:
                    if ftype in copiedDistMatrix:
                        feature2d[ftype] = np.expand_dims(
                            copiedDistMatrix[ftype])
                    elif ftype in copiedOriMatrix:
                        feature2d[ftype] = np.expand_dims(
                            copiedOriMatrix[ftype])

                # CaCa_norm, CbCb_norm, NO_norm, shape: (L, L, 1)
                if ftype.endswith('norm'):
                    _ftype = ftype.split('_')[0]
                    dist_norm = np.where(
                        copiedDistMatrix[_ftype] > 0,
                        copiedDistMatrix[_ftype]/20,
                        copiedDistMatrix[_ftype])
                    dist_norm = np.where(
                        dist_norm > 0, np.clip(dist_norm, 0, 1),
                        dist_norm)
                    feature2d[ftype] = np.expand_dims(dist_norm, -1)

                # CaCa_disc, CbCb_disc, NO_disc, shape: (L, L, 39)
                if ftype.endswith('disc'):
                    _ftype = ftype.split('_')[0]
                    lower_break = np.linspace(3.25, 50.75, 39)
                    upper_break = np.concatenate(
                        [lower_break[1:], np.array([1e8], dtype=np.float32)],
                        axis=-1)
                    dist_map = copiedDistMatrix[_ftype]
                    dist_disc = (
                        (dist_map[:, :, None] > lower_break).astype(
                            np.float32) *
                        (dist_map[:, :, None] < upper_break).astype(
                            np.float32))

                    feature2d[ftype] = dist_disc

                # CaCa_disc_value, CbCb_dis_value, NO_disc_value,
                # shape: (L, L, 41)
                if ftype.endswith('disc_value'):
                    _ftype = ftype.split('_')[0]
                    lower_break = np.linspace(3.25, 50.75, 39)
                    upper_break = np.concatenate(
                        [lower_break[1:], np.array([1e8], dtype=np.float32)],
                        axis=-1)
                    dist_map = copiedDistMatrix[_ftype]
                    dist_disc = (
                        (dist_map[:, :, None] > lower_break).astype(
                            np.float32) *
                        (dist_map[:, :, None] < upper_break).astype(
                            np.float32))
                    # gap in template
                    gap = np.where(dist_map < 0, 1, 0).astype(np.float32)
                    # distance larger than 50.75
                    over = np.where(dist_map >= lower_break[-1], 1, 0)
                    tmp_break = np.concatenate(
                        [lower_break, np.array([1e8], dtype=np.float32)],
                        axis=-1)
                    value = ((dist_map -
                              tmp_break[np.digitize(dist_map, lower_break)]) /
                             1.25)
                    value = np.where(gap, -1, value)
                    value = np.where(over, 0, value)

                    dist_disc = np.concatenate(
                        [dist_disc, np.expand_dims(gap, -1),
                         np.expand_dims(value, -1)], axis=-1)
                    feature2d[ftype] = dist_disc

                # Ca1Cb1Cb2_sincos, N1Ca1Cb1Cb2_sincos, Ca1Cb1Cb2Ca2_sincos,
                # shape (L, L, 2)
                if ftype.endswith('sincos'):
                    _ftype = ftype.split('_')[0]
                    sin_ftype = np.where(
                        copiedOriMatrix[_ftype] > np.deg2rad(np.float32(-720)),
                        np.sin(copiedOriMatrix[_ftype]), 0)
                    cos_ftype = np.where(
                        copiedOriMatrix[_ftype] > np.deg2rad(np.float32(-720)),
                        np.cos(copiedOriMatrix[_ftype]), 0)
                    feature2d[ftype] = np.stack((sin_ftype, cos_ftype),
                                                axis=-1)

        template_feat2d.append(
            np.concatenate([feature2d[_] for _ in feature2d], axis=-1))

    return np.stack(template_feat2d)

def calc_angle_map(data, nan_fill=-4):
    """
    Calc the ange for a map with three points.
        data: shape: [N, N, 3, 3].
    """
    ba = data[:, :, 0, :] - data[:, :, 1, :]
    bc = data[:, :, 2, :] - data[:, :, 1, :]
    angle_radian = np.arccos(np.einsum('ijk,ijk->ij', ba, bc) /
                             (np.linalg.norm(ba, axis=-1) *
                              np.linalg.norm(bc, axis=-1)))
    # angle_degree = np.degrees(angle_radian)
    return np.nan_to_num(angle_radian, nan=nan_fill)

def calc_dihedral_map(data, nan_fill=-4):
    """
    Calc the dihedral ange for a map with four points.
        data: shape: [N, N, 4, 3].
    """
    b01 = -1.0 * (data[:, :, 1, :] - data[:, :, 0, :])
    b12 = data[:, :, 2, :] - data[:, :, 1, :]
    b23 = data[:, :, 3, :] - data[:, :, 2, :]

    b12 = b12 / np.linalg.norm(b12, axis=-1)[:, :, None]

    v = b01 - np.einsum('ijk,ijk->ij', b01, b12)[:, :, None]*b12
    w = b23 - np.einsum('ijk,ijk->ij', b23, b12)[:, :, None]*b12

    x = np.einsum('ijk,ijk->ij', v, w)
    y = np.einsum('ijk,ijk->ij', np.cross(b12, v, axis=-1), w)

    return np.nan_to_num(np.arctan2(y, x), nan=nan_fill)

