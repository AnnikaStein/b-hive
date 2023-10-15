import gc
import uproot
import numpy as np
import pandas as pd

import awkward as ak

print("Imported libs")

inputs_root = "/eos/cms/store/group/phys_btag/ParticleTransformer/merged/ntuple_merged_23*.root:deepntuplizer/tree"

cpf_branches = ['Cpfcan_BtagPf_trackEtaRel',
                'Cpfcan_BtagPf_trackPtRel',
                'Cpfcan_BtagPf_trackPPar',
                'Cpfcan_BtagPf_trackDeltaR',
                'Cpfcan_BtagPf_trackPParRatio',
                'Cpfcan_BtagPf_trackSip2dVal',
                'Cpfcan_BtagPf_trackSip2dSig',
                'Cpfcan_BtagPf_trackSip3dVal',
                'Cpfcan_BtagPf_trackSip3dSig',
                'Cpfcan_BtagPf_trackJetDistVal',
                'Cpfcan_ptrel',
                'Cpfcan_drminsv',
                'Cpfcan_VTX_ass',
                'Cpfcan_puppiw',
                'Cpfcan_chi2',
                'Cpfcan_quality'
]

npf_branches = ['Npfcan_ptrel',
                'Npfcan_deltaR',
                'Npfcan_isGamma',
                'Npfcan_HadFrac',
                'Npfcan_drminsv',
                'Npfcan_puppiw'
]

vtx_branches = ['sv_deltaR',
                'sv_mass',
                'sv_ntracks',
                'sv_chi2',
                'sv_normchi2',
                'sv_dxy',
                'sv_dxysig',
                'sv_d3d',
                'sv_d3dsig',
                'sv_costhetasvpv',
                'sv_enratio',
                'sv_pt'
]

cpf_pts_branches = ['Cpfcan_pt',
                    'Cpfcan_eta',
                    'Cpfcan_phi',
                    'Cpfcan_e'
]

npf_pts_branches = ['Npfcan_pt',
                    'Npfcan_eta',
                    'Npfcan_phi',
                    'Npfcan_e'
]

vtx_pts_branches = ['sv_pt',
                    'sv_eta',
                    'sv_phi',
                    'sv_e'
]

df_cpf = uproot.concatenate(inputs_root, cpf_branches, library="ak")
df_npf = uproot.concatenate(inputs_root, npf_branches, library="ak")
df_vtx = uproot.concatenate(inputs_root, vtx_branches, library="ak")
df_cpf_pts = uproot.concatenate(inputs_root, cpf_pts_branches, library="ak")
df_npf_pts = uproot.concatenate(inputs_root, npf_pts_branches, library="ak")
df_vtx_pts = uproot.concatenate(inputs_root, vtx_pts_branches, library="ak")

print("Loaded data")

hflav = uproot.concatenate(inputs_root, 'jet_hflav', library="ak")['jet_hflav']
hFlav = ak.to_numpy(ak.flatten(hflav, axis=0))

df_cpf_clip = df_cpf #ak.flatten(df_cpf)
df_npf_clip = df_npf #ak.flatten(df_npf)
df_vtx_clip = df_vtx #ak.flatten(df_vtx)
df_cpf_pts_clip = df_cpf_pts #ak.flatten(df_cpf_pts)
df_npf_pts_clip = df_npf_pts #ak.flatten(df_npf_pts)
df_vtx_pts_clip = df_vtx_pts #ak.flatten(df_vtx_pts)

print("Prepared the data")

gc.collect()

print(df_vtx_clip.type)
print(df_vtx_pts_clip.type)

def quantile_min_max(feature,group='cpf',candidate=None):
    if group=='cpf':
        print(feature,group,candidate)
        array_np = ak.to_numpy(ak.flatten(df_cpf_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        print(mini_)
        print(maxi_)
        print(np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)]))
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
    elif group=='npf':
        print(feature,group,candidate)
        #array_np = ak.to_numpy(df_npf_clip[feature][:,candidate])
        array_np = ak.to_numpy(ak.flatten(df_npf_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
    elif group=='vtx':
        print(feature,group,candidate)
        #array_np = ak.to_numpy(df_vtx_clip[feature][:,candidate])
        array_np = ak.to_numpy(ak.flatten(df_vtx_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
    elif group=='cpf_pts':
        print(feature,group,candidate)
        #array_np = ak.to_numpy(df_cpf_pts_clip[feature][:,candidate])
        array_np = ak.to_numpy(ak.flatten(df_cpf_pts_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
    elif group=='npf_pts':
        print(feature,group,candidate)
        #array_np = ak.to_numpy(df_npf_pts_clip[feature][:,candidate])
        array_np = ak.to_numpy(ak.flatten(df_npf_pts_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
    elif group=='vtx_pts':
        print(feature,group,candidate)
        #array_np = ak.to_numpy(df_vtx_pts_clip[feature][:,candidate])
        array_np = ak.to_numpy(ak.flatten(df_vtx_pts_clip[feature]))
        array_np = np.where(array_np == -999, 0, array_np)
        array_np = np.where(array_np ==   -1, 0, array_np)
        mini, maxi = np.quantile(array_np,0.01),np.quantile(array_np,0.99)
        mini_, maxi_ = np.quantile(array_np,0.2),np.quantile(array_np,0.8)
        return [mini, maxi], np.std(array_np[(array_np >= mini_) & (array_np <= maxi_)])
        
        
cpf_epsilons = np.zeros((len(cpf_branches),1))
cpf_standardized_epsilons = np.zeros((len(cpf_branches),1))
cpf_ranges = np.zeros((len(cpf_branches),1, 2))

for (i,key) in enumerate(cpf_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'cpf',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        cpf_epsilons[i,cand] = scale_epsilon
        cpf_standardized_epsilons[i,cand] = standardized_epsilon
        cpf_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)
        
npf_epsilons = np.zeros((len(npf_branches),1))
npf_standardized_epsilons = np.zeros((len(npf_branches),1))
npf_ranges = np.zeros((len(npf_branches),1, 2))

for (i,key) in enumerate(npf_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'npf',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        npf_epsilons[i,cand] = scale_epsilon
        npf_standardized_epsilons[i,cand] = standardized_epsilon
        npf_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)

vtx_epsilons = np.zeros((len(vtx_branches),1))
vtx_standardized_epsilons = np.zeros((len(vtx_branches),1))
vtx_ranges = np.zeros((len(vtx_branches),1, 2))

for (i,key) in enumerate(vtx_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'vtx',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        vtx_epsilons[i,cand] = scale_epsilon
        vtx_standardized_epsilons[i,cand] = standardized_epsilon
        vtx_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)

cpf_pts_epsilons = np.zeros((len(cpf_pts_branches),1))
cpf_pts_standardized_epsilons = np.zeros((len(cpf_pts_branches),1))
cpf_pts_ranges = np.zeros((len(cpf_pts_branches),1, 2))

for (i,key) in enumerate(cpf_pts_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'cpf_pts',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        cpf_pts_epsilons[i,cand] = scale_epsilon
        cpf_pts_standardized_epsilons[i,cand] = standardized_epsilon
        cpf_pts_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)

npf_pts_epsilons = np.zeros((len(npf_pts_branches),1))
npf_pts_standardized_epsilons = np.zeros((len(npf_pts_branches),1))
npf_pts_ranges = np.zeros((len(npf_pts_branches),1, 2))

for (i,key) in enumerate(npf_pts_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'npf_pts',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        npf_pts_epsilons[i,cand] = scale_epsilon
        npf_pts_standardized_epsilons[i,cand] = standardized_epsilon
        npf_pts_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)

vtx_pts_epsilons = np.zeros((len(vtx_pts_branches),1))
vtx_pts_standardized_epsilons = np.zeros((len(vtx_pts_branches),1))
vtx_pts_ranges = np.zeros((len(vtx_pts_branches),1, 2))

for (i,key) in enumerate(vtx_pts_branches):
    for cand in range(1):
        range_inputs, standardized_epsilon = quantile_min_max(key,'vtx_pts',cand)
        scale_epsilon = (range_inputs[1] - range_inputs[0])/2
        vtx_pts_epsilons[i,cand] = scale_epsilon
        vtx_pts_standardized_epsilons[i,cand] = standardized_epsilon
        vtx_pts_ranges[i,cand] = range_inputs
        print(range_inputs, scale_epsilon, standardized_epsilon)

dest = '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/'

np.save(dest+'cpf_epsilons.npy',cpf_epsilons)
np.save(dest+'cpf_standardized_epsilons.npy',cpf_standardized_epsilons)
np.save(dest+'cpf_ranges.npy',cpf_ranges)

np.save(dest+'npf_epsilons.npy',npf_epsilons)
np.save(dest+'npf_standardized_epsilons.npy',npf_standardized_epsilons)
np.save(dest+'npf_ranges.npy',npf_ranges)

np.save(dest+'vtx_epsilons.npy',vtx_epsilons)
np.save(dest+'vtx_standardized_epsilons.npy',vtx_standardized_epsilons)
np.save(dest+'vtx_ranges.npy',vtx_ranges)

np.save(dest+'cpf_pts_epsilons.npy',cpf_pts_epsilons)
np.save(dest+'cpf_pts_standardized_epsilons.npy',cpf_pts_standardized_epsilons)
np.save(dest+'cpf_pts_ranges.npy',cpf_pts_ranges)

np.save(dest+'npf_pts_epsilons.npy',npf_pts_epsilons)
np.save(dest+'npf_pts_standardized_epsilons.npy',npf_pts_standardized_epsilons)
np.save(dest+'npf_pts_ranges.npy',npf_pts_ranges)

np.save(dest+'vtx_pts_epsilons.npy',vtx_pts_epsilons)
np.save(dest+'vtx_pts_standardized_epsilons.npy',vtx_pts_standardized_epsilons)
np.save(dest+'vtx_pts_ranges.npy',vtx_pts_ranges)

print(cpf_epsilons)
print(cpf_standardized_epsilons)
print(cpf_ranges)
