cands_per_variable = {
#    'glob' : 1,
    'cpf' : 26,
    'npf' : 25,
    'vtx' : 5,
    'cpf_pts' : 26,
    'npf_pts' : 25,
    'vtx_pts' : 5,
    #'pxl' : ,
}
vars_per_candidate = {
 #   'glob' : 15,
    'cpf' : 16,#17,
    'npf' : 6,
    'vtx' : 12,
    'cpf_pts' : 4, #10,
    'npf_pts' : 4,
    'vtx_pts' : 4,
    #'pxl' : ,
}
defaults_per_variable_before_prepro = {
  #  'glob' : [None,None,None,None,None,None,-999,-999,-999,-999,-999,-999,-999,None,None],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    'cpf_pts' : [0 for i in range(vars_per_candidate['cpf_pts'])],
    'npf_pts' : [0 for i in range(vars_per_candidate['npf_pts'])],
    'vtx_pts' : [0 for i in range(vars_per_candidate['vtx_pts'])],
    #'pxl' : ,
}
epsilons_per_feature = {
#    'glob' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/global_standardized_epsilons.npy',
    'cpf' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/cpf_standardized_epsilons.npy',
    'npf' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/npf_standardized_epsilons.npy',
    'vtx' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/vtx_standardized_epsilons.npy',
    'cpf_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/cpf_pts_standardized_epsilons.npy',
    'npf_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/npf_pts_standardized_epsilons.npy',
    'vtx_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/vtx_pts_standardized_epsilons.npy',
    #'pxl' : ,
}
#epsilons_per_feature = {
#    'glob' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/global_standardized_epsilons.npy',                                                                                            
#    'cpf' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/cpf_epsilons.npy',
 #   'npf' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/npf_epsilons.npy',
  #  'vtx' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/vtx_epsilons.npy',
#    'cpf_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/cpf_pts_epsilons.npy',
 #   'npf_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/npf_pts_epsilons.npy',
  #  'vtx_pts' : '/eos/cms/store/group/phys_btag/ParticleTransformer/auxiliary/vtx_pts_epsilons.npy',
    #'pxl' : ,                                                                                                                                                                         
#}
defaults_per_variable_ = {
   # 'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    'cpf_pts' : [0 for i in range(vars_per_candidate['cpf_pts'])],
    'npf_pts' : [0 for i in range(vars_per_candidate['npf_pts'])],
    'vtx_pts' : [0 for i in range(vars_per_candidate['vtx_pts'])],
    #'pxl' : ,
}
defaults_per_variable = {
    #'glob' : [[None],[None],[None],[None],[None],[None],[-999,0],[-999],[-999],[-999,-1],[-999,-1],[-999,-1],[-999,-1],[None],[None]],
    'cpf' : [[0],[0],[0],[0],[0],[-1,0],[-1,0],[-1,0],[-1,0],[0],[0],[0],[0],[0],[0],[0]],
    'npf' : [[0,1,5],[0],[0],[0],[0],[0]],
    'vtx' : [[0],[0],[0],[0],[0],[-1000,0],[0],[0],[0],[0],[0],[0]],
    'cpf_pts' : [[0],[0],[0],[0]],
    'npf_pts' : [[0],[0],[0],[0]],
    'vtx_pts' : [[0],[0],[0],[0]],
    #'pxl' : ,
}
integer_variables_by_candidate = {
    #'glob' : [2,3,4,5,8,13,14],
    'cpf' : [12,13,14,15],#[13,14,15,16],
    'npf' : [2],
    'vtx' : [2],
    'cpf_pts' : [],
    'npf_pts' : [],
    'vtx_pts' : [],
    #'pxl' : ,
}
