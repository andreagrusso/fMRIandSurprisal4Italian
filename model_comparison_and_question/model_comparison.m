clear all
close all 
clc
%% Loading data (Models: SwS, LS) Forward and Backward

sws_sdm = xff('fw_sem.sdm');
%skip the constant
fw_sws_preds = sws_sdm.SDMMatrix(:,1:end-1);
ls_sdm = xff('fw_lex.sdm');
%skip the constant
fw_ls_preds = ls_sdm.SDMMatrix(:,1:end-1);

bw_sws_sdm = xff('bw_sem.sdm');
bw_sws_preds = bw_sws_sdm.SDMMatrix(:,1:end-1);
bw_ls_sdm = xff('bw_lex.sdm');
bw_ls_preds = bw_ls_sdm.SDMMatrix(:,1:end-1);

% sws_preds = [bw_sws_preds;fw_sws_preds];
% ls_preds = [bw_ls_preds; fw_ls_preds];
sws_preds = zscore([bw_sws_preds;fw_sws_preds]);
ls_preds = zscore([bw_ls_preds; fw_ls_preds]);



%% Get the fMRI data

%VVD_FileName = 'BW_FW_Lex_clc_p001';
VVD_FileName = 'BW_FW_Sem_clc_p001';
vvd = xff([VVD_FileName,'.vvd']);
voi_names = vvd.VOINames;
%voi_names{3} = 'R-cerebellum';

NT = size(vvd.VTC(1).Values,1);
%t = 0:TR:TR*(NT-1); % time points

NrOfVOIs = vvd.NrOfVOIs; VOIs_included = 1; %1:NrOfVOIs;
NrOfVTCs = vvd.NrOfVTCs; VTCs_included = 1; %1:NrOfVTCs;


sigs = zeros(NrOfVTCs, NT, NrOfVOIs);
for i = 1 : NrOfVTCs
    sigs(i,:,:) = zscore(vvd.VTC(i).Values);
end


%% Built models
subs_id = [];

%vector that assigns a numerical id to the subject
for i = 1: NrOfVTCs
    subs_id = [subs_id;i*ones(NT,1)];
end

results.VoiNames = voi_names;
results.comparisons.names = {'SwS vs LS','Base SwS vs SwS','Base LS vs Ls'};
results.comparisons.regions = {};
results.models.names = {'base_sws','base_ls','sws_m','ls_m'};
results.models.regions = {};


for voi_idx = 1:NrOfVOIs
    disp(voi_names{voi_idx});
    %select the data of a VOI
    data = sigs(:,:,voi_idx);
    %vectorize the data (response variable)
    y = data(:);
    %y = reshape(data,NrOfVTCs*NT,1);
    %create a replication of the matrix
    sws_data = repmat(sws_preds,27,1);
    ls_data = repmat(ls_preds,27,1);
    %picking the fixed effects of each model (the fixed effects are
    %different because of the Gram-Schmidt othogonalization
    sws_fixed = table(y,subs_id,sws_data(:,2),sws_data(:,3),sws_data(:,4),'VariableNames',{'SubjData','SubsName','Freq','Dur','Env'});
    ls_fixed = table(y,subs_id,ls_data(:,2),ls_data(:,3),ls_data(:,4),'VariableNames',{'SubjData','SubsName','Freq','Dur','Env'});
    %tables of the two models
    tbl_sws = table(y,subs_id,sws_data(:,1),sws_data(:,2),sws_data(:,3),sws_data(:,4), 'VariableNames',{'SubjData','SubsName','SwS','Freq','Dur','Env'});
    tbl_ls = table(y,subs_id,ls_data(:,1),ls_data(:,2),ls_data(:,3),ls_data(:,4), 'VariableNames',{'SubjData','SubsName','LS','Freq','Dur','Env'});
    %models with fixed effects. Subjects are treated as random effect
    fixed_sws = fitlme(sws_fixed, 'SubjData ~ Freq + Dur + Env + (1|SubsName)');
    fixed_ls = fitlme(ls_fixed, 'SubjData ~ Freq + Dur + Env + (1|SubsName)');
    %mixed models with surprisal. Subjects are treated as random effect
    sws_m =  fitlme(tbl_sws, 'SubjData ~ SwS + Freq + Dur + Env + (1|SubsName)');
    ls_m =  fitlme(tbl_ls, 'SubjData ~ LS + Freq + Dur + Env + (1|SubsName)');
    results.models.regions{voi_idx} = {fixed_sws, fixed_ls, sws_m, ls_m};
    
    %the two models are not nested so we need to add the simulation 
    disp('Estimating comparison SwS vs LS......')
    results.comparisons.regions{voi_idx,1} = compare(ls_m,sws_m,'NSim',10000);    %the base models (no surprisal) are nested into the model with
    %surprisal
    disp('Estimating comparison SwS vs base model......')
    results.comparisons.regions{voi_idx,2} = compare(fixed_sws,sws_m);

    disp('Estimating comparison LS vs base model......')
    results.comparisons.regions{voi_idx,3} = compare(fixed_ls,ls_m);

end
    
namesave = [VVD_FileName,'.mat'];
save(namesave,'results')