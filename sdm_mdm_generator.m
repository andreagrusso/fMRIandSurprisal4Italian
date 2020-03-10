clear all
close all; close all hidden;
clc

%% Raw data loading


timestamps_struct = load('timestamps.mat');
surp_data = dlmread(uigetfile('*.txt'));

timestamps = timestamps_struct.timestamps;
durations = timestamps_struct.durations;
freq = timestamps_struct.freq;
sem_surp = surp_data(:,1);
lex_surp = surp_data(:,2);


%% Path

outdir = 'BVanalysis';


%% Acoustic predictors
[y_bw, fss_bw] = audioread('gianna_rev.wav'); nss_bw = length(y_bw);
[y_fw, fss_fw] = audioread('gianna.wav'); nss_fw = length(y_fw);

fss = fss_fw;

audio_start = 5*fss;
audio_end = (750)*fss;

y_fw = [zeros(audio_start,1);y_fw;zeros(audio_end-audio_start-length(y_fw),1)];
y_bw = [zeros(audio_start,1);y_bw;zeros(audio_end-audio_start-length(y_bw),1)];


%% Lingusitic data

ls = zscore(-log10(lex_surp)); %lexical

ss = zscore(-log10(sem_surp)); %semantic

fr = zscore(log10(freq/222153649)); %log of the relative frequency
%fr = zscore(log10(freq)); %log of the relative frequency

dr = zscore(durations);


sp_min = min(prob_cs(prob_cs>0));
prob_cs(prob_cs==0) = sp_min;
sp = zscore(-log10(prob_cs));

sd = zscore(sem_dist);

disp(corrcoef(ss,ls))

%%
NT = 750;  % number of time points
TR = 1; % TR
t = 0:TR:TR*(NT-1); % volumes

dt = 0.01; % resolution of time (for processing timestamps)
tt = 0:dt:TR*(NT-1);
n = neuroelf;
hrf = n.hrf('twogamma',dt,5,15,6,0,1,1,[0,tt(end)],0,'area');

%%

%forward predictors
fw_data.ss = ss;
fw_data.ls = ls;
fw_data.fr = fr;
fw_data.env = y_fw;
fw_data.dr = dr;
fw_data.timestamps = timestamps;
fw_data.durations = durations;

[fw_des_full,fw_des_sem,fw_des_lex] = create_preds(hrf,tt,t,fw_data,fss);




%backward predictors
timestamps_bw = zeros(size(timestamps));
durations_bw = flipud(durations);

for i = 0:length(durations)-1
    timestamps_bw(i+1) = 710 - timestamps(end-i) - durations(end-i);
end

bw_fr = flipud(fr);
bw_ss = flipud(ss);
bw_ls = flipud(ls);
bw_dr = flipud(dr);

bw_data.ss = bw_ss;
bw_data.ls = bw_ls;
bw_data.fr = bw_fr;
bw_data.dr = bw_dr;
bw_data.env = y_bw;
bw_data.timestamps = timestamps_bw;
bw_data.durations = durations_bw;


[bw_des_full,bw_des_sem,bw_des_lex]= create_preds(hrf,tt,t,bw_data,fss);



full_list_names = {'Sem','InvSem','LS','InvLS','Freq','InvFreq','Dur','InvDur','Env','InvEnv','Costant'};
sem_list_names = {'Sem','InvSem','Freq','InvFreq','Dur','InvDur','Env','InvEnv','Costant'};
lex_list_names = {'LS','InvLS','Freq','InvFreq','Dur','InvDur','Env','InvEnv','Costant'};

%fw SDM
create_sdm('fw_env.sdm','fw_full.sdm',outdir,fw_des_full,full_list_names,'fw')
create_sdm('fw_env.sdm','fw_sem.sdm',outdir,fw_des_sem,sem_list_names,'fw')
create_sdm('fw_env.sdm','fw_lex.sdm',outdir,fw_des_lex,lex_list_names,'fw')



create_sdm_noenv('fw_env.sdm','fw_full_noenv.sdm',outdir,fw_des_full,full_list_names,'fw')
create_sdm_noenv('fw_env.sdm','fw_sem_noenv.sdm',outdir,fw_des_sem,sem_list_names,'fw')
create_sdm_noenv('fw_env.sdm','fw_lex_noenv.sdm',outdir,fw_des_lex,lex_list_names,'fw')


%bw SDM
create_sdm('bw_env.sdm','bw_full.sdm',outdir,bw_des_full,full_list_names,'bw')
create_sdm('bw_env.sdm','bw_sem.sdm',outdir,bw_des_sem,sem_list_names,'bw')
create_sdm('bw_env.sdm','bw_lex.sdm',outdir,bw_des_lex,lex_list_names,'bw')


create_sdm_noenv('bw_env.sdm','bw_full_noenv.sdm',outdir,bw_des_full,full_list_names,'bw')
create_sdm_noenv('bw_env.sdm','bw_sem_noenv.sdm',outdir,bw_des_sem,sem_list_names,'bw')
create_sdm_noenv('bw_env.sdm','bw_lex_noenv.sdm',outdir,bw_des_lex,lex_list_names,'bw')




%create MDM
create_mdm('natexp_env.mdm','fw_full.sdm','bw_full.sdm','full_model.mdm',outdir)
create_mdm('natexp_env.mdm','fw_sem.sdm','bw_sem.sdm','sem_model.mdm',outdir)
create_mdm('natexp_env.mdm','fw_lex.sdm','bw_lex.sdm','lex_model.mdm',outdir)


create_mdm('natexp_env.mdm','fw_full_noenv.sdm','bw_full_noenv.sdm','full_model_noenv.mdm',outdir)
create_mdm('natexp_env.mdm','fw_sem_noenv.sdm','bw_sem_noenv.sdm','sem_model_noenv.mdm',outdir)
create_mdm('natexp_env.mdm','fw_lex_noenv.sdm','bw_lex_noenv.sdm','lex_model_noenv.mdm',outdir)



%% Visualization

figure()
plot(zscore(fw_des_full(:,1:2:end-1)))
legend =['Sem','LS','Freq','Dur','Env'];
title = "Full Model";

figure()
subplot(1,2,1)
imagesc(abs(corrcoef(fw_des_full)),[0 0.7]), colorbar
disp(corrcoef(fw_des_full))
subplot(1,2,2)
imagesc(abs(corrcoef(bw_des_full)),[0 0.7]), colorbar
disp(corrcoef(bw_des_full))



%% Function for Preds Matrix creation 

function [des_full,des_sem,des_lex]= create_preds(hrf,tt,t,data,fss)

    ref = zeros(size(hrf));
    ref_ss = zeros(size(hrf));
    ref_ls = zeros(size(hrf));
    ref_fr = zeros(size(hrf));
    ref_env = zeros(size(hrf));



    timestamps = data.timestamps;
    durations = data.durations;
    audio = data.env;
    
    for i = 1 : length(timestamps)

        [~,iws] = min(abs(tt-timestamps(i)));
        [~,iwe] = min(abs(tt-timestamps(i)-durations(i)));
        ref(iws:iwe) = 1;
        ref_ss(iws:iwe) = data.ss(i);
        ref_ls(iws:iwe) = data.ls(i);
        ref_fr(iws:iwe) = data.fr(i);
        ref_env(iws:iwe) = rms(audio(round(iws/100*fss):round(iwe/100*fss))); % RMS signal envelope

    end

    %% convolution 
    
    refc = conv(ref,hrf,'full'); refc = refc(1:length(tt));
    ssc = conv(ref_ss,hrf,'full'); ssc = ssc(1:length(tt));
    lsc = conv(ref_ls,hrf,'full'); lsc = lsc(1:length(tt));
    frc = conv(ref_fr,hrf,'full'); frc = frc(1:length(tt));
    envc = conv(ref_env,hrf,'full'); envc = envc(1:length(tt));
    
    %% interpolation
    ssci = interp1(tt,ssc,t);
    lsci = interp1(tt,lsc,t);
    frci = interp1(tt,frc,t);
    refci = interp1(tt,refc,t);
    envci = interp1(tt,envc,t);

    
    des_full = [ssci',lsci',frci',refci',envci'];
    %GS orthogonalization
    [des_sem,~] = gsog([ssci',frci',refci',envci']);
    [des_lex,~] = gsog([lsci',frci',refci',envci']);


    
%     des_full = [ssci',lsci',frci',refci',envci'];
%     des_sem = [ssci',frci',refci',envci'];
%     des_lex = [lsci',frci',refci',envci'];
%     des_semprob = [spci',frci',refci',envci'];
%     des_semdist = [sdci',frci',refci',envci'];

    
    
    
    

end


function create_sdm(input_name,output_name,outdir,data,list_names,direction)

    sdm = xff(input_name);
    TotPreds = size(list_names,2);
    sdm.SDMMatrix = zeros(size(data,1),TotPreds);
    sdm.NrOFPredictors = TotPreds;
    sdm.PredictorNames = list_names;
    sdm.FirstConfoundPredictor = TotPreds;
    sdm.SDMMatrix(:,end) = ones(size(data,1),1);

    sdm.PredictorColors = zeros(TotPreds,3);
    for i=1:TotPreds-1
        sdm.PredictorColors(i,:) = [randi([0,255]) randi([0,255]) randi([0,255])];
    end
    
    if strcmp(direction,'fw')
        sdm.SDMMatrix(:,1:2:end-1) = zscore(data);
    else
        sdm.SDMMatrix(:,2:2:end-1) = zscore(data);
    end
    
    sdm.SaveAs([outdir,'\',output_name]);
    sdm.ClearObject();
end


function create_sdm_noenv(input_name,output_name,outdir,data,list_names,direction)

    tmp_list_names = list_names(1:end-3);
    list_names = [tmp_list_names, 'Constant'];
    data = data(:,1:end-1);
    
    sdm = xff([outdir,'\',input_name]);
    TotPreds = size(list_names,2);
    sdm.SDMMatrix = zeros(size(data,1),TotPreds);
    sdm.NrOFPredictors = TotPreds;
    sdm.PredictorNames = list_names;
    sdm.FirstConfoundPredictor = TotPreds;
    sdm.SDMMatrix(:,end) = ones(size(data,1),1);

    sdm.PredictorColors = zeros(TotPreds,3);
    for i=1:TotPreds-1
        sdm.PredictorColors(i,:) = [randi([0,255]) randi([0,255]) randi([0,255])];
    end
    
    if strcmp(direction,'fw')
        sdm.SDMMatrix(:,1:2:end-1) = zscore(data);
    else
        sdm.SDMMatrix(:,2:2:end-1) = zscore(data);
    end
    
    sdm.SaveAs([outdir,'\',output_name]);
    sdm.ClearObject();
end


function create_mdm(template_name,fw_input,bw_input,output,outdir)
    mdm_template = xff([outdir,'\',template_name]);

    for i = 1:2:mdm_template.NrOfStudies
        mdm_template.XTC_RTC{i,2} = [outdir,'\',bw_input];
    end

    for i = 2:2:mdm_template.NrOfStudies
        mdm_template.XTC_RTC{i,2} = [outdir,'\',fw_input];
    end

    mdm_template.SaveAs([outdir,'\',output]);
    mdm_template.ClearObject();

end
