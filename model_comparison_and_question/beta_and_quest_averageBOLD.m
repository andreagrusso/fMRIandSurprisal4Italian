clear all;
close all;
clc
% Load Data

load('questions.mat') 
answers = questions;

%the subject #17 was discarded due to problems with fMRI data
answers(17,:) = [];

%vectorize answers for the analysis
%y = answers(:);
y = 100*(sum(answers,2)/size(answers,2));
%y = zscore(sum(answers,2));

%create a vector for the questions (fixed effect)
questions = [];
for i = 1:size(answers,2)
    questions = [questions; i*ones(size(answers,1),1)];
end
 
%create a predictor for the subjects
subs_id = 1:size(answers,1);
subs_id = repmat(subs_id',size(answers,2),1);

%output variable
results.data = {};
%%

surp = 'Sem';
%surp = 'Lex';
disp(['Surprisal model : ', surp])
disp(' ')


vvd = xff(['BW_FW_',surp,'_clc_p001.vvd']);

voi_names = vvd.VOINames;
results.VoiNames = voi_names;


%load predictors
sdm = xff(['fw_',lower(surp),'.sdm']);
preds = sdm.SDMMatrix(:,1:2:end-1);


for i=1:length(voi_names)
    
    
    betas = [];
    for j=2:2:vvd.NrOfVTCs
        
        %extracting the average bold signal from the ROI for each subject
        %we are considering only the scenario with the real speech
        vtc_data = vvd.VTC(j).Values(:,i);
        %we are interested in the beta relative to the surprisal
        b = regress(vtc_data,preds); 
        betas = [betas;b'];
    end
    
    %load the atd file
    %atd = xff([surp,'/',vois(i).name]);
    
    %extract and vectorize the subjects' betas
    %betas = atd.SubjectData;
    %betas = zscore(atd.SubjectData); %    repmat(zscore(atd.SubjectData),size(answers,2),1);   
    %create datatable
    
    data_table = table(y, betas(:,1),betas(:,2),betas(:,3),betas(:,4),....
        'VariableNames',{'y','Surp','Freq','Dur','Env'});
    
    results.data{i} = fitglm(data_table, 'y ~ 1 + Surp + Freq + Dur + Env',...
        'Distribution','Normal', 'Link', 'identity'); %,'DummyVarCoding','effects');



    %data_table = table(y,questions,betas,subs_id,...
        %'VariableNames',{'y','questions','betas','subjects'});
    %create and fit the model
    %results.data{i} = fitglme(data_table, 'y ~ 1 + betas + (1|questions) + (1|subjects)',...
     %   'Distribution','Binomial','DummyVarCoding','effects');
     
     disp(['VOI : ', results.VoiNames{i}])
     disp(results.data{i})
     
     %      [B,DEV,STATS] = mnrfit(betas,answers);
     %      disp('Multinomial fit (per question) : ')
     %      disp(STATS.p)
     disp('-------------------')
     
end

%%
%save([surp,'logit_questions.mat'],'results')
save([surp,'_norm_questions_averageBOLD.mat'],'results')
