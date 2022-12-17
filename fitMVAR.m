function [Weights, Inputs, Behaviour_Coeffs, Residuals] = fitMVAR(dF_F, behaviour, timewindow)

% inputs:
% dF_F = cell array of dF/F tensors [time x trials x vars]
% timewindow = indices of time samples to select from dF_F (a window about stim onset, usually -1 to +1 s)
% behaviour = cell array of behavioural measurements [time x trials]

% outputs:
% Weights: MVAR interaction weights
% Inputs: MVAR stimulus inputs
% Behaviour_Coeffs: behaviour coefficients 
% Residuals: mismatch between model and data

%% Get tensor shapes

Nvar = size(dF_F{1},3);  % number of variables (e.g., neurons, pixels, depending on dataset) 
Nt = length(timewindow);
Nstim = length(dF_F);
Nbehav = length(behaviour);

for s=1:Nstim
    Ntrials(s) = size(dF_F{s},2);
end

%% Reshape data tensors for regression analysis

for s=1:Nstim
    dF_F_concat{s} = reshape(dF_F{s}(timewindow,:,:), [Nt * Ntrials(s), Nvar,1]);  % concatenate trials
    dF_F_concat_negshift{s} = reshape(dF_F{s}(timewindow-1,:,:), [Nt * Ntrials(s), Nvar,1]);   
    for b=1:Nbehav
        behaviour_concat{b}{s} = reshape(behaviour{b}{s}(timewindow,:), [Nt*Ntrials(s),1]);
    end
end

    dF_Ftot = vertcat(dF_F_concat{:}).';  % concatenate stimuli
    dF_Ftot_negshift = vertcat(dF_F_concat_negshift{:}).'; 
for b=1:Nbehav    
    behaviourtot{b} =  vertcat(behaviour_concat{b}{:}).';              
end      
    
%% create design matrix for regression

Q = zeros([Nstim, sum(Ntrials)]);
N = cumsum([0,Ntrials]);
for s=1:Nstim
        Q(s, (N(s)+1):N(s+1)) = 1;   
end
stimblocks = kron(Q, eye(Nt));
    
Lambdatot = [dF_Ftot_negshift; stimblocks; vertcat(behaviourtot{:})]; % design matrix     

%% Perform least squares fit

MVAR_parameters = dF_Ftot / Lambdatot;  % least squares regression
    
%% Extract MVAR parameters
    
    Weights = MVAR_parameters(:, 1:Nvar); 
    
    for i=1:Nstim
        Inputs{i} = MVAR_parameters(:, (Nvar + 1 +(i-1)*Nt):(Nvar + i*Nt));
    end
    
    for b=1:Nbehav    
        Behaviour_Coeffs{b} =  MVAR_parameters(:, end-Nbehav+b);              
    end      

    Residuals = MVAR_parameters * Lambdatot - dF_Ftot;
    