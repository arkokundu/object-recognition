function [ROC,Model,yel,ye,ytest,TTS] = SSDBExperiment_CropwiseObjRec(Root,Template,variablename, objidx, options,savename);
%function [ROC,Model,yel,ye,ytest,TTS] = SSDBExperiment_CropwiseObjRec(Root,Template,variablename, objidx, options,savename);
%
%
% This example file shows how we have used the Cropwise object detection experiment paradigm.
% Using this code, one can train and test classfiers for three object types, car pedestrian
% and bicycle, which have been pre-cropped using SSDBExtractCropData.
% Additional features can be read in the comments next to the options section.
% We have used gentleboost, for the most part, in classfiction, but other learning
% algorithms can be replaced in a plug-n-play fashion by replicating the interface
% in the gentle boost example.
% 
% if Template and variable name are cell arrays of strings, instead of  just strings, then multiple
% cropped features will be loaded and concatinated before handing to the classifier
% 
% The template will be searched for under Root first, and then Root/CropData.  If neither of these contains
% files like Template, then there will be an error.
% 
% ROC is the Reciever Operator Characteristic curve of the classifier
% Model is the classifier output by the training mechanism
% yel is the emperical labels of the test data
% ye is the emperical confidence of the test data (1D output by classfier before threshold)
% ytest is the corresponding y values of the test points
% TTS is the training test split used, in terms of the original data order.
HelpTextInfo = 'SSDBExperiment_CropwiseObjRec';

d.bRandsplit = 1;        % 0 means that we will take the first 2/3 for training
d.TrainingPart = 2/3;    % The fraction of the crops that will be used for training
d.sPARAMS.Nrounds = 250; % sPARAMS is a set of paramters which are dedicated to the 
d.sPARAMS.R = 0;	 %      learning algorithm.
d.sPARAMS.T = 0;
d.sPARAMS.C = 1;
d.sPARAMS.KERNEL = 0;
d.nMaxFeatures = 4500;   % data with more features will be randomly decimated
d.ClassifierName = 'gentleBoost'; % this controls which sort of classifier will be used.
d.bNormalizeDataSamples = 1;      % what type of data normalization
d.caNormalizationRegions = [];    % If subsets of features should be independantly normd
d.NormalizationRelativeStrengths = [1]; % If subsets need to be normed stronger than others
d.bAbsVal = 0;                    % 1 for taking the absolute value of all data
d.featuresubset = [];             % allows you to take only a subset of available features.
d.bOverwrite = 0;                 % will overwrite the output savefile, instead of skipping
d.bPCAReduceSize = 0;             % data reduction via PCA

seedrand;
d.RandSeed = floor(rand*99999);   % if a random seed is provided externaly, then random runs can be repeated.
options = ResolveMissingOptions(options,d);

sPARAMS = options.sPARAMS;

objname{1} = 'car';
objname{2} = 'pedestrian';
objname{3} = 'bicycle';

objs2do = objidx;
for i = objs2do
  on = objname{i};
  if(not(options.bOverwrite))
    if(exist('savename'))
    if(exist(savename))
      fprintf('File exists, continuing\n');
      Model= [];, ye = [];, ROC = [];, yel = []; ytest = []; TTS = [];
      continue;
    end
    end
  end;
  [yfull, NewDataX] = GetNewDataX(Template, variablename, on,Root,options);
  yfull = yfull(:);
  %if(size(NewDataX,1) > options.nMaxFeatures), 
  %  rand('state',options.RandSeed);
  %  p = randperm(size(NewDataX,1));
  %  NewDataX = NewDataX(p(1:options.nMaxFeatures),:);
  %  fprintf('TOO MANY FEATURES: Randomly sampling features');
  %end
  L = length(yfull);
  if(options.bRandsplit)
    nTrain = round(options.TrainingPart * L);
    nTest = L - nTrain;
    rand('state',options.RandSeed);
    p = randperm(L);
    Test = p(1:nTest);
    Train = p((nTest+1):end);
  else
    Test = 1:(floor(L * (1-options.TrainingPart)));
    Train = (floor(L * (1-options.TrainingPart))+1):L;
  end
  DXtest = NewDataX(:,Test);
  ytest = yfull(Test);
  DXtrain = NewDataX(:,Train);
  ytrain = yfull(Train);
  [DXtrain, FeatureSubsample] = SubSampleFeatures(DXtrain, options);
  [DXtrain,NormalizationParams] = InternalNormalize(DXtrain, options);
  DXtest = SubSampleFeatures(DXtest, options, FeatureSubsample);
  DXtest = InternalNormalize(DXtest,options,NormalizationParams);
  if(options.bPCAReduceSize) 
     [U,S,V] = svd(DXtrain,0);
     DXtrain = U'*DXtrain;
     DXtest = U'*DXtest;
  end
  cmd = sprintf('Model = CLS%s(DXtrain,ytrain,options.sPARAMS);',options.ClassifierName);
  eval(cmd);
  cmd = sprintf('[yel, ye] = CLS%sC(DXtest,Model);',options.ClassifierName);
  eval(cmd)
  ROC = ROCetc2(ye' ,ytest);
  TTS.Train = Train;
  TTS.Test = Test;
  if(exist('savename'))
    save(savename,'HelpTextInfo','yel','ye','ROC','Root','Template','variablename','objidx','Model','options','TTS','FeatureSubsample','NormalizationParams');
  end
end



function [yfull, NewDataX] = GetNewDataX(Template, variablename,on,Root,options)
NewDataX = [];
yfull = [];
if(not(iscell(Template)))
  d2 = dir(fullfile(Root,sprintf(Template,on)));
  if(length(d2) == 0)
    d2 = dir(fullfile(Root,'CropData',sprintf(Template,on)));
    Root = fullfile(Root,'CropData');
    if(length(d2) == 0)
      error('File template not found %s\n',fullfile(Root,sprintf(Template,on)));
    end
  end
  for iFl = 1:length(d2);
    D = load(fullfile(Root,d2(iFl).name));%-->eval(variablename);,y
    Part = getfield(D,variablename);
    if(isfield(options,'featuresubset'))
         if(not(isempty(options.featuresubset)))
  	   Part = Part(options.featuresubset,:);
	 end
    end
    NewDataX = [NewDataX,Part];
    yfull = [yfull,D.y]; 
  end
else
  % In order to save memory here, we are going to have to estimate the full
  % size of the matrix to load, and decimate as we build.
  sn = 1;
  for iGroup = 1:length(Template)
    d2 = dir(fullfile(Root,sprintf(Template{iGroup},on)));
    if(length(d2) == 0)
      d2 = dir(fullfile(Root,'CropData',sprintf(Template{iGroup},on)));
      Root = fullfile(Root,'CropData');
      if(length(d2) == 0)
        error('File template not found %s\n',fullfile(Root,sprintf(Template{iGroup},on)));
      end
    end
    sq = 1;
    SizeCheckStruct = whos('-file',fullfile(Root,d2(1).name),variablename{iGroup});
    PartSizes(iGroup) = SizeCheckStruct.size(1);
    if(not(isempty(options.featuresubset{iGroup})))
      PartSizes(iGroup) = min(PartSizes(iGroup), length(options.featuresubset{iGroup}));
    end
  end
  nParts = length(Template);
  [newPartSizes,p,cap] = ComputeNewPartSizes(PartSizes,options.nMaxFeatures);
   %%%%%%%%%%%%%%%%%%%%%%%%%%BEFORE EDIT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sn = 1;
  for iGroup = 1:length(Template)
    d2 = dir(fullfile(Root,sprintf(Template{iGroup},on)));
    sq = 1;
    for iFl = 1:length(d2);
      D = load(fullfile(Root,d2(iFl).name));%-->eval(variablename),y
      Part = getfield(D,variablename{iGroup});
      if(isfield(options,'featuresubset'))
         if(not(isempty(options.featuresubset{iGroup})))
  	   Part = Part(options.featuresubset{iGroup},:);
	 end
      end
      Part = Part(cap{iGroup},:);
      [n,q] = size(Part);
      NewDataX(sn:(n+sn-1),sq:(sq+q-1)) = Part;
      sq = sq + q;
      if(iGroup == 1)
        yfull = [yfull,D.y];
      end 
    end %loop over breaks in data on disk
    sn = size(NewDataX,1) + 1;
    PartSizes(iGroup) = size(Part,1);
  end %loop over data types
end

function [X,NormalizationParams] = InternalNormalize(X,options,NormalizationParams)
if(nargin < 3)
  NormalizationParams = [];
end
if(options.bNormalizeDataSamples > 0)
  if(isempty(options.caNormalizationRegions))
    caNR{1} = 1:size(X,1);
  else
    caNR = options.caNormalizationRegions;
  end
  [X,NormalizationParams] = normalizeByIndex(X, caNR, options.bNormalizeDataSamples,NormalizationParams,options);
end
     

function [XTrainFull, SelectedFeatures] = SubSampleFeatures(XTrainFull, options, SelectedFeatures);
if(nargin < 3)
  SelectedFeatures = 1:size(XTrainFull,1);
  if(size(XTrainFull,1) > options.nMaxFeatures), 
    seedrand(options.RandSeed);
    p = randperm(size(XTrainFull,1));
    XTrainFull = XTrainFull(p(1:options.nMaxFeatures),:);
    SelectedFeatures = p(1:options.nMaxFeatures);
    fprintf('TOO MANY FEATURES: Randomly sampling %d features\n', options.nMaxFeatures);
  end
else
  XTrainFull = XTrainFull(SelectedFeatures,:);
end  

function [newPartSizes,p,cap] = ComputeNewPartSizes(PartSizes,nMaxFeatures);
%if each part is larger than nMaxFeatures / length(PartSizes)
%make each part equally big. Otherwise take all parts wich are less than this limit,
%add the extra to the bin and recurse.
nParts = length(PartSizes);
newPartSizes = zeros(nParts,1);
bPartRepresented = zeros(nParts,1);
done = 0;
while(not(done))
  maxTake = floor(nMaxFeatures / sum(not(bPartRepresented)));
  v = intersect(find(PartSizes < maxTake), find(not(bPartRepresented)));
  newPartSizes(v) = PartSizes(v);
  nMaxFeatures = nMaxFeatures - sum(PartSizes(v));
  bPartRepresented(v) = 1;
  if(not(any(v)))
    v = find(not(bPartRepresented));
    newPartSizes(v) = maxTake;
    bPartRepresented(v) = 1;
  end
  if(all(bPartRepresented))
    done = 1;
  end
end  
s = [0,cumsum(PartSizes)];
p = [];
for i = 1:nParts
  p_i = randperm(PartSizes(i));
  p_i = p_i(1:newPartSizes(i));
  cap{i} = p_i;
  p = [p,s(i)+p_i];
end    

function [X,normparams] = normalizeByIndex(X,ci,normtype,normparams,options);
%normtype 1 = sqrt of L1
%normtype 2 = X ./ norm(X)
%normtype 3 = outnormalizeX;
if(nargin < 3)
  normtype = 1;
end
if(nargin < 4)
  normparams = [];
end
if(normtype == 1) % sqrt of L1
  X = abs(X);
  for i = 1:length(ci),
    X(ci{i},:) = MatrixNormalize(X(ci{i},:),normtype);
  end
end
if(normtype == 2) % X ./ normX
  if(isempty(normparams))
    for i = 1:length(ci),
      normparams.cExpectedPartNorm{i} = norm(X(ci{i},:));
    end
  end
  for i = 1:length(ci),
    X(ci{i},:) = length(ci{i})* ( X(ci{i},:) / normparams.cExpectedPartNorm{i});
  end
end
if(normtype == 3) % outnormalizeData X
  if(isempty(normparams))
    clear normparams
    for i = 1:length(ci),
       [X(ci{i},:),normparams{i}] = outnormalizeDataX(X(ci{i},:));
    end
  else
    for i = 1:length(ci),
       [X(ci{i},:),normparams{i}] = outnormalizeDataX(X(ci{i},:),normparams{i});
    end
  end
end
if(normtype == 4) % outnormalizeData X And Reweight Sections
  if(isempty(normparams))
    clear normparams
    for i = 1:length(ci),
       [X(ci{i},:),normparams{i}] = outnormalizeDataX(X(ci{i},:));
       X(ci{i},:) = X(ci{i},:) * options.NormalizationRelativeStrengths(i);
    end
  else
    for i = 1:length(ci),
       [X(ci{i},:),normparams{i}] = outnormalizeDataX(X(ci{i},:),normparams{i});
        X(ci{i},:) = X(ci{i},:) * options.NormalizationRelativeStrengths(i);
    end
  end
end
function X = MatrixNormalize(X, normType, normparams);
if(nargin < 3)
  normparams = [];
end
if(normType == 1)
 sX = sum(X);
 X = X * (spdiag(1./(sX+eps)));
 X = sqrt(X);
end
