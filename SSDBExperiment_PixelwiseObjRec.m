function ExperimentStructure = SSDBExperiment_PixelwiseObjRec(options);
%function ExperimentStructure = SSDBExperiment_PixelwiseObjRec(options);
%
% CBCL bileschi
load('DefaultRoot.mat'); %-->DefaultRoot
D.Root = DefaultRoot; % This is the database root, don't includ '/Originals'
D.WhichFilesToInclude = 1:300;  % There are a total of 3547 labeled images in the DB
D.TrainFraction = .5;            % Train test split
D.BorderPadding = 15;            % Samples are not included near the edge of objects.
D.MaxSamplesPerObj = 10;       % Samples are drawn more or less evenly across objects and images.
D.caObjectsToInclude = {'building','tree','road','sky'};

if(nargin < 1)
  options = [];
end
options = ResolveMissingOptions(options,D);
TTS = TTSplit_Make(options.WhichFilesToInclude, options.TrainFraction, 1-options.TrainFraction);
TTS.Train = sort(TTS.Train);
TTS.Test = sort(TTS.Test);
ExperimentStructure.TrainIdxs = TTS.Train;
ExperimentStructure.TestIdxs = TTS.Test;
[TestPoints,TestValues] = BuildTestPointStructure(options,ExperimentStructure.TestIdxs);
ExperimentStructure.TestPoints = TestPoints;
ExperimentStructure.TestValues = TestValues;
[TrainPoints,TrainValues] = BuildTestPointStructure(options,ExperimentStructure.TrainIdxs);
ExperimentStructure.TrainPoints = TrainPoints;
ExperimentStructure.TrainValues = TrainValues;
ExperimentStrucutre.options = options;

function [TestPoints,TestValues] = BuildTestPointStructure(options, TestIdxs);
%function [TestPoints,TestValues] = BuildTestPointStructure(options, TestIdxs);
TestPoints = [];
TestValues = [];
fPoints = [];
idxvec =[];
nObjects = length(options.caObjectsToInclude);
n = 0;
for iImgIdx = TestIdxs
  fprintf(' %d%% done... working image %d\r',floor(100*n/length(TestIdxs)),iImgIdx);
  yarr = SSDBLoadBaselineMask(iImgIdx, options.caObjectsToInclude,options.Root);
  yarr_occluders = SSDBLoadBaselineMask(iImgIdx, {'car','bicycle','pedestrian'},options.Root);
  yarr_occluders = max(yarr_occluders,[],3); 
  yarr = yarr > .95;
  yarr = EliminateAllButUniqueLabels(yarr,yarr_occluders);
  yarr = padimage(yarr,1,0);
  for i = 1:nObjects
    yarr(:,:,i) = imerode(logical(yarr(:,:,i)),fspecial('disk',options.BorderPadding)>0);
  end
  yarr = unpadimage(yarr,1);
  yRollout = nLayerImage2MatrixOfPixels(yarr); 
  for i = 1:nObjects
    objMask = yRollout(i,:);
    [f] = find(objMask);
    nToTake = min(length(f),options.MaxSamplesPerObj);
    if(length(f) < 10000)%  if the number of pixels is large enough we are better off sampling anohter way
      p = randperm(length(f));
      p = p(1:nToTake);
    else
      p = ceil(rand(nToTake,1) * length(f));
    end
    TestPoints = [TestPoints,[repmat(iImgIdx,[1,nToTake]);mod(f(p),size(yarr,1)); ceil(f(p) / size(yarr,1))]];
    TestValues = [TestValues,yRollout(:, f(p))];
  end
  n = n+1;
end
 fprintf(' 100%% done                           \n');
 
function yarr = EliminateAllButUniqueLabels(yarr,yarr_occluders);
d = size(yarr,3);
s = sum(yarr,3);
s = (s == 1);
yarr_occluders = not(yarr_occluders);
if(nargin > 1)
  for i = 1:d
    yarr(:,:,i) = yarr(:,:,i) & s & yarr_occluders;
  end
else
  for i = 1:d
    yarr(:,:,i) = yarr(:,:,i) & s;
  end
end
  
