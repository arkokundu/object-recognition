function [ExperimentOutput] = SSDBExperiment_BBoxwiseObjRec(sDetStruct, options);
% function [ExperimentOutput] = SSDBExperiment_BBoxwiseObjRec(sDetStruct, options);
% 
% BBoxwise experiments are used to detect objects in the full imagee-search sceneario.
% The measure is provided by a set of dections and confidences, and outputs measures of performance
% such as ROC and PR.  The paradigm used here is inspired by the Pascal Object Detction Challenge
% among others.
% 
% http://www.pascal-network.org/challenges/VOC/voc2006/examples/index.html
% 
% Correct detections must have the same label, bounding boxes which overlap by some threshold
% (default 50%) and don't match the same true positive example more than once.  Detections in images
% outside of options.WhichFilesToInclude are simply discarded.  Also, all bboxs are by default reshaped
% to be the smallest square bbox which has the same center as the original bbox.  This is due to the 
% difficulty in matching bboxs with elongated structures.  This feature can be turned off, though.
% 
% sDetStruct has the following structure.  It is a 1D struct array where each element has the following
% names.  *ObjName* is a string containing the name of the detected object.  *ImgIdx* is the index of the 
% street scene image in which the detection occured.  *BBox* is the bounding box of the detection.
% *Confidence* is the strength or confidence of this detection.
% 
% It should be noted that there are particularly contrived cases of true positive and detection
% configurations in which matching each detection to a true positive requires solving a binary 
% constraint satisfaction problem for each individual confidence.  This occurs very rarely when there 
% are large numbers of the same type of object within a small region and there is either a large 
% degree of overlap or the Overlap threshold is set very low.  Since we don't bother to solve this very rare
% problem, the PR curve must be considered to be a lower bound.  

load DefaultRoot;
D.ImagesAvailable = 1:10;
% D.caObjectNames = {'car','pedestrian','bicycle'};
D.caObjectNames = {'car'};
D.BBoxOverlapThreshold = .5;
D.bMandateSquareBBox = 1;
D.Root = DefaultRoot; % This is the database root, don't includ '/Originals'
D.MinPositiveSize = 64; % the linear size of the minimum dimension of the bounding box
D.CenteringStrategy = 'center';  % can be 'center' or 'bottom'

if(nargin < 2)
  options = [];
end
options = ResolveMissingOptions(options,D);

% Build the set of true detections
  sTrueDetStruct = CollectBaselineDetections(options);
% Remove detections from images we don't count.
  aImgIdx = sField2Arr(sDetStruct, 'ImgIdx');
  sDetStruct = sDetStruct(find(ismember(aImgIdx,options.ImagesAvailable)));
% Match each Emperical detection to a true detection
  [sDetStruct,sTrueDetStruct] = FindMatches(sDetStruct, sTrueDetStruct,options);
% For each discreet confidence in the set of true detections determine the PR
  PR = BuildPR(sDetStruct,sTrueDetStruct,options);
% Record the results in the experiment structure.  
  ExperimentOutput.sDetStruct = sDetStruct;
  ExperimentOutput.sTrueDetStruct = sTrueDetStruct;  
  ExperimentOutput.PR = PR;
  
function sTrueDetStruct = CollectBaselineDetections(options)
% function sTrueDetStruct = CollectBaselineDetections(options)
% 
load('CAfilelist.mat'); %--> CAfilelist;
%% First, Generate the set of positive locations from which to extract.
pBBoxs = [];
pIdxs = [];
n = 0;
for IMidx = sort(options.ImagesAvailable);
  ol = QReadOList2(CAfilelist{IMidx}.olistname);
  if(mod(IMidx,20)==0)
    fprintf('building positives for image %d of %d\r',IMidx,length(options.ImagesAvailable));
  end
  for iObjType = 1:length(options.caObjectNames);
    ObjectName = options.caObjectNames{iObjType};
    if(~isfield(ol,ObjectName)), continue, end;
    for j = 1:length(ol.(ObjectName))
      bbox = poly2bbox(ol.(options.caObjectNames{iObjType}){j});
      if(options.bMandateSquareBBox)
        bbox = bboxEnforceAspectRatio(bbox,[1 1],options.CenteringStrategy);
      end
      if(not(BBoxIsInside(bbox,[.5 .5 960 1280])));
        continue;   % Only take boxes which aren't truncated by the image border
      end
      if(min(bbox(3),bbox(4)) < options.MinPositiveSize)
        continue;
      end
      n = n+1;
      sTrueDetStruct(n).IsMatched = 0;
      sTrueDetStruct(n).ImgIdx = IMidx;
      sTrueDetStruct(n).ObjName = ObjectName;
      sTrueDetStruct(n).BBox = bbox;
    end
  end
end  
fprintf('\n%d positives were detected\n',n);

function [sDetStruct,sTrueDetStruct] = FindMatches(sDetStruct, sTrueDetStruct,options)
% function sDetStruct = FindMatches(sDetStruct, sTrueDetStruct,options)
% Our strategy is to assign the highest confidence detection with its best match in each image,
% and then to continue with lower confidences until either there are no more true positives or no more
% detections
% The output is a copy of sDetStruct with an additional field indicating the index of the best match
% in sTrueDetStruct

aDetConfidences = sField2Arr(sDetStruct,'Confidence');
aDetIdxs = sField2Arr(sDetStruct,'ImgIdx');
aTruIdxs = sField2Arr(sTrueDetStruct,'ImgIdx');
aDetObjs = Objname2Int(sDetStruct, options.caObjectNames);
aTrueObjs = Objname2Int(sTrueDetStruct, options.caObjectNames);
aDetConf = sField2Arr(sDetStruct,'Confidence');
[sDetStruct(:).TrueMatch] = deal(NaN);
[sDetStruct(:).MatchStrength] = deal(NaN);
[sTrueDetStruct(:).BestDetMatch] = deal(NaN);
[sTrueDetStruct(:).DetConfidence] = deal(NaN);
for iImgIdx = options.ImagesAvailable  % loop over images
  bDetOkIdx = (aDetIdxs == iImgIdx);
  bTruOkIdx = (aTruIdxs == iImgIdx);
  for iObjType = 1:length(options.caObjectNames)  % loop over object names
     bDetOkIdxObj = (aDetObjs == iObjType) .* bDetOkIdx;
     bTruOkIdxObj = (aTrueObjs == iObjType) .* bTruOkIdx;
     WhichDetections = find(bDetOkIdxObj);
     [sConfOk,siConfOk] = sort(-aDetConfidences(WhichDetections)); % sort from largest to smallest conf
     sConfOk = -sConfOk;
     TruObjectList = find(bTruOkIdxObj);
     WhichDetections = WhichDetections(siConfOk); 
     for iCandidate = WhichDetections
        MatchStrengths = zeros(length(TruObjectList),1);
	for iTrueObj = 1:length(TruObjectList);
	  if(sTrueDetStruct(TruObjectList(iTrueObj)).IsMatched)
	    MatchStrengths(iTrueObj) = 0;
	  else
 	    MatchStrengths(iTrueObj) = ...
	      MatchStrength(sDetStruct(iCandidate),sTrueDetStruct(TruObjectList(iTrueObj)));
	  end
	end
        [M,Mi] = max(MatchStrengths);
	if(M >= options.BBoxOverlapThreshold)
	  TruMatchIdx = TruObjectList(Mi);
          sTrueDetStruct(TruMatchIdx).BestDetMatch = iCandidate;
          sTrueDetStruct(TruMatchIdx).IsMatched = 1;
          sTrueDetStruct(TruMatchIdx).DetConfidence = sDetStruct(iCandidate).Confidence;
	  sDetStruct(iCandidate).TrueMatch = TruMatchIdx;
	  sDetStruct(iCandidate).MatchStrength = M;
	end
     end
  end
end  

  
function ms = MatchStrength(Det1, Det2);
% MatchStrength is defined as area of intersection / area of union
if(Det2.IsMatched)
  ms = 0;
  return;
end
bbox1 = Det1.BBox;
bbox2 = Det2.BBox;
bbi= BBoxIntersect(bbox1,bbox2);
ms = bbi(3)*bbi(4) / (bbox1(3)*bbox1(4) + bbox2(3)*bbox2(4) - bbi(3)*bbi(4));

function n = Objname2Int(sDetStruct, caObjectNames);
for i= 1:length(sDetStruct)
    n(i) = strmatch(sDetStruct(i).ObjName,caObjectNames);
end
  
function sPR = BuildPR(sDetStruct,sTrueDetStruct,options);
% function PR = BuildPR(sDetStruct,sTrueDetStruct,options);
% 
% each point on the precision and recall curve corresponds to a threshold on the confidence
% of the detected responses.  PR curves are recorded for each object independantly
% along with one curve for the combination of all objects.

nObjects = length(options.caObjectNames);
% first build the object independant list
sPR.independant = BuildPRHelper(sDetStruct, sTrueDetStruct);
aDetObjs = Objname2Int(sDetStruct, options.caObjectNames);
aTrueObjs = Objname2Int(sTrueDetStruct, options.caObjectNames);
for i = 1:nObjects
  on = options.caObjectNames{i};
  fDet = find(aDetObjs == i);
  fTrue = find(aTrueObjs == i);
  sPR.(on) = BuildPRHelper(sDetStruct(fDet),sTrueDetStruct(fTrue));
end

function PR = BuildPRHelper(sDet,sTru);
nPos = length(sTru);
nDet = length(sDet);
istp = zeros(nDet,1);
conf = zeros(nDet,1);
for i = 1:nDet
  istp(i) = sDet(i).TrueMatch > 0;
  conf(i) = sDet(i).Confidence;
end
u = unique(conf);
n=0;
for iu = 1:length(u);
  n=n+1;
  confenuf = (conf >= u(iu));
  f = find(confenuf);
  PR.p(n) = sum(istp(f)) / length(f);
  PR.r(n) = sum(istp(f)) / sum(istp);
  PR.f(n) = geomean([PR.p(n),PR.r(n)]);
end  
PR.maxf = max(f);
