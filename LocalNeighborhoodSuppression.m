function DL = LocalNeighborhoodSuppression(caImages,options)
% DL returns a list of possible detection locations.
% the process proceeds by detecting the global maximum, and then suppressing the area
% in location and scale.
% ca images is a cell array of images perportedly from the same detector being activated
% on the same image at different scales. The size of the suppression and the stopping 
% conditions are contained within the options structure.
% results are returned as fractions of the whole image size and index of the scale

D.max_n_return = 25;        % the maximum number of detections that will be returned.
D.min_return_val = -inf;    % the minimum detection strength allowed (-inf for all dets ok).
D.fSuppressionRadius = 64;  % the tightness of the spacial suppression in the xy direction.
D.SuppressScaleStructure = [1, .85, .5, .15]; % the first value refers to the strenth 
                  % of the suppression in this scale.  The second value the relative
		  % suppression strength in the adjacent scales. etc.             
D.OriginalImageSize = [960 1280];
D.ScaleStructure = ((1/2).^(1/4)) .^ [0:11];
D.WindowSize = [128 128];
D.bSuppressTruncatedBoxes = 1; % suppresses the detection of bounding boxes which would exit the
                  % original image window
if(nargin < 2), options = [];, end
options = ResolveMissingOptions(options,D);

nReturnSoFar = 0;
previousReturnVal = -inf;
% while not done
while ~((nReturnSoFar == options.max_n_return) || ...
   (previousReturnVal < options.min_return_val))
   nReturnSoFar = nReturnSoFar + 1;
   i = nReturnSoFar;
   % return the strongest detection
   [DetectionList(i).str, DetectionList(i).xy_loc, DetectionList(i).iScale] = RecordMaximum(caImages);
   DetectionList(i).rel_loc_xy = CoordReal2Relative(DetectionList(i).xy_loc,size(caImages{DetectionList(i).iScale}));
   DetectionList(i).bboxOrig = CalcBBoxOrig(DetectionList(i),options);
   % suppress the neighborhood of this detection
   caImages = LocalSupression(caImages, DetectionList(i),options);
   if(options.bSuppressTruncatedBoxes)
     if(not(BBoxIsInside(DetectionList(i).bboxOrig,[.5, .5, options.OriginalImageSize])))
       nReturnSoFar = nReturnSoFar - 1;
       continue;
     end
   end
   previousReturnVal = DetectionList(i).str;
end
DL = DetectionList;
if(previousReturnVal < options.min_return_val)
  DL = DL(1:(end-1));
end

function bbox = CalcBBoxOrig(D,options)
% find the bounding box around this example
bbox = [0 0 options.WindowSize(2) options.WindowSize(1)] * 1 / options.ScaleStructure(D.iScale);
bbox(1:2) = CoordRelative2Real(D.rel_loc_xy, options.OriginalImageSize);
% this point is at the center though, must move it to the top left
bbox(1:2) = bbox(1:2) - .5 * (bbox(3:4) - [1,1]);

function [str,xyloc,iScale] = RecordMaximum(caImages);
for i = 1:length(caImages);
  [submax(i), Locs{i}] = max2d(caImages{i});
  [str,Mi] = max(submax);
  iScale = Mi;
  loc = Locs{Mi};
  xyloc = loc([2,1]); % switch to xy rather than yx format.
end

function caImg = LocalSupression(caImg,DL,options)
% caImg is the detection scale space.  The local suppression occurs over both location and scale.
nScale = length(caImg);
for i = 1:nScale
  abs_scal_dif = abs(i - DL.iScale);
  if(abs_scal_dif) > (length(options.SuppressScaleStructure) - 1);
     continue; % no suppression in this scale
  end
  SuppressStr = options.SuppressScaleStructure(abs_scal_dif + 1);
  m = min(min(caImg{i}));
  caImg{i} = caImg{i} - m;
  DetScale = 2 * options.fSuppressionRadius / (960 * options.ScaleStructure(i)) * size(caImg{i},1);
  s1 = size(caImg{DL.iScale});
  s2 = size(caImg{i});
  G = GausImg([size(caImg{i},1), size(caImg{i},2)],CoordinateRespace(DL.xy_loc([2,1]),s1,s2),DetScale);
  G = G * ( SuppressStr);
  caImg{i} = caImg{i} .* (1-G);
  caImg{i} = caImg{i} + m;
end

function [v,d] = max2d(X);
%function [v,d] = max2d(X);
%
%for finding maximum of an array X
% v = max(max(X)).  X(d(1),d(2)) = v;

[v,i] = max(X);
[v,j] = max(v);
d = [i(j),j];
