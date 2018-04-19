function sTrueDetStruct = CollectBaselineDetections(options)
% function sTrueDetStruct = CollectBaselineDetections(options)
% 
% returns a set of objects in a list of street scenes images.  Each object is annotated
% with its type, the source image, and its bounding box

if(nargin < 1), options = [];, end

% Set Default Options
D.ImagesAvailable = 1:10;
% D.caObjectNames = {'car','pedestrian','bicycle'};
D.caObjectNames = {'car'};
D.BBoxOverlapThreshold = .5;
D.bMandateSquareBBox = 1;
load DefaultRoot;
D.Root = DefaultRoot; % This is the database root, don't includ '/Originals'
D.MinPositiveSize = 64; % the linear size of the minimum dimension of the bounding box
D.CenteringStrategy = 'center';  % can be 'center' or 'bottom'

options = ResolveMissingOptions(options,D);

load('CAfilelist.mat'); %--> CAfilelist;
%% Generate the set of positive locations from which to extract.
pBBoxs = [];
pIdxs = [];
n = 0;
% For every image
for IMidx = sort(options.ImagesAvailable);
  ol = QReadOList2(CAfilelist{IMidx}.olistname);
  if(mod(IMidx,20)==0)
    fprintf('building positives for image %d of %d\r',IMidx,length(options.ImagesAvailable));
  end
  % for every object type
  for iObjType = 1:length(options.caObjectNames);
    ObjectName = options.caObjectNames{iObjType};
    if(~isfield(ol,ObjectName)), continue, end;
    % for every example of this object
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
fprintf('%d positives were detected\n',n);
