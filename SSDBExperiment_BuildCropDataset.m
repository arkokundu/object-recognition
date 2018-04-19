function SSDBExperiment_BuildCropDataset(options)
% function SSDBExperiment_BuildCropDataset(options)
% 
load('DefaultRoot.mat'); %-->DefaultRoot
D.Root = DefaultRoot; % This is the database root, don't includ '/Originals'
D.ObjectName = 'car';
D.UsableIndicies = 1:300;
D.FilterMinSize = 64; % the linear size of the minimum dimension of the bounding box
D.OutputSize = [128 128];
D.SamplesPerRecordedFile = 1000;
D.NegativesPerPositive = 6;
D.CenteringStrategy = 'center';  % can be 'center' or 'bottom'
D.SaveDir = fullfile(D.Root, 'CropData');
D.DataName = 'MyFirstData';
if(nargin < 1)
  options = [];
end
options = ResolveMissingOptions(options,D);
load('CAfilelist.mat'); %--> CAfilelist;
%% First, Generate the set of positive locations from which to extract.
fprintf('\nGenerating the set of positive crop locations ...\n');
pBBoxs = [];
pIdxs = [];
for IMidx = options.UsableIndicies
  ol = QReadOList2(CAfilelist{IMidx}.olistname);
  if(~isfield(ol,options.ObjectName)), continue, end;
   if(mod(IMidx,20)==0)
     fprintf('reading image %d of %d\r',IMidx,length(options.UsableIndicies));
   end
  for j = 1:length(ol.(options.ObjectName))
    bbox_orig = poly2bbox(ol.(options.ObjectName){j});
    bbox = bboxEnforceAspectRatio(bbox_orig,[.1 .1],options.CenteringStrategy);
    %for padding around the crop.
    bbox = [bbox(1) - bbox(3)/6,bbox(2) - bbox(4)/6,bbox(3)*4/3,bbox(4)*4/3];
    if(BBoxIsInside(bbox,[.5 .5 1280 960]));  % take if its inside the image and not too small
      if(min(bbox(3),bbox(4)) > options.FilterMinSize) 
        pBBoxs(end+1,:) = bbox;
        pIdxs(end+1) = IMidx;
      end
    end
  end
end  
fprintf('%d positives were selected\n',size(pBBoxs,1));
% Second, Build the negative extraction locations so that the distributions look about the same
fprintf('\nGenerating the set of negative crop locations at %d per positive...',options.NegativesPerPositive);
pBBoxs = round(pBBoxs);
ntodo = length(pIdxs);
% nperpart = ceil(ntodo / ofhowmanyparts);
% whichdoido = (nperpart*(partno-1)):(nperpart*(partno)) + 1;
% whichdoido = intersect(whichdoido,1:ntodo);
% myBBoxs = pBBoxs(whichdoido,:);
[nBBoxs,nIdxs] = GenerateRandNegativeBBoxSet(pBBoxs, ...
               options.UsableIndicies,options.ObjectName,options.NegativesPerPositive,CAfilelist);
savename = fullfile(options.SaveDir,sprintf('%s_%s_CroppingInfo.mat',options.DataName,options.ObjectName));
save(savename,'nBBoxs','nIdxs','pBBoxs','pIdxs','options');
fprintf('done\n saved progress in %s\n',savename);

% Third, combine these two sets and extract them from the data.
fprintf('Actutally cropping the data now...\n');
BBoxs = [pBBoxs;nBBoxs];
BBoxs = round(BBoxs);
Idxs = [pIdxs,nIdxs];
Ys = [ones(1,length(pIdxs)),-ones(1,length(nIdxs))];
[i,si] = sort(Idxs);
BBoxs = BBoxs(si,:);
Idxs = Idxs(si);
Ys = Ys(si);
MAXFRACTIONSIZE = options.SamplesPerRecordedFile;
resolution = options.OutputSize;
X = [];
y= [];
myBBox = [];
myIdx = [];
previm = -1;
FRACTION_NUM = 1
for i = 1:length(Idxs)
   imidx = Idxs(i);
   if(imidx ~= previm);
      im = imread(CAfilelist{imidx}.imagename);
      here = pwd;
      previm = imidx;
   end
   gray_cr = imresize(rgb2gray(imcrop_pad(im,BBoxs(i,:),'symmetric')),resolution,'bilinear');
   X(:,end+1) = gray_cr(:);
   y(end+1) = Ys(i);
   myBBox((end+1),:) = BBoxs(i,:);
   myIdx((end+1)) = Idxs(i);
   if(mod(i,1)==0)
     fprintf('%d of %d\r',i,length(Idxs));
   end
   if(size(X,2) == MAXFRACTIONSIZE)
      savename = fullfile(options.SaveDir,sprintf('%s_%s_f%.3d.mat',options.DataName,options.ObjectName,FRACTION_NUM));
      save(savename,'X','y','myBBox','myIdx');
      FRACTION_NUM = FRACTION_NUM + 1;
      X = [];
      y = [];
      myBBox = [];
      myIdx = [];
      fprintf('\nSAVE FRACTION SAVE FRACTION SAVE FRACTION %d\n',FRACTION_NUM-1);
   end
end
savename = fullfile(options.SaveDir,sprintf('%s_%s_f%.3d.mat',options.DataName,options.ObjectName,FRACTION_NUM));
save(savename,'X','y','myBBox','myIdx');

function [selected_nonobjs_bboxs,idxout] = GenerateRandNegativeBBoxSet(selected_bboxs, idxin,objectname,negperpos,CAfilelist);
% function [selected_nonobjs_bboxs,idxout] = GenerateRandNegativeBBoxSet(selected_bboxs, idxin,objectname,negperpos,CAfilelist);
% 
% selects a negative image location set which is similar in a sense to the positive image locations.
nbox = size(selected_bboxs,1);
nidx = length(idxin);
n=1;
t1 = clock;
for i = 1:nbox
   for j = 1:negperpos
      bFail = 1;
      while(bFail)
        rand_idx = ceil(rand*nidx);
        idx = idxin(rand_idx);
        rand_box = ceil(rand*nbox);
        box = selected_bboxs(rand_box,:);
        [selected_nonobjs_bboxs(n,:),bFail] = Get10PercentBox(box,idx,CAfilelist,objectname);
        idxout(n) = idx;
      end
      n = n+1;
   end
   e = etime(clock,t1);
   eper = e/ i;
   etot = eper * (nbox-i-1);
   if(mod(i,1)==0)
     fprintf('built negatives for box %d of %d, cost is %.1f sec per box: %.1f mins left\r',i,nbox,eper,etot/60);
   end
end     
fprintf('\n');
function [outbox,bFail] = Get10PercentBox(inbox,idx,CAfilelist,objectname)
% attempts to find a position to put the box within the image which doesn't overlap the actual object too much.

bFail = 0;
img = imread(CAfilelist{idx}.imagename);
olist = QReadOList2(CAfilelist{idx}.olistname);
if(isfield(olist,objectname))
  mask = QGetObjectMask(olist.(objectname),[960 1280]);
else
  mask = zeros([960 1280]);
end
done = 0;
retrycount = 0;
while(not(done))
   outbox =  SelectRandomBBox([960 1280], inbox([4,3])+1);
   overlap = CalculateOverlap(outbox, mask);
   if(overlap < .1)
      done = 1;
   else
      retrycount = retrycount + 1;
   end
   if (retrycount == 35)
      done = 1;
      bFail = 1;
   end
end

function overlap = CalculateOverlap(box,mask)
% overlap is defined as the part of the bounding box which contains the target object.
support = sum(sum(mask(box(2):(box(2)+box(4)),box(1):(box(1)+box(3)))));
maxsupport = (box(4)+1)*(box(3)+1);
overlap = support / maxsupport;
