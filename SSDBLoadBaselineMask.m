function yimg = SSDBLoadBaselineMask(iImgIdx, caObjectNames,Root);
%function yimg = SSDBLoadBaselineMask(iImgIdx, caObjectNames,Root);
%
%returns an image containing masks for each selected object type.
%
if(nargin < 3)
  Root = '/cbcl/scratch03/bileschi/Release';
end
fn = fullfile(Root,'Anno_OList',sprintf('SSDB%.5d_olist.mat',iImgIdx));
olist = QReadOList2(fn);
if (not(isstruct(olist)))
  error('Could not find annotation %s\n', fn);
end
ReducedOlist = {};
yimg = zeros([960,1280,length(caObjectNames)]);
for i = 1:length(caObjectNames)
  if(isfield(olist,caObjectNames{i}))
      yimg(:,:,i) = QGetObjectMask(olist.(caObjectNames{i}),[960,1280]);
   else
      yimg(:,:,i) = zeros(960,1280);
   end
end

