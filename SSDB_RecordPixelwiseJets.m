function SSDB_RecordPixelwiseJets(PxExp,Root, Template, savename,options);
%function SSDB_RecordPixelwiseJets(PxExp,Root, Template, savename,options);
%
% Root is where the files are stored
% each image is stored as a matrix named options.inputvariablename within the file named
% fullfile(Root, sprintf(Template,ImgIdx)) where ImgIdx is the number of the image.

if(nargin < 5), options = [];, end
D.bOverwrite = 0;
D.ReceptiveFieldSize = [1 1];
D.BorderStrategy = 'symmetric';
D.inputvariablename = 'X';
D.bUseImRead = 0;
options = ResolveMissingOptions(options,D);

uIdxs = unique([PxExp.TrainPoints(1,:),PxExp.TestPoints(1,:)]);
ivn = options.inputvariablename;
rfs = options.ReceptiveFieldSize;
nTr = 0;
nTe = 0;
% check if the file already exists
if(exist(savename) & not(options.bOverwrite))
  fprintf('ABORTING extraction: savename exists, and overwrite is turned off\n');
  return;
end
nImgIdx = 0;
t1 = clock;
for ImgIdx = uIdxs;
  nImgIdx = nImgIdx + 1;
  fprintf('Extracting from image %d of %d, etime is %.1f seconds\r',nImgIdx, length(uIdxs),etime(clock,t1));
  if(options.bUseImRead)
    JetImage.(ivn) = imread(fullfile(Root,sprintf(Template,ImgIdx)));
  else
    JetImage = load(fullfile(Root,sprintf(Template,ImgIdx)),ivn);
  end
  sz = size(JetImage.(ivn));
  if(ImgIdx == uIdxs(1))
    nDims = size(JetImage.(ivn),3) * prod(options.ReceptiveFieldSize);
    X_Train = zeros(nDims,size(PxExp.TrainPoints,2));
    X_Test = zeros(nDims,size(PxExp.TestPoints,2));
    % Try saveing the savename, to make sure it is writable and there is enough space
    try
      X_Placeholder = rand(size(X_Train));
      save(savename,'X_Placeholder');
    catch
      fprintf('ABORTING: it looks like either the savename is broken or there isn''t enough space\n');
      return;
    end
  end
  fTr = find(PxExp.TrainPoints(1,:) == ImgIdx);
  fTe = find( PxExp.TestPoints(1,:) == ImgIdx);
  for i = fTr
    c_xy = PxExp.TrainPoints([3,2],i);
    c_xy = round(CoordinateRespace(c_xy',[960 1280],sz([1,2])));
    bbox = [c_xy(1)-(floor(rfs(2)-1)/2), c_xy(2)-(floor(rfs(2)-1)/2), rfs(2)-1, rfs(1)-1];
    x = imcrop_pad(JetImage.(ivn), bbox, options.BorderStrategy);
    X_Train(:,i) = x(:);
  end
  for i = fTe
    c_xy = PxExp.TestPoints([3,2],i);
    c_xy = round(CoordinateRespace(c_xy',[960 1280],sz([1,2])));
    bbox = [c_xy(1)-(floor(rfs(2)-1)/2), c_xy(2)-(floor(rfs(2)-1)/2), rfs(2)-1, rfs(1)-1];
    x = imcrop_pad(JetImage.(ivn), bbox, options.BorderStrategy);
    X_Test(:,i) = x(:);
  end
end  
try
  save(savename,'X_Train','X_Test','PxExp','options');
catch
  fprintf('ERROR saving, going to keyboard...\n');
  keyboard;
end
  
  
  
  
