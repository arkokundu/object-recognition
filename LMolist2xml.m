function LMolist2xml(olistfilename, xmlfilename, AdditionalInfo);
% function LMolist2xml(olistfilename, xmlfilename, AdditionalInfo);
% 
% reads the olist from olist filename and saves it in xml format in xmlfilename
% 
[p,n,e] = fileparts(olistfilename);
D.filename = strrep([n,e],'_olist.mat','.JPG');
D.folder = 'StreetScenes/images';
D.source.sourceImage = 'CBCL Street Scenes Database';
D.source.sourceAnnotation = 'Imagelabeler1 Matlab Image labeling tool';
if(nargin < 3)
  AdditionalInfo = [];
end
AdditionalInfo = ResolveMissingOptions(AdditionalInfo,D);

xml.annotation.filename = AdditionalInfo.filename;
xml.annotation.folder = AdditionalInfo.folder;
xml.annotation.source = AdditionalInfo.source;
olist = QReadOList2(olistfilename);
fns = fieldnames(olist);
nObjectTypes = length(fns);
xml.annotation.object = [];
for iObjType  = 1:nObjectTypes
  caobjs = olist.(fns{iObjType});
  nObjExamples = length(caobjs);
  objectName = strrep(fns{iObjType},'_',' ');
  for iObjExample = 1:nObjExamples
    xml.annotation.object(end+1).name = objectName;
    xml.annotation.object(end).deleted = '0';
    xml.annotation.object(end).verified = '0';
    xml.annotation.object(end).date = [];
    polygonStruct = PolyToStruct(olist.(fns{iObjType}){iObjExample});
    xml.annotation.object(end).polygon = polygonStruct;
  end
end
writeXML(xmlfilename, xml)

function s = PolyToStruct(p)
  s.username = 'CenterFoBiologicalAndComputationalLearning';
  npoints = size(p,1);
  for i = 1:(npoints-1)
    s.pt(i).x = p(i,1);
    s.pt(i).y = p(i,2);
  end
  