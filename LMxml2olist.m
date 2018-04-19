function LMxml2olist(xmlfilename, olistfilename);
%function LMxml2olist(xmlfilename, olistfilename);
%
%converts an xml file into olist format.
%
%bileschi 2006

xml = loadXML(xmlfilename);
olist = struct;
nObjs = length(xml.annotation.object);
for iObj = 1:nObjs
  name = xml.annotation.object(iObj).name;
  name = strrep(name,' ','_');
  if(not(isfield(olist,name)))
    olist.(name) = {};
  end
  polyarray = GetPolyArray(xml.annotation.object(iObj).polygon);
  olist.(name){end+1} = polyarray;
end
keyboard
QWriteOList2(olistfilename,olist);

function polyarray = GetPolyArray(polygon)
np = length(polygon.pt);
polyarray = zeros(np+1,2);
for i = 1:np
  polyarray(i,1) = str2num(polygon.pt(i).x);
  polyarray(i,2) = str2num(polygon.pt(i).y);
end
polyarray(end+1,:) = polyarray(1,:);
  
