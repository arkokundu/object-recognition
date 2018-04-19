function olist = QReadOList2(filename)
%read olist from disk format 2.00

fid = fopen(filename,'r');
if(fid == -1)
    olist = 0;
    return;
else
    temp = fscanf(fid, '%s',1);
end
if (not(strcmp(temp, 'ObjectListFormat2.0')))
    %the fileformat recorded differently, best to just quit now.
    fprintf('wrongfilefomrat %s\n', filename);
    olist = 0;
%    olist = struct;
    return;
end
%the file format is at least nominally correct, continue with processing
%fprintf('rightfileformat\n');
olist = struct;
%check how many objects in the olist
nobjects = fscanf(fid, '%d objectypes\n',1);
nobjects = nobjects';
if (nobjects > 0)
    % for every type of object
    for i = 1:nobjects
        tempcellarray = {};
        fieldnames = {};
        % for every mask
        %read the fieldname
        fieldnames{i} = fscanf(fid,'%s\n',1);
        nmasks = fscanf(fid, '%d masks\n',1);
        for j = 1:nmasks
           % read the number of coordinates in the mask
           ncoords = fscanf(fid, 'poly %d ',1); %read human-centric label garbage
           ncoords = fscanf(fid, 'has %d coordinates\n',1); %totally needs to be fixed :p
           poly = zeros(ncoords,2);
           for k = 1:ncoords
               poly(k,1) = fscanf(fid, '%d ', 1);
               %fprintf('xi %d ', poly(k,1));
           end           
           for k = 1:ncoords
               poly(k,2) = fscanf(fid, '%d ', 1);    
               %fprintf('xi %d ', poly(k,1));
           end           
           olist.(fieldnames{i}){j} = poly;    
       end
    end%for every object
end%if there are any objects
error = fclose(fid);
