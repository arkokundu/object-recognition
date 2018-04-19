function SSDB_InitializeDirectories(DBRoot);
% function SSDB_InitializeDirectories(DBRoot);
% calling this function with the parameter the full path of the root of the database
% will initialize all the default path variables with the appropriate values.
% This step is necessary for proper functioning of many database methods.
 
DefaultRoot = DBRoot;
addpath(fullfile(DBRoot,'code'));
addpath(DBRoot);
save(fullfile(DBRoot,'DefaultRoot.mat'),'DefaultRoot');
for i= 1:3547
  CAfilelist{i}.imagename = fullfile(DBRoot,'Original',sprintf('SSDB%.5d.JPG',i));
  CAfilelist{i}.olistname = fullfile(DBRoot,'Anno_OList',sprintf('SSDB%.5d_olist.mat',i));
end
save(fullfile(DBRoot,'CAfilelist.mat'),'CAfilelist');
  
