function edshMakelist(varargin)
% EDSHMAKELIST makes a train list and val list for EDSH
%   edshMakelist(VARARGIN) e.g.'dataDir'

opts.dataDir = fullfile('data', 'EDSH') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imgDir = fullfile(opts.dataDir, 'JPEG', 'EDSH1_') ;
opts.expTrain = fullfile(opts.dataDir, 'train.txt') ;
opts.expVal = fullfile(opts.dataDir, 'val.txt') ;

dirs = dir(opts.imgDir) ;
[length,~] = size(dirs) ;

traintxt = fopen(opts.expTrain, 'w') ;
valtxt = fopen(opts.expVal, 'w');

% distribute the images
for x = 3:length
    if rand > 0.1667
        fprintf(traintxt, '%s\n', dirs(x).name(1:6)) ;
    else
        fprintf(valtxt, '%s\n', dirs(x).name(1:6)) ;
    end
end

fclose(traintxt) ;
fclose(valtxt) ;