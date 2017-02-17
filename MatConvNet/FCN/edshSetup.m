function imdb = edshSetup(varargin)
% edshSetup setup an imdb for EDSH
% CALL: imdb = edshSetup('dataDir', DATADIR)


% set opts
opts.dataDir = fullfile('data','EDSH') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDirs{1,1} = fullfile(opts.dataDir, 'JPEG', 'EDSH1') ;
opts.dataDirs{1,2} = fullfile(opts.dataDir, 'JPEG', 'EDSH2') ;
opts.dataDirs{1,3} = fullfile(opts.dataDir, 'JPEG', 'EDSHK') ;

opts.dataDirs{2,1} = fullfile(opts.dataDir, 'Segmentation', 'EDSH1') ;
opts.dataDirs{2,2} = fullfile(opts.dataDir, 'Segmentation', 'EDSH2') ;
opts.dataDirs{2,3} = fullfile(opts.dataDir, 'Segmentation', 'EDSHK') ;

% new directories
opts.dataDirs_{1,1} = fullfile(opts.dataDir, 'JPEG', 'EDSH1_') ;
opts.dataDirs_{1,2} = fullfile(opts.dataDir, 'JPEG', 'EDSH2_') ;
opts.dataDirs_{1,3} = fullfile(opts.dataDir, 'JPEG', 'EDSHK_') ;

opts.dataDirs_{2,1} = fullfile(opts.dataDir, 'Segmentation', 'EDSH1_') ;
opts.dataDirs_{2,2} = fullfile(opts.dataDir, 'Segmentation', 'EDSH2_') ;
opts.dataDirs_{2,3} = fullfile(opts.dataDir, 'Segmentation', 'EDSHK_') ;


% mkdir and resize the jpg data to 540 * 960
for x1 = 1:3
    if ~exist(opts.dataDirs_{1,x1}, 'dir') mkdir(opts.dataDirs_{1,x1}) ; end
    if ~exist(opts.dataDirs_{2,x1}, 'dir') mkdir(opts.dataDirs_{2,x1}) ; end
    
    dirs = dir(opts.dataDirs{2,x1}) ;
    [length,~] = size(dirs) ;
    
    if ~exist(fullfile(opts.dataDirs_{1,x1}, dirs(end).name), 'file')
        for x2 = 3:length
            im = imread(fullfile(opts.dataDirs{1,x1}, dirs(x2).name)) ;
            im = imresize(im, [540 960]) ;
            fprintf('%s: resizing %s in %s\n', mfilename, dirs(x2).name, opts.dataDirs{1,x1}) ;
            imwrite(im, fullfile(opts.dataDirs_{1,x1}, dirs(x2).name)) ;
        end
    end
    
    if ~exist(fullfile(opts.dataDirs_{2,x1},strrep(dirs(end).name, 'jpg', 'png')), 'file')
        for x2 = 3:length
            im = imread(fullfile(opts.dataDirs{2,x1}, dirs(x2).name)) ;
            im = imresize(im, [540 960]) ;
            im = uint8(im > 0) ;
            fprintf('%s: resizing and making %s in %s\n', mfilename, dirs(x2).name, opts.dataDirs{2,x1}) ;
            imwrite(im, fullfile(opts.dataDirs_{2,x1}, strrep(dirs(x2).name, 'jpg', 'png'))) ;
        end
    end
end

% make imdb
% just make the EDSH1 first fot test
imdb.paths.image = esc(fullfile(opts.dataDirs_{1,1},'%s.jpg')) ;
imdb.paths.classSegmentation = esc(fullfile(opts.dataDirs_{2,1},'%s.png')) ;
imdb.sets.id = uint8([1 2]) ;
imdb.sets.name = {'train', 'val'} ;
imdb.classes.id = uint8(1) ;
imdb.classes.name = {'hand'} ;
imdb.images.id = [] ;
imdb.images.name = {} ;
imdb.images.set = [] ;

trainPath = fullfile(opts.dataDir, 'train.txt') ;
valPath = fullfile(opts.dataDir, 'val.txt') ;

% make the train list and val list
if ~exist(trainPath, 'file') | ~exist(valPath, 'file')
    edshMakelist('dataDir', opts.dataDir) ;
end
 
% add the images' info to imdb
imdb = addImageSet(imdb, trainPath, 1) ;
imdb = addImageSet(imdb, valPath, 2) ;

% compress data type
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;

% check images on disk and get their size
imdb = getImageSizes(imdb) ;

% -------------------------------------------------------------------------
% Add images to imdb.images.set
function imdb = addImageSet(imdb, setPath, setCode)
% -------------------------------------------------------------------------
cnt = length(imdb.images.id) ;
fprintf('%s: reading %s\n', mfilename, setPath);
names = textread(setPath, '%s') ;
for x = 1:length(names)
    cnt = cnt + 1 ;
    imdb.images.id(cnt) =  cnt ;
    imdb.images.name{cnt} = names{x} ;
    imdb.images.set(cnt) = setCode;
end

% -------------------------------------------------------------------------
function imdb = getImageSizes(imdb)
% -------------------------------------------------------------------------
for j=1:numel(imdb.images.id)
  %imf: image info
  info = imfinfo(sprintf(imdb.paths.image, imdb.images.name{j})) ;
  %this is to store the size info of the images
  %(:,j) means the jth column
  imdb.images.size(:,j) = uint16([info.Width ; info.Height]) ;
  fprintf('%s: checked image %s [%d x %d]\n', mfilename, imdb.images.name{j}, info.Height, info.Width) ;
end

% -------------------------------------------------------------------------
%replace \ with \\ for windows, but no influence for linux because of /
function str=esc(str)
% -------------------------------------------------------------------------
str = strrep(str, '\', '\\') ;