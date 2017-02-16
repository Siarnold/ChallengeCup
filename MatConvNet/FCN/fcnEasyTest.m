function info = fcnEasyTest( varargin )
%fcnTest tests the fcn on an image
%hanse: hand segmentation in ego-centric videos

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn8s-edsh' ;
opts.dataDir = 'data/EDSH/JPEG/' ;
opts.modelPath = 'data/fcn8s-edsh/pascal-fcn8s-dag.mat' ;
opts.modelFamily = 'ModelZoo' ;

% experiment setup
opts.imdbPath = fullfile( opts.dataDir, 'EDSH1_' );
opts.gpus = [2,3,1] ;% use gpu No. 2
opts = vl_argparse(opts, varargin) ;

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
  %trained in this original example of FCNTrain
  case 'matconvnet'
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net) ;
    net.mode = 'test' ;
    annotated for pascal model downloaded directly
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  %those from pretrained models
  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    %get the index of the output layer for use afterwards
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  %that from pretrained models
  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end

%set GPU if exists
if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1)) ;
  net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

imdb = dir(opts.imdbPath) ;
[ total, ~ ] = size( imdb )

%test all images in the validation
for i = 3:total
    
% Load an image and gt segmentation
rgbPath = fullfile( opts.imdbPath, imdb(i).name );
rgb = vl_imreadjpeg({rgbPath}) ;
rgb = rgb{1} ;

% Subtract the mean (color)
im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;

% Some networks requires the image to be a multiple of 32 pixels
% im_ will be the renewed image
if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
else
    im_ = im ;
end

% change im_ to gpu array
if ~isempty(opts.gpus)
    im_ = gpuArray(im_) ;
end

%evaluate when the input layer is im_
net.eval({inputVar, im_}) ;
% gather: fetch to the current workspace from gpu
scores_ = gather(net.vars(predVar).value) ;
[~,pred_] = max(scores_,[],3) ;

% restore the size if the im_ has been resized
% pred is the renewed prediction
if imageNeedsToBeMultiple
    pred = imresize(pred_, sz, 'method', 'nearest') ;
else
    pred = pred_ ;
end

% Plots
% Print segmentation
figure(100) ;clf ;
%image, label, prediction
displayImage(rgb/255, pred) ;
drawnow ;

% Save segmentation
imPath = fullfile(opts.expDir, [ 'seg_' imdb(i).name '.jpg']) ;
imwrite(pred,labelColors(),imPath,'png');
end



% -------------------------------------------------------------------------
function displayImage(im, pred)
% -------------------------------------------------------------------------
subplot(1,2,1) ;
image(im) ;
axis image ;
title('source image') ;


%define a new colormap for better demostration
cmap = labelColors() ;
subplot(1,2,2) ;
image(uint8(pred-1)) ;
axis image ;
title('predicted') ;
%by a map, the colormap can change the style of displayed color
colormap(cmap) ;

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    %bit-or, bitshift(A,k) means bit-wise shift A to the left by k bits
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
