function info = fcnTest(varargin)
%fcnTest tests the fcn on an image
%hanse: hand segmentation in ego-centric videos

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn8s-edsh' ;
opts.dataDir = 'data/EDSH' ;
opts.modelPath = 'data/fcn8s-edsh/pascal-fcn8s-dag.mat' ;
opts.modelFamily = 'ModelZoo' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.vocEdition = '12' ;
opts.vocAdditionalSegmentations = false ;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
opts.gpus = [2,3,1] ;% use gpu No. 2
opts = vl_argparse(opts, varargin) ;

% result path
resPath = fullfile(opts.expDir, 'results.mat') ;
if exist(resPath)
  info = load(resPath) ;
  return ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get EDSH for segmentation test
% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = edshSetup('dataDir', opts.dataDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get validation subset
val = find(imdb.images.set == 2) ;

% Compare the validation set to the one used in the FCN paper
% valNames = sort(imdb.images.name(val)') ;
% valNames = textread('data/seg11valid.txt', '%s') ;
% valNames_ = textread('data/seg12valid-tvg.txt', '%s') ;
% assert(isequal(valNames, valNames_)) ;

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

%temporarily set to be 2
confusion = zeros(2) ;

%test all images in the validation
for i = 1:numel(val)
  imId = val(i) ;
  name = imdb.images.name{imId} ;
  rgbPath = sprintf(imdb.paths.image, name) ;
  labelsPath = sprintf(imdb.paths.classSegmentation, name) ;

  % Load an image and gt segmentation
  rgb = vl_imreadjpeg({rgbPath}) ;
  rgb = rgb{1} ;
  lb = imread(labelsPath) ;

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

  % Accumulate errors
  %zero area as the edge will not be counted
  lb_ = lb + 1 ;
  pred_ = uint8(pred > 1) ;
  pred_ = pred_ + 1 ;
  lb = lb * 255;
  
  %temporarily set 2
  confusion = confusion + accumarray([lb_(:),pred_(:)],1,[2 2]) ;

  % Plots
  clear info ;
  [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
  fprintf('IU ') ;
  fprintf('%4.1f ', 100 * info.iu) ;
  fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
      100*info.miu, 100*info.pacc, 100*info.macc) ;
  
  %print the accuracy
  %clf for clear figure
  figure(1) ; clf;
  %image scale
  %the normalization means the sum along the column in each row is 1
  imagesc(normalizeConfusion(confusion)) ;
  %axis image means equal dx length corresponding to equal dy length
  axis image ;
  %set graphics object properties
  %gca: get current axis; ydir means set y-axis to be normal
  set(gca,'ydir','normal') ;
  %set the colormap to be 'jet', which is default
  colormap(jet) ;
  %update figure windows and process callbacks
  drawnow ;
  
  % Print segmentation
  figure(100) ;clf ;
  %image, label, prediction
  displayImage(rgb/255, lb, pred) ;
  drawnow ;
  
  % Save segmentation
  imPath = fullfile(opts.expDir, [name '.png']) ;
  imwrite(pred,labelColors(),imPath,'png');
end

% Save results
save(resPath, '-struct', 'info') ;

% -------------------------------------------------------------------------
function nconfusion = normalizeConfusion(confusion)
% -------------------------------------------------------------------------
% normalize confusion by row (each row contains a gt label)
%bsxfun means binary singleton expansion function, which expands the
%singleton of the latter in this mfile
%the result will contain many NaN, which will not be displayed later
nconfusion = bsxfun(@rdivide, double(confusion), double(sum(confusion,2))) ;

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
%sum along dimension 2
pos = sum(confusion,2) ;
res = sum(confusion,1)' ;
tp = diag(confusion) ;
IU = tp ./ max(1, pos + res - tp) ;
meanIU = mean(IU) ;
%confusion(:) makes a column array
pixelAccuracy = sum(tp) / max(1,sum(confusion(:))) ;
meanAccuracy = mean(tp ./ max(1, pos)) ;

% -------------------------------------------------------------------------
function displayImage(im, lb, pred)
% -------------------------------------------------------------------------
subplot(2,2,1) ;
image(im) ;
axis image ;
title('source image') ;

subplot(2,2,2) ;
image(uint8(lb-1)) ;
axis image ;
title('ground truth')

%define a new colormap for better demostration
cmap = labelColors() ;
subplot(2,2,3) ;
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
