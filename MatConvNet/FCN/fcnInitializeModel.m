function net = fcnInitializeModel(varargin)
%FCNINITIALIZEMODEL Initialize the HAND-FCN model from PASCAL-FCN8S

opts.sourceModelPath = 'data/fcn32s-hand-edsh/fcn32s-epoch10.mat' ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                    Load the source model
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath, 'file')
  fprintf('ERROR!\nSource Model is not found.\n') ;
  return ;
end
net = load(opts.sourceModelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;


% for convt (deconv) layers, cuDNN seems to be slower?
% set NVIDIA CUDA Deep Neural Network library for GPU acceleration
net.meta.cudnnOpts = {'cudnnworkspacelimit', 512 * 1024^3} ;
%net.meta.cudnnOpts = {'nocudnn'} ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the HAND-FCN version
% -------------------------------------------------------------------------

% Modify the last fully-connected layer to have 2 output classes
% Initialize the new filters to zero
for i = [1 2]
  p = net.getParamIndex(net.layers(end-3).params{i}) ;
  if i == 1
    %fcn32s: [1,1,4096,21]
    sz = size(net.params(p).value) ;
    sz(end) = 2 ;
  else
    %fcn32s: [21,1]
    sz = [2 1] ;
  end
  net.params(p).value = zeros(sz, 'single') ;
end
%accordingly, change the size of the conv layer
net.layers(end-3).block.size = size(...
  net.params(net.getParamIndex(net.layers(end-3).params{1})).value) ;

% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------

%bilinear_u creates a binear interpolation filter
%(SIZE_K, NUMGROUPS, NUMCLASSES)
%filters is 4-D [64 64 1 2]
filters = single(bilinear_u(64, 2, 2)) ;
f = net.getLayerIndex('deconv32') ;
net.layers(f).block.size = size(filters) ;
net.layers(f).block.numGroups = 2 ;
f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;


