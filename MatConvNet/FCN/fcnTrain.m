function fcnTrain(varargin)
%FNCTRAIN Train FCN model using MatConvNet
%   Modified particularly for fcns32-hand-edsh

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = 'data/fcn32s-hand-edsh' ;
opts.dataDir = 'data/EDSH' ;
opts.modelType = 'hand-fcn8s' ;% not used yet
opts.sourceModelPath = fullfile(opts.expDir, 'fcn32s-epoch10.mat') ;

% imdbPath
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;

% remain default, used in the getBatch for cnn_train_dag
opts.numFetchThreads = 1 ; % not used yet

% training options (SGD for stochastic gradient descent)
opts.train = struct ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% used in cnn_train_dag
opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = [2] ;% use 3 gpus to accelerate
opts.train.prefetch = true ;% use in getBatch NOT UNDERSTAND
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,10) ;% set 10 epochs for now
opts.train.numEpochs = numel(opts.train.learningRate) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get edsh imdb
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = edshSetup('dataDir', opts.dataDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

% Get dataset statistics
if exist(opts.imdbStatsPath)
  stats = load(opts.imdbStatsPath) ;
else
  stats = getDatasetStatistics(imdb) ;
  save(opts.imdbStatsPath, '-struct', 'stats') ;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from pascal-fcn8s
net = fcnInitializeModel('sourceModelPath', opts.sourceModelPath) ;

%set net.meta
net.meta.normalization.rgbMean = stats.rgbMean ;
net.meta.classes = imdb.classes.name ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup bopts, i.e. batch options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,2,'single') ;
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
%function [net,stats] = cnn_train_dag(net, imdb, getBatch, varargin)
%getBatchWrapper is a function handle requiring inputs (imdb, batch)
%opts.train is a structure array, which can be used in varargin
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), ...
                     opts.train, ....
                     'train', train, ...
                     'val', val, ...
                     opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;
