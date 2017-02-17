function stats = getDatasetStatistics(imdb)

%get the train and segment-labeled images
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;

% Class statistics
%class means class segmentation: not instance-aware
%set 21 because of 20 classes and the default class background(others)-0
classCounts = zeros(21,1) ;%init: zero 21*1 column array
for i = 1:numel(train)
  fprintf('%s: computing segmentation stats for training image %d\n', mfilename, i) ;
  lb = imread(sprintf(imdb.paths.classSegmentation, imdb.images.name{train(i)})) ;
  ok = lb < 255 ;%255 means white, 0 means black
  classCounts = classCounts + accumarray(lb(ok(:))+1, 1, [21 1]) ;
end
%classCounts means the total pixels of each class
stats.classCounts = classCounts ;

% Image statistics
for t=1:numel(train)
  fprintf('%s: computing RGB stats for training image %d\n', mfilename, t) ;
  rgb = imread(sprintf(imdb.paths.image, imdb.images.name{train(t)})) ;
  rgb = single(rgb) ;
  %permute to rearrange the dimension, reshape to get a 3*[AUTO] matrix
  z = reshape(permute(rgb,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
end
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;

%the rgbMean of all train images
stats.rgbMean = rgbm1 ;
stats.rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
