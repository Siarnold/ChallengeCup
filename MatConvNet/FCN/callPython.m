import py.sys.path ;
disp(path) ;

[mPath,~,~] = fileparts(mfilename('fullpath')) ;
if count(py.sys.path, mPath) == 0
    insert(py.sys.path,int32(0),mPath) ;
end
disp(path) ;

py.test.hello() ;