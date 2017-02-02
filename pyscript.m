fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
addpath('/Applications/MATLAB_R2015b.app/toolbox/spm12');
addpath(genpath('/Users/dlchang/img_pipe/surface_warping_scripts'));                              plot_recon_anatomy_compare_warped('EC_test2','cvs_avg35_inMNI152','lh');
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;