function elecmatrix = get_elecs_fromlabel(subj, labelprefix,fsdir)
% function elecmatrix = get_elecs_fromlabel(subj, labelprefix)
% 
% Gets electrode coordinates from all label files with a specified label
% prefix.  Usually you are looking for 256 electrodes on a grid, though
% this is not explicitly set.
%
% Inputs:
%      subj: e.g. 'EC79' or 'cvs_avg35_inMNI152'
%      labelprefix: e.g. 'rh.hd_grid.chan' or 'EC79.to.cvs_avg35_inMNI152.rh.hd_grid.chan' 
%
% Written 2/10/15 by Liberty Hamilton
%

atlas = 'cvs_avg35_inMNI152';
all_labels = dir(sprintf('%s/%s/label/%s*', fsdir, subj, labelprefix));
nelecs = length(all_labels);
elecmatrix = nan(nelecs,3,10); % last dimension is larger than necessary 
npts = zeros(nelecs,1);
for i=1:length(all_labels)
    label = sprintf('%s/%s/label/%s', fsdir, subj, all_labels(i).name);
    fid = fopen(label);
    C=textscan(fid, '%f %f %f %f %f', 'headerlines',2);
    npts(i) = length(C{1});
    if npts(i) > 1
        fprintf(1,'Electrode %d was warped into %d voxels\n', i, npts(i));
    end
    for coord=1:3
        for n=1:npts(i)
            elecmatrix(i,coord,n) = C{coord+1}(n);
        end
    end
    fclose(fid);
end