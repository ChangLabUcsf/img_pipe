function [elecmatrix] = warp_elecs(subj, hem, labelprefix,fsdir,fsBinDir,atlas)
%function [elecmatrix] = warp_ofc_elecs(subj, hem, labelprefix)
% 
% warp OFC electrodes from subject [subj] in hemisphere [hem] 
% to the cvs_avg35_inMNI152 brain using a surface warping algorithm.
%
% You will need the freesurfer directory for your subject with the correct directory
% structure, including the surf and label directories.
%
% Example: 
  %labelprefix = 'OFC_minigrid.mat';
  %subj = 'EC118';
  %hem = 'lh';
  %elecmatrix = warp_ofc_elecs(subj, hem, labelprefix);
%
% Written 2015 by Liberty Hamilton
 
%fsdir = '../'

debug_plot=0;

% get cortical surface meshes
cortex_src = load(sprintf('%s/%s/Meshes/%s_%s_pial.mat', fsdir, subj, subj, hem)); 
cortex_targ = load(sprintf('%s/%s/Meshes/%s_%s_pial.mat', fsdir, atlas, atlas, hem));

%labelprefix = 'hd_grid'; % for mat file and for naming label files
load(sprintf('%s/%s/elecs/%s.mat', fsdir, subj, labelprefix));

indices = find(~strcmp(anatomy(:,3),'depth'));
%indices = intersect(find(~strcmp(anatomy(:,3),'depth')),intersect(find(~strcmp(anatomy(:,3),'EKG')),find(~strcmp(anatomy(:,3),'NaN'))));
elecmatrix = elecmatrix(indices,:); 

% make label files from the electrode matrix
fprintf(1,'Making labels...\n');
make_elec_label(elecmatrix, cortex_src.cortex, subj, hem, lower(labelprefix),fsdir);

% warp labels to the atlas
fprintf(1,'warping labels from %s to %s, labelprefix: %s\n', subj, atlas, lower(labelprefix));
run_label2label(subj, hem, labelprefix, atlas,fsdir,fsBinDir);

labelprefix2=sprintf('%s.%s.chan', hem, lower(labelprefix));
proj_direction = hem;
[elecmatrix_targ, elecmatrix_targ_chull] = compare_warped_elecs(subj, hem, elecmatrix, cortex_src.cortex, atlas, cortex_targ.cortex, labelprefix2, 0, proj_direction,fsdir);

% Save electrode file (only the convex hull version, but could change this)
outfile = sprintf('%s/%s/elecs/%s_surface_warped.mat', fsdir, subj, labelprefix);
elecmatrix = nanmean(elecmatrix_targ,3); % Take the mean across points in case an electrode was split into multiple voxels

save(outfile, 'elecmatrix', 'eleclabels', 'anatomy');

