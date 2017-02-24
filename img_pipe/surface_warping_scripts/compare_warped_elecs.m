function [elecmatrix_targ, elecmatrix_targ_chull] = compare_warped_elecs(subj, hem, elecmatrix_src, cortex_src, atlas, cortex_targ, labelprefix, saveflag, proj_direction,fsdir)
% [elecmatrix_targ, elecmatrix_targ_chull] = compare_warped_elecs(subj, hem, elecmatrix_src, cortex_src, atlas, cortex_targ, labelprefix, saveflag)
%
% Compares the warped electrodes from mri_label2label to the originals
% by plotting the electrodes on the brain in native space as well as the
% warped electrodes on the atlas brain.
%
%   Inputs:
%       subj:   'EC79'
%       hem:    'rh'
%       elecmatrix_src: [256 x 3] matrix of electrode coordinates in native
%               space
%       cortex_src: cortex structure (with tri and vert) for plotting pial
%               mesh
%       atlas:  'cvs_avg35_inMNI152'
%       cortex_targ:  pial surface for atlas
%       labelprefix:  for ls step to get all matching labels (see warp_elecs_wrapper)
%       saveflag: 0 or 1, whether to save out electrode matrix
%
%   Outputs:
%       elecmatrix_targ: [256 x 3] matrix of electrode locations in warp
%               space, on surface
%       elecmatrix_targ_chull: [256 x 3] matrix of electrode locations in
%               warp space, projected out to the convex hull
%
% 2/12/15: Liberty Hamilton
%

atlaslabelprefix = sprintf('%s.to.%s.%s', subj, atlas, labelprefix);

addpath(genpath(fsdir));

fprintf(1,'Loading warped electrode data from labels\n');
elecmatrix_targ = get_elecs_fromlabel(atlas, atlaslabelprefix, fsdir);

elecmatrix_targ_mean = nanmean(elecmatrix_targ,3);  % mean across multiple points (sometimes label is warped from 1 voxel to several, this takes the mean)

% color map
%cc=rand(256,3);
cc = jet(size(elecmatrix_src,1));

% Plot the electrodes before projecting atlas electrodes to the convex hull
% and before averaging across multiple voxels in the same electrode label
figure('visible','off');
subplot(1,2,1);
ctmr_gauss_plot(cortex_src, [0 0 0], 0, hem);
title('Native space');
subplot(1,2,2);
ctmr_gauss_plot(cortex_targ, [0 0 0], 0, hem);

title('Atlas space without conv hull');

for chan=1:size(elecmatrix_src,1)
    clr=cc(chan,:);
    subplot(1,2,1);
    el_add(elecmatrix_src(chan,:), 'elcol',clr);
    tt=text(elecmatrix_src(chan,1), elecmatrix_src(chan,2), elecmatrix_src(chan,3), num2str(chan));
    set(tt,'color','y');
    % find first index where there is a NaN (to determine number of duplicate points)
    [~,j]=ind2sub([size(elecmatrix_targ,2), size(elecmatrix_targ,3)], find(isnan(squeeze(elecmatrix_targ(chan,:,:))),1));
    for n=1:j-1
        subplot(1,2,2);
        el_add(elecmatrix_targ(chan,:,n), 'elcol', clr);
        tt=text(elecmatrix_targ(chan,1,n), elecmatrix_targ(chan,2,n), elecmatrix_targ(chan,3,n), num2str(chan));
        set(tt,'color','k');
    end
    if j>2
        el_add(elecmatrix_targ_mean(chan,:), 'elcol','r');
    end
end

% Now project the warped electrodes onto the convex hull of the atlas

if strcmp(hem, 'lh')
    hem2 = 'left';
elseif strcmp(hem, 'rh')
    hem2 = 'right';
end
debug_plot=0;

[elecmatrix_targ_chull] = project_electrodes_anydirection(cortex_targ, elecmatrix_targ_mean, proj_direction, debug_plot);

figure('visible','off');
subplot(1,2,1);
ctmr_gauss_plot(cortex_src, [0 0 0], 0, hem);
title('Native space');

subplot(1,2,2);
ctmr_gauss_plot(cortex_targ, [0 0 0], 0, hem);
title('Atlas space with averaging and conv. hull proj');

% Plot each channel with a different color (same colors as in the source brain plot)

for chan=1:size(elecmatrix_src,1)
    clr=cc(chan,:);
    subplot(1,2,1);
    el_add(elecmatrix_src(chan,:), 'elcol', clr);
    %tt=text(elecmatrix_src(chan,1), elecmatrix_src(chan,2), elecmatrix_src(chan,3), num2str(chan));
    %set(tt,'color','y');
    
    subplot(1,2,2);
    el_add(elecmatrix_targ_chull(chan,:), 'elcol', clr);
    %tt=text(elecmatrix_targ_chull(chan,1), elecmatrix_targ_chull(chan,2), elecmatrix_targ_chull(chan,3), num2str(chan));
    %set(tt,'color','y');
end

%saveas(gcf,sprintf('%s/%s/elecs/%s_surface_warp_QC.pdf',fsdir,subj,subj),'pdf');