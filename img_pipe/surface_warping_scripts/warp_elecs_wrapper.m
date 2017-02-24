fsdir = '/Applications/freesurfer/subjects';

% atlas to warp to
atlas = 'cvs_avg35_inMNI152';

% subject info
%subj='EC58';
%subj = 'EC6';
%subj = 'EC96';
subj = 'EC108';

hem='lh';
proj_direction = hem;

debug_plot=1;

% get cortical surface meshes
cortex_src = load(sprintf('%s/%s/Meshes/%s_%s_pial.mat', fsdir, subj, subj, hem)); 
cortex_targ = load(sprintf('%s/%s/Meshes/%s_%s_pial.mat', fsdir, atlas, atlas, hem));

labelprefix = 'hd_grid'; % for mat file and for naming label files
load(sprintf('%s/%s/elecs/%s.mat', fsdir, subj, labelprefix));

% project electrodes onto the nearest surface point
[elecs_proj] = project_electrodes_anydirection(cortex_src.cortex, elecmatrix, hem, debug_plot);

% make label files from the electrode matrix
fprintf(1,'Making labels...\n');
make_elec_label(elecs_proj, cortex_src.cortex, subj, hem, labelprefix);

% warp labels to the atlas
fprintf(1,'warping labels from %s to %s\n', subj, atlas);
run_label2label(subj, hem, labelprefix, atlas);

labelprefix2=sprintf('%s.%s.chan', hem, labelprefix);
[elecmatrix_targ, elecmatrix_targ_chull] = compare_warped_elecs(subj, hem, elecmatrix, cortex_src.cortex, atlas, cortex_targ.cortex, labelprefix2, [], proj_direction);

% Save electrode file (only the convex hull version, but could change this)
outfile = sprintf('%s/%s/elecs/%s_to_%s_%s.mat', fsdir, subj, subj, atlas, labelprefix);
elecmatrix = elecmatrix_targ_chull;
save(outfile, 'elecmatrix');

