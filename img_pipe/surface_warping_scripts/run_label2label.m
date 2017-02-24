function [] = run_label2label(subj, hem, labelprefix, atlas,fsdir,fsBinDir)
% function [] = run_label2label(subj, hem, labelprefix, atlas)
% 
% Use with make_elec_label.m
% Transforms a label file into the cvs_avg35_inMNI152 atlas space (or other
% atlas space you specify)
%
% Written 2/10/15 by Liberty Hamilton

if nargin < 4
    atlas = 'cvs_avg35_inMNI152';
end

all_labels = dir(sprintf('%s/%s/label/%s.%s.*', fsdir, subj, hem, lower(labelprefix)));

for i = 1:length(all_labels)
    srclabel = sprintf('%s/%s/label/%s', fsdir, subj, all_labels(i).name);
    trglabel = sprintf('%s/%s/label/%s.to.%s.%s', fsdir, atlas, subj, atlas, all_labels(i).name);
    system('source ~/.bash_profile')
    cmd=['export SUBJECTS_DIR=' fsdir '; export FREESURFER_HOME=' fsBinDir '; ' fsBinDir '/bin/mri_label2label --srclabel ' srclabel ' --srcsubject ' subj ' --trgsubject ' atlas ' --trglabel ' trglabel ' --regmethod surface --hemi ' hem ' --trgsurf pial --paint 6 pial --sd ' fsdir];
    disp(cmd);
    system(cmd);
end