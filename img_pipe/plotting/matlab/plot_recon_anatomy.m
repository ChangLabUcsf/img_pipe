function [elecmatrix, anatomy] = plot_recon_anatomy(fs_dir, subjects_dir,subj, hem, elecfile_prefix, zero_indexed)
% [elecmatrix, anatomy] = plot_recon_anatomy(subj, hem)
% 
% Plots anatomy 

subj_dir = sprintf('%s/%s/', subjects_dir, subj);

load(sprintf('%s/Meshes/%s_%s_pial.mat', subj_dir, subj, hem));
load(sprintf('%s/elecs/%s.mat', subj_dir,elecfile_prefix));

figure;
c_h = ctmr_gauss_plot(cortex, [0 0 0], 0); 

if strcmp(hem, 'rh') %commented out b/c segfaulting...?
    loc_view(90,0);
end

brain_areas = unique(anatomy(:,4));
brain_areas(strcmp(brain_areas,'')) = []; %remove empty string segment from brain_areas
brain_areas(strcmp(brain_areas,'NaN')) = []; %remove empty string segment from brain_areas

% Make a jet color map with the same # of colors as brain areas
cm = colorcube(length(brain_areas));

% Loop through brain areas
for i=1:length(brain_areas)
    fprintf(1,'Brain area: %s\n', brain_areas{i});
    inds = find(strcmp(brain_areas{i}, anatomy(:,4)));
    if strcmp(zero_indexed,'True')
        inds_labels = inds-1;
    else
        inds_labels = inds;
    end
    % Have to change label for LUT
    if ~strcmp('CC_Anterior',strtrim(brain_areas{i})) && ~strcmp('CC_Mid_Anterior',strtrim(brain_areas{i})) && ~strcmp('Brain-Stem',strtrim(brain_areas{i})) && ~strcmp('WM',strtrim(brain_areas{i}(1:2))) && ~strcmp('ctx',brain_areas{i}(1:3)) && (length(brain_areas{i})<4 || ~strcmp('Left',brain_areas{i}(1:4))) && (length(brain_areas{i})<5 || ~strcmp('Right',brain_areas{i}(1:5)))
        this_area = sprintf('ctx-%s-%s', hem, brain_areas{i});
    else
        this_area = brain_areas{i};
    end

    grep_cmd = sprintf('grep ''%s'' %s/FreeSurferColorLUT.txt', this_area,fs_dir);
    [stat, result] = system(grep_cmd);

    if ~isempty(result)
        LUT = strsplit(result, '\s','DelimiterType','RegularExpression');
        elec_color = [str2double(LUT{3}) str2double(LUT{4}) str2double(LUT{5})]./255;
    else
        elec_color = cm(i,:);
    end
    
    el_add(elecmatrix(inds,:), 'elcol', elec_color, 'numbers', inds_labels, 'msize', 8);
    
end

leg = legend(['brain';brain_areas]);