function [] = make_elec_label(elecmatrix, cortex, subject, hem, labelprefix,fsdir)
% function [] = make_elec_label(elecmatrix, subject, hem, labelnameprefix)
% Create a freesurfer label for each electrode in an electrode matrix
% 
% Inputs:
%   elecmatrix:     [n x 3] matrix of electrode coordinates (usu. in native
%                           space)
%   cortex:         cortex struct from EC79_rh_pial.mat
%   subject:        'EC79'  string, subject ID
%   hem:            'lh' or 'rh'
%   labelprefix:    'hd_grid' or some other string describing the label
%
% Outputs:
%   Writes to a file named [hem].[labelprefix].chan[n].label (one label per channel)
%       This is a freesurfer label file
%       (https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles), 
%       where the first two header rows contain information about the subject, 
%       registration from surface to voxel space, and number of coordinates 
%       in the file.
%       The 3rd row and on should have 5 columns:
%           [vertex #]  [x coord]  [y coord]  [z coord]  [dummy value - freesurfer says "don't know" what this is]
%   Also writes to a file named [hem].[labelprefix].allchan.label (all electrodes in one label, easier to view when debugging)
% 
% Written 2/9/15 Liberty Hamilton
% 

n = size(elecmatrix,1);

% Getting nearest vertex
[vert_inds, coords] = nearest_electrode_vert(cortex, elecmatrix);
elecmatrix = coords;

% check for label directory!!
fprintf(1,'Making labels for each electrode separately:\n');
for chan=1:n
    % Open label file for writing
    labelname = sprintf('%s/%s/label/%s.%s.chan%03d.label', fsdir, subject, hem, labelprefix, chan);
    fid=fopen(labelname,'w');
    fprintf(1,'%s\n',labelname);
    
    % Print header of label file
    fprintf(fid,'#!ascii label  , from subject %s vox2ras=TkReg\n1\n', subject);
    fprintf(fid,'%i %.9f %.9f %.9f 0.0000000', vert_inds(chan), elecmatrix(chan,1), elecmatrix(chan,2), elecmatrix(chan,3));
    
    fclose(fid);
end

% Make a label with all the channels on it
fprintf(1,'Making a label for all the channels: ');

% Open label file for writing
labelname2 = sprintf('%s/%s/label/%s.%s.allchan.label', fsdir, subject, hem, labelprefix);
fprintf(1,'%s\n', labelname2);
fid2=fopen(labelname2,'w');
% Print header of label file
fprintf(fid2,'#!ascii label  , from subject %s vox2ras=TkReg\n256\n', subject);

for chan=1:n
    fprintf(fid2,'%i %.9f %.9f %.9f 0.0000000\n', vert_inds(chan), elecmatrix(chan,1), elecmatrix(chan,2), elecmatrix(chan,3));
end
fclose(fid2);