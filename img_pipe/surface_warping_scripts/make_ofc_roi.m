function ofc = make_ofc_roi(subj, hem, cortex, small_roi)
%           [vertex #]  [x coord]  [y coord]  [z coord]  [dummy value - freesurfer says "don't know" what this is]
%

if nargin<3
    cortex =[];
end
if nargin<4
    small_roi = 0;
end

fsdir =  '/data_store2/imaging/subjects/';
subjdir = sprintf('%s/%s', fsdir, subj);
%recondir = sprintf('/Users/liberty/Documents/UCSF/data/timit/%s/recon', subj);
recondir = sprintf('%s/Meshes', subjdir); % for cvs_avg35_inMNI152

labeldir = sprintf('%s/label/gyri', subjdir);

if small_roi
    label_list = {'lateralorbitofrontal','medialorbitofrontal'};
else
    label_list = {'lateralorbitofrontal','medialorbitofrontal','rostralmiddlefrontal','parsorbitalis','parstriangularis','superiorfrontal','rostralanteriorcingulate','caudalanteriorcingulate','frontalpole','insula'};   
end

outfile = sprintf('%s/%s_%s_ofc_pial.mat', recondir, subj, hem);

if 1%~exist(outfile, 'file')
    if isempty(cortex)
        cortex = get_fsbrain_image(subj, hem);
    end
    pial_surf = struct();
    pial_surf.cortex = cortex;
    %pial_surf = load(sprintf('%s/%s_%s_pial.mat', recondir, subj, hem));
    ofc = struct();
    ofc.tri = [];
    ofc.vert = [];
    vertnums = [];
    for lab = 1:length(label_list)
        this_label = sprintf('%s/%s.%s.label', labeldir, hem, label_list{lab})
        fid=fopen(this_label);
        C = textscan(fid, '%f %f %f %f %f','Headerlines',2);
        verts = C{1}+1;
        
        % Find the vertices and add them to the ofc lobe vertices
        vertnums = [verts; vertnums];
        fclose(fid);
    end
    
    % Sort the vertices so they're drawn in the correct order
    vertnums = sort(vertnums);
    ofc.vert = pial_surf.cortex.vert(vertnums,:);
    
    vnum_new = 1:length(vertnums); % Index of the vertex (new, relative to ofc lobe)
    tri_list = [];
    % Find the triangles for these vertex numbers
    tri_row_inds = find(sum(ismember(pial_surf.cortex.tri, vertnums),2)==3);
    tri_list = pial_surf.cortex.tri(tri_row_inds,:);
    
    [i,j]=ismember(tri_list,vertnums);
    
    ofc.tri=j;
    
    fprintf(1,'Saving OFC recon %s\n', outfile);
    cortex = ofc;
    save(outfile, 'cortex');
else
    load(outfile)
end


