function [vert_inds, coords] = nearest_electrode_vert(cortex,elecmatrix,matlab_order,yzplane)
%function [vert_inds, coords] = NEAREST_ELECTRODE_VERT(cortex,elecmatrix)
%
% Find the vertex on a mesh that is closest to the given electrode
% coordinates
%
% Returns vertex indices and the new coordinates
%
if nargin<3
    matlab_order = 0;
end
if nargin<4
    yzplane = 0;
end

% Find distance between each electrode coordinate and the surface vertices
d=zeros(size(elecmatrix,1), size(cortex.vert,1));
for chan=1:size(elecmatrix,1)
    if yzplane
        d(chan,:)=sqrt((elecmatrix(chan,2)-cortex.vert(:,2)).^2+(elecmatrix(chan,3)-cortex.vert(:,3)).^2);
    else
        d(chan,:)=sqrt((elecmatrix(chan,1)-cortex.vert(:,1)).^2+(elecmatrix(chan,2)-cortex.vert(:,2)).^2+(elecmatrix(chan,3)-cortex.vert(:,3)).^2);
    end
end
[i,vert_inds]=min(d'); % Find the indices where the distance is minimized

coords = cortex.vert(vert_inds,:);

if matlab_order==0
    % Important!! Vertex indices are 0 indexed for freesurfer, so we have to
    % subtract 1 from them here!
    vert_inds = vert_inds - 1; % for freesurfer compatibility
end