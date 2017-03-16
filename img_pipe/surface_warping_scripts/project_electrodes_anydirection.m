function [elecs_proj,chull] = project_electrodes_anydirection(cortex, elecmatrix, proj_direction, debug_plot, surf_type)
% function [elecs_proj] = PROJECT_ELECTRODES_ANYDIRECTION(cortex, elecmatrix, proj_direction, debug_plot, surf_type)
%
% Projects electrode locations onto the convex hull of a cortical surface.
% This allows for the electrode locations to wrap smoothly around the
% cortical surface instead of being outside the brain or going into the
% sulci.
%
% Inputs: cortex:       cortical surface struct (with fields cortex.vert
%                       and cortex.tri)
%         elecmatrix:   a [nchans] x 3 electrode position array of locations
%                       of electrodes on the cortical surface (don't use
%                       this script for depth electrodes)
%         proj_direction:    a string with value
%                       'lh','rh','top','bottom','front','back' depending
%                       on where you want electrodes to project to
%         debug_plot:   0 or 1, whether to plot debugging plots or not
%         surf_type:    'alphavol', 'convex_hull', or 'none' to project
%                       directly to the nearest cortical surface point
%
% Output: elecs_proj:   A [nchans] x 3 position array of locations of
%                       electrodes that have been projected to the convex
%                       hull of the cortical surface.
%
% Written Dec. 2014 by Liberty Hamilton
%
% See also: TRIANGLERAYINTERSECTION, written by Jarek Tuszynski, available
% at http://www.mathworks.com/matlabcentral/fileexchange/33073-triangle-ray-intersection
hem=proj_direction;
if nargin < 4
    debug_plot = 0;
    if nargin < 3
        proj_direction = lower(input('lh/rh/top/bottom/front/back?', 's'));
    end
end
if nargin<5
    surf_type = 'alphavol';
end

% This flag is for calculating light direction as well as direction of line
% of intersection
if isnumeric(proj_direction)
%     proj_direction(1) = -proj_direction(1);
%     proj_direction(3) = -proj_direction(3);
    direction = proj_direction;
elseif strcmp(proj_direction, 'lh')
    direction = [-1000 0 0]; % IMPORTANT! This is not a point.
    % This is the direction to add to the
    % "orig" coordinate to get the line you
    % wish to intersect
elseif strcmp(proj_direction, 'rh')
    direction = [1000 0 0];
elseif strcmp(proj_direction, 'top')
    direction = [0 0 1000];
elseif strcmp(proj_direction, 'top_right')
    direction = [-1000 0 -800];
elseif strcmp(proj_direction, 'bottom')
    direction = [0 0 -1000];
elseif strcmp(proj_direction, 'front')
    direction = [0 1000 0];
elseif strcmp(proj_direction, 'back')
    direction = [0 -1000 0];
else % Assume grid on top
    direction = [0 0 1000];
end

direction

x = cortex.vert(:,1);
y = cortex.vert(:,2);
z = cortex.vert(:,3);

% create Delaunay triangulation of cortical surface
dt=delaunayTriangulation(x,y,z);

% Calculate the convex hull (or alpha volume)
chull = struct();

switch surf_type
    case 'convex_hull'
        k=convexHull(dt);
        [faces,vertices]=freeBoundary(dt);
        vert1 = vertices(faces(:,1),:);
        vert2 = vertices(faces(:,2),:);
        vert3 = vertices(faces(:,3),:);
        chull.vert = dt.Points;
        chull.tri = k;
    case 'alphavol'
        [V,S]=alphavol(cortex.vert,15,1);
        vert1 = cortex.vert(S.bnd(:,1),:);
        vert2 = cortex.vert(S.bnd(:,2),:);
        vert3 = cortex.vert(S.bnd(:,3),:);
        chull.vert = cortex.vert;
        chull.tri = S.bnd;
    case 'none'
        vert1 = cortex.vert(cortex.tri(:,1),:);
        vert2 = cortex.vert(cortex.tri(:,2),:);
        vert3 = cortex.vert(cortex.tri(:,3),:);
        chull.vert = cortex.vert;
        chull.tri = cortex.tri;
end

elecs_proj = zeros(size(elecmatrix));
elec_intersect = zeros(size(elecmatrix,1),1);
for i=1:size(elecmatrix,1) % Loop through all electrodes
    
    % Define a line with original (off-brain) electrode position as the
    % origin and pointing in a direction parallel to the x axis (so the
    % grid will only move left-right)
    orig = elecmatrix(i,:);
    
    % Calculate the intersection of the electrode with the convex hull mesh
    [elec_intersect, ~, ~, ~, xcoor] = TriangleRayIntersection(orig, direction, vert1, vert2, vert3, 'linetype','line','planetype','two sided');
    % Plot the convex hull mesh with electrodes, if desired
    % The face containing the point of intersection will be colored and a
    % blue dot will be placed along the ray at the intersection.
    if debug_plot
        figure(3); clf(); %set(gcf,'visible','on'); clf();
        trisurf(chull.tri, chull.vert(:,1), chull.vert(:,2), chull.vert(:,3), elec_intersect*1.0, 'FaceAlpha', 0.9, 'edgecolor', [0.3 0.3 0.3]); hold all;
        axis equal;
        line('XData',orig(1)+[0 direction(1)],'YData',orig(2)+[0 direction(2)],'ZData',...
            orig(3)+[0 direction(3)],'Color','r','LineWidth',3);
        scatter3(xcoor(elec_intersect,1), xcoor(elec_intersect,2), xcoor(elec_intersect,3), 100, 'b', 'o', 'filled');
        if i>1
            scatter3(elecs_proj(1:i,1),elecs_proj(1:i,2),elecs_proj(1:i,3), 50, 'y', 'o', 'filled');
        end
        colormap('Bone');
        if strcmp(proj_direction, 'lh')
            %view([-120+i/2.5 8]);
            %view([-168+i/2.5 8+i/4]);
            view([-120 8]);
        else
            view([120 8]);
            %view([120+i/2.5 8]);
        end
        axis off;
        %saveas(gcf,sprintf('~/Documents/UCSF/wikipics/proj_movie/elec_%03d.png',i), 'png');
        %pause(0.1);
    end
    
    % If there is more than one intersection of this ray with the cortical
    % mesh, only get the most negative coordinate for the left hemisphere,
    % or only the most positive coordinate for the right hemisphere (this
    % is for surface electrodes only!)
    if sum(elec_intersect)>1
        xctmp = xcoor(elec_intersect,:); % all intersecting coordinates
        [~,inds] = sort(xctmp(:,1));
        if strcmp(proj_direction,'lh')
            x = xctmp(inds(1),:);
        else % right hemisphere
            x = xctmp(inds(end),:);
        end

    elseif sum(elec_intersect)==1
        x = xcoor(elec_intersect,:);
    else
        x = NaN;
    end
    elecs_proj(i,:) = x;
end

if debug_plot
    figure(4); clf();
    ctmr_gauss_plot(cortex, [0 0 0], 0, 'lh');
    el_add(elecs_proj,'color','b');
end

% Not currently implemented: (this could replace the convex hull step)
%[V,S]=alphavol(cortex.vert,20,1); % May consider using this for electrodes
% on bottom surface or concave surfaces... need to develop!!
%figure; trisurf(S.bnd, cortex.vert(:,1), cortex.vert(:,2), cortex.vert(:,3))
