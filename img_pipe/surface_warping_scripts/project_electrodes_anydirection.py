def project_electrodes_anydirection(tri, vert, elecmatrix, proj_direction, surf_type = 'convex_hull', debug_plot=None):
	''' 
	Projects electrode locations onto the convex hull of a cortical surface.
	This allows for the electrode locations to wrap smoothly around the
	cortical surface instead of being outside the brain or going into the
	sulci.

	 Inputs: cortex:       cortical surface struct (with fields cortex.vert
	                       and cortex.tri)
	         elecmatrix:   a [nchans] x 3 electrode position array of locations
	                       of electrodes on the cortical surface (don't use
	                       this script for depth electrodes)
	         proj_direction:    a string with value
	                       'lh','rh','top','bottom','front','back' depending
	                       on where you want electrodes to project to
	         debug_plot:   0 or 1, whether to plot debugging plots or not
	         surf_type:    'alphavol', 'convex_hull', or 'none' to project
	                       directly to the nearest cortical surface point

	 Output: elecs_proj:   A [nchans] x 3 position array of locations of
	                       electrodes that have been projected to the convex
	                       hull of the cortical surface.

	 Written Dec. 2014 by Liberty Hamilton

	 See also: TRIANGLERAYINTERSECTION, written by Jarek Tuszynski, available
	 at http://www.mathworks.com/matlabcentral/fileexchange/33073-triangle-ray-intersection
	'''

# This flag is for calculating light direction as well as direction of line
# of intersection
if type(proj_direction) is list:
	direction = proj_direction
elif proj_direction=='lh':
	direction = [1000 0 0] # IMPORTANT! This is not a point.
	# This is the direction to add to the
	# "orig" coordinate to get the line you
	# wish to intersect
elif proj_direction=='rh':
	direction = [-1000 0 0]
elif proj_direction=='top':
	direction = [0 0 1000]
elif proj_direction=='topright':
	direction = [-1000 0 -800]
elif proj_direction=='bottom':
	direction = [0 0 -1000]
elif proj_direction=='front':
	direction = [0 1000 0]
elif proj_direction=='back'
	direction = [0 -1000 0]
else # Assume grid on left
	direction = [1000 0 0]

if surf_type == 'convex_hull':
	# Use dural surface?  
else:
	vert1 = vert[tri[:,0],:]
    vert2 = vert[tri[:,1],:]
    vert3 = vert[tri[:,2],:]

######### THE REST OF THIS IS MATLAB ########

elecs_proj = zeros(size(elecmatrix));
elec_intersect = zeros(size(elecmatrix,1),1);
for i=1:size(elecmatrix,1) # Loop through all electrodes
    
    # Define a line with original (off-brain) electrode position as the
    # origin and pointing in a direction parallel to the x axis (so the
    # grid will only move left-right)
    orig = elecmatrix(i,:);
    
    # Calculate the intersection of the electrode with the convex hull mesh
    [elec_intersect, ~, ~, ~, xcoor] = TriangleRayIntersection(orig, direction, vert1, vert2, vert3, 'linetype','line','planetype','one sided');
    
    # Plot the convex hull mesh with electrodes, if desired
    # The face containing the point of intersection will be colored and a
    # blue dot will be placed along the ray at the intersection.
    if debug_plot
        figure(3); clf(); #set(gcf,'visible','on'); clf();
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
            #view([-120+i/2.5 8]);
            #view([-168+i/2.5 8+i/4]);
            view([-120 8]);
        else
            view([120 8]);
            #view([120+i/2.5 8]);
        end
        axis off;
        #saveas(gcf,sprintf('~/Documents/UCSF/wikipics/proj_movie/elec_#03d.png',i), 'png');
        #pause(0.1);
    end
    
    # If there is more than one intersection of this ray with the cortical
    # mesh, only get the most negative coordinate for the left hemisphere,
    # or only the most positive coordinate for the right hemisphere (this
    # is for surface electrodes only!)
    if sum(elec_intersect)>1
        xctmp = xcoor(elec_intersect,:); # all intersecting coordinates
        [~,inds] = sort(xctmp(:,1));
        if strcmp(proj_direction,'left')
            x = xctmp(inds(1),:);
        else # right hemisphere
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
    plot_electrodes(cortex, elecs_proj, hem, 'Projected electrodes');
end

# Not currently implemented: (this could replace the convex hull step)
#[V,S]=alphavol(cortex.vert,20,1); # May consider using this for electrodes
# on bottom surface or concave surfaces... need to develop!!
#figure; trisurf(S.bnd, cortex.vert(:,1), cortex.vert(:,2), cortex.vert(:,3))
