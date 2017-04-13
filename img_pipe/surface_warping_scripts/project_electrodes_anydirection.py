import numpy as np

def project_electrodes_anydirection(tri, vert, elecmatrix, proj_direction):
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
    	direction = [1000, 0, 0] # IMPORTANT! This is not a point.
    	# This is the direction to add to the
    	# "orig" coordinate to get the line you
    	# wish to intersect
    elif proj_direction=='rh':
    	direction = [-1000, 0, 0]
    elif proj_direction=='top':
        direction = [0, 0, -1000]
    elif proj_direction=='bottom':
        direction = [0, 0, 1000]
    elif proj_direction=='front':
        direction = [0, 1000, 0]
    elif proj_direction=='back':
        direction = [0, -1000, 0]

    vert1 = vert[tri[:,0],:]
    vert2 = vert[tri[:,1],:]
    vert3 = vert[tri[:,2],:]

    elecs_proj = np.zeros(elecmatrix.shape)
    elec_intersect = np.zeros((elecmatrix.shape[0],1))
    for i in range(elecmatrix.shape[0]): # Loop through all electrodes  
        # Define a line with original (off-brain) electrode position as the
        # origin and pointing in a direction parallel to the x axis (so the
        # grid will only move left-right)
        orig = elecmatrix[i,:]
        # Calculate the intersection of the electrode with the convex hull mesh
        elec_intersect,_,_,_,xcoor = TriangleRayIntersection(orig, direction, vert1, vert2, vert3)
        #print xcoor[elec_intersect,:] , elec_intersect.sum(), np.where(elec_intersect>0), xcoor.shape
        # If there is more than one intersection of this ray with the cortical
        # mesh, only get the most negative coordinate for the left hemisphere,
        # or only the most positive coordinate for the right hemisphere (this
        # is for surface electrodes only!)

        if np.sum(elec_intersect)>1:
            xctmp = xcoor[elec_intersect,:] # all intersecting coordinates
            # #code block that selects based on lh or rh
            # sorted_xctmp = np.sort(xctmp,axis=0)
            # if proj_direction=='lh':
            #     x = sorted_xctmp[0,:]
            # else: # right hemisphere
            #     x = sorted_xctmp[-1,:]

            #instead, select from competing intersection coordinates using least euclidean distance metric
            x = xctmp[np.argmin(np.sum(np.abs(orig-xctmp)**2,axis=1)**(1./2)),:]
            elecs_proj[i,:] = x
        elif np.sum(elec_intersect)==1:
            x = xcoor[elec_intersect,:]
        else:
            x = np.nan
        elecs_proj[i,:] = x

    return elecs_proj

    # Not currently implemented: (this could replace the convex hull step)
    #[V,S]=alphavol(cortex.vert,20,1) # May consider using this for electrodes
    # on bottom surface or concave surfaces... need to develop!!
    #figure trisurf(S.bnd, cortex.vert(:,1), cortex.vert(:,2), cortex.vert(:,3))

def TriangleRayIntersection (orig, direction, vert0, vert1, vert2, planeType='two sided', border='normal', eps=1e-5, fullReturn=False):
    # TRIANGLERAYINTERSECTION Ray/triangle intersection.
    #    INTERSECT = TriangleRayIntersection(ORIG, DIRECTION, VERT1, VERT2, VERT3) 
    #      calculates ray/triangle intersections using the algorithm proposed
    #      BY Moller and Trumbore (1997), implemented as highly vectorized
    #      MATLAB code. The ray starts at ORIG and points toward DIRECTION. The 
    #      triangle is defined by vertix points: VERT1, VERT2, VERT3. All input  
    #      arrays are in Nx3 or 1x3 format, where N is number of triangles or 
    #      rays.
    # 
    #   [INTERSECT, T, U, V, XCOOR] = TriangleRayIntersection(...) 
    #     Returns:
    #     * Intersect - boolean array of length N informing which line and
    #                 triangle pair intersect
    #     * t   - distance from the ray origin to the intersection point in 
    #             units of |dir|. Provided only for line/triangle pair that 
    #             intersect unless 'fullReturn' parameter is true.
    #     * u,v - barycentric coordinates of the intersection point 
    #     * xcoor - carthesian coordinates of the intersection point
    #
    #   TriangleRayIntersection(...,'param','value','param','value'...) allows
    #    additional param/value pairs to be used. Allowed parameters:
    #    * planeType - 'one sided' or 'two sided' (default) - how to treat
    #        triangles. In 'one sided' version only intersections in single
    #        direction are counted and intersections with back facing
    #           tringles are ignored
    #    * border - controls border handling:
    #        - 'normal'(default) border - triangle is exactly as defined. 
    #           Intersections with border points can be easily lost due to
    #           rounding errors. 
    #        - 'inclusive' border - triangle is marginally larger.
    #           Intersections with border points are always captured but can
    #           lead to double counting when working with surfaces.
    #        - 'exclusive' border - triangle is marginally smaller. 
    #           Intersections with border points are not captured and can
    #           lead to under-counting when working with surfaces.
    #    * epsilon - (default = 1e-5) controls border size
    #    * fullReturn - (default = false) controls returned variables t, u, v, 
    #        and xcoor
    #
    # ALGORITHM:
    #  Function solves
    #        |t|
    #    M * |u| = (o-v0)
    #        |v|
    #  for [t; u; v] where M = [-d, v1-v0, v2-v0]. u,v are barycentric coordinates
    #  and t - the distance from the ray origin in |d| units
    #  ray/triangle intersect if u>=0, v>=0 and u+v<=1
    #
    # NOTE:
    #  The algorithm is able to solve several types of problems:
    #  * many faces / single ray  intersection
    #  * one  face  / many   rays intersection
    #  * one  face  / one    ray  intersection
    #  * many faces / many   rays intersection
    #  In order to allow that to happen all imput arrays are expected in Nx3
    #  format, where N is number of vertices or rays. In most cases number of
    #  vertices is different than number of rays, so one of the imputs will
    #  have to be cloned to have the right size. Use "repmat(A,size(B,1),1)".
    #
    # Based on:
    #  *"Fast, minimum storage ray-triangle intersection". Tomas Moller and
    #    Ben Trumbore. Journal of Graphics Tools, 2(1):21--28, 1997.
    #    http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    #  * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/
    #  * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/raytri/raytri.c
    #
    # Author:
    #    Jarek Tuszynski (jaroslaw.w.tuszynski@leidos.com)
    #
    # License: BSD license (http://en.wikipedia.org/wiki/BSD_licenses)
    # 
    # Converted to python by Liberty Hamilton 2017

    # In case of single points clone them to the same size as the rest
    orig, direction, vert0, vert1, vert2 = np.array(orig),np.array(direction),np.array(vert0),np.array(vert1),np.array(vert2)

    N = max(orig.shape[0], direction.shape[0], vert0.shape[0], vert1.shape[0],vert2.shape[0])
    if orig.shape[0]==3:
        orig = np.tile(orig,(N,1))
    if direction.shape[0]==3:
        direction = np.tile(direction,(N,1))
    #if (size(orig ,1)==1 && N>1 && size(orig ,2)==3), orig  = repmat(orig , N, 1); end
    #if (size(direction  ,1)==1 && N>1 && size(direction  ,2)==3), direction   = repmat(direction  , N, 1); end
    #if (size(vert0,1)==1 && N>1 && size(vert0,2)==3), vert0 = repmat(vert0, N, 1); end
    #if (size(vert1,1)==1 && N>1 && size(vert1,2)==3), vert1 = repmat(vert1, N, 1); end
    #if (size(vert2,1)==1 && N>1 && size(vert2,2)==3), vert2 = repmat(vert2, N, 1); end

    # Check if all the sizes match
    # SameSize = (any(size(orig)==size(vert0)) && ...
    #   any(size(orig)==size(vert1)) && ...
    #   any(size(orig)==size(vert2)) && ...
    #   any(size(orig)==size(dir  )) );
    # assert(SameSize && size(orig,2)==3, ...
    #   'All input vectors have to be in Nx3 format.');

    # Set up border parameter
    if border=='normal':
        zero=0.0
    elif border=='inclusive':
        zero=eps
    elif border=='exclusive':
        zero=-eps
    else:
        print("Using 'normal' border parameter")
        zero=0.0

    # initialize default output
    intersect = np.zeros((orig.shape[0],), dtype=bool) # by default there are no intersections
    t = np.inf + np.zeros((orig.shape[0],))
    u = t
    v = t 

    # Find faces parallel to the ray
    edge1 = vert1 - vert0          # find vectors for two edges sharing vert0
    edge2 = vert2 - vert0
    tvec  = orig - vert0          # vector from vert0 to ray origin
    pvec  = np.cross(direction, edge2) # begin calculating determinant - also used to calculate U parameter
    det   = np.sum(edge1*pvec, axis=1)   # determinant of the matrix M = dot(edge1,pvec)


    if planeType=='two sided':    # treats triangles as two sided
        angleOK = np.abs(det)>eps  # if determinant is near zero then ray lies in the plane of the triangle
    elif planeType=='one sided':  # treats triangles as one sided
        angleOK = det>eps
    else:
        print("Using two sided plane")
        angleOK = np.abs(det)>eps

    # if all parallel than no intersections
    if not np.any(angleOK):
        intersect = False

    # Different behavior depending on one or two sided triangles
    det[np.invert(angleOK)] = np.nan              # change to avoid division by zero
    u    = np.sum(tvec*pvec, axis=1)/det    # 1st barycentric coordinate

    if fullReturn:
        # calculate all variables for all line/triangle pairs
        qvec = np.cross(tvec, edge1)    # prepare to test V parameter
        v    = np.sum(direction  *qvec,axis=1)/det   # 2nd barycentric coordinate
        t    = np.sum(edge2*qvec,axis=1)/det   # 'position on the line' coordinate
        # test if line/plane intersection is within the triangle
        ok   = (angleOK) & (u>=-zero) & (v>=-zero) & ((u+v)<=(1.0+zero))
    else:
        # limit some calculations only to line/triangle pairs where it makes
        # a difference. It is tempting to try to push this concept of
        # limiting the number of calculations to only the necessary to "u"
        # and "t" but that produces slower code
        v = np.nan+np.zeros((u.shape[0], ))
        t = v
        ok = (angleOK) & (u>=-zero) & (u<=1.0+zero) # mask
        
        # if all line/plane intersections are outside the triangle than no intersections
        if not any(ok):
            intersect = ok

        qvec = np.cross(tvec[ok,:], edge1[ok,:]) # prepare to test V parameter
        v[ok] = np.sum(direction[ok,:]*qvec,axis=1) / det[ok] # 2nd barycentric coordinate

        # test if line/plane intersection is within the triangle
        ok = (ok) & (v>=-zero) & ((u+v)<=(1.0+zero))


    # Test where along the line the line/plane intersection occurs
    intersect = ok

    # calculate intersection coordinates if requested
    xcoor = np.nan+np.zeros((orig.shape))
    ok = intersect
    #xcoor = vert0[ok,:] + edge1[ok,:]*u[ok] + edge2[ok,:]*v[ok]
    for o in np.where(ok)[0]:
        xcoor[o,:] = (vert0[o,:] + edge1[o,:]*u[o] + edge2[o,:]*v[o])

    return np.array(intersect), np.array(t), np.array(u), np.array(v), xcoor
