import numpy as np

def TriangleRayIntersection (orig, direction, vert0, vert1, vert2, \
                             planeType='one sided', lineType='line', \
                             border='normal', eps=1e-5, fullReturn=False):
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
    #    * lineType - 'ray' (default), 'line' or 'segment' - how to treat rays:
    #        - 'line' means infinite (on both sides) line; 
    #        - 'ray' means infinite (on one side) ray comming out of origin; 
    #        - 'segment' means line segment bounded on both sides
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
    #N = np.max(orig.shape[0], dir.shape[0], vert0.shape[0], vert1.shape[0],vert2.shape[0])
    #if (size(orig ,1)==1 && N>1 && size(orig ,2)==3), orig  = repmat(orig , N, 1); end
    #if (size(dir  ,1)==1 && N>1 && size(dir  ,2)==3), dir   = repmat(dir  , N, 1); end
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
        print u.shape
        v = np.nan+np.zeros((u.shape[0], u.shape[1]))
        t = v
        ok = (angleOK) & (u>=-zero) & (u<=1.0+zero) # mask
        
        # if all line/plane intersections are outside the triangle than no intersections
        if not any(ok):
            intersect = ok

        qvec = np.cross(tvec[ok,:], edge1[ok,:]) # prepare to test V parameter
        v[ok,:] = np.sum(dir[ok,:]*qvec,axis=1) / det[ok,:] # 2nd barycentric coordinate
        
        if lineType != 'line': # 'position on the line' coordinate
            t[ok,:] = np.sum(edge2[ok,:]*qvec, axis=1)/det[ok,:]

        # test if line/plane intersection is within the triangle
        ok = (ok) & (v>=-zero) & ((u+v)<=(1.0+zero))


    # Test where along the line the line/plane intersection occurs
    if lineType == 'line':      # infinite line
        intersect = ok
    elif lineType == 'ray':
        intersect = (ok) & (t>=-zero) # intersection on the correct side of the origin
    elif lineType == 'segment':   # segment is bound on two sides
        intersect = (ok) & (t>=-zero) & (t<=1.0+zero) # intersection between origin and destination
    else:
        print("assuming line type")
        intersect = ok

    # calculate intersection coordinates if requested
    #xcoor = np.nan+np.zeros((orig.shape))
    xcoor = []
    ok = intersect
    #xcoor = vert0[ok,:] + edge1[ok,:]*u[ok] + edge2[ok,:]*v[ok]
    for o in np.where(ok)[0]:
        xcoor.append(vert0[o,:] + edge1[o,:]*u[o] + edge2[o,:]*v[o])

    return intersect, t, u, v, xcoor