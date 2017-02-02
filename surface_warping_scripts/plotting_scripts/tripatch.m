function handle=tripatch(struct, nofigure, lims, varargin)
% TRIPATCH handle=tripatch(struct, nofigure)

if nargin<2 | isempty(nofigure)
   figure('visible','off');
end% if
if nargin<3
   handle=trisurf(struct.tri, struct.vert(:, 1), struct.vert(:, 2), struct.vert(:, 3));

else
   if isnumeric(varargin{1})
      col=varargin{1};
      %this block added by dchang 6/20/16
      if isstr(lims) && strcmp(lims,'auto')
          if ~(-max(abs(col))== max(abs(col)))
              set(gca,'CLim',[-max(abs(col)) max(abs(col))])
          end
      else
          lower_clim = lims(1);
          upper_clim =  lims(2);
          zero_indices = find(col<0.001);
          col(zero_indices) = mean([lower_clim upper_clim]);
          set(gca,'CLim',[lower_clim upper_clim]);     
      end
      %end block
      varargin(1)=[];
      if [1 3]==sort(size(col))
         col=repmat(col(:)', [size(struct.vert, 1) 1]);
      end% if
      handle=trisurf(struct.tri, struct.vert(:, 1), struct.vert(:, 2), struct.vert(:, 3), ...
         'FaceVertexCData', col, varargin{:});
      if length(col)==size(struct.vert, 1)
         set(handle, 'FaceColor', 'interp');
      end% if

   else
      handle=trisurf(struct.tri, struct.vert(:, 1), struct.vert(:, 2), struct.vert(:, 3), varargin{:});
   end% if
end% if
axis tight
axis equal
hold on
if version('-release')>=12
   cameratoolbar('setmode', 'orbit')
else
   rotate3d on
end% if
