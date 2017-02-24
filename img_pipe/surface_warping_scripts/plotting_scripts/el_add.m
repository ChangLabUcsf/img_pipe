function [h] = el_add(els, varargin)
% function [h] = EL_ADD(els,varargin)
%
% Optional arguments:
%   'elcol': string, or [electrodes x 3] matrix of colors
%   'msize': marker size (default = 6)
%   'mtype': marker type (default = 'o')
%   'edgecol': edge color, (default = 'none')
%   'numbers': whether to plot numbers next to the electrode markers or not
%   (default = [], no numbers)
%
% Edited by Liberty Hamilton 2015
% Original code (C) 2006  K.J. Miller

p = inputParser;
p.addOptional('elcol','r');
p.addOptional('msize',6);
p.addOptional('mtype','o');
p.addOptional('edgecol','none');
p.addOptional('numbers',[]);

p.parse(varargin{:});

elcol = p.Results.elcol;
msize = p.Results.msize;
mtype = p.Results.mtype;
edgecol = p.Results.edgecol;
numbers = p.Results.numbers;

hold on;

if ~isstr(elcol) && size(elcol,1)>1 && size(elcol,2)==3
    for i=1:size(els,1)
        h(i)=plot3(els(i,1),els(i,2),els(i,3),mtype,'MarkerFaceColor', elcol(i,:),'MarkerSize',msize,'MarkerEdgeColor',edgecol,'LineWidth',1);
    end
else
    h=plot3(els(:,1),els(:,2),els(:,3), mtype,'MarkerFaceColor', elcol,'MarkerSize',msize,'MarkerEdgeColor',edgecol,'LineWidth',1);
end

if ~isempty(numbers)
    for i=1:size(els,1)
        tt=text(els(i,1),els(i,2),els(i,3),num2str(numbers(i)));
        set(tt,'color','b');
    end
end