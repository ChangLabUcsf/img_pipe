function [c_h] = ctmr_gauss_plot(cortex,electrodes,weights,hemi,do_lighting)
% function [electrodes]=ctmr_gauss_plot(cortex,electrodes,weights)
% projects electrode locations onto their cortical spots in the
% left hemisphere and plots about them using a gaussian kernel
% for only cortex use:
% ctmr_gauss_plot(cortex,[0 0 0],0)
% rel_dir=which('loc_plot');
% rel_dir((length(rel_dir)-10):length(rel_dir))=[];
% addpath(rel_dir)

%     Copyright (C) 2009  K.J. Miller & D. Hermes, Dept of Neurology and Neurosurgery, University Medical Center Utrecht
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

%   Version 1.1.0, released 26-11-2009

if nargin<4
    hemi = 'lh';
end
if nargin<5
    do_lighting=1;
end

%load in colormap
% load('loc_colormap')
load('loc_colormap_thresh')
% load('BlWhRdYl_colormap')
% load('BlGyOrCp_colormap')
% cm = flipud(cbrewer('div','RdGy',64));

brain=cortex.vert;
v='l';
% %view from which side?
% temp=1;
% while temp==1
%     disp('---------------------------------------')
%     disp('to view from right press ''r''')
%     disp('to view from left press ''l''');
%     v=input('','s');
%     if v=='l'
%         temp=0;
%     elseif v=='r'
%         temp=0;
%     else
%         disp('you didn''t press r, or l try again (is caps on?)')
%     end
% end

if length(weights)~=length(electrodes(:,1))
    error('you sent a different number of weights than electrodes (perhaps a whole matrix instead of vector)')
end
%gaussian "cortical" spreading parameter - in mm, so if set at 10, its 1 cm
%- distance between adjacent electrodes
gsp=10; %zg edited from 50

c=zeros(length(cortex(:,1)),1);
for i=1:length(electrodes(:,1))
    b_z=abs(brain(:,3)-electrodes(i,3));
    b_y=abs(brain(:,2)-electrodes(i,2));
    b_x=abs(brain(:,1)-electrodes(i,1));
    %     d=weights(i)*exp((-(b_x.^2+b_z.^2+b_y.^2).^.5)/gsp^.5); %exponential fall off
    d=weights(i)*exp((-(b_x.^2+b_z.^2+b_y.^2))/gsp); %gaussian
    c=c+d';
end

% c=(c/max(c));
c_h=tripatch(cortex, 'nofigure', c');
shading interp;
a=get(gca);
%%NOTE: MAY WANT TO MAKE AXIS THE SAME MAGNITUDE ACROSS ALL COMPONENTS TO REFLECT
%%RELEVANCE OF CHANNEL FOR COMPARISON's ACROSS CORTICES
d=a.CLim;
set(gca,'CLim',[-max(abs(d)) max(abs(d))])
colormap(cm)
%colormap(jet)
lighting phong; %play with lighting...
%material shiny;
material dull;
%material([.3 .8 .1 10 1]);
%     material([.2 .9 .2 50 1]); %  BF: editing mesh viewing attributes
axis off

%set(gcf,'Renderer', 'zbuffer','Position',[500 500 900 900]); % BF: added for lateral. view
%set(gcf,'Renderer', 'zbuffer','Position',[400 400 500 900]); % BF: added for inf. view
%set(gcf,'Renderer', 'zbuffer','Position',[400 400 950 550]); % BF: added figure size for movie

% if v=='l'
if do_lighting
    l=light;
    
    if strcmp(hemi,'lh')
        view(270, 0);
        % set(l,'Position',[-1 0 1])
        set(l,'Position',[-1 0 0],'Color',[0.8 0.8 0.8]);
        % elseif v=='r'
    elseif strcmp(hemi,'rh')
        view(90, 0);
        % set(l,'Position',[1 0 1])
        set(l,'Position',[1 0 0],'Color',[0.8 0.8 0.8]);
    end
end
% %exportfig
% exportfig(gcf, strcat(cd,'\figout.png'), 'format', 'png', 'Renderer', 'painters', 'Color', 'cmyk', 'Resolution', 600, 'Width', 4, 'Height', 3);
% disp('figure saved as "figout"');
