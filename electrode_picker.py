#!/usr/env/pytho
import matplotlib
matplotlib.use('Qt4Agg') 
from matplotlib import pyplot as plt
plt.rcParams['keymap.save'] = '' # Unbind 's' key saving
from matplotlib import cm
import numpy as np
import nibabel as nib
import scipy.ndimage

img = nib.load('/Applications/freesurfer/subjects/EC121_test/mri/brain.mgz')
ct = nib.load('/Applications/freesurfer/subjects/EC121_test/CT/rCT.nii')
#img = nib.load('/Applications/freesurfer/subjects/EC55/mri/brain.mgz')
#ct = nib.load('/Applications/freesurfer/subjects/EC55/CT/rCT.nii')

class electrode_picker:
	def __init__(self, img=img, ct=ct):
		self.img = img
		self.ct = ct
		self.affine = img.affine
		self.codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(self.affine))
		img_data = nib.orientations.apply_orientation(img.get_data(), self.codes)
		self.voxel_sizes = nib.affines.voxel_sizes(self.affine)
		nx,ny,nz = np.array(img_data.shape, dtype='float')

		self.inv_affine = np.linalg.inv(self.affine)
		self.img_clim = np.percentile(img_data, (1., 99.))

		#ct_data = ct.get_data()
		self.ct_codes =nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(ct.affine))
		ct_data = nib.orientations.apply_orientation(ct.get_data(), self.ct_codes)
		ct_data[ct_data<1000] = np.nan
		cx,cy,cz=np.array(ct.shape, dtype='float')
		

		# Resample both images to the highest resolution
		voxsz = (256, 256, 256)
		if ct.shape != voxsz:
			print("Resizing voxels in CT")
			ct_data = scipy.ndimage.zoom(ct_data, [voxsz[0]/cx, voxsz[1]/cy, voxsz[2]/cz])
		if img.shape != voxsz:
			print("Resizing voxels in MRI")
			img_data = scipy.ndimage.zoom(img_data, [voxsz[0]/nx, voxsz[1]/ny, voxsz[2]/nz])
		
		self.ct_data = ct_data
		self.img_data = img_data
		self.elec_data = np.nan+np.zeros((img_data.shape))
		self.imsz = [256, 256, 256]
		self.ctsz = [256, 256, 256]

		self.current_slice = np.array([self.imsz[0]/2, self.imsz[1]/2, self.imsz[2]/2], dtype=np.float)
		
		self.fig=plt.figure(figsize=(12,10))
		self.im = []
		self.ct_im = []
		self.elec_im = []

		self.cursor = []
		self.cursor2 = []

		im_ranges = [[0, self.imsz[1], 0, self.imsz[2]],
					 [0, self.imsz[0], 0, self.imsz[2]],
					 [0, self.imsz[0], 0, self.imsz[1]]]
		im_labels = [['Inferior','Posterior'],
					 ['Inferior','Left'],
					 ['Posterior','Left']]

		self.ax = []
		cs = np.round(self.current_slice).astype(np.int)
		for i in np.arange(3):
			self.ax.append(self.fig.add_subplot(2,2,i+1))
			self.ax[i].set_axis_bgcolor('k')
			if i==0:
				imdata = img_data[cs[0],:,:].T
				ctdat = ct_data[cs[0],:,:].T
				edat = self.elec_data[cs[0],:,:].T
			elif i==1:
				imdata = img_data[:,cs[1],:].T
				ctdat = ct_data[:,cs[1],:].T
				edat = self.elec_data[:,cs[1],:].T
			elif i==2:
				imdata = img_data[:,:,cs[2]].T
				ctdat = ct_data[:,:,cs[2]].T
				edat = self.elec_data[:,:,cs[2]].T

			self.im.append(plt.imshow(imdata, cmap=cm.gray, aspect='auto'))
			self.ct_im.append(plt.imshow(ctdat, cmap=cm.hot, aspect='auto',alpha=0.5, vmin=1000, vmax=3000))
			self.elec_im.append(plt.imshow(edat, cmap=cm.Set1, aspect='auto', alpha=1, vmin=0, vmax=10))
			self.cursor.append(plt.plot([cs[1], cs[1]], [self.ax[i].get_ylim()[0]+1, self.ax[i].get_ylim()[1]-1], color=[0, 1, 0] ))
			self.cursor2.append(plt.plot([self.ax[i].get_xlim()[0]+1, self.ax[i].get_xlim()[1]-1], [cs[2], cs[2]], color=[0, 1, 0] ))
			plt.gca().invert_yaxis()
			self.ax[i].set_xticks([])
			self.ax[i].set_yticks([])
			self.ax[i].set_xlabel(im_labels[i][0])
			self.ax[i].set_ylabel(im_labels[i][1])
			self.ax[i].axis(im_ranges[i])

		self.ct_slice = 's' # Show sagittal MIP to start
		self.ax.append(self.fig.add_subplot(2,2,4))
		self.ax[3].set_axis_bgcolor('k')
		self.im.append(plt.imshow(np.nanmax(ct_data[cs[0]-50:cs[0]-20,:,:], axis=0).T, cmap=cm.gray, aspect='auto'))
		self.cursor.append(plt.plot([cs[1], cs[1]], [self.ax[3].get_ylim()[0]+1, self.ax[3].get_ylim()[1]-1], color=[0, 1, 0] ))
		self.cursor2.append(plt.plot([self.ax[3].get_xlim()[0]+1, self.ax[3].get_xlim()[1]-1], [cs[2], cs[2]], color=[0, 1, 0] ))
		self.ax[3].set_xticks([])
		self.ax[3].set_yticks([])
		plt.gca().invert_yaxis()
		self.ax[3].axis([0,self.imsz[1],0,self.imsz[2]])

		self.elec_im.append(plt.imshow(self.elec_data[cs[0],:,:].T, cmap=cm.Set1, aspect='auto', alpha=1, vmin=0, vmax=10))

		plt.tight_layout()
		cid2 = self.fig.canvas.mpl_connect('scroll_event',self.on_scroll)
		cid3 = self.fig.canvas.mpl_connect('button_press_event',self.on_click)
		cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

		plt.show()
		self.fig.canvas.draw()

	def on_key(self, event):
		#print('You pressed', event.key)
		bb1=self.ax[0].get_position()
		bb2=self.ax[1].get_position()
		bb3=self.ax[2].get_position()
		bb4=self.ax[3].get_position()

		# Transform coordinates to figure coordinates
		fxy = self.fig.transFigure.inverted().transform((event.x, event.y))
		print event.x, event.y
		print self.current_slice

		if bb1.contains(fxy[0],fxy[1]):
			slice_num = 0
		if bb2.contains(fxy[0],fxy[1]):
			slice_num = 1
		if bb3.contains(fxy[0],fxy[1]):
			slice_num = 2
		if bb4.contains(fxy[0],fxy[1]):
			slice_num = 3
		this_ax = self.ax[slice_num]

		if event.key == 'n':
			self.ename = raw_input("Enter name of electrode to be added: ")

		if event.key == 'e':
			print("Adding electrode at (%3.3f, %3.3f, %3.3f)"%(self.current_slice[0], self.current_slice[1], self.current_slice[2]))
			self.add_electrode()

		# Maximum intensity projection in another dimension
		if slice_num == 3:
			ct_slice = dict()
			ct_slice['s'] = 0
			ct_slice['c'] = 1
			ct_slice['a'] = 2
			if event.key == 's' or event.key == 'c' or event.key == 'a':
				self.ct_slice = event.key

		# Scrolling through slices
		if event.key == 'pageup' or event.key == 'pagedown':
			if event.key == 'pagedown':
				sgn = -1
			else:
				sgn = 1
			if slice_num < 3:
				self.current_slice[slice_num] = self.current_slice[slice_num] + 1*sgn
			if slice_num == 3:
				self.current_slice[ct_slice[self.ct_slice]] = self.current_slice[ct_slice[self.ct_slice]] + 1*sgn

		# Panning left/right/up/down
		if event.key == 'up' or event.key == 'down' or event.key == 'left' or event.key == 'right':
			if event.key == 'up' or event.key == 'down':
				if event.key == 'up':
					sgn = -1
				else:
					sgn = 1
				ylims = this_ax.get_ylim()
				this_ax.set_ylim([ylims[0]+1*sgn, ylims[1]+1*sgn])
				
			elif event.key == 'left' or event.key == 'right':
				if event.key == 'right':
					sgn = -1
				else:
					sgn = 1
				xlims = this_ax.get_xlim()
				this_ax.set_xlim([xlims[0]+1*sgn, xlims[1]+1*sgn])

		# Draw the figure
		self.update_figure_data()
		plt.gcf().canvas.draw()

	def on_scroll(self, event):
		# Zoom on scroll
		stepsz = 10.
		bb1=self.ax[0].get_position()
		bb2=self.ax[1].get_position()
		bb3=self.ax[2].get_position()
		bb4=self.ax[3].get_position()

		# Transform coordinates to figure coordinates
		fxy = self.fig.transFigure.inverted().transform((event.x, event.y))
		
		# Find which subplot had the mouse focus
		if bb1.contains(fxy[0],fxy[1]):
			this_ax = self.ax[0]
			x = self.current_slice[1]
			y = self.current_slice[2]
		if bb2.contains(fxy[0],fxy[1]):
			this_ax = self.ax[1]
			x = self.current_slice[0]
			y = self.current_slice[2]
		if bb3.contains(fxy[0],fxy[1]):
			this_ax = self.ax[2]
			x = self.current_slice[0]
			y = self.current_slice[1]
		if bb4.contains(fxy[0],fxy[1]):
			this_ax = self.ax[3]
			x = self.current_slice[1]
			y = self.current_slice[2]

		xlims = this_ax.get_xlim()
		ylims = this_ax.get_ylim()

		xstep = event.step*stepsz
		ystep = event.step*stepsz

		if x > xlims[0] and x < xlims[1]:
			xratio = np.abs(xlims[1]-x)/np.abs(xlims[0]-x)
		else:
			xratio = 0.5
		if y > ylims[0] and y < ylims[1]:
			yratio = np.abs(ylims[1]-y)/np.abs(ylims[0]-y)
		else:
			yratio = 0.5

		xratio = 0.5
		yratio = 0.5

		this_ax.set_xlim(xlims[0]+xstep*2*(1-xratio), xlims[1] - xstep*2*(xratio))
		this_ax.set_ylim(ylims[0]+ystep*2*(1-yratio), ylims[1] - ystep*2*(yratio))

		plt.gcf().canvas.draw()

	def on_click(self, event):
		#print('You scrolled %d steps at x: %d, y: %d', event.step, event.x, event.y)
		# Get the bounding box for each of the subplots
		bb1=self.ax[0].get_position()
		bb2=self.ax[1].get_position()
		bb3=self.ax[2].get_position()
		bb4=self.ax[3].get_position()

		print event.xdata, event.ydata
		# Transform coordinates to figure coordinates
		fxy = self.fig.transFigure.inverted().transform((event.x, event.y))
		
		x = np.int(np.round(event.xdata))
		y = np.int(np.round(event.ydata))

		# If you clicked the first subplot
		if bb1.contains(fxy[0],fxy[1]):
			self.current_slice[1] = event.xdata
			self.current_slice[2] = event.ydata
			
		# If you clicked the second subplot
		elif bb2.contains(fxy[0],fxy[1]):
			self.current_slice[0] = event.xdata
			self.current_slice[2] = event.ydata

		# If you clicked the third subplot
		elif bb3.contains(fxy[0],fxy[1]):
			self.current_slice[0] = event.xdata
			self.current_slice[1] = event.ydata

		# If you clicked the third subplot
		elif bb4.contains(fxy[0],fxy[1]):
			if self.ct_slice == 's':
				self.current_slice[1] = event.xdata
				self.current_slice[2] = event.ydata
			elif self.ct_slice == 'c':
				self.current_slice[0] = event.xdata
				self.current_slice[2] = event.ydata
			elif self.ct_slice == 'a':
				self.current_slice[0] = event.xdata
				self.current_slice[1] = event.ydata
		
		self.update_figure_data()

		print("Current slice: %3.2f %3.2f %3.2f"%(self.current_slice[0], self.current_slice[1], self.current_slice[2]))
		plt.gcf().canvas.draw()

	def update_figure_data(self):
		cs = np.round(self.current_slice).astype(np.int) # Make integer for indexing the volume
		self.im[0].set_data(self.img_data[cs[0],:,:].T)
		self.im[1].set_data(self.img_data[:,cs[1],:].T)
		self.im[2].set_data(self.img_data[:,:,cs[2]].T)

		if self.ct_slice == 's':
			self.im[3].set_data(np.nanmax(self.ct_data[cs[0]-15:cs[0]+15,:,:], axis=0).T)
		elif self.ct_slice == 'c':
			self.im[3].set_data(np.nanmax(self.ct_data[:,cs[1]-15:cs[1]+15,:], axis=1).T)
		elif self.ct_slice == 'a':
			self.im[3].set_data(np.nanmax(self.ct_data[:,:,cs[2]-15:cs[2]+15], axis=2).T)


		self.ct_im[0].set_data(self.ct_data[cs[0],:,:].T)
		self.ct_im[1].set_data(self.ct_data[:,cs[1],:].T)
		self.ct_im[2].set_data(self.ct_data[:,:,cs[2]].T)

		self.elec_im[0].set_data(self.elec_data[cs[0],:,:].T)
		self.elec_im[1].set_data(self.elec_data[:,cs[1],:].T)
		self.elec_im[2].set_data(self.elec_data[:,:,cs[2]].T)
		if self.ct_slice == 's':
			self.elec_im[3].set_data(self.elec_data[cs[0],:,:].T)
			self.cursor[3][0].set_xdata ([self.current_slice[1], self.current_slice[1]]) 
			self.cursor2[3][0].set_ydata([self.current_slice[2], self.current_slice[2]])

		elif self.ct_slice == 'c':
			self.elec_im[3].set_data(self.elec_data[:,cs[1],:].T)
			self.cursor[3][0].set_xdata ([self.current_slice[0], self.current_slice[0]]) 
			self.cursor2[3][0].set_ydata([self.current_slice[2], self.current_slice[2]])
		elif self.ct_slice == 'a':
			self.elec_im[3].set_data(self.elec_data[:,:,cs[2]].T)
			self.cursor[3][0].set_xdata ([self.current_slice[0], self.current_slice[0]]) 
			self.cursor2[3][0].set_ydata([self.current_slice[1], self.current_slice[1]])

		self.cursor[0][0].set_xdata ([self.current_slice[1], self.current_slice[1]]) 
		self.cursor2[0][0].set_ydata([self.current_slice[2], self.current_slice[2]])
		self.cursor[1][0].set_xdata ([self.current_slice[0], self.current_slice[0]])
		self.cursor2[1][0].set_ydata([self.current_slice[2], self.current_slice[2]])
		self.cursor[2][0].set_xdata ([self.current_slice[0], self.current_slice[0]])
		self.cursor2[2][0].set_ydata([self.current_slice[1], self.current_slice[1]])

	def add_electrode(self):
		cs = np.round(self.current_slice).astype(np.int) # Make integer for indexing the volume
		#self.ax[0].plot(self.current_slice[1], self.current_slice[2], '.r', ms=12)
		#self.ax[1].plot(self.current_slice[0], self.current_slice[2], '.r', ms=12)
		#self.ax[2].plot(self.current_slice[0], self.current_slice[1], '.r', ms=12)

		# create a sphere centered around the current point
		radius = 3
		r2 = np.arange(-radius, radius+1)**2
		dist2 = r2[:,None,None]+r2[:,None]+r2
		bin_mat = np.array(dist2<=radius**2, dtype=np.float)
		bin_mat[bin_mat==0] = np.nan
		
		self.elec_data[cs[0]-radius:cs[0]+radius+1, cs[1]-radius:cs[1]+radius+1, cs[2]-radius:cs[2]+radius+1] = bin_mat

		self.elec_im[0].set_data(self.elec_data[cs[0],:,:].T)
		self.elec_im[0].set_data(self.elec_data[:,cs[1],:].T)
		self.elec_im[0].set_data(self.elec_data[:,:,cs[2]].T)


	#fig.canvas.setFocus()
electrode_picker()