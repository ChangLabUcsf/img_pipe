import xmltodict
import scipy.io
import nipy
import numpy as np
import os

#elecfile = '/Users/benspeidel/Documents/dura/data_store2/imaging/subjects/EC187/elecs/TDT_elecs_all_warped.mat'
subjects_dir = '/Users/benspeidel/Documents/dura/data_store2/imaging/subjects/'

subject_list = ['EC183','EC186']

#subject_list = ['EC91','EC96','EC100','EC107','EC108','EC125','EC136','EC142','EC143','EC155','EC159','EC166','EC175','EC179','EC92','EC99','EC113','EC131','EC133','EC135','EC137','EC139','EC148','EC152','EC153','EC156','EC158','EC160','EC162','EC115','EC170','EC84','EC87','EC150','EC82','EC178','EC112','EC111','EC119','EC132','EC144','EC154','KP09','KP01','KP20','KP27','EC130','KP06','EC129']
#EC118 taken out because of strange audio electrodes
#EC110 has a serious problem with the warping and needs to be looked at again. it has been removed for now

LUT_path = '/Users/benspeidel/Documents/GitHub/img_pipe/img_pipe/SupplementalFiles/FreeSurferLUT.xml'



# Define the affine transform to go from surface coordinates to volume coordinates (as CRS, which is
# the slice *number* as x,y,z in the 3D volume.
affine = np.array([[-2., 0., 0., 91],
                   [0., 2., 0., -109.],
                   [0., 0., 2., -91.],
                   [0., 0., 0., 1.]])

# with open(LUT_path) as fd:
#     dd = xmltodict.parse(fd.read())['atlas']['data']['label']
#     id_to_lbl = {}
#     for sel in dd:
#         id_to_lbl[int(sel['index'])] = sel['name']


with open(LUT_path) as fd:
    dd = xmltodict.parse(fd.read())['labelset']['label']
    id_to_lbl = {}
    for sel in dd:
        id_to_lbl[int(sel['@id'])] = sel['@fullname']

for subj in range(0,len(subject_list)):
    clinicalwarpexists=True
    atlas_path=os.path.join(subjects_dir,subject_list[subj],'mri/aparc+aseg.mgz')
    atlas = nipy.load_image(atlas_path)

    if os.path.isfile(os.path.join(subjects_dir,subject_list[subj],'elecs/clinical_elecs_all.mat')):
        elecfile = os.path.join(subjects_dir,subject_list[subj],'elecs/clinical_elecs_all.mat')
        elecmontage = scipy.io.loadmat(elecfile)['anatomy'][:,0:3]
    elif os.path.isfile(os.path.join(subjects_dir, subject_list[subj], 'elecs/clinical_TDT_elecs_all.mat')):
        elecfile = os.path.join(subjects_dir, subject_list[subj], 'elecs/clinical_TDT_elecs_all.mat')
        elecmontage = scipy.io.loadmat(elecfile)['eleclabels']
    else:
        clinicalwarpexists = False
        print('%s did not have necessary files to generate labels' %(subject_list[subj]))

    if clinicalwarpexists:

        short_label = []
        long_label = []
        grid_or_depth = []

        elecmatrix = scipy.io.loadmat(elecfile)['elecmatrix']

        for r in elecmontage:
            short_label.append(r[0][0])  # This is the shortened electrode montage label
            long_label.append(r[1][0])  # This is the long form electrode montage label
            grid_or_depth.append(r[2][0])  # This is the label for grid, depth, or strip




        #now is when we need to split up depths and grids and label them separately. Do nearest nonzero for grids and current location for depths


        isnotdepth = []

        # These are the indices that won't be used for labeling
        # dont_label = ['EOG','ECG','ROC','LOC','EEG','EKG','NaN','EMG','scalpEEG']
        indices = [i for i, x in enumerate(long_label) if (
                    'EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x == np.nan or 'scalpEEG' in x)]
        indices.extend([i for i, x in enumerate(short_label) if (
                    'EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x == np.nan or 'scalpEEG' in x)])
        indices.extend([i for i, x in enumerate(grid_or_depth) if (
                    'EOG' in x or 'ECG' in x or 'ROC' in x or 'LOC' in x or 'EEG' in x or 'EKG' in x or 'NaN' in x or 'EMG' in x or x == np.nan or 'scalpEEG' in x)])
        indices.extend(np.where(np.isnan(elecmatrix) == True)[0])
        indices = list(set(indices))
        indices_to_use = list(set(range(len(long_label))) - set(indices))

        # Initialize the cell array that we'll store electrode labels in later
        elec_labels_orig = np.empty((len(long_label), 4), dtype=np.object)
        elec_labels_orig[:, 0] = short_label
        elec_labels_orig[:, 1] = long_label
        elec_labels_orig[:, 2] = grid_or_depth
        elec_labels = np.empty((len(indices_to_use), 4), dtype=np.object)
        elecmatrix_orig = elecmatrix
        elecmatrix = elecmatrix[indices_to_use, :]

        short_label_orig, long_label_orig, grid_or_depth_orig = short_label, long_label, grid_or_depth
        short_label = [i for j, i in enumerate(short_label) if j not in indices]
        long_label = [i for j, i in enumerate(long_label) if j not in indices]
        grid_or_depth = [i for j, i in enumerate(grid_or_depth) if j not in indices]
        elec_labels[:, 0] = short_label
        elec_labels[:, 1] = long_label
        elec_labels[:, 2] = grid_or_depth


        isnotdepth = np.array([r!='depth' for r in grid_or_depth])


        #electrodes should now be identified as either surface or depth


        intercept = np.ones(len(elecmatrix))
        elecs_ones = np.column_stack((elecmatrix, intercept))

        # Find voxel CRS
        VoxCRS = np.dot(np.linalg.inv(affine), elecs_ones.transpose()).transpose().astype(int)

        nchans = VoxCRS.shape[0]
        anatomy = np.empty((nchans, 1), dtype=np.object)

        atlas_nonzeros = np.transpose(np.asarray(np.where(atlas._data>0), dtype='f4'))
        d = np.zeros((1,atlas_nonzeros.shape[0]))

        for elec in np.arange(nchans):
            if isnotdepth[elec]:
                d = np.sqrt((VoxCRS[elec, 0] - atlas_nonzeros[:, 0])**2 + (VoxCRS[elec, 1] - atlas_nonzeros[:, 1])**2 + (VoxCRS[elec, 2] - atlas_nonzeros[:, 2])**2)
                vox_ind = np.argmin(d)
                nearest_vox = atlas_nonzeros[vox_ind, :].astype('int')
                anatomy[elec] = id_to_lbl[atlas._data[nearest_vox[0], nearest_vox[1], nearest_vox[2]]]
            else:
                if atlas._data[VoxCRS[elec,0],VoxCRS[elec, 1], VoxCRS[elec, 2]] == 0 or abs(atlas._data[VoxCRS[elec, 0], VoxCRS[elec, 1], VoxCRS[elec, 2]]) > 10000:
                    anatomy[elec] = 'No_Label'
                else:
                    anatomy[elec] = id_to_lbl[atlas._data[VoxCRS[elec,0],VoxCRS[elec,1],VoxCRS[elec,2]]]

        AALanatomy = np.empty((nchans,4), dtype=np.object)
        if nchans == elecmontage.shape[0] and nchans==anatomy.shape[0]:
            AALanatomy[:, 0:3] = elecmontage[:, 0:3]
            AALanatomy[:, 3] = anatomy[:, 0]

            scipy.io.savemat(os.path.join(subjects_dir,subject_list[subj],'elecs/aparc+aseg_clinical_elecs_all.mat'), {'elecmatrix' : elecmatrix, 'anatomy': AALanatomy, 'eleclabels' : elecmontage})
        else:
            print('%s was skipped because of inconsistencies in elecs files' % (subject_list[subj]))

print('hello')


