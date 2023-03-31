from nd2reader import ND2Reader
import pandas as pd 
import numpy as np
import scipy.ndimage

class ND2ImageLoading:
    def __init__(self):
        self.load_data()

    def load_data(self,data_filename = 'data/raw_nd2_image.nd2',
                  annotation_filename = 'data/raw_nd2_image.csv'):

        # load data
        print('** Loading image file:',data_filename)
        images =ND2Reader(data_filename)
        print('** Loading image annotation file:',annotation_filename)
        annotation = pd.read_csv(annotation_filename)

        # get parameters 
        print('** Loading metadata')
        pixel_microns = images.metadata['pixel_microns']
        z_stack_location = images.metadata['z_coordinates']
        z_depth = [x-z_stack_location[0] for x in z_stack_location]
        thickness_microns = (z_stack_location[-1]-z_stack_location[0])/(len(z_stack_location)-1)
        spacing_resolution = pixel_microns
        im =  np.array(images)
        
        self.images = images
        self.metadata = images.metadata
        self.im = im
        self.thickness_microns = thickness_microns
        self.spacing_resolution = spacing_resolution
        self.pixel_microns = pixel_microns
        

        # cropping
        print('** cropping image')
        cropping_vector = annotation[annotation['Type']=='Rectangle'][['X','Y','Width','Height']].to_numpy()[0]
        _crop_xlim = [cropping_vector[0],cropping_vector[0]+cropping_vector[2]]
        _crop_ylim = [cropping_vector[1],cropping_vector[1]+cropping_vector[3]]
        axon_vector = annotation[annotation['Type']=='Axon'][['X','Y','Width','Height']].to_numpy()[0]
        _axon_x = np.array([axon_vector[0],axon_vector[0]+axon_vector[2]])
        _axon_y = np.array([axon_vector[1],axon_vector[1]+axon_vector[3]])
        spine_head_vector = annotation[annotation['Type']=='Spine'][['X','Y','Width','Height']].to_numpy()[0]
        _spine_x = np.array([spine_head_vector[0],spine_head_vector[0]+spine_head_vector[2]])
        _spine_y = np.array([spine_head_vector[1],spine_head_vector[1]+spine_head_vector[3]])
        im_crop = im[:,_crop_ylim[0]:_crop_ylim[1],_crop_xlim[0]:_crop_xlim[1]].copy()
        self.im_crop = im_crop

        # rotate the cropped image
        print('** rotating image')
        im_zstack = im.max(axis=0)
        _im_annot = np.zeros(im_zstack.shape)
        _im_annot[_axon_y[0],_axon_x[0]] = 1 
        _im_annot[_axon_y[1],_axon_x[1]] = 1
        _im_annot[_spine_y[0],_spine_x[0]] = -1
        _im_annot[_spine_y[1],_spine_x[1]] = -1

        _im_crop_annot = _im_annot[_crop_ylim[0]:_crop_ylim[1],_crop_xlim[0]:_crop_xlim[1]]
        _rotation_degree = np.arctan(axon_vector[3]/axon_vector[2])/np.pi*180
        _rotation_degree_set = [x+_rotation_degree for x in [0,90,180,270]]

        _rotated_annot_set = []
        _vertical_diff_set = []
        _horizontal_diff_set = []
        for _test_degree in _rotation_degree_set:
            _im_rotate_annot = scipy.ndimage.rotate(_im_crop_annot,_test_degree,reshape = False,cval=0)
            _rotated_axon =  np.where(_im_rotate_annot>_im_rotate_annot.max()*0.5)
            _rotated_spine =  np.where(_im_rotate_annot<_im_rotate_annot.min()*0.5)
            _vertical_diff = _rotated_axon[0].mean() - _rotated_spine[0].mean() 
            _horizontal_diff = _rotated_axon[1].mean() - _rotated_spine[1].mean() 
            _vertical_diff_set.append(_vertical_diff)
            _horizontal_diff_set.append(_horizontal_diff)
            _rotated_annot_set.append([_rotated_axon,_rotated_spine])
        _choosen_index = np.array(range(4))[(_vertical_diff_set==np.array(_vertical_diff_set).min())][0]
        _rotation_degree = _rotation_degree_set[_choosen_index]
        _rotated_annot = _rotated_annot_set[_choosen_index]
        _rotated_axon = _rotated_annot[0]
        _rotated_spine = _rotated_annot[1]
        rotation_z_angle = _rotation_degree
        self.rotation_z_angle = rotation_z_angle
