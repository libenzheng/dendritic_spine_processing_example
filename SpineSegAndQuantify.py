import numpy as np
import scipy.ndimage
import skimage.morphology 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
import skimage.measure
import matplotlib as mpl
from plotly import figure_factory as FF
from plotly.offline import download_plotlyjs, init_notebook_mode
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True) 

class SpineSegmentationAndQuantification:
    
    def __init__(self,preprocess):
        self.preprocess = preprocess
        print('** Processing spine segmentation')
        self.segmentation()
        print('** Calculating metrics')
        self.quantification()
        
    def segmentation(self):
        
        # rotating of x-y plane on z-axis 

        im_rotate = self.preprocess.im_thres_closing.copy()
        spacing_resolution = self.preprocess.data.spacing_resolution
        im_z_rotated = scipy.ndimage.rotate(im_rotate.astype(float),
                                               self.preprocess.data.rotation_z_angle,
                                               axes = (1,2),
                                               reshape = False)
        im_z_rotated = im_z_rotated.astype(float)
        im_z_rotated -= im_z_rotated.min()
        im_z_rotated /= im_z_rotated.max()
        im_z_rotated = im_z_rotated>(im_z_rotated.mean()+im_z_rotated.std())

        
        _im_compute = im_z_rotated.copy()
        self.im_compute = _im_compute
        
        _z,_y,_x = np.mgrid[0:_im_compute.shape[0],0:_im_compute.shape[1],0:_im_compute.shape[2]]
        _df_matrix = pd.DataFrame([])
        _df_matrix['x'] = _x.flatten()
        _df_matrix['y'] = _y.flatten()
        _df_matrix['z'] = _z.flatten()
        _df_matrix['value'] = _im_compute.flatten()
        
        _df = _df_matrix[_df_matrix['value']!=0].copy()
        _df['yPosition']=_df['y'].apply(lambda x:x*spacing_resolution)
        
        # gaussian mixture model based segmentation

        def gauss_function(x, amp, x0, sigma):
            return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


        _y_samples = _df['y'].to_numpy()
        gmm = GaussianMixture(n_components=2, covariance_type="full", tol=0.001)
        gmm = gmm.fit(X=np.expand_dims(_y_samples, 1))


        # obtain fitted parameters 
        _mean_set = gmm.means_.ravel()
        _cov_set = gmm.covariances_.ravel()
        _weight_set = gmm.weights_.ravel()
        _std_set = [np.sqrt(x) for x in _cov_set]

        if (_mean_set[0] - _mean_set[1])<0:
            _class_name_set = ['Shaft','Spine']
        else:
            _class_name_set = ['Spine','Shaft']

        _prob_set = [gmm.predict_proba(np.array([[_y]])) for _y in _df['y'].unique()]
        _prob_mapping_A = dict(zip(_df['y'].unique(),[_prob[0][0] for _prob in _prob_set]))
        _prob_mapping_B = dict(zip(_df['y'].unique(),[_prob[0][1] for _prob in _prob_set]))

        _df[_class_name_set[0]+'_prob'] = _df['y'].map(_prob_mapping_A)
        _df[_class_name_set[1]+'_prob'] = _df['y'].map(_prob_mapping_B)


        _df_prob = _df[['y','yPosition','Shaft_prob','Spine_prob']].drop_duplicates()
        _df_prob = _df_prob.sort_values('y').reset_index(drop=True)

        _f_map_A = (lambda x: gauss_function(x,_weight_set[0],_mean_set[0],_std_set[0]))
        _f_map_B = (lambda x: gauss_function(x,_weight_set[1],_mean_set[1],_std_set[1]))
        _df_prob[_class_name_set[0]+'_dist'] = _df_prob['y'].apply(_f_map_A)
        _df_prob[_class_name_set[1]+'_dist'] = _df_prob['y'].apply(_f_map_B)

        _mean = _mean_set.min()
        _std = np.array(_std_set)[(_mean_set==_mean)][0]
        _region_adjust = 2 ## for full width at half maximum using np.sqrt(2*np.log(2))
        # _region_adjust =  np.sqrt(2*np.log(2))
        _shaft_thres = _mean+_std*_region_adjust
        _shaft_thres_start  = _mean-_std*_region_adjust
        if _shaft_thres>_mean_set.max():
            _shaft_thres = _mean_set.max()

        _shaft_set = [_shaft_thres_start,_shaft_thres]
        
        _shaft_y_end = int(np.floor(_shaft_thres))
        _im_spine = _im_compute[:,_shaft_y_end:,:]
        _im_shaft = _im_compute[:,:_shaft_y_end,:]

        # selected longest contour as the mask for spine 
        _region = _im_spine.max(axis=0)
        _region[0,:]= 0 
        _region[_region.shape[0]-1,:]= 0 
        _region[:,0]= 0 
        _region[:,_region.shape[1]-1]= 0 
        _contours = skimage.measure.find_contours(_region, 0.8,fully_connected = 'high')
        _contour_length_set =np.array([ len(x) for x in _contours])
        _i_contour = np.arange(len(_contour_length_set))[_contour_length_set == _contour_length_set.max()][0]
        _contour = _contours[_i_contour]
        _im_thres_mask = np.zeros_like(_region, dtype='bool')
        _im_thres_mask[np.round(_contour[:, 0]).astype('int'), np.round(_contour[:, 1]).astype('int')] = 1
        _im_thres_mask = scipy.ndimage.binary_fill_holes(_im_thres_mask)

        # mask the original spine image 
        _im_spine_masked = np.zeros_like(_im_spine, dtype='bool')
        for _i_im in range(_im_spine.shape[0]):
            _im_z_slice = _im_spine[_i_im,:,:]
            _im_masked_z_slice = np.logical_and(_im_thres_mask,_im_z_slice)
            _im_spine_masked[_i_im,:,:] = _im_masked_z_slice

        # selected longest contour as the mask for shaft
        _region = _im_shaft.max(axis=0)
        _region[0,:]= 0 
        _region[_region.shape[0]-1,:]= 0 
        _region[:,0]= 0 
        _region[:,_region.shape[1]-1]= 0 
        _contours = skimage.measure.find_contours(_region, 0.8,fully_connected = 'high')
        _contour_length_set =np.array([ len(x) for x in _contours])
        _i_contour = np.arange(len(_contour_length_set))[_contour_length_set == _contour_length_set.max()][0]
        _contour = _contours[_i_contour]
        _im_thres_mask = np.zeros_like(_region, dtype='bool')
        _im_thres_mask[np.round(_contour[:, 0]).astype('int'), np.round(_contour[:, 1]).astype('int')] = 1
        _im_thres_mask = scipy.ndimage.binary_fill_holes(_im_thres_mask)

        # mask the original spine image 
        _im_shaft_masked = np.zeros_like(_im_shaft, dtype='bool')
        for _i_im in range(_im_shaft_masked.shape[0]):
            _im_z_slice = _im_shaft[_i_im,:,:]
            _im_masked_z_slice = np.logical_and(_im_thres_mask,_im_z_slice)
            _im_shaft_masked[_i_im,:,:] = _im_masked_z_slice


        # longest contour masked image and dataframe
        _im_compute_masked = np.zeros_like(_im_compute, dtype='bool')
        _im_compute_masked[:,:_shaft_y_end,:] = _im_shaft_masked
        _im_compute_masked[:,_shaft_y_end:,:] = _im_spine_masked

        _z,_y,_x = np.mgrid[0:_im_compute_masked.shape[0],0:_im_compute_masked.shape[1],0:_im_compute_masked.shape[2]]
        _df_matrix_masked = pd.DataFrame([])
        _df_matrix_masked['x'] = _x.flatten()
        _df_matrix_masked['y'] = _y.flatten()
        _df_matrix_masked['z'] = _z.flatten()
        _df_matrix_masked['value'] = _im_compute_masked.flatten()


        # rotate the spine to find the optimal angle for the segmentation
        _rotation_angle_set = np.arange(-90,90,10)
        _y_projection_length_set = []
        for rotation_z_angle in _rotation_angle_set:
            _im_z_rotated = scipy.ndimage.rotate(_im_spine_masked.astype(float),
                                                rotation_z_angle,
                                                axes = (1,2),reshape = True)

            _im_z_rotated -= _im_z_rotated.min()
            _im_z_rotated /= _im_z_rotated.max()
            _im_z_rotated = _im_z_rotated>(_im_z_rotated.mean()+_im_z_rotated.std())

            _im_z_rotated_stack = _im_z_rotated.max(axis=0)
            _y_projection = _im_z_rotated_stack.sum(axis =1 )
            _y_projection_length = (_y_projection!=0).sum()
            _y_projection_length_set.append(_y_projection_length)

        _x = _rotation_angle_set
        _y = _y_projection_length_set
        _z = np.polyfit(_x, _y, 7)
        _poly = np.poly1d(_z)
        _x_fine = np.linspace(_x[0], _x[-1], 200)
        # compute optimal angle in degrees -> as base angle of the spine 
        _optimal_angle = _x_fine[_poly(_x_fine)==_poly(_x_fine).max()][0]
        self.optimal_angle = _optimal_angle
        _sin_optimal_angle = np.sin(_optimal_angle/180*np.pi)


        _mean = _mean_set[0]
        _std = _std_set[0]
        # _region_adjust = 2 ## for full width at half maximum using np.sqrt(2*np.log(2))
        _region_adjust = np.sqrt(2*np.log(2))
        _head_thres = _mean-_std*_region_adjust
        _head_thres_end = _mean+_std*_region_adjust
        _head_thres_set = [_head_thres,_head_thres_end]

        # summary 
        _df_seg = _df_matrix_masked.copy()
        _df_seg = _df_seg[_df_seg['value']!=0].reset_index(drop = True)

        _anchor_region = _im_spine_masked[:,:3,:]
        _z,_y,_x = np.mgrid[0:_anchor_region.shape[0],0:_anchor_region.shape[1],0:_anchor_region.shape[2]]
        _df_anchor = pd.DataFrame([])
        _df_anchor['x'] = _x.flatten()
        _df_anchor['y'] = _y.flatten()
        _df_anchor['z'] = _z.flatten()
        _df_anchor['value'] = _anchor_region.flatten()
        _df_anchor = _df_anchor[_df_anchor['value']!=0]
        _x_0 = _df_anchor.x.mean()
        _y_0 = _shaft_thres
        _x_1 = _x_0+100
        _y_1 = (_x_1- _x_0)*np.tan(-(_optimal_angle)/180*np.pi+0.5*np.pi)



        # find anchor points and line for the spine/neck segmentation and compute distance from that junctional plane 
        _z,_y,_x = np.mgrid[0:_im_spine_masked.shape[0],0:_im_spine_masked.shape[1],0:_im_spine_masked.shape[2]]
        _df_spine = pd.DataFrame([])
        _df_spine['x'] = _x.flatten()
        _df_spine['y'] = _y.flatten()
        _df_spine['z'] = _z.flatten()
        _df_spine['value'] = _im_spine_masked.flatten()
        _df_spine = _df_spine[_df_spine['value']!=0]
        _df_anchor = _df_spine[_df_spine['y']<3]

        _x_0 = _df_anchor.x.max()
        if _df_spine.x.mean()>_df_anchor.x.mean():
            _x_0 = _df_anchor.x.min() 

        _y_0 = _shaft_thres
        _x_1 = _x_0 + 20
        _y_1 = (_x_1- _x_0)*np.tan(np.deg2rad(_optimal_angle))+_shaft_thres


        _p_0 = np.array([_x_0,_y_0]).astype(float)
        _p_1 = np.array([_x_1,_y_1]).astype(float)
        _p_measured = np.array([5,25]).astype(float)

        def get_point2line_dist(row):
            _x = row['x']
            _y = row['y']
            _p_0 = np.array([_x_0,_y_0])
            _p_1 = np.array([_x_1,_y_1])
            _p_measured = np.array([_x,_y])
            _dist = np.linalg.norm(np.cross(_p_1-_p_0, _p_0-_p_measured))/np.linalg.norm(_p_1-_p_0)
            return _dist
        _df_seg['SpineDist'] = _df_seg.apply(get_point2line_dist,axis = 1 )

        _df_seg['IsShaft'] = _df_seg['y'].apply(lambda x: x<_shaft_thres)
        _df_seg['IsSpine'] = _df_seg['y'].apply(lambda x: x>_shaft_thres)
        _df_seg['IsSpineHead'] = np.logical_and(_df_seg['IsSpine'] ,_df_seg['SpineDist']>_head_thres)
        _df_seg['IsSpineNeck'] = np.logical_and(_df_seg['IsSpine'] ,_df_seg['SpineDist']<_head_thres)
        self.df_seg =  _df_seg
        
    def quantification(self):
        # measurements 

        _label_set = ['Shaft','Spine','Neck','Head']
        _seg_set =  ['IsShaft','IsSpine','IsSpineNeck','IsSpineHead']
        _length_term_set = ['x','SpineDist','SpineDist','SpineDist']
        _estimation_alpha = 0.99


        _column_names = [
                        'Object',
                        'Volume(um^3)',
                        'SurfaceArea(um^2)',
                        'Length(um)',
                        'SectionArea(um^2)',
                        'SectionDiameter(um)',
                        'SpineBaseAngle(degree)',
                        'ConvexHullRatio',
                        ] 



        _spine_base_angle = self.optimal_angle

        _df_measurement_result = []
        
        _df_seg = self.df_seg.copy()
        _im_compute = self.im_compute.copy()
        spacing_resolution = self.preprocess.data.spacing_resolution

        for _label,_seg,_length_term in zip(_label_set,_seg_set,_length_term_set):

            print('Computing metrics in ',_label)
            _im_seg = np.zeros_like(_im_compute, dtype='bool')
            _df_seg_part = _df_seg[_df_seg[_seg]]
            _im_seg[_df_seg_part.z,_df_seg_part.y,_df_seg_part.x] = True

            if _im_seg.max()!=0:


                # marching cube mesh with verts, faces, norm, val
                _surface_mesh = skimage.measure.marching_cubes_lewiner(_im_seg)  # verts, faces, norm, val
                _verts,_faces,_norm,_val = _surface_mesh


                # compute volume in um^3
                _volume = _im_seg.sum()*np.power(spacing_resolution,3) 

                # compute surface area in micrometer^2
                _surface_area = skimage.measure.mesh_surface_area(_verts, _faces)*np.power(spacing_resolution,2) 


                # convex hull volume and ratio
                _im_convex = skimage.morphology.convex_hull_image(_im_seg)
                _convex_hull_volume = _im_convex.sum()
                _convex_hull_ratio = (_convex_hull_volume - _im_seg.sum())/_im_seg.sum()

                # compute length, section area, and section diameter

                _length_start = _df_seg_part[_length_term].quantile(1-_estimation_alpha)
                _length_end = _df_seg_part[_length_term].quantile(_estimation_alpha)
                _length =  (_length_end - _length_start)*spacing_resolution

                _df_seg_part_width = _df_seg_part.copy()
                _df_seg_part_width['Metric'] = _df_seg_part_width[_length_term].apply(round)
                _area_vector = _df_seg_part_width.pivot_table(index = 'Metric', values ='value',aggfunc=len)
                _min_area = _area_vector.value.quantile(0.05)
                _max_area = _area_vector.value.quantile(0.95)
                _area_vector_trimmed = _area_vector[(_area_vector.value>_min_area)&(_area_vector.value<_max_area)]
                _mean_section_area = _area_vector_trimmed.to_numpy().mean()*np.power(spacing_resolution,2) 
                _width_diameter_mean = 2*np.sqrt(_mean_section_area/np.pi)


                _result_vector = [
                            _label,
                            _volume,
                            _surface_area,
                            _length,
                            _mean_section_area,
                            _width_diameter_mean,
                            _spine_base_angle,
                            _convex_hull_ratio,
                        ]

                _df_measurement_result.append(pd.DataFrame([_result_vector],columns=_column_names))

            else:
                # no volume detected 
                _result_vector = list(np.zeros(len(_column_names)))
                _result_vector[0] = _label
                _df_measurement_result.append(pd.DataFrame([_result_vector],columns=_column_names))

        _df_measurement_result = pd.concat(_df_measurement_result)
        
        self.df_measurement_result = _df_measurement_result
        
    def plotting_3D_spine_sample(self):
        # plot whole image without segmentation
        
        _im_compute = self.im_compute.copy()
        _df_seg = self.df_seg.copy()
        spacing_resolution = self.preprocess.data.spacing_resolution
        _label = ''
        
        # Universal configs 

        # parameters
        _lw = 2 # axis line width
        _font_size = 18
        _font_size_minor = 12
        _cm_data = 'jet'
        _cm_dipole = 'inferno'
        _cbar_label = 'Not defined'

        _fig_width= 600
        _fig_height= 400
        _plt_width= 5
        _plt_height= 4
        _dpi = 300

        _d_voxel = 50j

        # axis 
        axis_config = dict( visible = True,
                            showbackground= False,
                            backgroundcolor="white",
                            showgrid = True,
                            gridwidth = _lw,
                            gridcolor="lightgray",
                            color = 'black',
                            showline = True,
                            dtick = 1,
                            ticks='inside',
                            ticklen = 5,
                            tickwidth = _lw,
                            tickcolor = 'black',
                            tickfont_size = _font_size-5,
                            linewidth = _lw,
                            linecolor = 'black',
                            showspikes = False,
                            mirror = True,
                          )


        # colorbar
        colorbar_config = dict(
                                title = _cbar_label,
                                title_font_color = 'black',
                                tickfont_color = 'black',
                                title_font_size = _font_size,
                                tickfont_size = _font_size,
                                outlinewidth = 0,
                                x = 0.8,
                                yanchor = 'top',
                                y = 0.75
                              )
        colorbar_config['thickness'] = 15
        colorbar_config['len'] = 0.4 

        # camera and scenes

        camera_config = dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=-1.5, z=3)
                            )

        scene_config = dict(camera=camera_config,
                            aspectratio=dict(x=1,y=1,z=1),
                            xaxis = axis_config,
                            yaxis = axis_config,
                            zaxis = axis_config,
                            xaxis_title='x (um)',
                            yaxis_title='y (um)',
                            zaxis_title='z (um)',
                            xaxis_title_font_size = _font_size,
                            yaxis_title_font_size = _font_size,
                            zaxis_title_font_size = _font_size,
                           )
        
         

        _color = 'lightslategrey'
        _fig_data = []
        _im_seg = np.zeros_like(_im_compute, dtype='bool')
        _df_seg_part = _df_seg # all image
        _im_seg[_df_seg_part.z,_df_seg_part.y,_df_seg_part.x] = True

        im_plot = _im_seg

        colormap=[mpl.colors.to_rgb(_color)]

        verts, faces, norm, val = skimage.measure.marching_cubes_lewiner(im_plot) 
        _aspect = np.array(im_plot.shape)
        _aspect_ratio = [x/_aspect.max() for x in _aspect]
        _aspect_ratio = dict(z=_aspect_ratio[0],y=_aspect_ratio[1],x=_aspect_ratio[2])

        _dtick = 1
        x,y,z = zip(*verts) 

        x = np.array(x)*spacing_resolution
        y = np.array(y)*spacing_resolution
        z = np.array(z)*spacing_resolution

        _surface = FF.create_trisurf(x=z,
                            y=y, 
                            z=x, 
                            colormap=colormap,
                            simplices=faces,
                            showbackground=False, 
                            plot_edges=False,
                            show_colorbar   = False,)


        _surface_data = _surface.data[0]
        _surface_data.name = _label
        #     _surface_data.opacity = 1
        _fig_data.append(_surface_data)

        fig = go.Figure( data = _fig_data)
        fig.update_layout(scene = scene_config,
                          width = _fig_width,
                          height = _fig_height)
        fig.update_layout(scene_aspectratio = _aspect_ratio,
                        scene_xaxis_dtick = _dtick,
                         scene_yaxis_dtick = _dtick,
                         scene_zaxis_dtick = _dtick,)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        fig.update_layout(scene = dict(  xaxis = dict(range=[0,im_plot.shape[2]*spacing_resolution],),
                                         yaxis = dict(range=[0,im_plot.shape[1]*spacing_resolution],),
                                         zaxis = dict(range=[0,im_plot.shape[0]*spacing_resolution],),),)

        fig.show()


       

    def plotting_3D_spine_sample_two_seg(self):
        # plot whole image without segmentation
        
        _im_compute = self.im_compute.copy()
        _df_seg = self.df_seg.copy()
        spacing_resolution = self.preprocess.data.spacing_resolution

        
        
        # Universal configs 

        # parameters
        _lw = 2 # axis line width
        _font_size = 18
        _font_size_minor = 12
        _cm_data = 'jet'
        _cm_dipole = 'inferno'
        _cbar_label = 'Not defined'

        _fig_width= 600
        _fig_height= 400
        _plt_width= 5
        _plt_height= 4
        _dpi = 300

        _d_voxel = 50j

        # axis 
        axis_config = dict( visible = True,
                            showbackground= False,
                            backgroundcolor="white",
                            showgrid = True,
                            gridwidth = _lw,
                            gridcolor="lightgray",
                            color = 'black',
                            showline = True,
                            dtick = 1,
                            ticks='inside',
                            ticklen = 5,
                            tickwidth = _lw,
                            tickcolor = 'black',
                            tickfont_size = _font_size-5,
                            linewidth = _lw,
                            linecolor = 'black',
                            showspikes = False,
                            mirror = True,
                          )


        # colorbar
        colorbar_config = dict(
                                title = _cbar_label,
                                title_font_color = 'black',
                                tickfont_color = 'black',
                                title_font_size = _font_size,
                                tickfont_size = _font_size,
                                outlinewidth = 0,
                                x = 0.8,
                                yanchor = 'top',
                                y = 0.75
                              )
        colorbar_config['thickness'] = 15
        colorbar_config['len'] = 0.4 

        # camera and scenes

        camera_config = dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=-1.5, z=3)
                            )

        scene_config = dict(camera=camera_config,
                            aspectratio=dict(x=1,y=1,z=1),
                            xaxis = axis_config,
                            yaxis = axis_config,
                            zaxis = axis_config,
                            xaxis_title='x (um)',
                            yaxis_title='y (um)',
                            zaxis_title='z (um)',
                            xaxis_title_font_size = _font_size,
                            yaxis_title_font_size = _font_size,
                            zaxis_title_font_size = _font_size,
                           )

        _label_set = ['Shaft','Spine']
        _two_seg_set =  ['IsShaft','IsSpine']
        _color_set = ['dodgerblue','crimson']

        _fig_data = []

        for _label,_seg,_color in zip(_label_set,_two_seg_set,_color_set):


            _im_seg = np.zeros_like(_im_compute, dtype='bool')
            _df_seg_part = _df_seg[_df_seg[_seg]]
            _im_seg[_df_seg_part.z,_df_seg_part.y,_df_seg_part.x] = True

            im_plot = _im_seg

            colormap=[mpl.colors.to_rgb(_color)]

            verts, faces, norm, val = skimage.measure.marching_cubes_lewiner(im_plot) 
            _aspect = np.array(im_plot.shape)
            _aspect_ratio = [x/_aspect.max() for x in _aspect]
            _aspect_ratio = dict(z=_aspect_ratio[0],y=_aspect_ratio[1],x=_aspect_ratio[2])

            _dtick = 1
            x,y,z = zip(*verts) 

            x = np.array(x)*spacing_resolution
            y = np.array(y)*spacing_resolution
            z = np.array(z)*spacing_resolution


            _surface = FF.create_trisurf(x=z,
                                y=y, 
                                z=x, 
                                colormap=colormap,
                                simplices=faces,
                                showbackground=False, 
                                plot_edges=False,
                                show_colorbar   = False,)


            _surface_data = _surface.data[0]
            _surface_data.name = _label
        #     _surface_data.opacity = 1
            _fig_data.append(_surface_data)

        fig = go.Figure( data = _fig_data)
        fig.update_layout(scene = scene_config,
                          width = _fig_width,
                          height = _fig_height)
        fig.update_layout(scene_aspectratio = _aspect_ratio,
                        scene_xaxis_dtick = _dtick,
                         scene_yaxis_dtick = _dtick,
                         scene_zaxis_dtick = _dtick,)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        fig.update_layout(scene = dict(  xaxis = dict(range=[0,im_plot.shape[2]*spacing_resolution],),
                                         yaxis = dict(range=[0,im_plot.shape[1]*spacing_resolution],),
                                         zaxis = dict(range=[0,im_plot.shape[0]*spacing_resolution],),),)




        fig.show()

    def plotting_3D_spine_sample_three_seg(self):
        # plot whole image without segmentation
        
        _im_compute = self.im_compute.copy()
        _df_seg = self.df_seg.copy()
        spacing_resolution = self.preprocess.data.spacing_resolution

        
        
        # Universal configs 

        # parameters
        _lw = 2 # axis line width
        _font_size = 18
        _font_size_minor = 12
        _cm_data = 'jet'
        _cm_dipole = 'inferno'
        _cbar_label = 'Not defined'

        _fig_width= 600
        _fig_height= 400
        _plt_width= 5
        _plt_height= 4
        _dpi = 300

        _d_voxel = 50j

        # axis 
        axis_config = dict( visible = True,
                            showbackground= False,
                            backgroundcolor="white",
                            showgrid = True,
                            gridwidth = _lw,
                            gridcolor="lightgray",
                            color = 'black',
                            showline = True,
                            dtick = 1,
                            ticks='inside',
                            ticklen = 5,
                            tickwidth = _lw,
                            tickcolor = 'black',
                            tickfont_size = _font_size-5,
                            linewidth = _lw,
                            linecolor = 'black',
                            showspikes = False,
                            mirror = True,
                          )


        # colorbar
        colorbar_config = dict(
                                title = _cbar_label,
                                title_font_color = 'black',
                                tickfont_color = 'black',
                                title_font_size = _font_size,
                                tickfont_size = _font_size,
                                outlinewidth = 0,
                                x = 0.8,
                                yanchor = 'top',
                                y = 0.75
                              )
        colorbar_config['thickness'] = 15
        colorbar_config['len'] = 0.4 

        # camera and scenes

        camera_config = dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=-1.5, z=3)
                            )

        scene_config = dict(camera=camera_config,
                            aspectratio=dict(x=1,y=1,z=1),
                            xaxis = axis_config,
                            yaxis = axis_config,
                            zaxis = axis_config,
                            xaxis_title='x (um)',
                            yaxis_title='y (um)',
                            zaxis_title='z (um)',
                            xaxis_title_font_size = _font_size,
                            yaxis_title_font_size = _font_size,
                            zaxis_title_font_size = _font_size,
                           )


        #segmentation to three objects
        _label_set = ['Shaft','Neck','Head']
        _three_seg_set =  ['IsShaft','IsSpineNeck','IsSpineHead']
        _color_set = ['dodgerblue','mediumseagreen','crimson']


        _fig_data = []

        for _label,_seg,_color in zip(_label_set,_three_seg_set,_color_set):


            _im_seg = np.zeros_like(_im_compute, dtype='bool')
            _df_seg_part = _df_seg[_df_seg[_seg]]
            _im_seg[_df_seg_part.z,_df_seg_part.y,_df_seg_part.x] = True

            im_plot = _im_seg

            colormap=[mpl.colors.to_rgb(_color)]


            verts, faces, norm, val = skimage.measure.marching_cubes_lewiner(im_plot) 
            _aspect = np.array(im_plot.shape)
            _aspect_ratio = [x/_aspect.max() for x in _aspect]
            _aspect_ratio = dict(z=_aspect_ratio[0],y=_aspect_ratio[1],x=_aspect_ratio[2])

            _dtick = 1
            x,y,z = zip(*verts) 

            x = np.array(x)*spacing_resolution
            y = np.array(y)*spacing_resolution
            z = np.array(z)*spacing_resolution


            _surface = FF.create_trisurf(x=z,
                                y=y, 
                                z=x, 
                                colormap=colormap,
                                simplices=faces,
                                showbackground=False, 
                                plot_edges=False,
                                show_colorbar   = False,)


            _surface_data = _surface.data[0]
            _surface_data.name = _label
        #     _surface_data.opacity = 1
            _fig_data.append(_surface_data)

        fig = go.Figure( data = _fig_data)
        fig.update_layout(scene = scene_config,
                          width = _fig_width,
                          height = _fig_height)
        fig.update_layout(scene_aspectratio = _aspect_ratio,
                        scene_xaxis_dtick = _dtick,
                         scene_yaxis_dtick = _dtick,
                         scene_zaxis_dtick = _dtick,)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        fig.update_layout(scene = dict(  xaxis = dict(range=[0,im_plot.shape[2]*spacing_resolution],),
                                         yaxis = dict(range=[0,im_plot.shape[1]*spacing_resolution],),
                                         zaxis = dict(range=[0,im_plot.shape[0]*spacing_resolution],),),)




        fig.show()

