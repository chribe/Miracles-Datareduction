'''
This version of the reduction script is a cleaned up 

data is obtained from the Mcstas model using a banana monitor.
Detector tubes are mimiked by splitting up the signal into artificial tubes.

'''
#%% import packages
import math
from collections.abc import Sequence
from typing import NewType, TypeVar, get_type_hints
import sciline
import inspect
import warnings
from IPython.display import display
import time
import numpy as np
import os
import tempfile
import h5py
import pandas as pd
import scipy as sp
import ast
import json
import lmfit as lm
import scipp as sc
import scippneutron as scn
import tqdm
import copy
import lmfit as lm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#%% define some golbal flags
debugging=True
debugging_figures={'AcceptanceDiagram':False,
                   'Peakfinding':False,
                   'Monitor':False}
debugging_dettube=150
#%% define the main class
class Reduction:
    '''
    The reduction class is simplified to a maximum at the moment.
    Missing elements are e.g.:
        - correct background subtraction (data missing)
        - normalization
        - merging files
        - masking (with scn.MaskingTool)
        - ...
    
    written by C. Beck, April 2026
    '''
    def __init__(self, sample_file: str="Sample", background_file: str="bkg", vanadium_file: str="van", 
                scale: float = 2.0,
                bins={},ReductionFolder='./Reduction_Files/',Reductionfiles={}):
        """Initialize Reduction parameters."""

        if bins=={}:
            bins={'hw':sc.linspace('hw',-100,100,num=1000,unit='ueV'),
                'qval':sc.linspace('qval',0,2,num=10,unit="1/Angstrom")}
        if Reductionfiles=={}:
            Reductionfiles={'Vana':ReductionFolder+'Vana.hdf',
                            'Bkg':ReductionFolder+'Bkg.hdf',
                            'Peakcenters':ReductionFolder+'Peakcenters.hdf'}
        self.params = {
            self.ScaleFactor: scale,
            self.bins:bins,
            self.FolderNames[self.Sample]: sample_file,
            self.FolderNames[self.Background]: background_file,
            self.FolderNames[self.Vanadium]: vanadium_file,
            self.Reductionfiles:Reductionfiles,
            self.instparamfile:ReductionFolder+'Instrumentparameters.txt'
        }
        self.providers = self.get_fully_annotated_methods()
        self.pipeline = None
        self.graph = None
        self.intermediate={}
        if not os.path.isdir(ReductionFolder):
            os.mkdir(ReductionFolder)
        self.ReductionFolder=ReductionFolder
        self.result = None
    #%%% define domain types for sciline
    Sample = NewType('Sample', int)
    Background = NewType('Background', int)
    Vanadium = NewType('Vanadium', int)

    #generic domains
    RunType = TypeVar('RunType', Sample, Background, Vanadium)
    # RunTypenorm = TypeVar('RunTypenorm', Sample, Background)
    class FolderName(sciline.Scope[RunType, str], str): ...
    class FolderNames(sciline.Scope[RunType, list], list): ...
    class RawData(sciline.Scope[RunType, dict], dict): ...
    class convertedUnits(sciline.Scope[RunType, list], list): ...
    class BinnedData(sciline.Scope[RunType, list], list): ...
    class NormedData(sciline.Scope[RunType, list], list): ...

    # normal domain types
    ScaleFactor = NewType('ScaleFactor', float)
    BackgroundSubtractedData = NewType('BackgroundSubtractedData', dict)
    VanadiumPeak=NewType('VanadiumPeak',dict)
    Result = NewType('Result', float)
    bins = NewType('bins',dict)
    Reductionfiles=NewType('Reductionfiles',dict)
    instparamfile=NewType('instparamfile',str)
    # simplify life with sciline
    def get_fully_annotated_methods(cls):
        """Return a list of function objects for methods where all args (except self/cls) are type-hinted."""
        fully_annotated = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            sig = inspect.signature(method)
            hints = get_type_hints(method)
            params = [p for p in sig.parameters.values() if p.name not in ('self', 'cls')]

            if params and all(p.annotation != inspect._empty for p in params):
                fully_annotated.append(method)  # append the actual function handle

        return fully_annotated


    def chopper_times(phase_deg,
        frequency_hz,
        opening_angle_deg,
        t_min=0.0,
        t_max=1.0,
        offset=0.0,
        direction=1,
    ):
        """
        Calculate chopper opening and closing times.

        Parameters
        ----------
        phase_deg : float
            Reference phase in degrees.
        frequency_hz : float
            Chopper rotation frequency.
        opening_angle_deg : float
            Slit opening width in degrees.
        direction : int
            +1 or -1
        """

        if direction not in (+1, -1):
            raise ValueError("direction must be +1 or -1")

        period = 1.0 / frequency_hz

        if direction == +1:
            open_phase = phase_deg
            close_phase = phase_deg + opening_angle_deg

        else:
            open_phase = phase_deg - opening_angle_deg
            close_phase = phase_deg

        t_open0 = open_phase / 360.0 * period + offset
        t_close0 = close_phase / 360.0 * period + offset

        # Ensure proper ordering in time
        if t_close0 < t_open0:
            t_close0 += period

        n_start = int(np.floor((t_min - t_close0) / period))
        n_stop = int(np.ceil((t_max - t_open0) / period))

        n = np.arange(n_start, n_stop + 1)

        open_times = t_open0 + n * period
        close_times = t_close0 + n * period

        mask = (close_times >= t_min) & (open_times <= t_max)

        return open_times[mask], close_times[mask]

    @staticmethod
    def acceptanceDiagram(instparamfile,position,chopper):
        instrumentdict=Reduction.get_instrument_params(instparamfile)
        params={}
        openingangle=[12,12,15,105]
        plt.figure()
        for chop in range(4):
            
            if chop==1:
                direction=-1
                delta=12
            else:
                direction=1      
                delta=0
            open,close=Reduction.chopper_times(chopper[chop]['phase']+delta,
                                               np.abs(chopper[chop]['speed']),
                                               openingangle[chop],
                                               direction=direction)
            params.update({f'chopper{chop+1}':{'open':open,
                                            'close':close,
                                            'distance':instrumentdict[f'chopper{chop+1}_distance']
                                            }
                        })
        choppers = sc.DataGroup(
            {name: scn.tof.chopper_cascade.Chopper(
                distance=sc.scalar(param['distance'], unit='m'),
                time_open=sc.array(dims=('cutout',), values=param['open'], unit='s'),
                time_close=sc.array(dims=('cutout',), values=param['close'], unit='s')
            ) for name, param in params.items()}
        )
        frames = scn.tof.chopper_cascade.FrameSequence.from_source_pulse(
            time_min=sc.scalar(0.0, unit='ms'),
            time_max=sc.scalar(4.0, unit='ms'),  # ESS pulse is 3 ms, but it has a tail
            wavelength_min=sc.scalar(0.0, unit='angstrom'),
            wavelength_max=sc.scalar(10.0, unit='angstrom'),
        )
        frames = frames.chop(choppers.values())
        at_sample = frames.propagate_to(sc.scalar(instrumentdict[position], unit='m'))
        if debugging:
            if debugging_figures['AcceptanceDiagram']==False:
                at_sample.draw()
                debugging_figures['AcceptanceDiagram']=True
        #print(at_sample[-1])
        ac_dia=at_sample[-1].subframes[0]
        
        return ac_dia

    @staticmethod
    def determine_lambda_acceptancediagram(time,ac_dia):
        wlvalues=Reduction.polygon_vertical_intersections(ac_dia.time.values,ac_dia.wavelength.values,time)
        wl=np.mean(wlvalues)
        dwl=np.abs(wl-wlvalues[0])
        return wl,dwl

    @staticmethod
    def determine_time_acceptancediagram(lbda,ac_dia):
        values=Reduction.polygon_vertical_intersections(ac_dia.wavelength.values,ac_dia.time.values,lbda)
        t=np.mean(values)
        dt=np.abs(t-values[0])
        return t,dt
    
    @staticmethod
    def polygon_vertical_intersections(x, y, x0):
        """
        x, y: arrays of polygon vertices (must be closed: last==first or "wrap")
        x0:   the vertical constant value to find intersections with
        
        Returns: list of y values where the vertical line x=x0 crosses the polygon
        """
        if np.isnan(x0):
            intersections = [np.nan,np.nan]
        else:
            intersections = []
            N = len(x)
            # Iterate over polygon edges
            for i in range(N):
                x1, y1 = x[i], y[i]
                x2, y2 = x[(i+1)%N], y[(i+1)%N]  # wrap to first for closed polygon
                # Check if edge crosses x0
                if (x1 - x0) * (x2 - x0) < 0:  # If x0 is between x1 and x2
                    # Linear interpolation to find intersection y
                    t = (x0 - x1) / (x2 - x1)
                    y_int = y1 + t * (y2 - y1)
                    intersections.append(y_int)
                # Optionally, handle if x0 exactly at a vertex
                elif x1 == x0:
                    intersections.append(y1)
        if intersections==[]:
            intersections = [np.nan,np.nan]
        return intersections
    
    #%%%Detector to 2theta calculation
    @staticmethod
    def detpos2theta(instparamfile):
        """
        This function converts the scattering angle (the polar angle theta in the spherical coordinate system) to a detector pixel on a detector tube. 
        The secondary spectrometer is defined by the following quantities which are saved in the file "Instrumentparameters.txt" (point like Sample in origin):
        - detector_x: horizontal distance of the detector to the sample axis 
        - detector_bottom: vertical lower edge of the detector
        - detector_top: vertical upper edge of the detector
        - analyzer_focus_x: horizontal position of focus
        - analyzer_focus_y: vertical position of focus
        - analyzer_radius: radius of analyzer
        """
        print(10*'$'+'calculate detpos2theta')
        instrumentdict=Reduction.get_instrument_params(instparamfile)
        end_angle=20.43 # based on geogebra geometry
        start_angle=-np.rad2deg(np.arcsin(instrumentdict['analyzer_focus_y']/instrumentdict['analyzer_radius']))
        def x_of_phi(phi, y0=instrumentdict['analyzer_focus_y'], r=instrumentdict['analyzer_radius'], sign=+1):
            return np.cos(phi) * (y0*np.sin(phi) + sign*np.sqrt(r**2 - (y0*np.cos(phi))**2))
                    
        def y_of_phi(phi, y0=instrumentdict['analyzer_focus_y'], r=instrumentdict['analyzer_radius'], sign=+1):
            sqrt_term = np.sqrt(np.maximum(0, r**2 - (y0*np.cos(phi))**2))
            return np.sin(phi) * (y0*np.sin(phi) + sign*sqrt_term)
        
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)
        
        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2' """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        def reflect_vector(v,u):
            """Reflect vector v across vector u. and changes direction"""
            v = np.array(v, dtype=float)
            u = np.array(u, dtype=float)
            return -(2 * np.dot(v, u) / np.dot(u, u) * u - v)

        # now do calibration curve:
        theta =np.linspace(0,22,1000)
        calcurve={'theta':theta,
                'angleatanalyzer':[],
                'detectorpos':[],
                'flightpath':[]}
        if debugging:
            calcurve.update({
                'vec_Sample_Ana':[],
                'vec_AnaDet':[],
                'vec_analyzed':[],
                'vec_Ana_Focus':[]})
        plt.figure()
        for thet in theta:
            thet=np.deg2rad(thet)
            x=x_of_phi(thet)
            y=y_of_phi(thet)
            plt.plot(x,y,'o')
            vec_Sample_Ana=(x,y)
            vec_Ana_Focus=(instrumentdict['analyzer_focus_x']-x,instrumentdict['analyzer_focus_y']-y)
            calcurve['angleatanalyzer'].append(float(angle_between(vec_Sample_Ana,vec_Ana_Focus)))
            vec_analyzed=reflect_vector(vec_Sample_Ana,vec_Ana_Focus)
            # calculate now scaling parameter for vec_analyzed based on x coord to reach detector
            scaling=(instrumentdict['detector_x']-vec_Sample_Ana[0])/vec_analyzed[0]
            vec_AnaDet=scaling*vec_analyzed
            vec_SamAnaDet=vec_Sample_Ana+vec_AnaDet
            calcurve['flightpath'].append(np.linalg.norm(vec_Sample_Ana)+np.linalg.norm(vec_AnaDet))
            if vec_SamAnaDet[1]>instrumentdict['detector_bottom'] and vec_SamAnaDet[1]<instrumentdict['detector_top']:
                calcurve['detectorpos'].append(vec_SamAnaDet[1]-instrumentdict['detector_bottom'])
            else:
                calcurve['detectorpos'].append(np.nan)
            if debugging:
                calcurve['vec_Sample_Ana'].append(vec_Sample_Ana)
                calcurve['vec_analyzed'].append(vec_analyzed)
                calcurve['vec_Ana_Focus'].append(vec_Ana_Focus)
                calcurve['vec_AnaDet'].append(vec_AnaDet)
        dataframe=pd.DataFrame.from_dict({'detectorpos':calcurve['detectorpos'],'theta':calcurve['theta'],'flightpath':calcurve['flightpath'],'angleatanalyzer':calcurve['angleatanalyzer']})
        dataframe.to_csv("detector_angle_calibration.txt")
        circle = Circle((0, 0.075), 2.5, fill=False)
        circle1 = Circle((0, 0.075), 2.4, fill=False)
        ax=plt.gca()
        ax.set_aspect('equal')
        ax.add_patch(circle)
        ax.add_patch(circle1)
        return calcurve
    #%% determine kf
    @staticmethod
    def determine_kf_flighttime(detectorheight,idxtube,instparamfile):
        """
        determines first the vector pointing to the analyzer position and then rescales it to have the correct q value.
        Returns also the flight time in the secondary spectrometer
        """
        #check input of detectorheight
        try:
            detheight=detectorheight.value
        except:    
            detheight=detectorheight
        if not os.path.isfile('detector_angle_calibration.txt'):
            Reduction.detpos2theta(instparamfile)
        dataframe=pd.read_csv('detector_angle_calibration.txt')
        instrument_params=Reduction.get_instrument_params(instparamfile)
        #determine now the angle phi of tube
        phideg=instrument_params['delta_angle_tube']*idxtube+instrument_params['angle_tube0']
        phi=np.deg2rad(phideg)
        theta=np.deg2rad(np.interp(detheight,np.array(dataframe['detectorpos']),np.array(dataframe['theta']),left=np.nan,right=np.nan))
        theta_sphercord=np.deg2rad(90-theta)
        #calculate now x,y,z on sphere with r=1
        x=np.sin(theta_sphercord)*np.cos(phi)
        y=np.sin(theta_sphercord)*np.sin(phi)
        z=np.cos(theta_sphercord)
        kfnorm=(x,y,z)
        # determine wavelength via bragg and angle
        if instrument_params['ana_crystal']=='Si111':
            dspacing=3.135 #AA
        else:
            warnings.warn(f'analyzer crystal f{instrument_params['ana_crystal']} not defined. Falling back on Si111.')
            dspacing=3.135 #AA
        angle=np.deg2rad(90-np.interp(detheight,np.array(dataframe['detectorpos']),np.array(dataframe['angleatanalyzer']),left=np.nan,right=np.nan))
        lambda_reflected=2*np.sin(angle)*dspacing
        k_abs=2*np.pi/lambda_reflected #1/AA
        flightpath=np.interp(detheight,np.array(dataframe['detectorpos']),np.array(dataframe['flightpath']),left=np.nan,right=np.nan)
        hbar =sp.constants.hbar # J s
        m_neutron = sp.constants.neutron_mass # kg
        v_neutron=hbar*k_abs/m_neutron*1e10 #m/s
        tof=flightpath/v_neutron
        kf=tuple(k_abs * x for x in kfnorm)
        return kf,tof
    

    @staticmethod
    def get_instrument_params(instparamfile):
        instrument_params=pd.read_csv(instparamfile,sep=":",header=None)
        instrument_params[1] = instrument_params[1].apply(lambda x: ast.literal_eval(str(x).strip()) if isinstance(x, str) else x) # convert entry to float if not string
        instrument_params=instrument_params.set_index(0)[1].to_dict()
        return instrument_params
    
    #%%% loading files
    @staticmethod
    def load_normalize_merge_folders(
        folders: "Reduction.FolderNames[Reduction.RunType]",
    ) -> "Reduction.RawData[Reduction.RunType]":
        if debugging: print(5*'#'+'load several files')
        merged = []
        filename=[]
        print(folders)
        if 'Vana' in folders:
            treatment='Vanadium; '
            folders.remove('Vana')
        elif "Bkg" in folders:
            treatment='Bkg; '
            folders.remove('Bkg')
        else:
            treatment=''
        for f in folders:
            print(f)
            if os.path.exists(f+'mccode.h5'):
                data=Reduction.loadeventshdf(f+'mccode.h5')
                data=Reduction.determine_nearest_tof_peak(data)
                data,monitor=Reduction.monitornormalization({'data': data,'treatment':treatment+'loaded from McStas nexus'},f)
                print('nexus file found and loaded')
            else:
                raise Exception(f'No files found in {f}')
            if merged==[]:
                merged=data['data']
            else:
                merged=sc.concat([merged, data['data']],dim='event')
            filename.append(f)
        with h5py.File(f+'mccode.h5','r') as f:
            chopperdata=np.array(f['/entry1/data/chopper_settings_dat/data'])
        choppers=[]
        for c in range(4):
            choppers.append({'speed':chopperdata[c,0],'phase':chopperdata[c,1]})
        print(chopperdata)
        print(choppers)
        print("Merging", len(folders), "folder(s)")
        data={'data': merged, 'meta': {'filename': filename},'treatment':data['treatment'],'monitor':monitor,'choppers':choppers}
        return Reduction.RawData[Reduction.RunType](data)

    @staticmethod
    def loadeventshdf(filename):
        path='/entry1/data/caen_events_list_p_U0_U1_U2_U3_U4_U5_U6_U7/events'
        with h5py.File(filename, "r") as f:
            if path in f:
                kafkatube=True
            else:
                kafkatube=False
        if kafkatube:
            warnings.warn('Assuming McStas results from Miracles with detector tubes from Detector_tubes in combination with modified ReadoudCaen \n \
                          Assuming the following entries: weight; ring; fen; time; channel; a; b; c; d\n \
                          a and b are the voltages from the tubes\n \
                          Assuming 150m tube length')
            L=0.150
            with h5py.File(filename, 'r') as f:
                dat=np.array(f[path])
            pos=['t','b']
            tubes=sc.array(dims=['event'], values=dat[:,4])
            height=sc.array(dims=['event'],values=L*(0.5+(dat[:,5]-dat[:,6])/(dat[:,5]+dat[:,6])),unit='m')
            tof=sc.array(dims=['event'],values=dat[:,3],unit='s')
            weights = sc.array(dims=['event'], unit='counts', values=dat[:,0])
            topbottom=sc.array(dims=['event'], values=[pos[int(tb-1)] for tb in dat[:,2]])
        else:
            warnings.warn('Assuming McStas results from Miracles with detector tubes from monitornd \n Assuming the following entries: Intensity Wavelength Time_Of_Flight Position')
            tube=np.array([])
            tb=[]
            data=np.array([])
            for hi in range(1,49):
                for pos in ['t','b']:
                    try:
                        with h5py.File(filename, 'r') as f:
                            dat = np.array(f[f'/entry1/data/tube{hi}t_list_p_L_t_y/events'])
                            if dat.shape==(4,):
                                dat=np.array([dat])
                    except:
                        print('no data loaded for ',hi,pos)
                        dat=np.array([])
                    if dat.shape==(0,):
                        pass
                    elif data.shape==(0,):
                        data=dat
                    else:
                        data=np.concatenate((data, dat), axis=0)
                    tube=np.append(tube,np.ones(dat.shape[0])*hi)
                    #print(f'overall events: {tube.shape}')
                    tb+=dat.shape[0]*[pos]
            tubes=sc.array(dims=['event'], values=tube)
            height=sc.array(dims=['event'],values=data[:,3],unit='m')
            tof=sc.array(dims=['event'],values=data[:,2],unit='s')
            weights = sc.array(dims=['event'], unit='counts', values=data[:,0])
            topbottom=sc.array(dims=['event'], values=tb)
        usepoisson=False
        if usepoisson:
            idc=Reduction.detect_event_indices_poisson(weights.values)
            vals=np.ones(data[:,0].shape)[idc]
        else:
            idc=np.arange(len(weights.values))
            vals=weights.values
        print(f'neutrons reduced from {tubes.shape} events to {vals.shape} events')
        counts=sc.DataArray(data=sc.array(dims=['event'], unit='counts', values=vals), coords={'tof': tof[idc],'height':height[idc],'tube':tubes[idc],'topbottom':topbottom[idc]})
        print(counts)
        return counts

    @staticmethod
    def findplateau(x, y, tol, n=3):
        """
        Detect plateaus by checking the slope between points i and i+n.
        Args:
            x: numpy array, x-values
            y: numpy array, y-values
            tol: float, slope threshold to define plateaus
            n: int, displacement for slope calculation
        Returns:
            plateau_starts, plateau_ends, centers_x, centers_y
        """
        # check if n< len(y)/10
        if n>len(y)/10:
            warnings.warn(f'parameter n set from {n} to {int(len(y)/10)}')
            n=int(len(y)/10)
        print(n)
        x = np.asarray(x)
        y = np.asarray(y)
        # Slope between i and i+n
        print((y[n:]))
        print((x[n:] - x[:-n]))
        slope = (y[n:] - y[:-n]) / (x[n:] - x[:-n])
        print(slope)
        # Pad to match y's length (or assign nan to unmatched points)
        slope_padded = np.concatenate((
            [slope[0]] * (n//2), 
            slope, 
            [slope[-1]] * (len(y) - len(slope) - (n//2))
        ))
        # Plateau if absolute slope is below tol
        on_plateau = np.abs(slope_padded) < tol
        # Find transitions
        change_points = np.where(np.diff(on_plateau.astype(int)) != 0)[0]
        plateau_starts = np.insert(change_points + 1, 0, 0)
        plateau_ends = np.append(change_points, len(on_plateau)-1)
        centers_x = []
        centers_y = []
        for start, end in zip(plateau_starts, plateau_ends):
            if on_plateau[start]:
                mid_idx = (start + end) // 2
                centers_x.append(x[mid_idx])
                centers_y.append(y[mid_idx])
        return plateau_starts, plateau_ends, centers_x, centers_y
    
    @staticmethod
    def find_nearest(array, values):
        array = np.asarray(array)
        values = np.atleast_1d(values)
        idx = np.abs(array[np.newaxis, :] - values[:, np.newaxis]).argmin(axis=1)
        return array[idx]

    def gauss(x,center,width,scaling,bkg):
        return scaling/np.sqrt(2*width**2*np.pi)*np.exp(-(x-center)**2/width**2/2)+bkg
    
    def sigmoid(x,b,c):
        return sp.special.expit((x-c)/b)
    
    def beamtof(x,scaling,b1,c1,b2,c2):
        y=scaling*Reduction.sigmoid(x,b1,c1)*(1-Reduction.sigmoid(x,b2,c2))
        return y

    def beamtof3sig(x,s1,b1,c1,s2,b2,c2,b3,c3):
        y=s1*Reduction.sigmoid(x,b1,c1)*(1-s2*Reduction.sigmoid(x,b2,c2))*(1-Reduction.sigmoid(x,b3,c3))
        return y

    def savehdf(data, filename):
        """
        Saves each entry of a dictionary to a temporary HDF5 file, then merges to final file.
        Scipp arrays saved with sc.io.save_hdf5, others as attributes or JSON dataset.
        """
        print('start saveing routine')
        # Step 1. Save each entry to a temp hdf5
        tempfiles = {}
        for key, val in data.items():
            # Create temporary file
            fd, tmpname = tempfile.mkstemp(suffix='.hdf5')
            os.close(fd)

            if hasattr(val, 'dims') and hasattr(val, 'dtype'):  # crude: is Scipp-like?
                sc.io.save_hdf5(val, tmpname)
            else:
                # Save as HDF5 dataset (for string/int/float) or as JSON string in an attribute
                with h5py.File(tmpname, 'w') as f:
                    if isinstance(val, (dict, list)):
                        f.attrs['value'] = json.dumps(val)
                    elif isinstance(val, (str, int, float, bool)):
                        f.attrs['value'] = str(val)
                    else:
                        # If you have more exotic classes, maybe use pickle here
                        f.attrs['value'] = json.dumps(str(val))
            tempfiles[key] = tmpname
        # Step 2. Merge into one hdf5 file with subgroups
        with h5py.File(filename, 'w') as out:
            for key, tmpname in tempfiles.items():
                with h5py.File(tmpname, 'r') as src:
                    # For Scipp, will find 'data', 'coords' etc at the root
                    grp = out.create_group(key)
                    # Copy everything from src into grp
                    def copyall(srcnode, dstnode):
                        for name, item in srcnode.items():
                            srcnode.copy(name, dstnode, name=name)
                        for attr in srcnode.attrs:
                            dstnode.attrs[attr] = srcnode.attrs[attr]
                    copyall(src, grp)
                os.remove(tmpname)

    def loadhdf(filename):
        import scipp as sc, h5py, json
        result = {}
        with h5py.File(filename, 'r') as f:
            for key in f:
                # try reading as scipp first:
                try:
                    result[key] = Reduction.load_group_as_scipp(filename, group=key)
                except Exception:
                    v = f[key].attrs.get("value", None)
                    if v is not None:
                        if isinstance(v, bytes):
                            v = v.decode("utf-8")
                        try:
                            v = json.loads(v)
                        except Exception:
                            pass
                    result[key] = v
        return result


    @staticmethod
    def monitorTOF(monitorname,center,eps=0.015,threshold=1e5,model='beamtof3sig'):
        if model=='beamtof3sig':
            model=lm.Model(Reduction.beamtof3sig)
            params=lm.Parameters()
            params.add('b1',min=0,value=5e-3)
            params.add('b2',min=0,value=5e-3)
            params.add('b3',min=0,value=5e-3)
            params.add('c3',min=0,value=center+eps)
            params.add('c2',min=0,value=center)
            params.add('c1',min=0,value=center-eps)
            params.add('s1',min=0,value=1e5)
            params.add('s2',min=0,max=1,value=0.5)
        else:
            model=lm.Model(Reduction.beamtof)
            params=lm.Parameters()
            params.add('b1',min=0,value=5e-3)
            params.add('b2',min=0,value=5e-3)
            params.add('c2',value=center)
            params.add('deltac',min=0,value=0.01)
            params.add('c1',expr='c2-deltac')
            params.add('scaling',min=0,value=7e7)
        if os.path.exists(monitorname):
            df=pd.read_csv(monitorname,sep=' ',comment='#',header=None)
            x=df[0].values/1e6
            y=df[1].values
            print('Monitor from .dat file loaded!')
        elif os.path.exists(monitorname.replace('tof_BGAS.dat','mccode.h5')):
            with h5py.File(monitorname.replace('tof_BGAS.dat','mccode.h5'), 'r') as f:
                x = np.array(f['/entry1/data/tof_BGAS_dat/Time_of_flight___gms_'])/1e6
                y = np.array(f['/entry1/data/tof_BGAS_dat/data'])
            print('Monitor from nexus file loaded!')
        else:
            raise Exception('No monitor data found!')

        idx=np.abs(x-center+eps)<1/14/2
        fit=model.fit(y[idx],params=params,x=x[idx])
        # 1. Smooth
        smoothed_y = sp.signal.savgol_filter(y, window_length=600, polyorder=3)
        # 2. Find minima
        minima_idx, _ = sp.signal.find_peaks(-smoothed_y, distance=200, prominence=0.2)
        minima_idx=minima_idx[y[minima_idx]>threshold]
        tofmins=x[minima_idx]
        tofmindiff=np.mean(np.abs(np.diff(tofmins)))
        #print(tofmins)
        if len(tofmins)==0:
            tofmins=[0]
        try:
            tmin=max(tofmins[tofmins<center])
        except:
            tmin=min(tofmins)-tofmindiff
        try:
            tmax=min(tofmins[tofmins>center])
        except:
            tmax=max(tofmins)+tofmindiff
        if debugging:
            if not debugging_figures['Monitor']:
                plt.figure()
                plt.plot(x,y)
                plt.plot(x[idx],y[idx])
                plt.plot(x,fit.eval(x=x))
                plt.plot(x[idx],fit.init_fit)
                plt.plot(2*[center-eps],[0,max(y)])
                plt.plot(x[minima_idx],y[minima_idx],'v')
                plt.title('monitor fit')
                plt.xlabel('time [s]')
                plt.show()
                plt.savefig('Monitorfit.pdf')
                debugging_figures['Monitor']=True
        return fit,tmin,tmax

    @staticmethod
    def monitorTOF_load(monitorname):
        if os.path.exists(monitorname):
            df=pd.read_csv(monitorname,sep=' ',comment='#',header=None)
            x=df[0].values/1e6
            y=df[1].values
            print('Monitor from .dat file loaded!')
        elif os.path.exists(monitorname.replace('tof_BGAS.dat','mccode.h5')):
            with h5py.File(monitorname.replace('tof_BGAS.dat','mccode.h5'), 'r') as f:
                x = np.array(f['/entry1/data/tof_BGAS_dat/Time_of_flight___gms_'])/1e6
                y = np.array(f['/entry1/data/tof_BGAS_dat/data'])
            print('Monitor from nexus file loaded!')
        else:
            raise Exception('No monitor data found!')
        return x,y
    
    @staticmethod
    def monitorTOF_fit(x,y,center,eps=0.015,threshold=1e5,model='beamtof3sig'):
        if model=='beamtof3sig':
            model=lm.Model(Reduction.beamtof3sig)
            params=lm.Parameters()
            params.add('b1',min=0,value=5e-3)
            params.add('b2',min=0,value=5e-3)
            params.add('b3',min=0,value=5e-3)
            params.add('c3',min=0,value=center+eps)
            params.add('c2',min=0,value=center)
            params.add('c1',min=0,value=center-eps)
            params.add('s1',min=0,value=1e5)
            params.add('s2',min=0,max=1,value=0.5)
        else:
            model=lm.Model(Reduction.beamtof)
            params=lm.Parameters()
            params.add('b1',min=0,value=5e-3)
            params.add('b2',min=0,value=5e-3)
            params.add('c2',value=center)
            params.add('deltac',min=0,value=0.01)
            params.add('c1',expr='c2-deltac')
            params.add('scaling',min=0,value=7e7)
        idx=np.abs(x-center+eps)<1/14/2
        fit=model.fit(y[idx],params=params,x=x[idx])
        # 1. Smooth
        smoothed_y = sp.signal.savgol_filter(y, window_length=600, polyorder=3)
        # 2. Find minima
        minima_idx, _ = sp.signal.find_peaks(-smoothed_y, distance=200, prominence=0.2)
        minima_idx=minima_idx[y[minima_idx]>threshold]
        tofmins=x[minima_idx]
        tofmindiff=np.mean(np.abs(np.diff(tofmins)))
        #print(tofmins)
        if len(tofmins)==0:
            tofmins=[0]
        try:
            tmin=max(tofmins[tofmins<center])
        except:
            tmin=min(tofmins)-tofmindiff
        try:
            tmax=min(tofmins[tofmins>center])
        except:
            tmax=max(tofmins)+tofmindiff
        if debugging:
            if not debugging_figures['Monitor']:
                plt.figure()
                plt.plot(x,y)
                plt.plot(x[idx],y[idx])
                plt.plot(x,fit.eval(x=x))
                plt.plot(x[idx],fit.init_fit)
                plt.plot(2*[center-eps],[0,max(y)])
                plt.plot(x[minima_idx],y[minima_idx],'v')
                plt.title('monitor fit')
                plt.xlabel('time [s]')
                plt.show()
                plt.savefig('Monitorfit.pdf')
                debugging_figures['Monitor']=True
        return fit,tmin,tmax


    @staticmethod
    def determine_nearest_tof_peak(rawdata):
        x=rawdata.coords['tof'].values
        y=rawdata.values
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        plateau_starts,plateau_ends,plateau_centers_x,plateau_centers_y=Reduction.findplateau(np.arange(len(x_sorted)),x_sorted,1e-4,n=100)
        if debugging:
            if not debugging_figures['Peakfinding']:
                plt.figure()
                plt.plot(x_sorted,y_sorted,'o')
                for center in plateau_centers_x:
                    plt.plot(2*[x_sorted[center]],[0,max(y_sorted)])
                plt.title('peak identification')
                plt.xlabel('time [s]')
                plt.show()
                debugging_figures['Peakfinding']=True
        tofpeakcenters=Reduction.find_nearest(plateau_centers_y,x)
        rawdata.coords.update({'tofpeakcenters':sc.array(dims=['event'],values=tofpeakcenters,unit='s')})
        return rawdata

    @staticmethod
    def monitornormalization(data,monitor):
        if debugging: print(5*'#'+'save monitornormalization values')
        centers=np.unique(data['data'].coords['tofpeakcenters'].values)
        area=[]
        time_min=[]
        time_max=[]
        print(f'detected centers: {centers}')
        monitorparams={}
        x,y=Reduction.monitorTOF_load(monitor+'tof_BGAS.dat')
        for center in centers:
            monitormodel='beamtof3sig'
            monitorfit,tmin,tmax=Reduction.monitorTOF_fit(x,y,center,model=monitormodel)
            p = monitorfit.params
            b1      = p['b1'].value
            c1      = p['c1'].value
            b2      = p['b2'].value
            c2      = p['c2'].value
            time_min.append(tmin)
            time_max.append(tmax)
            if monitormodel=='beamtof3sig':
                b3      = p['b3'].value
                c3      = p['c3'].value
                s1 = p['s1'].value
                s2 = p['s2'].value
                area_quad, abs_err = sp.integrate.quad(
                    lambda x: Reduction.beamtof3sig(x, s1, b1, c1, s2,b2, c2,b3,c3),
                    -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8
                )
                fitparams=[s1, b1, c1, s2,b2, c2,b3,c3]
            # integrate with scipy.quad (can use infinite bounds) ---
            # choose bounds; -inf..inf is often fine for logistic-like pulses
            else:
                scaling = p['scaling'].value
                area_quad, abs_err = sp.integrate.quad(
                    lambda x: Reduction.beamtof(x, scaling, b1, c1, b2, c2),
                    -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8
                )
                fitparams=[scaling, b1, c1, b2, c2]
            monitorparams.update({f'center{center}':{'Area':area_quad,
                                                     'fitparams':fitparams}})
        centers=np.array(centers)
        time_min=np.array(time_min)
        time_max=np.array(time_max)
        for hi,tofc in enumerate(data['data'].coords['tofpeakcenters'].values):
            if  data['data'].coords['tof'].values[hi]<time_min[centers==tofc]:
                data['data'].coords['tof'].values[hi]=0
            elif data['data'].coords['tof'].values[hi]>time_max[centers==tofc]:
                data['data'].coords['tof'].values[hi]=0
        data['treatment']+='; monitor normalization value saved'
        return data,monitorparams
    

    @staticmethod
    def loadevents(filename):
        warnings.warn('Assuming McStas results from Miracles with banana monitor at the moment \n Assuming the following entries: Intensity Angle Wavelength Time_Of_Flight Position')
        df=pd.read_csv(filename,sep=' ',comment='#',header=None,names=['Intensity', 'Angle', 'Wavelength', 'Time_Of_Flight', 'Position',""])#,names=['Intensity', 'Angle', 'Wavelength', 'Time_Of_Flight', 'Position']
        # convert angle now into tube:
        df['tube']=np.round((df['Angle']-13.06)/3.43)
        angle=sc.array(dims=['event'], values=df['Angle'])
        tube=sc.array(dims=['event'], values=df['tube'])
        height=sc.array(dims=['event'],values=df['Position'],unit='m')
        tof=sc.array(dims=['event'],values=df['Time_Of_Flight'],unit='s')
        weights = sc.array(dims=['event'], unit='counts', values=df['Intensity'])
        counts=sc.DataArray(data=weights, coords={'tof': tof,'height':height,'tube':tube,'angle':angle})
        return counts
    
    @staticmethod
    def determineVanapeak(dat:"Reduction.BinnedData[Reduction.Vanadium]",
                        Reductionfiles: "Reduction.Reductionfiles") -> "Reduction.VanadiumPeak":
        if debugging: print(5*'#'+'Vanapeak')
        if os.path.isfile(Reductionfiles['Peakcenters']):
            vanapeak=Reduction.loadhdf(Reductionfiles['Peakcenters'])
            if debugging:print(5*'#'+'Peakcenters loaded')
        else:
            dense = dat['data'].bins.sum()  # Now shape (hw, q), values are counts/floats
            idx_max = np.argmax(dense.values, axis=0)
            hw_coords = dense.coords['hw'].values
            q_coords = dense.coords['qval'].values
            q_centers = 0.5 * (q_coords[:-1] + q_coords[1:])
            hw_centers = 0.5 * (hw_coords[:-1] + hw_coords[1:])
            hw_max_per_q = hw_coords[idx_max]
            vanapeak={'hw_max':hw_max_per_q,'gaussfits':[],'integratedfit':[]}
            model=lm.Model(Reduction.gauss)
            params=lm.Parameters()
            params.add('center',value=0)
            params.add('width',value=10,min=0)
            params.add('scaling',value=1,min=0)
            params.add('bkg',value=1)
            for hiq,q in enumerate(q_centers):
                y=dense['qval',hiq].values
                params['bkg'].value=np.mean(y)
                params['scaling'].value=np.max(y)
                params['center'].value=hw_max_per_q[hiq]
                gaussfit=model.fit(y,x=hw_centers,params=params)
                fitparams={}
                for name, param in gaussfit.params.items():
                    fitparams.update({name:(param.value,param.stderr)})
                vanapeak['gaussfits'].append(fitparams)
                if debugging:
                    plt.figure()
                    plt.semilogy(hw_centers,y)
                    plt.semilogy(hw_centers,gaussfit.eval())
                    plt.savefig(f'Vanafit{q}.pdf')
                    print('figure saved')
                # integration for 3 sigma of fit including bkg: Equation based on https://www.wolframalpha.com/input?i2d=true&i=Integrate%5B%5C%2840%29Divide%5Bs%2CSqrt%5B2*%CF%80*Power%5Bw%2C2%5D%5D%5D*exp%5C%2840%29-Divide%5BPower%5Bx-c%2C2%5D%2C2*Power%5Bw%2C2%5D%5D%5C%2841%29%2Bb%5C%2841%29%2C%7Bx%2Cc-3w%2Cc%2B3w%7D%5D
                vanapeak['integratedfit'].append(6*gaussfit.params['bkg'].value*gaussfit.params['width'].value+sp.special.erf(3/np.sqrt(2))*gaussfit.params['scaling'].value)
            Reduction.savehdf(vanapeak,Reductionfiles['Peakcenters'])
        return Reduction.VanadiumPeak(vanapeak)

    @staticmethod
    def binning(
        data:"Reduction.convertedUnits[Reduction.RunType]",
        binning:"Reduction.bins",
    ) -> "Reduction.BinnedData[Reduction.RunType]":
        '''
        This function is binning the event data to sqw. It requires as inputs the event data as well as the bins for hw and q.
        The bins have to be provided as a dictionary with the keys 'hw' and 'qval'.
        the values should be one of the following options:
        - list with three values [a,b,c] indicating the minimum, maximum and number of bins
        - list with two values [a,b]: Same as [a,b,100]
        - scipp array, scipp linspace, scipp logspace, etc
        '''
        if debugging: print(5*'#'+'binning')
        "check type of binnings"
        bins={}
        for param in ['hw','qval']:
            if param in binning.keys():
                if type(binning[param])==list:
                    if len(binning[param])==2:
                        bin=sc.linspace(param,binning[param][0],binning[param][1],num=100) 
                    elif len(binning[param])==3:
                        bin=sc.linspace(param,binning[param][0],binning[param][1],num=binning[param][2])
                    else:
                        raise ValueError(f'length of binning list for {param} not adapted')
                elif type(binning[param])==type(sc.linspace('event',0,2,num=100)):
                    bin=binning[param]
            else:
                raise ValueError(f'parameter {param} not in binning!')
            bins.update({param:bin})
        data['data']=sc.bin(data['data'],hw=bins['hw'],qval=bins['qval'])
        data['treatment']+='; masked; binned'
        return Reduction.BinnedData(data)
    

    @staticmethod
    def transformation_coordinates(
        raw_data: "Reduction.RawData[Reduction.RunType]",
        instparamfile:"Reduction.instparamfile",
        Reductionfiles: "Reduction.Reductionfiles"
    ) -> "Reduction.convertedUnits[Reduction.RunType]":
        """
        Multiply the data by a scale factor
        applies mask
        transforms pixel coordinates into kf assuming a point sample and perfect spherical analysers
        """
        if debugging: print(5*"#"+'transformation_coordinates')
        #calculate kf and tof SAD
        qvec=[]
        hw=[]
        height=150
        tubes=48
        filename='detectorvalues.npy'
        if os.path.isfile(filename):
            detectorvalues=np.load(filename,allow_pickle=True)
        else:
            detectorvalues= np.empty((tubes+2,height+2), dtype=object)
            for tube in tqdm.tqdm(range(tubes+2)):
                if tube==0:
                    detectorvalues[tube,-1]=((np.nan,np.nan,np.nan),np.nan)
                for detheight in range(height+2):
                    if tube==0 and detheight==0:
                        detectorvalues[-1,detheight]=((np.nan,np.nan,np.nan),np.nan)
                    kfv,tof=Reduction.determine_kf_flighttime(detheight/1000,tube,instparamfile)
                    detectorvalues[tube,detheight]=(kfv,tof)
            np.save(filename, detectorvalues)
        
        tubecoord=raw_data['data'].coords['tube'].values.astype(int)
        tubecoord[tubecoord>tubes]=tubes+1
        pixelcoord=np.abs(raw_data['data'].coords['height'].values*1e3).astype(int)
        pixelcoord[pixelcoord>height]=height+1
        fill_value=((np.nan,np.nan,np.nan),np.nan)
        if debugging:
            print(5*'#'+f'artificially coarsening the detector tube to a resolution of {debugging_dettube} bin(s)')
            #debugging_dettube is number of detector bins
            # find now the centers of the bins
            delta=150/(2*debugging_dettube)
            binedges=np.linspace(delta,150-delta,debugging_dettube)
            pxcs=[]
            for pc in pixelcoord:
                pxcs.append(binedges[np.abs(binedges-pc).argmin()])
            pixelcoord=np.int32(pxcs)
        kf_tof=detectorvalues[np.abs(tubecoord),pixelcoord]
        mask=tubecoord<0
        replacement = np.empty(np.count_nonzero(mask), dtype=object)
        replacement[:] = [fill_value] * np.count_nonzero(mask)
        # kf_tof[mask] = replacement
        ac_dia=Reduction.acceptanceDiagram(instparamfile,'sample_distance',raw_data['choppers'])
        for hi in tqdm.tqdm(range(len(raw_data['data'].coords['height']))):
            kfvec=kf_tof[hi][0]
            t=kf_tof[hi][1]
            for idx in range(10):
                wl,dwl=Reduction.determine_lambda_acceptancediagram(raw_data['data'].coords['tof'][hi].values-t+(idx-5)/14,ac_dia)
                if not np.isnan(wl): break
            ki_abs=2*np.pi/wl
            ki=(ki_abs,0,0)
            q=tuple(ai - bi for ai, bi in zip(kfvec, ki))
            kf_abs=np.linalg.norm(kfvec)
            dE=sp.constants.hbar**2/2/sp.constants.neutron_mass*(kf_abs**2-ki_abs**2)# units: (J s)^2/kg/AA^2
            #convert units into ueV:  
            dE*=1e26/sp.constants.e
            #q,dE=Reduction.determine_q_dE(kfvec,raw_data['data'].coords['tof'][hi].values-t,instparamfile)
            hw.append(dE)
            qvec.append(q)
        raw_data['data'].coords.update({'kfx':sc.array(dims=['event'],values=np.array([k[0][0] for k in kf_tof])),
                                        'kfy':sc.array(dims=['event'],values=np.array([k[0][1] for k in kf_tof])),
                                        'kfz':sc.array(dims=['event'],values=np.array([k[0][2] for k in kf_tof])),
                                        'tof_sad':sc.array(dims=['event'],values=[k[1] for k in kf_tof], unit='s'),
                                        'qx':sc.array(dims=['event'],values=np.array([q[0] for q in qvec])),
                                        'qy':sc.array(dims=['event'],values=np.array([q[1] for q in qvec])),
                                        'qz':sc.array(dims=['event'],values=np.array([q[2] for q in qvec])),
                                        'qval':sc.array(dims=['event'],values=np.array([np.linalg.norm(q) for q in qvec]),unit="1/Angstrom"),
                                        'hw':sc.array(dims=['event'],values=hw,unit='ueV')
                                        })
        raw_data['treatment']+='; coordinates transformed'
        return Reduction.convertedUnits(raw_data)
    

    @staticmethod
    def norm_subtract_sample(
        dat: "Reduction.BinnedData[Reduction.Sample]",
        bkg: "Reduction.BinnedData[Reduction.Background]",
        vana: "Reduction.VanadiumPeak",
        param: "Reduction.ScaleFactor",
        Reductionfiles: "Reduction.Reductionfiles"
    ) -> "Reduction.BackgroundSubtractedData":
        """Subtract scaled background, Normalize data using vanadium. Afterwards, apply mask"""
        data={key:dat[key] for key in dat.keys()}
        data['data']=dat['data'].bins.concatenate(-param*bkg['data'])
        data['treatment']+='; subtract scaled background'
        data['data']=data['data'] / sc.array(dims=['qval'], values=vana['integratedfit'])
        data['treatment']+='; vanadium normalized'
        centers=sc.array(dims=['event'],values=[gf['center'][0] for gf in vana['gaussfits']],unit='ueV')
        data['data'].coords['hw']-=sc.mean(centers)
        if debugging: print(f'center by mean value of vanadium peaks of q values')
        data['treatment']+=f'; centered based on vanadium by "{sc.mean(centers).values}"'
        # TODO masking
        return Reduction.BackgroundSubtractedData(
            data
        )
    #%%----------------------------
    # Pipeline Management
    # ----------------------------

    def create_pipeline(self):
        """Create or recreate the Sciline pipeline."""
        self.pipeline = sciline.Pipeline(self.providers, params=self.params)
        self.graph = None
        self.result = None
        return self.pipeline

    def run(self) -> "Reduction.BackgroundSubtractedData":
        """Execute the pipeline and return the result and possible intermediates which are defined via save_intermediate."""
        debugging_figures['Peakfinding']=False
        if self.pipeline==None:
            self.pipeline=self.create_pipeline()
        # save requested intermediate steps
        if len(self.intermediate)!=0:
            for itm,val in self.intermediate.items():
                evitem=eval('self.'+itm.replace('[','[self.'))
                if len(self.intermediate[itm])==0:
                    self.intermediate[itm].append(copy.deepcopy(self.pipeline.compute(evitem)))
                else:
                    self.intermediate[itm][0]=copy.deepcopy(self.pipeline.compute(evitem))
                self.pipeline[evitem]=copy.deepcopy(self.intermediate[itm][0])
                #print(f"{itm} complete")
        # calculate now final result
        self.result = self.pipeline.compute(self.BackgroundSubtractedData)
        return self.result

    def visualize(self, filename: str | None = None, format:str='pdf', **kwargs):
        """
        Visualize the pipeline graph. 
        Can be saved as file by providing a filename. 
        Format is provided individually with format=... (default: pdf)
        """
        if not self.pipeline:
            self.create_pipeline()
        if not self.graph:
            self.graph = self.pipeline.get(self.BackgroundSubtractedData)

        g = self.graph.visualize(graph_attr={'rankdir': 'TB', **kwargs})
        if filename:
            g.render(filename, format=format, cleanup=True)
        return g

    def save_intermediate(self,interm):
        """
        provide framework that intermediate steps are saved. 
        Items which should be saved have to be provided as 
        -string e.g. 'BackgroundSubtractedData'
        -list of strings e.g. ['BackgroundSubtractedData','mask']
        -as "Datatype(workspace)" e.g. "NormedData(Sample)"
        """
        if isinstance(interm,str):
            interm=[interm]
        if not isinstance(interm,list) or any(not isinstance(x, str) for x in interm):
            warnings.warn(f"Wrong input for save_intermediate: {interm}. Entry must be string or list of strings")
        #case 1: NewType domain
        newtype_names = [
            name for name, value in Reduction.__dict__.items()
            if callable(value) and hasattr(value, "__supertype__")
        ]
        #case 2 generic domain
        nested_classes = [
            name for name, value in Reduction.__dict__.items()
            if isinstance(value, type)
        ]
        for itm in interm:
            #case 1
            if itm in newtype_names:
                self.intermediate.update({itm:[]})
            # case 2: generic domains
            elif '[' in itm and itm.split("[")[0] in nested_classes:
                self.intermediate.update({itm:[]})
            else:
                warnings.warn(f'Parameter "{itm}" not found or incomplete')

    def get_intermediate(self, data_type):
        """
        Fetch intermediate data (e.g., NormedData[Sample]) from the pipeline. 
        Everything is calculated again!
        If data has been saved as intermediate result via save_intermediate, use values from reduction.intermediate to avoid calculating.
        """
        if not self.pipeline:
            self.create_pipeline()
        return self.pipeline.get(data_type).compute()