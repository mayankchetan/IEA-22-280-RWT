from openfast_io.FAST_output_reader import FASTOutputFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yaml
import numpy as np
import pandas as pd

def setOutputs(fst_vt):
    '''
    Set outputs for the FAST input file
    Requested data and units
    '''

    '''
    4.1.1 Loads and deflections
    '''

    # lets set the nodal outputs
    fst_vt['AeroDyn']['BldNd_BladesOut'] = 3
    fst_vt['AeroDyn']['BldNd_BlOutNd'] = 'All'
    fst_vt['BeamDyn']['BldNd_BlOutNd'] = 'All'


    # Fax : Axial force on the rotor [N].
    fst_vt['outlist']['AeroDyn']['RtAeroFxh'] = 'True'
    fst_vt['outlist']['ElastoDyn']['RotThrust'] = 'True'

    # Torque: Aerodynamic torque of the rotor [Nm].
    fst_vt['outlist']['AeroDyn']['RtAeroMxh'] = 'True'
    fst_vt['outlist']['ElastoDyn']['RotTorq'] = 'True'

    # For a minimum of 10 (preferably more) radial locations (starting from rotor center at r=0 onwards):
    # Deflflap [m] Flapwise deflection in the direction of the rotorplane coordinate system from 
    # Fig. 4 (positive pointing downwind). Note that deflection is 0 for case V3.1
    fst_vt['outlist']['BeamDyn_Nodes']['TDxr'] = 'True'
    

    # Defledge [m] Edgewise deflection in the direction of the rotor plane coordinate system from 
    # Fig. 4 (positive pointing into the rotational direction ). Note that deflection is 0 for case V3.1
    fst_vt['outlist']['BeamDyn_Nodes']['TDyr'] = 'True'

    # Detors [deg] Torsional deformation wrt the local deformed blade axis (positive nose point up). 
    # We define this as the last angle in the “x-y-z sequence of rotation’’ when we go from the undeflected 
    # to a deflected blade section. It would be the torsion around dzB from Fig. 5 in case the blade would be 
    # straight and undeflected. Note that the deflection is 0 for case V3.1
    fst_vt['outlist']['BeamDyn_Nodes']['RDxr'] = 'True'
    fst_vt['outlist']['BeamDyn_Nodes']['RDyr'] = 'True'
    fst_vt['outlist']['BeamDyn_Nodes']['RDzr'] = 'True'
    

    # Fn_c (i.e. the aerodynamic force normal to the local chord, positive pointing in downwind direction, 
    # i.e. in x-direction from Fig. 6) [N/m]
    fst_vt['outlist']['AeroDyn_Nodes']['Fn'] = 'True'


    # Ft_c (i.e. the aerodynamic force parallel (tangential) to the local chord, positive pointing from 
    # trailing to leading edge, i.e. in - y-direction from Fig. 6) [N/m].
    fst_vt['outlist']['AeroDyn_Nodes']['Ft'] = 'True'


    # Fn_r (i.e. the aerodynamic force normal to the rotor plane, positive pointing in downwind direction, 
    # i.e. in XR direction from Fig. 4) [N/m]
    fst_vt['outlist']['AeroDyn_Nodes']['Fxp'] = 'True'

    # Ft_r (i.e. the aerodynamic force parallel (tangential) to the rotor plane, positive pointing in rotational 
    # direction i.e. in -YR direction from Fig. 4) [N/m].
    fst_vt['outlist']['AeroDyn_Nodes']['Fyp'] = 'True'


    '''
    4.1.2 Lifting line variables
    '''

    # For a minimum of 10 (preferably more) radial locations:
    # Veff (i.e. the resultant incoming velocity at the blade section) [m/s]
    fst_vt['outlist']['AeroDyn_Nodes']['VRel'] = 'True'

    # alpha, Angle of attack [deg]
    fst_vt['outlist']['AeroDyn_Nodes']['Alpha'] = 'True'

    # ui, the local axial induced velocity [m/s]
    # The sign is positive pointing in upwind direction.
    fst_vt['outlist']['AeroDyn_Nodes']['Uin'] = 'True'
    fst_vt['outlist']['AeroDyn_Nodes']['Vindxp'] = 'True'  

    # utan, the local velocity induced in tangential (rotational) direction [m/s]
    # The sign is positive pointing opposite to the direction of rotation.
    fst_vt['outlist']['AeroDyn_Nodes']['Uit'] = 'True'
    fst_vt['outlist']['AeroDyn_Nodes']['Vindyp'] = 'True'

    # cn aerodynamic normal force coefficient = Fn_c/(0.5 rho Veff2 c) [-]
    # Orientation in agreement with Fn_c from loads (section 4.1.1)
    fst_vt['outlist']['AeroDyn_Nodes']['Cn'] = 'True'

    # ct aerodynamic tangential force coefficient = Ft_c(0.5 rho Veff2 c) [-]
    # Orientation in agreement with Ft_c from loads (section 4.1.1)
    fst_vt['outlist']['AeroDyn_Nodes']['Ct'] = 'True'

    return fst_vt


def runOpenFAST(fast_main_file, OPENFAST_EXE):
    '''
    Run OpenFAST with the given main file path
    '''
    if not os.path.exists(fast_main_file):
        raise FileNotFoundError(f"FAST main file '{fast_main_file}' does not exist.")
    if not os.path.exists(OPENFAST_EXE):
        raise FileNotFoundError(f"OpenFAST executable '{OPENFAST_EXE}' does not exist.")
    
    # Create log file name by removing '.fst' extension and adding '.log'
    log_file = fast_main_file[:-4] + '.log'
    
    # Command to run OpenFAST and redirect both stdout and stderr to the log file
    command = f"{OPENFAST_EXE} {fast_main_file} > {log_file} 2>&1"

    os.system(command)


def RotDef(rx, ry, rz):

    # RotDef.py Converts Wiener-Milenkovic parameters to rotation matrix
    #
    # Converted from Matlab:
    # wieMilToR.m, Winstroth 08 July 2016, Version 1.0
    # by Evan Gaertner 17 June, 2020
    #
    # This function converts the Wiener-Milenkovic rotation parameters used
    # by NREL OpenFAST to a rotation matrix and also to Tait-Bryan angles
    # with the rotation sequence x-y'-z''.
    # ---------------------------------------------------------------------
    # Input
    # rx : x-component of the Wiener-Milenkovic parameter
    # ry : y-component of the Wiener-Milenkovic parameter
    # rz : z-component of the Wiener-Milenkovic parameter
    # ---------------------------------------------------------------------
    # Output
    # R  3x3 Rotation matrix that corresponds to the Wiener-Milenkovic 
    #    parameters [rx; ry; rz]
    #
    # Xphi  Rotation angle of the first rotation about the x-axis
    #
    # Ytheta  Rotation angle of the second rotation about the y'-axis
    #
    # Zpsi  Rotation angle of the third rotation about the z''-axis
    # ---------------------------------------------------------------------

    # Create Wiener-Milenkovic vector
    c = np.row_stack((rx, ry, rz))
    
    # Convert to rotation matrix
    c0 = 2.0 - 1.0/8.0*np.matmul(np.transpose(c), c)
    # c0 = float(c0[0])

    R = np.zeros((3, 3))

    R[0,0] = (c0**2 + rx**2 - ry**2 - rz**2)
    R[0,1] = (2.0*(rx*ry - c0*rz))
    R[0,2] = (2.0*(rx*rz + c0*ry))

    R[1,0] = (2.0*(rx*ry + c0*rz))
    R[1,1] = (c0**2 - rx**2 + ry**2 - rz**2)
    R[1,2] = (2.0*(ry*rz - c0*rx))

    R[2,0] = (2.0*(rx*rz - c0*ry))
    R[2,1] = (2.0*(ry*rz + c0*rx))
    R[2,2] = (c0**2 - rx**2 - ry**2 + rz**2)

    sf     = 1.0/(4.0 - c0)**2
    R      = sf * R

    Xphi   = np.degrees(np.arctan2(-R[1,2], R[2,2]))
    Ytheta = np.degrees(np.arcsin(R[0,2]))
    Zpsi   = np.degrees(np.arctan2(-R[0,1], R[0,0]))

    return Xphi, Ytheta, Zpsi

def processResults(RESULTS_FOLDER, fast_output_files, nodalR, results_string, startTime = 120.):
    '''
    Process the OpenFAST output file and save to a results file

    Parameters
    ----------
    fast_output_file : dict
        dictionary with keys v3.1 and v3.2, each containing the path to the OpenFAST output file for that case.
    nodalR : list
        List of nodal R locations for the AD and BD nodes. This is given in terms of rotor radius.
    results_string : str
        String to be used in the results file name, e.g. 'case_v3_1' or 'case_v3_2'.
    startTime : float, optional
        Start time for processing the data, by default 120.0 seconds.

        
    Notes from Task47:
    4.1.1 Loads and deflections

    Files to be supplied:
    
    Please supply the data in one ASCII file which should contain the data for the two cases

    Format: Each row contains 15 columns with data. Separate the columns by tabs or blanks. 
    The first row gives the identification of data. The second row gives the axial force for 
    the cases V3.1 and V3.2 and the third row gives the torque. Please duplicate the values for 
    Fax and Torque to the other columns of row 2 and 3 since 7 columns are available for each case. 
    The next rows give the data for the two cases for the chosen radial locations 
    (a total of n with n >10), increasing from root to tip. The variable r is defined to start 
    at the rotor center (r=0m) increasing towards the blade tip. Hence, the format is as follows.

    Note that this makes the total number of rows to be n + 3 (1 (header) + 1 (axial force) + 1 (torque) + n radial locations)

    '''
    # So, looks like our BeamDyn and AeroDyn nodal points match with a max(abs(error)) of 2e-4, i.e 0.2 mm for a 138m blade!
    # Create a DataFrame for the results, index is Fax, Torque, and the radial locations (nodalR)
    results_df_loads = pd.DataFrame(columns=['Fn_c1.1', 'Ft_c1.1', 'Fn_r1.1', 'Ft_r1.1', 'Deflflap1.1', 'Defledge1.1', 'Detors1.1',
                                        'Fn_c1.2', 'Ft_c1.2', 'Fn_r1.2', 'Ft_r1.2', 'Deflflap1.2', 'Defledge1.2', 'Detors1.2'],
                                        index = ['Fax', 'Torque'] + [f'r_{r:.2f}' for r in nodalR],
                                    dtype=float)
    
    results_df_lifting_line = pd.DataFrame(columns=['Veff1.1', 'Alpha1.1', 'ui1.1', 'utan1.1', 'cn1.1', 'ct1.1',
                                                'Veff1.2', 'Alpha1.2', 'ui1.2', 'utan1.2', 'cn1.2', 'ct1.2'],
                                                index = [f'r_{r:.2f}' for r in nodalR],
                                            dtype=float)

    of_out = {}

    for case, fast_output_file in fast_output_files.items():
        if not os.path.exists(fast_output_file):
            raise FileNotFoundError(f"FAST output file '{fast_output_file}' does not exist.")

        # Read the OpenFAST output file, and prune the data
        fst_output = FASTOutputFile(fast_output_file)
        data_df = fst_output.toDataFrame()

        of_out[case] = data_df

        data_df = data_df[data_df['Time_[s]'] >= startTime]  # Prune data to start from startTime

        #### Now we start processing the data, and time average it

        # Fax : Axial force on the rotor [N]. 
        if case == 'v3.1': # fill 7 columns with the same value
            results_df_loads.iloc[0, :7] = data_df['RtAeroFxh_[N]'].mean()  # or data_df['RotThrust'].mean()
        elif case == 'v3.2': # fill 7 columns with the same value
            results_df_loads.iloc[0, 7:] = data_df['RtAeroFxh_[N]'].mean()

        # results_df_loads.loc['Fax', :] = data_df['RtAeroFxh_[N]'].mean() # or data_df['RotThrust'].mean()

        # Torque: Aerodynamic torque of the rotor [Nm].
        if case == 'v3.1': # fill 1st 7 columns with the same value
            results_df_loads.iloc[1, :7] = data_df['RtAeroMxh_[N-m]'].mean()
        elif case == 'v3.2': # fill next 7 columns with the same value
            results_df_loads.iloc[1, 7:] = data_df['RtAeroMxh_[N-m]'].mean()
        # results_df_loads.loc['Torque', :] = data_df['RtAeroMxh_[N-m]'].mean() # or data_df['RotTorq'].mean()

        # Looping over the radial locations
        # For a minimum of 10 (preferably more) radial locations (starting from rotor center at r=0 onwards):
        for i, r in enumerate(nodalR):

            # Deflflap [m] Flapwise deflection in the direction of the rotorplane coordinate system from 
            # Fig. 4 (positive pointing downwind). Note that deflection is 0 for case V3.1
            results_df_loads.loc[f'r_{r:.2f}', 'Deflflap1.'+case[-1]] = data_df[f'B1N{i+1:03d}_TDxr_[m]'].mean() if f'B1N{i+1:03d}_TDxr_[m]' in data_df.columns else 0. 


            # Defledge [m] Edgewise deflection in the direction of the rotor plane coordinate system from 
            # Fig. 4 (positive pointing into the rotational direction ). Note that deflection is 0 for case V3.1
            results_df_loads.loc[f'r_{r:.2f}', 'Defledge1.'+case[-1]] = data_df[f'B1N{i+1:03d}_TDyr_[m]'].mean() if f'B1N{i+1:03d}_TDyr_[m]' in data_df.columns else 0.

            # Detors [deg] Torsional deformation wrt the local deformed blade axis (positive nose point up). 
            # We define this as the last angle in the “x-y-z sequence of rotation’’ when we go from the undeflected 
            # to a deflected blade section. It would be the torsion around dzB from Fig. 5 in case the blade would be 
            # straight and undeflected. Note that the deflection is 0 for case V3.1

            # TEMP, NEED TO DO THE WM 2 Rad 2 Deg calculations!!!
            results_df_loads.loc[f'r_{r:.2f}', 'Detors1.'+case[-1]] = np.rad2deg(data_df[f'B1N{i+1:03d}_RDzr_[-]'].mean()) if f'B1N{i+1:03d}_RDzr_[-]' in data_df.columns else 0.


            # Fn_c (i.e. the aerodynamic force normal to the local chord, positive pointing in downwind direction, 
            # i.e. in x-direction from Fig. 6) [N/m]
            results_df_loads.loc[f'r_{r:.2f}', 'Fn_c1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Fn_[N/m]'].mean() if f'AB1N{i+1:03d}Fn_[N/m]' in data_df.columns else 0.


            # Ft_c (i.e. the aerodynamic force parallel (tangential) to the local chord, positive pointing from 
            # trailing to leading edge, i.e. in - y-direction from Fig. 6) [N/m].
            results_df_loads.loc[f'r_{r:.2f}', 'Ft_c1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Ft_[N/m]'].mean() if f'AB1N{i+1:03d}Ft_[N/m]' in data_df.columns else 0.


            # Fn_r (i.e. the aerodynamic force normal to the rotor plane, positive pointing in downwind direction, 
            # i.e. in XR direction from Fig. 4) [N/m]
            results_df_loads.loc[f'r_{r:.2f}', 'Fn_r1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Fxp_[N/m]'].mean() if f'AB1N{i+1:03d}Fxp_[N/m]' in data_df.columns else 0.

            # Ft_r (i.e. the aerodynamic force parallel (tangential) to the rotor plane, positive pointing in rotational 
            # direction i.e. in -YR direction from Fig. 4) [N/m].
            results_df_loads.loc[f'r_{r:.2f}', 'Ft_r1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Fyp_[N/m]'].mean() if f'AB1N{i+1:03d}Fyp_[N/m]' in data_df.columns else 0.


            '''
            4.1.2 Lifting line variables
            '''

            # For a minimum of 10 (preferably more) radial locations:
            # Veff (i.e. the resultant incoming velocity at the blade section) [m/s]
            results_df_lifting_line.loc[f'r_{r:.2f}', 'Veff1.'+case[-1]] = data_df[f'AB1N{i+1:03d}VRel_[m/s]'].mean() if f'AB1N{i+1:03d}VRel_[m/s]' in data_df.columns else 0.

            # alpha, Angle of attack [deg]    
            results_df_lifting_line.loc[f'r_{r:.2f}', 'Alpha1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Alpha_[deg]'].mean() if f'AB1N{i+1:03d}Alpha_[deg]' in data_df.columns else 0.

            # ui, the local axial induced velocity [m/s]
            # The sign is positive pointing in upwind direction.
            results_df_lifting_line.loc[f'r_{r:.2f}', 'ui1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Uin_[m/s]'].mean() if f'AB1N{i+1:03d}Uin_[m/s]' in data_df.columns else 0. # Vindxp


            # utan, the local velocity induced in tangential (rotational) direction [m/s]
            # The sign is positive pointing opposite to the direction of rotation.
            results_df_lifting_line.loc[f'r_{r:.2f}', 'utan1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Uit_[m/s]'].mean() if f'AB1N{i+1:03d}Uit_[m/s]' in data_df.columns else 0. # Vindyp


            # cn aerodynamic normal force coefficient = Fn_c/(0.5 rho Veff2 c) [-]
            # Orientation in agreement with Fn_c from loads (section 4.1.1)
            results_df_lifting_line.loc[f'r_{r:.2f}', 'cn1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Cn_[-]'].mean() if f'AB1N{i+1:03d}Cn_[-]' in data_df.columns else 0.

            # ct aerodynamic tangential force coefficient = Ft_c(0.5 rho Veff2 c) [-]
            # Orientation in agreement with Ft_c from loads (section 4.1.1)
            results_df_lifting_line.loc[f'r_{r:.2f}', 'ct1.'+case[-1]] = data_df[f'AB1N{i+1:03d}Ct_[-]'].mean() if f'AB1N{i+1:03d}Ct_[-]' in data_df.columns else 0.


    # Now we have the results for both cases, we can save them to a CSV file
    results_df_loads.to_csv(os.path.join(RESULTS_FOLDER, f'{results_string}_loads.csv'), index=True, sep='\t')
    results_df_lifting_line.to_csv(os.path.join(RESULTS_FOLDER, f'{results_string}_lifting_line.csv'), index=True, sep='\t')

    return results_df_loads, results_df_lifting_line, of_out


def reporting(results_df_loads, results_df_lifting_line, of_out, REPORT_FOLDER, vtk=False, vtk_folder=None):
    '''
    Generate reports for the results
    
    Parameters
    ----------
    results_df_loads : DataFrame or dict of DataFrames
        DataFrame(s) containing load results for different models (BEM, OLAF)
    results_df_lifting_line : DataFrame or dict of DataFrames
        DataFrame(s) containing lifting line results for different models
    of_out : dict or dict of dicts
        Dictionary containing OpenFAST output data
    REPORT_FOLDER : str
        Path to folder where reports will be saved
    '''

    # Check if inputs are dictionaries or single DataFrames/dicts
    is_dict_loads = isinstance(results_df_loads, dict)
    is_dict_lifting_line = isinstance(results_df_lifting_line, dict)
    is_dict_of_out = isinstance(next(iter(of_out.values())) if of_out else None, dict)

    # Convert single DataFrames to dictionaries for uniform handling
    if not is_dict_loads:
        results_df_loads = {'BEM': results_df_loads}
    if not is_dict_lifting_line:
        results_df_lifting_line = {'BEM': results_df_lifting_line}
    if not is_dict_of_out:
        of_out = {'BEM': of_out}

    # Channels for time series
    chansMapping = {
            'Axial Thrust (F_ax) [N]': ['RtAeroFxh_[N]'],
            'Torque [N-m]': ['RtAeroMxh_[N-m]'],
            'Deflection Flap (Defl_flap) [m]': [f'B1N{chanEnum:03d}_TDxr_[m]' for chanEnum in range(1,60)],
            'Deflection Edge (Defl_edge) [m]': [f'B1N{chanEnum:03d}_TDyr_[m]' for chanEnum in range(1,60)],
            'Blade Torsion (DeTors) [deg]': [f'B1N{chanEnum:03d}_RDzr_[-]' for chanEnum in range(1,60)],
            'aerodynamic force normal, local chord (Fn_c) [N/m]': [f'AB1N{chanEnum:03d}Fn_[N/m]' for chanEnum in range(1,60)],
            'aerodynamic force parallel, local chord (Ft_c) [N/m]': [f'AB1N{chanEnum:03d}Ft_[N/m]' for chanEnum in range(1,60)],
            'aerodynamic force normal, rotor plane (Fn_r) [N/m]': [f'AB1N{chanEnum:03d}Fxp_[N/m]' for chanEnum in range(1,60)],
            'aerodynamic force parallel, rotor plane (Ft_r) [N/m]': [f'AB1N{chanEnum:03d}Fyp_[N/m]' for chanEnum in range(1,60)],
            'resultant incoming velocity (Veff) [m/s]': [f'AB1N{chanEnum:03d}VRel_[m/s]' for chanEnum in range(1,60)],
            'Angle of attack (Alpha) [deg]': [f'AB1N{chanEnum:03d}Alpha_[deg]' for chanEnum in range(1,60)],
            'local axial induced velocity (ui) [m/s]': [f'AB1N{chanEnum:03d}Uin_[m/s]' for chanEnum in range(1,60)],
            'local velocity induced in tangential (utan) [m/s]': [f'AB1N{chanEnum:03d}Uit_[m/s]' for chanEnum in range(1,60)],
            'aerodynamic normal force coefficient (cn) [-]': [f'AB1N{chanEnum:03d}Cn_[-]' for chanEnum in range(1,60)],
            'aerodynamic tangential force coefficient (ct) [-]': [f'AB1N{chanEnum:03d}Ct_[-]' for chanEnum in range(1,60)],
    }

    # Channel mapping for the loads and lifting line variables
    chansMapping_loads = {
        # 'Axial Thrust (F_ax)': ['Fax'],   # This is to be included as tables
        # 'Torque': ['Torque'],
        'Deflection Flap (Defl_flap) [m]': [f'Deflflap1.{chanEnum}' for chanEnum in ['1', '2']],
        'Deflection Edge (Defl_edge) [m]': [f'Defledge1.{chanEnum}' for chanEnum in ['1', '2']],
        'Blade Torsion (DeTors) [deg]': [f'Detors1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic force normal, local chord (Fn_c) [N/m]': [f'Fn_c1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic force parallel, local chord (Ft_c) [N/m]': [f'Ft_c1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic force normal, rotor plane (Fn_r) [N/m]': [f'Fn_r1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic force parallel, rotor plane (Ft_r) [N/m]': [f'Ft_r1.{chanEnum}' for chanEnum in ['1', '2']],
    }
    chansMapping_lifting_line = {
        'resultant incoming velocity (Veff) [m/s]': [f'Veff1.{chanEnum}' for chanEnum in ['1', '2']],
        'Angle of attack (Alpha) [deg]': [f'Alpha1.{chanEnum}' for chanEnum in ['1', '2']],
        'local axial induced velocity (ui) [m/s]': [f'ui1.{chanEnum}' for chanEnum in ['1', '2']],
        'local velocity induced in tangential (utan) [m/s]': [f'utan1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic normal force coefficient (cn) [-]': [f'cn1.{chanEnum}' for chanEnum in ['1', '2']],
        'aerodynamic tangential force coefficient (ct) [-]': [f'ct1.{chanEnum}' for chanEnum in ['1', '2']],
    }

    # Descriptions for the load variables based on comments in the code
    loads_descriptions = {
        'Deflection Flap (Defl_flap) [m]': 'Flapwise deflection in the direction of the rotorplane coordinate system from Fig. 4 (positive pointing downwind). Note that deflection is 0 for case V3.1',
        'Deflection Edge (Defl_edge) [m]': 'Edgewise deflection in the direction of the rotor plane coordinate system from Fig. 4 (positive pointing into the rotational direction ). Note that deflection is 0 for case V3.1',
        'Blade Torsion (DeTors) [deg]': 'Torsional deformation wrt the local deformed blade axis (positive nose point up). We define this as the last angle in the `x-y-z sequence of rotation` when we go from the undeflected to a deflected blade section. It would be the torsion around dzB from Fig. 5 in case the blade would be straight and undeflected. Note that the deflection is 0 for case V3.1',
        'aerodynamic force normal, local chord (Fn_c) [N/m]': '(i.e. the aerodynamic force normal to the local chord, positive pointing in downwind direction, i.e. in x-direction from Fig. 6) [N/m]',
        'aerodynamic force parallel, local chord (Ft_c) [N/m]': '(i.e. the aerodynamic force parallel (tangential) to the local chord, positive pointing from trailing to leading edge, i.e. in - y-direction from Fig. 6) [N/m].',
        'aerodynamic force normal, rotor plane (Fn_r) [N/m]': '(i.e. the aerodynamic force normal to the rotor plane, positive pointing in downwind direction, i.e. in XR direction from Fig. 4) [N/m]',
        'aerodynamic force parallel, rotor plane (Ft_r) [N/m]': '(i.e. the aerodynamic force parallel (tangential) to the rotor plane, positive pointing in rotational direction i.e. in -YR direction from Fig. 4) [N/m].'
    }

    # Descriptions for the lifting line variables based on comments in the code
    lifting_line_descriptions = {
        'resultant incoming velocity (Veff) [m/s]': '(i.e. the resultant incoming velocity at the blade section) [m/s]',
        'Angle of attack (Alpha) [deg]': 'Angle of attack in degrees.',
        'local axial induced velocity (ui) [m/s]': 'The local axial induced velocity. The sign is positive pointing in upwind direction.',
        'local velocity induced in tangential (utan) [m/s]': 'The local velocity induced in tangential (rotational) direction. The sign is positive pointing opposite to the direction of rotation.',
        'aerodynamic normal force coefficient (cn) [-]': 'Aerodynamic normal force coefficient = Fn_c/(0.5 rho Veff^2 c).',
        'aerodynamic tangential force coefficient (ct) [-]': 'Aerodynamic tangential force coefficient = Ft_c/(0.5 rho Veff^2 c).'
    }

    # Create time series HTML header
    html_time_series = """
    <html>
    <head>
        <title>IEA Wind Task 47 Case v3.x</title>
        <style>
            h1, p {
                text-align: center;
            }
            .description {
                margin: 10px auto;
                max-width: 800px;
                text-align: center;
                font-style: italic;
                color: #555;
            }
        </style>
    </head>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    <body>
        <h1>IEA Wind Task 47 Case v3.x - Time series plots</h1>
        <p>This report contains the time series plots of the IEA-22-280-RWT-Monopile turbine for IEA Wind Task 47 cases v3.1 and v3.2.</p>
    """
    if is_dict_of_out and len(of_out) > 1:
        html_time_series += f'<p>Models included: {", ".join(of_out.keys())}</p>'
    html_time_series += """
    </body>
    </html>
    """

    # Dictionary to store colors for each node position and models
    node_colors = {0: 'blue', 1: 'red', 2: 'green'}
    model_markers = {'BEM': 'circle', 'OLAF': 'triangle-up'}

    for chan_name, chan_list in chansMapping.items():
        # Create a single plot for all cases and models
        fig = go.Figure()
        
        # Determine which nodes to display
        if chan_name in ['Axial Thrust (F_ax) [N]', 'Torque [N-m]']:
            selected_nodes = [0]
        else:
            num_nodes = len(chan_list)
            selected_nodes = [0, num_nodes // 2, num_nodes - 1]
        
        # Loop through each model
        for model_name, model_data in of_out.items():
            # Loop through the cases
            for case_idx, (case, data_df) in enumerate(model_data.items()):
                # Line style based on case
                line_style = 'solid' if case == 'v3.1' else 'dot'
                
                for node_idx, node in enumerate(selected_nodes):
                    chan = chan_list[node]
                    if chan in data_df.columns:
                        node_description = 'Root' if node == 0 else ('Middle' if node == len(chan_list) // 2 else 'Tip')
                        fig.add_trace(
                            go.Scatter(
                                x=data_df['Time_[s]'], 
                                y=data_df[chan],
                                mode='lines',
                                name=f'{model_name} - Case {case} - {node_description}',
                                line=dict(
                                    width=1.5,
                                    color=node_colors[node_idx],
                                    dash=line_style
                                ),
                                marker=dict(symbol=model_markers.get(model_name, 'circle')),
                            )
                        )
                    else:
                        print(f"Warning: Channel '{chan}' not found in data for {model_name} case '{case}'. Skipping.")

        # Update layout
        fig.update_layout(
            title_text=f'Time Series for {chan_name}',
            xaxis_title='Time [s]',
            yaxis_title=chan_name,
            height=600,
            width=1000,
            legend=dict(
                title='Models, Cases and Nodes',
                orientation='v',  # Vertical orientation
                yanchor='middle',
                y=0.5,  # Centered vertically
                xanchor='left',
                x=1.02  # Positioned to the right of the plot
            )
        )

        # Save the figure to HTML
        html_time_series += '<div style="margin-bottom: 20px; display: flex; justify-content: center; flex-direction: column; align-items: center;">'
        html_time_series += f'<h2 style="text-align: center;">{chan_name}</h2>'
        html_time_series += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_time_series += '</div>'
    
    # Save the HTML report
    with open(os.path.join(REPORT_FOLDER, 'time_series_report.html'), 'w') as f:
        f.write(html_time_series)
    print("Time series report generated successfully.")
    # Create a single HTML report for radial plots and comparisons
    html_report = """
    <html>
    <head>
        <title>IEA Wind Task 47 Case v3.x - Radial Analysis</title>
        <style>
            h1, h2, h3, p {
                text-align: center;
            }
            .description {
                margin: 10px auto;
                max-width: 800px;
                text-align: center;
                font-style: italic;
                color: #555;
            }
            table {
                border-collapse: collapse;
                margin: 20px auto;
            }
            th, td {
                padding: 8px 15px;
                text-align: center;
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            .plot-container {
                margin-bottom: 30px;
                display: flex;
                justify-content: center;
                flex-direction: column;
                align-items: center;
            }
        </style>
    </head>
    <body>
        <h1>IEA Wind Task 47 Case v3.x - Radial Analysis</h1>
        <p>This report contains the radial analysis of the IEA-22-280-RWT-Monopile turbine for IEA Wind Task 47 cases v3.1 and v3.2.</p>
    """
    
            # lets make a different HTML for the VTK plots
    html_vtk = """
        <html>
        <head>
            <title>IEA Wind Task 47 Case v3.x - VTK Visualization</title>
            <style>
                h1, h2, p {
                    text-align: center;
                }
                .description {
                    margin: 10px auto;
                    max-width: 800px;
                    text-align: center;
                    font-style: italic;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <h1>IEA Wind Task 47 Case v3.x - VTK Visualization</h1>
            <p>This report contains the VTK visualization of the IEA-22-280-RWT-Monopile turbine for IEA Wind Task 47 cases v3.1 and v3.2.</p>
        """


    # Add models included if we have multiple models
    is_multi_model = is_dict_loads and len(results_df_loads) > 1
    if is_multi_model:
        html_report += f'<p>Models included: {", ".join(results_df_loads.keys())}</p>'
    
    # Add the loads summary table
    html_report += '<h2>Loads Summary</h2>'
    html_report += '<div class="description">Axial forces and torque summary for all cases and models.</div>'
    html_report += '<div style="display: flex; justify-content: center;">'
    html_report += '<table><tr><th>Model</th><th>Variable</th><th>Case v3.1</th><th>Case v3.2</th></tr>'
    
    # Add data for each model
    for model_name, df_loads in results_df_loads.items() if is_dict_loads else {'BEM': results_df_loads}.items():
        if "Fax" in df_loads.index and "Torque" in df_loads.index:
            html_report += f'<tr><td rowspan="2">{model_name}</td><td>Fax</td>'
            html_report += f'<td>{df_loads.loc["Fax", "Fn_c1.1"]:.2f} N</td>'
            html_report += f'<td>{df_loads.loc["Fax", "Fn_c1.2"]:.2f} N</td></tr>'
            html_report += f'<tr><td>Torque</td>'
            html_report += f'<td>{df_loads.loc["Torque", "Fn_c1.1"]:.2f} Nm</td>'
            html_report += f'<td>{df_loads.loc["Torque", "Fn_c1.2"]:.2f} Nm</td></tr>'
    
    html_report += '</table>'
    html_report += '</div>'
    
    # Define model styles
    model_colors = {'BEM': 'blue', 'OLAF': 'red'}
    model_markers = {'BEM': 'circle', 'OLAF': 'triangle-up'}
    case_styles = {'1': 'solid', '2': 'dot'}
    
    # Create radial plots sections
    # First for loads
    html_report += '<h2>Loads Radial Analysis</h2>'
    html_report += '<div class="description">Loads include blade deflections and aerodynamic forces along the blade span.</div>'
    
    for variable_name, columns in chansMapping_loads.items():
        fig = go.Figure()
        
        # Create a combined plot with all models
        for model_name, df_loads in results_df_loads.items() if is_dict_loads else {'BEM': results_df_loads}.items():
            df_loads_plot = df_loads.copy()
            # Remove Fax and Torque rows for plotting
            if "Fax" in df_loads_plot.index and "Torque" in df_loads_plot.index:
                df_loads_plot = df_loads_plot.drop(['Fax', 'Torque'])
            
            # Extract radial locations
            radial_locations = [float(r.split('_')[1]) for r in df_loads_plot.index if r.startswith('r_')]
            
            # Add traces for each case
            for case_idx, (case, column) in enumerate(zip(['1', '2'], columns)):
                if column in df_loads_plot.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=radial_locations,
                            y=df_loads_plot[column],
                            mode='lines+markers',
                            name=f'{model_name} - Case v3.{case}',
                            line=dict(
                                width=1.5, 
                                dash=case_styles[case],
                                color=model_colors.get(model_name, 'green')
                            ),
                            marker=dict(
                                symbol=model_markers.get(model_name, 'circle'),
                                color=model_colors.get(model_name, 'green')
                            ),
                        )
                    )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Radial Location [m]',
            yaxis_title=variable_name,
            height=600,
            width=1000,
            legend=dict(
                title='Model and Case',
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02
            )
        )
        
        # Add to HTML
        html_report += '<div class="plot-container">'
        html_report += f'<h3>{variable_name}</h3>'
        
        # Add description
        if variable_name in loads_descriptions:
            html_report += f'<div class="description">{loads_descriptions[variable_name]}</div>'
        
        html_report += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_report += '</div>'
    
    # Then for lifting line variables
    html_report += '<h2>Lifting Line Variables Analysis</h2>'
    html_report += '<div class="description">Lifting line variables include velocities, angles of attack, induced velocities, and aerodynamic force coefficients.</div>'
    
    for variable_name, columns in chansMapping_lifting_line.items():
        fig = go.Figure()
        
        # Create a combined plot with all models
        for model_name, df_lifting_line in results_df_lifting_line.items() if is_dict_lifting_line else {'BEM': results_df_lifting_line}.items():
            if df_lifting_line is None:
                continue
                
            # Extract radial locations
            radial_locations = [float(r.split('_')[1]) for r in df_lifting_line.index if r.startswith('r_')]
            
            # Add traces for each case
            for case_idx, (case, column) in enumerate(zip(['1', '2'], columns)):
                if column in df_lifting_line.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=radial_locations,
                            y=df_lifting_line[column],
                            mode='lines+markers',
                            name=f'{model_name} - Case v3.{case}',
                            line=dict(
                                width=1.5, 
                                dash=case_styles[case],
                                color=model_colors.get(model_name, 'green')
                            ),
                            marker=dict(
                                symbol=model_markers.get(model_name, 'circle'),
                                color=model_colors.get(model_name, 'green')
                            ),
                        )
                    )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Radial Location [m]',
            yaxis_title=variable_name,
            height=600,
            width=1000,
            legend=dict(
                title='Model and Case',
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02
            )
        )
        
        # Add to HTML
        html_report += '<div class="plot-container">'
        html_report += f'<h3>{variable_name}</h3>'
        
        # Add description
        if variable_name in lifting_line_descriptions:
            html_report += f'<div class="description">{lifting_line_descriptions[variable_name]}</div>'
        
        html_report += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_report += '</div>'
    

    # CREATE interactive pyvista view for the OLAF vtk files.

    if vtk and vtk_folder is not None:

        html_vtk += plotVTK(html_vtk, of_out, vtk_folder=vtk_folder)


    else:
        print("VTK visualization is not enabled or vtk_folder is not provided. Skipping VTK plots.")

    if vtk and vtk_folder is not None:
        html_vtk += "</body></html>"
        # Save the VTK HTML report
        with open(os.path.join(REPORT_FOLDER, 'vtk_visualization_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_vtk)

    # Close HTML
    html_report += "</body></html>"
    
    # Save the HTML report with UTF-8 encoding
    with open(os.path.join(REPORT_FOLDER, 'radial_analysis_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_report)
    print("Combined radial analysis report generated successfully.")

    return None


def plotVTK(html_report, of_out, vtk_folder=None):

    import pyvista as pv
    from pyvista import set_plot_theme
    import glob

    # # Set the plot theme
    # set_plot_theme('document')

    # Read the VTK files, the shared path to the highlevel folder, lets use the nested dict of of_out to
    # build the path to the vtk files, we will show only the OLAF ones seperate for each case, 
    # but place the first and last timestep next to each other.
    vtk_files_string = {}
    for model_name, model_data in of_out.items():
        if model_name == 'OLAF':
            for case, data_df in model_data.items():
                caseName = 'case_v3_1' if case == 'v3.1' else 'case_v3_2'
                vtk_files_string[case] = os.path.join(vtk_folder, 'OLAF', caseName, 'vtk_fvw', f'{caseName}_OLAF.FVW_Glb')

    # Loop over the cases
    for case, vtk_file in vtk_files_string.items():
        
        # creating html header for the vtk section
        html_report += f'<h2>OLAF VTK Mesh Visualization for Case {case}</h2>'
        # now create the plotter object 
        plotter = pv.Plotter( shape=(1, 2))  # Create a plotter with 1 row and 2 columns

        for timestep in ['000000000','999999999']:
            # lets swap the subplot if we are at the last timestep
            if timestep == '999999999':
                plotter.subplot(0, 1)

            # We need to use glob to get the paths 
            for file in glob.glob(f'{vtk_file}.*.{timestep}.vtk'):

                # Read the VTK file
                mesh = pv.read(file)
                
                # Add the mesh to the plotter
                plotter.add_mesh(mesh, show_edges=True, color='lightblue', opacity=0.5)

        # Show the plot in a separate window
        # plotter.show()

        # Save the plot to HTML
        html_report += '<div class="plot-container">'
        html_buffer = plotter.export_html(filename=None)
        html_report += html_buffer.getvalue()
        html_report += '</div>'

    return html_report

# Lets add a test for the VTK
if __name__ == "__main__":
    of_out = {'BEM': {'v3.1': pd.DataFrame(), 'v3.2': pd.DataFrame()},
              'OLAF': {'v3.1': pd.DataFrame(), 'v3.2': pd.DataFrame()}}
    
    # temp html report
    html_report = """
    <html>
    <head>
        <title>VTK Test</title>
        <style>
            h1, p {
                text-align: center;
            }
            .description {
                margin: 10px auto;
                max-width: 800px;
                text-align: center;
                font-style: italic;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>VTK Test</h1>
        <p>This is a test for the VTK plotting functionality.</p>
    """

    html_report = plotVTK(html_report, of_out, vtk_folder='_model_omp')