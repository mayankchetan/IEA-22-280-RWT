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
    # trailing to leading edge, i.e. in – y-direction from Fig. 6) [N/m].
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
            # trailing to leading edge, i.e. in – y-direction from Fig. 6) [N/m].
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


def reporting(results_df_loads, results_df_lifting_line, of_out, REPORT_FOLDER):
    '''
    Generate reports for the results
    '''

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
        'aerodynamic force parallel, local chord (Ft_c) [N/m]': '(i.e. the aerodynamic force parallel (tangential) to the local chord, positive pointing from trailing to leading edge, i.e. in – y-direction from Fig. 6) [N/m].',
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

    # Creating the time series plots, each chansMapping item is a separate plot, other than the first two, we do only 1st, middle and last nodes
    # We also have subplots for the two differnt cases, so we can compare them, and sharing x-axis, Since we will save this as HTML reports,
    # we should accumulate the figures. 

    # creating header and intro in HTML format
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
        <p>This report contains the radial plots of the IEA-22-280-RWT-Monopile tubine for IEA Wind Task 47 cases v3.1 and v3.2.</p>
    </body>
    </html>
    """
    # Dictionary to store colors for each node position
    node_colors = {0: 'blue', 1: 'red', 2: 'green'}

    for chan_name, chan_list in chansMapping.items():
        # Create a single plot for both cases
        fig = go.Figure()
        
        # Determine which nodes to display
        if chan_name in ['Axial Thrust (F_ax)', 'Torque']:
            selected_nodes = [0]
        else:
            num_nodes = len(chan_list)
            selected_nodes = [0, num_nodes // 2, num_nodes - 1]
        
        # Loop through the cases
        for case_idx, (case, data_df) in enumerate(of_out.items()):
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
                            name=f'Case {case} - {node_description}',
                            line=dict(
                                width=1.5,
                                color=node_colors[node_idx],
                                dash=line_style
                            ),
                        )
                    )
                else:
                    print(f"Warning: Channel '{chan}' not found in data for case '{case}'. Skipping.")

        # Update layout
        fig.update_layout(
            title_text=f'Time Series for {chan_name}',
            xaxis_title='Time [s]',
            yaxis_title=chan_name,
            height=600,
            width=1000,
            legend=dict(
            title='Cases and Nodes',
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


    # Now we can also generate the radial plots for the loads and lifting line variables,
    # This will be different, First we go over the colums of the results_df_loads and then results_df_lifting_line,
    # in results_df_loads, for Fax and Torque, we will have straight lines, and for the rest we will have radial plots,
    # here its important to extract the radial locations from the index, and also we have two different cases in each DF.
    # We'll plot them in the same figure, and not subplots.
    html_radial_plots = """
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
        <h1>IEA Wind Task 47 Case v3.x - Radial plots</h1>
        <p>This report contains the radial plots of the IEA-22-280-RWT-Monopile tubine for IEA Wind Task 47 cases v3.1 and v3.2.</p>
    </body>
    </html>
    """

    # Starting with the loads DataFrame, Lets just report the Fax and Torque as a table and not a plot
    html_radial_plots += '<h2 style="text-align: center;">Loads Radial Plots</h2>'
    html_radial_plots += '<div class="description">Loads include axial forces, torque, blade deflections and aerodynamic forces along the blade span.</div>'
    html_radial_plots += '<div style="display: flex; justify-content: center;">'
    html_radial_plots += '<table border="1" style="margin: 20px auto;"><tr><th>Variable</th><th>Case v3.1</th><th>Case v3.2</th></tr>'
    html_radial_plots += f'<tr><td>Fax</td><td>{results_df_loads.loc["Fax", "Deflflap1.1"]:.2f} N</td><td>{results_df_loads.loc["Fax", "Deflflap1.2"]:.2f} N</td></tr>'
    html_radial_plots += f'<tr><td>Torque</td><td>{results_df_loads.loc["Torque", "Deflflap1.1"]:.2f} Nm</td><td>{results_df_loads.loc["Torque", "Deflflap1.2"]:.2f} Nm</td></tr>'
    html_radial_plots += '</table>'
    html_radial_plots += '</div>'

    # we can delete the Fax and Torque rows from the results_df_loads, since we will not plot them
    results_df_loads = results_df_loads.drop(['Fax', 'Torque'])

    # we need to extract the radial locations from the index, but they are of the format r_{r:.2f}
    radial_locations = [float(r.split('_')[1]) for r in results_df_loads.index if r.startswith('r_')]

    # Loop through each dictionary for loads and lifting line separately
    for mapping, df, df_name, descriptions in zip([chansMapping_loads, chansMapping_lifting_line], 
                                  [results_df_loads, results_df_lifting_line], 
                                  ['Loads', 'Lifting Line'],
                                  [loads_descriptions, lifting_line_descriptions]):
        
        # Add section header with description for this group of variables
        if df_name == 'Lifting Line':
            html_radial_plots += '<h2 style="text-align: center;">Lifting Line Variables</h2>'
            html_radial_plots += '<div class="description">Lifting line variables include velocities, angles of attack, induced velocities, and aerodynamic force coefficients.</div>'
        
        # Loop through each variable type in the mapping
        for variable_name, columns in mapping.items():
            # Create a new figure for each variable type
            fig = go.Figure()
            
            # We know we have two cases: '1' and '2'
            for case, column in zip(['1', '2'], columns):
                if column in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=radial_locations,
                            y=df[column],
                            mode='lines+markers',
                            name=f'Case v3.{case}',
                            line=dict(width=1.5),
                        )
                    )
                else:
                    print(f"Warning: Column '{column}' not found in DataFrame '{df_name}'. Skipping.")
            
            # Update layout with proper title from the mapping
            fig.update_layout(
                # title=f'{df_name} - {variable_name}',
                xaxis_title='Radial Location [m]',
                yaxis_title=variable_name,
                height=600,
                width=1000,
                legend=dict(
                    title='Cases',
                    orientation='v',     # Change to vertical orientation
                    yanchor='middle',    # Anchor to middle of the legend
                    y=0.5,               # Center vertically
                    xanchor='left',      # Anchor to left side of the legend
                    x=1.02               # Position slightly right of the plot area
                )
            )
            
            # Save the figure to HTML
            html_radial_plots += '<div style="margin-bottom: 20px; display: flex; justify-content: center; flex-direction: column; align-items: center;">'
            html_radial_plots += f'<h2 style="text-align: center;">{df_name} - {variable_name}</h2>'
            
            # Add the description for this specific variable if it exists
            if variable_name in descriptions:
                html_radial_plots += f'<div class="description">{descriptions[variable_name]}</div>'
            
            html_radial_plots += fig.to_html(full_html=False, include_plotlyjs='cdn')
            html_radial_plots += '</div>'

    # Save the HTML report
    with open(os.path.join(REPORT_FOLDER, 'radial_plots_report.html'), 'w') as f:
        f.write(html_radial_plots)
    print("Radial plots report generated successfully.")

    return None