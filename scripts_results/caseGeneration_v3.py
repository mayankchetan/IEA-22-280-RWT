from openfast_io.FAST_reader import InputReader_OpenFAST
from openfast_io.FAST_writer import InputWriter_OpenFAST
import os
import copy
import sys
import multiprocessing

from utils import *

sys.path.append('../../welib/welib/fast')
from olaf import OLAFParams



BASE_FOLDER = '../OpenFAST/IEA-22-280-RWT-Monopile'
BASE_FILE = 'IEA-22-280-RWT-Monopile.fst'

OPENFAST_EXE = '../../openfast-dev-tc/build/glue-codes/openfast/openfast'

MODEL_FOLDER = '_model_omp'
REPORT_FOLDER = '_report'
RESULTS_FOLDER = '_results'

THREADS = 4  # Number of threads for OpenFAST

RUN_FILE_WRITE      = False  # Set to True to write the OpenFAST input files
RUN_OPENFAST        = False  # Set to True to run OpenFAST simulations
RUN_REPORTING       = True  # Set to True to generate reports
RUN_RESULTS_GEN     = True  # Set to True to process results


def set_case_vars(fst_vt, case='v3.1', OLAF = False):
    '''
    Case V3.1 (Axi-symmetric)
    '''

    # sim parameters
    fst_vt['Fst']['TMax'] = 240.0 # 120s transient + 120s steady state
    fst_vt['Fst']['DT'] = 0.01
    fst_vt['Fst']['CompElast'] = 1 # Setting ElastoDyn
    fst_vt['Fst']['CompServo'] = 0 # No servo control
    fst_vt['Fst']['CompSeaSt'] = 0 # No sea state  -> Rigid monopile/Tower
    fst_vt['Fst']['CompHydro'] = 0 # No hydrodynamics -> Rigid monopile/Tower
    fst_vt['Fst']['CompSub']   = 0 # No substructure -> Rigid monopile/Tower
    fst_vt['Fst']['SttsTime'] = 0.1
    fst_vt['Fst']['TStart'] = 0.0
    fst_vt['Fst']['OutFileFmt'] = 2 # binary file since large files
    fst_vt['Fst']['OutFmt'] = "ES18.9E3"  # Single file output

    # AeroDyn parameters
    fst_vt['AeroDyn']['SectAvg'] = 'True'  # Use section average
    fst_vt['AeroDyn']['DBEMT_Mod'] = 2  # time-dependent tau1
        # a = 0.5*(1.-np.sqrt(1.-CT)), assume a = 0.3 or 0.33
        # tau_1 = 1.1 / (1.-1.3*np.min([a, 0.5])) * R / U0
    fst_vt['AeroDyn']['tau1_const'] = 1.1 / (1 - 1.3 * 0.3) * (fst_vt['ElastoDyn']['TipRad'] / 6.0)  # Assuming U0 = 6.0 m/s & a = 0.3
    fst_vt['AeroDyn']['UA_Mod'] = 4  # Use the B-L HGM 4-states UA model
    fst_vt['AeroDyn']['IntegrationMethod'] = 4

    if OLAF:
        fst_vt['AeroDyn']['Wake_Mod'] = 3

        # For OLAF, use the OLAF settings
        # fetch the parameters for OLAF
        dt_fvw, tMin, nNWPanels, nNWPanelsFree, nFWPanels, nFWPanelsFree = OLAFParams(
            omega_rpm = 3.693242031, 
            U0 = 6.0, 
            R = fst_vt['ElastoDyn']['TipRad'],
            dt_glue_code = fst_vt['Fst']['DT'],
        )

        # overide time to allow transients
        if tMin > fst_vt['Fst']['TMax']:
            fst_vt['Fst']['TMax'] = np.floor(tMin) + 120 #s

        fst_vt['AeroDyn']['OLAF']['DTfvw'] = dt_fvw  # Time step for the far wake
        fst_vt['AeroDyn']['OLAF']['nNWPanels'] = nNWPanels  
        fst_vt['AeroDyn']['OLAF']['nNWPanelsFree'] = nNWPanelsFree
        fst_vt['AeroDyn']['OLAF']['nFWPanels'] = nFWPanels
        fst_vt['AeroDyn']['OLAF']['nFWPanelsFree'] = nFWPanelsFree
        fst_vt['AeroDyn']['OLAF']['WakeRegFactor'] = 0.5
        fst_vt['AeroDyn']['OLAF']['WingRegFactor'] = 0.5
        fst_vt['AeroDyn']['OLAF']['WrVTk'] = 2
        fst_vt['AeroDyn']['OLAF']['nVTKBlades'] = 3

    else:
        # For OpenFAST, use the standard settings
        fst_vt['AeroDyn']['Wake_Mod'] = 1
    

    # Setting summaries for AeroDyn and BeamDyn
    fst_vt['AeroDyn']['SumPrint'] = 'True'  # Print summary
    fst_vt['BeamDyn']['SumPrint'] = 'True'  # Print summary

    # Rigid structure
    fst_vt['ElastoDyn']['FlapDOF1']   = 'False'
    fst_vt['ElastoDyn']['FlapDOF2']   = 'False'
    fst_vt['ElastoDyn']['EdgeDOF']    = 'False'
    fst_vt['ElastoDyn']['TeetDOF']    = 'False'
    fst_vt['ElastoDyn']['DrTrDOF']    = 'False'
    fst_vt['ElastoDyn']['GenDOF']     = 'False' # Generator DOF is True?
    fst_vt['ElastoDyn']['YawDOF']     = 'False'
    fst_vt['ElastoDyn']['TwFADOF1']   = 'False'
    fst_vt['ElastoDyn']['TwFADOF2']   = 'False'
    fst_vt['ElastoDyn']['TwSSDOF1']   = 'False'
    fst_vt['ElastoDyn']['TwSSDOF2']   = 'False'
    fst_vt['ElastoDyn']['PtfmSgDOF']  = 'False'
    fst_vt['ElastoDyn']['PtfmSwDOF']  = 'False'
    fst_vt['ElastoDyn']['PtfmHvDOF']  = 'False'
    fst_vt['ElastoDyn']['PtfmRDOF']   = 'False'
    fst_vt['ElastoDyn']['PtfmPDOF']   = 'False'
    fst_vt['ElastoDyn']['PtfmYDOF']   = 'False'

    # No tilt
    fst_vt['ElastoDyn']['ShftTilt'] = 0.0

    # No tower shadow
    fst_vt['AeroDyn']['TwrAero'] = 'False'
    fst_vt['AeroDyn']['TwrShadow'] = 0
    fst_vt['AeroDyn']['TwrPotent'] = 1

    # Pre-bend is included

    # Cone is included

    # Pitch angle: 2 degree (constant, note this purposely deviates from standard conditions)
    fst_vt['ElastoDyn']['BlPitch1'] = 2.0
    fst_vt['ElastoDyn']['BlPitch2'] = 2.0
    fst_vt['ElastoDyn']['BlPitch3'] = 2.0

    # Rotor speed: 3.693242031 rpm (constant)
    fst_vt['ElastoDyn']['RotSpeed'] = 3.693242031

    # Wind speed 6.0 m/s (constant), no shear
    fst_vt['InflowWind']['WindType'] = 1
    fst_vt['InflowWind']['HWindSpeed'] = 6.0 # Constant wind speed
    fst_vt['InflowWind']['PLexp'] = 0.0  # No shear

    # Air density rho = 1.225 kg/m3
    fst_vt['Fst']['AirDens'] = 1.225

    # Yaw angle: 0 degrees
    fst_vt['ElastoDyn']['NacYaw'] = 0.0

    if case == 'v3.2':
        '''
        Case V3.2 (Axi-symmetric with flexibilities)
        '''
        # As Case V3.1 including blade flexibilities
        fst_vt['Fst']['CompElast'] = 2  # Setting BeamDyn

    fst_vt = setOutputs(fst_vt)

    return fst_vt

def runOpenFAST_wrapper(case_path):
    runOpenFAST(case_path, OPENFAST_EXE)


def main():
    # Read the base FAST input file
    base_case = InputReader_OpenFAST()
    base_case.FAST_InputFile = BASE_FILE
    base_case.FAST_directory = BASE_FOLDER
    base_case.execute()

    # Using a single openfast writer
    of_writer = InputWriter_OpenFAST()

    # CASE v3.1 - BEM
    case_v3_1_BEM = copy.deepcopy(base_case)
    case_v3_1_BEM.fst_vt = set_case_vars(case_v3_1_BEM.fst_vt, case='v3.1', OLAF=False)

    of_writer.FAST_namingOut = 'case_v3_1_BEM'
    of_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'BEM','case_v3_1') #Output directory
    of_writer.fst_vt = case_v3_1_BEM.fst_vt
    case_v3_1_BEM_path = os.path.join(of_writer.FAST_runDirectory, of_writer.FAST_namingOut + '.fst')
    if RUN_FILE_WRITE:
        of_writer.execute()

    # CASE v3.1 - OLAF
    case_v3_1_OLAF = copy.deepcopy(base_case)
    case_v3_1_OLAF.fst_vt = set_case_vars(case_v3_1_OLAF.fst_vt, case='v3.1', OLAF=True)

    of_writer.FAST_namingOut = 'case_v3_1_OLAF'
    of_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'OLAF','case_v3_1') #Output directory
    of_writer.fst_vt = case_v3_1_OLAF.fst_vt
    case_v3_1_OLAF_path = os.path.join(of_writer.FAST_runDirectory, of_writer.FAST_namingOut + '.fst')
    if RUN_FILE_WRITE:
        of_writer.execute()

    # CASE v3.2 - BEM
    case_v3_2_BEM = copy.deepcopy(base_case)
    case_v3_2_BEM.fst_vt = set_case_vars(case_v3_2_BEM.fst_vt, case='v3.2', OLAF=False)

    of_writer.FAST_namingOut = 'case_v3_2_BEM'
    of_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'BEM','case_v3_2') #Output directory
    of_writer.fst_vt = case_v3_2_BEM.fst_vt
    case_v3_2_BEM_path = os.path.join(of_writer.FAST_runDirectory, of_writer.FAST_namingOut + '.fst')
    if RUN_FILE_WRITE:
        of_writer.execute()

    # CASE v3.2 - OLAF
    case_v3_2_OLAF = copy.deepcopy(base_case)
    case_v3_2_OLAF.fst_vt = set_case_vars(case_v3_2_OLAF.fst_vt, case='v3.2', OLAF=True)
    
    of_writer.FAST_namingOut = 'case_v3_2_OLAF'
    of_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'OLAF','case_v3_2') #Output directory
    of_writer.fst_vt = case_v3_2_OLAF.fst_vt
    case_v3_2_OLAF_path = os.path.join(of_writer.FAST_runDirectory, of_writer.FAST_namingOut + '.fst')
    if RUN_FILE_WRITE:
        of_writer.execute()

    if RUN_OPENFAST:
        # Create a pool with 2 processes for the two cases
        pool = multiprocessing.Pool(processes=THREADS)
        
        # Map the function to the case paths
        pool.map(runOpenFAST_wrapper, [case_v3_1_BEM_path, case_v3_1_OLAF_path, case_v3_2_BEM_path, case_v3_2_OLAF_path])
        
        # Close the pool
        pool.close()
        pool.join()

    if RUN_RESULTS_GEN:

        results_df_loads = {}
        results_df_lifting_line = {}
        of_out = {}

        out_ext = '.outb'  if case_v3_1_BEM.fst_vt['Fst']['OutFileFmt'] == 2 else '.out'

        # Process results for for BEM
        results_df_loads['BEM'], results_df_lifting_line['BEM'], of_out['BEM'] = processResults(
            RESULTS_FOLDER,
            fast_output_files = {
                'v3.1': os.path.join(case_v3_1_BEM_path[:-4] + out_ext),
                'v3.2': os.path.join(case_v3_2_BEM_path[:-4] + out_ext),
            },
            nodalR = np.array(base_case.fst_vt['AeroDynBlade']['BlSpn']) + base_case.fst_vt['ElastoDyn']['HubRad'],
            results_string='initialPass_BEM',
            startTime=120.0  # Start time for processing the data
        )

        # Process results for OLAF
        results_df_loads['OLAF'], results_df_lifting_line['OLAF'], of_out['OLAF'] = processResults(
            RESULTS_FOLDER,
            fast_output_files = {
                'v3.1': os.path.join(case_v3_1_OLAF_path[:-4] + out_ext),
                'v3.2': os.path.join(case_v3_2_OLAF_path[:-4] + out_ext),
            },
            nodalR = np.array(base_case.fst_vt['AeroDynBlade']['BlSpn']) + base_case.fst_vt['ElastoDyn']['HubRad'],
            results_string='initialPass_OLAF',
            startTime=591.0  # Start time for processing the data
        )

    if RUN_REPORTING:
        reporting(results_df_loads, results_df_lifting_line, of_out, REPORT_FOLDER, vtk=False, vtk_folder='_model_omp')



if __name__ == '__main__':
    main()