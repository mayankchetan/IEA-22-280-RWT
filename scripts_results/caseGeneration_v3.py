from openfast_io.FAST_reader import InputReader_OpenFAST
from openfast_io.FAST_writer import InputWriter_OpenFAST
import os
import copy
import multiprocessing

from utils import *


BASE_FOLDER = '../OpenFAST/IEA-22-280-RWT-Monopile'
BASE_FILE = 'IEA-22-280-RWT-Monopile.fst'

OPENFAST_EXE = '../../openfast-dev-tc/build/glue-codes/openfast/openfast'

MODEL_FOLDER = '_model'
REPORT_FOLDER = '_report'
RESULTS_FOLDER = '_results'


RUN_FILE_WRITE      = False  # Set to True to write the OpenFAST input files
RUN_OPENFAST        = False  # Set to True to run OpenFAST simulations
RUN_REPORTING       = True  # Set to True to generate reports
RUN_RESULTS_GEN     = True  # Set to True to process results


def set_case_vars(fst_vt, case='v3.1'):
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

    # CASE v3.1
    case_v3_1 = copy.deepcopy(base_case)
    case_v3_1.fst_vt = set_case_vars(case_v3_1.fst_vt)

    case_v3_1_writer = InputWriter_OpenFAST()
    case_v3_1_writer.FAST_namingOut = 'case_v3_1'
    case_v3_1_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'case_v3_1') #Output directory
    case_v3_1_writer.fst_vt = case_v3_1.fst_vt

    # CASE v3.2
    case_v3_2 = copy.deepcopy(base_case)
    case_v3_2.fst_vt = set_case_vars(case_v3_2.fst_vt, case='v3.2')

    case_v3_2_writer = InputWriter_OpenFAST()
    case_v3_2_writer.FAST_namingOut = 'case_v3_2'
    case_v3_2_writer.FAST_runDirectory = os.path.join(MODEL_FOLDER,'case_v3_2') #Output directory
    case_v3_2_writer.fst_vt = case_v3_2.fst_vt

    if RUN_FILE_WRITE:
        case_v3_1_writer.execute()
        case_v3_2_writer.execute()

    if RUN_OPENFAST:
        # Set outputs for both cases, and run them in parallel
        case_v3_1_path = os.path.join(case_v3_1_writer.FAST_runDirectory, case_v3_1_writer.FAST_namingOut + '.fst')
        case_v3_2_path = os.path.join(case_v3_2_writer.FAST_runDirectory, case_v3_2_writer.FAST_namingOut + '.fst')
        
        # Create a pool with 2 processes for the two cases
        pool = multiprocessing.Pool(processes=2)
        
        # Map the function to the case paths
        pool.map(runOpenFAST_wrapper, [case_v3_1_path, case_v3_2_path])
        
        # Close the pool
        pool.close()
        pool.join()

    if RUN_RESULTS_GEN:

        out_ext = '.outb'  if base_case.fst_vt['Fst']['OutFileFmt'] == 2 else '.out'

        results_df_loads, results_df_lifting_line, of_out = processResults(
            REPORT_FOLDER,
            fast_output_files = {
                'v3.1': os.path.join(case_v3_1_writer.FAST_runDirectory, case_v3_1_writer.FAST_namingOut + out_ext),
                'v3.2': os.path.join(case_v3_2_writer.FAST_runDirectory, case_v3_2_writer.FAST_namingOut + out_ext)
            },
            nodalR = np.array(base_case.fst_vt['AeroDynBlade']['BlSpn']) + base_case.fst_vt['ElastoDyn']['HubRad'],
            results_string='initialPass',
            startTime=120.0  # Start time for processing the data
        )

    if RUN_REPORTING:
        reporting(results_df_loads, results_df_lifting_line, of_out, REPORT_FOLDER)



if __name__ == '__main__':
    main()