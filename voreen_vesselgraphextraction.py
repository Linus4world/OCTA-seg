import os
import datetime
import pathlib
from shutil import copyfile


def extract_vessel_graph(volume_path: str, workdir: str, tempdir: str, cachedir:str, bulge_size: float, workspace_file: str, voreen_tool_path: str):
    bulge_size_identifier = f'{bulge_size}'
    bulge_size_identifier = bulge_size_identifier.replace('.','_')

    bulge_path = f'<Property mapKey="minBulgeSize" name="minBulgeSize" value="{bulge_size}"/>'

    # create temp directory
    temp_directory = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    pathlib.Path(temp_directory).mkdir(parents=True, exist_ok=True)

    voreen_workspace = 'feature-vesselgraphextraction_customized_command_line.vws'
    copyfile(workspace_file,os.path.join(temp_directory,voreen_workspace))

    # Read in the file
    with open(os.path.join(temp_directory,voreen_workspace), 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace("/home/voreen_data/volume.nii", volume_path)
    filedata = filedata.replace('<Property mapKey="minBulgeSize" name="minBulgeSize" value="3" />', bulge_path)


    # Write the file out again
    with open(os.path.join(temp_directory,voreen_workspace), 'w') as file:
        file.write(filedata)

    workspace_file = os.path.join(os.path.join(os. getcwd(),temp_directory),voreen_workspace)

    absolute_temp_path = os.path.join(os.getcwd(),temp_directory)

    # extract graph and delete temp directory
    os.system(f'cd {voreen_tool_path} ; ./voreentool \
        --workspace {workspace_file} \
        -platform minimal --trigger-volumesaves --trigger-geometrysaves  --trigger-imagesaves \
        --workdir {workdir} --tempdir {tempdir} --cachedir {cachedir} \
        ; rm -r {absolute_temp_path}'
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Control script for running coreen on a headless machine.')
    parser.add_argument('-i','--input_image', help='Specify input file path of a NIFTI image.', required=True)
    parser.add_argument('-b','--bulge_size',help='Specify bulge size',required=True)
    parser.add_argument('-vp','--voreen_tool_path',help="Specify the path where voreentool is located.",default='/home/shared/Software/voreen/')
    parser.add_argument('-wp','--workspace_file',default='/home/shared/Software/voreen/resource/voreenve/workspaces/feature-vesselgraphextraction.vws')
    # voreen settings
    # --workdir /home/voreen-work/ --tempdir /home/voreen-temp/ --cachedir /home/voreen-cache/

    parser.add_argument('-wd','--workdir', help='Specify the working directory.', required=True)
    parser.add_argument('-td','--tempdir', help='Specify the temporary data directory.', required=True)
    parser.add_argument('-cd','--cachedir', help='Specify the cache directory.', required=True)
    args = vars(parser.parse_args())

    extract_vessel_graph(args['input_image'], args['workdir'], args['tempdir'], args['cachedir'], float(args['bulge_size']), args['workspace_file'], args['voreen_tool_path'])