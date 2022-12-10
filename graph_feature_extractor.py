import os
import nibabel as nib
from PIL import Image
import numpy as np
from utils.voreen_vesselgraphextraction import extract_vessel_graph
from utils.visualizer import graph_file_to_img, node_edges_to_graph

voreen_config = {
    "image_file": "",
    "voreen_tool_path": "/home/shared/Software/Voreen-source/bin/",
    "workspace_file": "/home/lkreitner/OCTA-seg/voreen/feature-vesselgraphextraction_customized_command_line.vws",
    "outdir": "/home/lkreitner/OCTA-seg/voreen/workdir/",
    "tempdir": "/home/lkreitner/OCTA-seg/voreen/tmpdir/",
    "cachedir": "/home/lkreitner/OCTA-seg/voreen/cachedir/",
    "bulge_size": 1.5
}

def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths


image_dir = "/home/lkreitner/OCTA-seg/voreen_test/"
output_dir = "/home/lkreitner/OCTA-seg/voreen_test/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

image_paths = get_custom_file_paths(image_dir, ".png")
for image_path in image_paths:
    image_name = image_path.split("/")[-1].split(".")[0]
    a = np.array(Image.open(image_path))
    a = np.stack([np.zeros_like(a), a, np.zeros_like(a)], axis=-1)
    img_nii = nib.Nifti1Image(a.astype(np.uint8), np.eye(4))
    
    if not os.path.exists(voreen_config["tempdir"]):
        os.mkdir(voreen_config["tempdir"])
    nii_path = os.path.join(voreen_config["tempdir"], f'{image_name}.nii')
    nib.save(img_nii, nii_path)

    clean_seg = extract_vessel_graph(nii_path, 
        output_dir+"/",
        voreen_config["tempdir"],
        voreen_config["cachedir"],
        voreen_config["bulge_size"],
        voreen_config["workspace_file"],
        voreen_config["voreen_tool_path"],
        name=image_name
    )
    graph_file = os.path.join(output_dir, f'{image_name}_graph.png')
    nodes_file = os.path.join(output_dir, f'{image_name}_nodes.csv')
    edges_file = os.path.join(output_dir, f'{image_name}_edges.csv')
    graph_img = node_edges_to_graph(nodes_file, edges_file, a.shape[:2])
    Image.fromarray((graph_img*255).astype(np.uint8)).save(graph_file)