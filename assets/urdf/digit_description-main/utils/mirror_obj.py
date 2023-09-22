import sys
import os

def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(file_dir, "../urdf/meshes")

    obj_files = [fn for fn in os.listdir(mesh_dir) if (
        os.path.isfile(os.path.join(mesh_dir, fn)) and fn.endswith(".obj")) and "right" not in fn]
    
    mirror_obj_files = [fn[:-4]+"-right.obj" for fn in obj_files]

    for (fn, mirror_fn) in zip(obj_files, mirror_obj_files):
        mirror_obj(os.path.join(mesh_dir, fn), os.path.join(mesh_dir, mirror_fn))


def mirror_obj(obj_path, obj_scaled_path):
    with open(obj_path, "r") as src_file:
        with open(obj_scaled_path, "w") as target_file:
            for line in src_file:
                target_line = line

                if line.startswith("v "):
                    coord = [float(coord) for coord in line.split(" ")[1:]]
                    coord[1] *= -1
                    mirrored_str = " ".join([str(c) for c in coord])
                    target_line = "v " + mirrored_str + "\n"
                
                target_file.write(target_line)



if __name__=="__main__":
    main()