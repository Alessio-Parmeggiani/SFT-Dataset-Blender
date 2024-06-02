import bpy
import random
import os
import time
import numpy as np

#********* Parameters********
#Check all these aprameters before starting generation 

num_images=1 #How many images to create
object_number_range=(4,4) # (x,y)= create between x and y objects, if x=y create x objects
#*if in previous session you created 100 images, set img_offset=100
#!otherwise they will be overwritten
img_offset=0

#* Paths for train, test and validation
#! Change this paths to the ones in your system
root_path=r'C:\Users\alessio\Desktop\DatasetFlight' #Example for windows
#root_path=r'D:\DatasetSFT' 
#root_path="/home/alessio/Desktop/Altro/DatasetFlight/images" #Example for linux

dataset_path=os.path.join(root_path,'datasets')
train_path=os.path.join(dataset_path,'train')
test_path=os.path.join(dataset_path,'test')
val_path=os.path.join(dataset_path,'valid')

#* percentage of images for train, test and validation
train_percentage=1.0
test_percentage=0
val_percentage=0


#* parameters for generation
symbols="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
possible_materials = ['White', 'Black', 'Red',"Green","Blue", "Orange","Purple","Brown"]
possible_targets=["circle", "semi", "quarter", "triangle", "rectangle","pentagon", "star", "cross"]

classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
          'U', 'V', 'W', 'X', 'Y', 'Z', 
          'black', 'blue', 'brown', 'circle', 'cross', 'emergent', 
          'green', 'orange', 'pentagon', 'purple', 'quarter', 
          'rectangle', 'red', 'semi', 'star', 'triangle', 'white']

use_weight=False #if True, generate a random material and shape using weighted random choice
material_weights={
    "White":1,
    "Black":1,
    "Red":1,
    "Green":1,
    "Blue":1,
    "Orange":1,
    "Purple":1,
    "Brown":1
} #a value of X means that is X times more likely to be chosen
shape_weights={}
for i in range(len(possible_targets)):
    shape_weights[possible_targets[i]]=1
symbols_weight={}
for i in range(len(symbols)):
    symbols_weight[symbols[i]]=1

collection = bpy.data.collections['Shapes'] #Blender collection from which to take objects
#Check always that objects in collection and possible_targets have matching names


#* Camera position & rotation. to find good values, change in blender and note the values
#Position
camera_altitude_range =(32,36)
camera_x_range=(-2,2)
camera_y_range=(-30,-120) 
#Orientation
camera_rot_x_range=(-5,5) #Pitch
camera_rot_y_range=(-20,20) #Roll
camera_rot_z_range=(-15,15) #Yaw

#* Objects random positions
#Define the are near the camera where a target can spawn
area_x=16
area_y=20
min_x=-area_x/2
max_x=area_x/2
min_y=-area_y/2
max_y=area_y/2
object_z=0.1 #putting target slightly above the ground to avoid Z-fighting
object_scale_range=(1,1.1)
limits_x=(-1000,1000) #object should not be outside of this area


#used to avoid collisions between objects
object_positions=[]

#* Tile parameters
create_tiles=True #Create tiles from the images
tile_num=8 #Create NxN tiles from each image
do_clean_empty_tiles=False #Delete tiles with no targets
if create_tiles:
    try:
        from PIL import Image
    except: 
        print("Pillow not installed, installing now...")
        import subprocess
        import sys
        python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
        subprocess.call([python_exe, '-m', 'ensurepip'])
        subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pillow'])
        from PIL import Image
        print("Pillow installed successfully")
        


#*Debug flags (set all to true when generating the dataset)
generate_shapes=True #generate random targets
move_camera=True #move the camera to a random position for each image
render_image=True #render the image
clean_after_render=True #remove target from scene after rendering


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def camera_view_bounds_2d(scene, cam_ob, me_ob):
    
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:

            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        print("Invalid bounding box.")
        return (0, 0, 0, 0)

    print("Valid bounding box.")
    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )
    
def reset(keyword):
    #delete all objects starting with keyword
    for obj in bpy.data.objects:
        if obj.name.startswith(keyword):
            bpy.data.objects.remove(obj, do_unlink=True)

def is_colliding(pos, object_positions):
    #Check approxiamtely if an object is colliding with another object
    #print("checking collision of object at position: ",pos
    #      ," with objects at positions: ",object_positions)
    object_radius=0.3
    for obj in object_positions:
        distance=np.sqrt((pos[0]-obj[0])**2+(pos[1]-obj[1])**2)
        #print("Distance: ",distance)
        if distance < object_radius:
            return True
    return False

def create_object(i,camera_y):
    #create random objects from a collection,assign a random color and move them to a random position
    
    #* Choosing object and material
    obj =None
    shape=None
    material=None
    random_letter=None
    letter_material=None

    if use_weight:
        shape = random.choices(possible_targets, shape_weights.values())[0]
        obj = bpy.data.objects[shape]
        material = random.choices(possible_materials, material_weights.values())[0]
        random_letter=random.choices(symbols, symbols_weight.values())[0]
        #assign material for the letter different from the one of the object
        letter_material = random.choices(possible_materials, material_weights.values())[0]
        while letter_material == material:
            letter_material = random.choices(possible_materials, material_weights.values())[0]
    else:
        obj = random.choice(collection.objects)
        shape = obj.name
        material = random.choice(possible_materials)
        random_letter=random.choice(symbols)
        #assign material for the letter different from the one of the object
        letter_material = random.choice(possible_materials)
        while letter_material == material:
            letter_material = random.choice(possible_materials)

    print("Chosen: \n\tObject: ", obj.name, "|", material, 
        "\n\tLetter: ", random_letter, "|", letter_material)


    # Assign the material to the object
    obj.active_material = bpy.data.materials[material]

    # Create a new object with the same data
    new_obj = obj.copy()    
    #new_obj.name = "Dataset." + str(i) + "." + obj.name
    separator="."
    new_obj.name=separator.join(["Dataset",str(i),shape,material,random_letter,letter_material])
    new_obj.data = obj.data.copy()
    scene.collection.objects.link(new_obj)

    # Move the new object to a random position
    #objects must be in an area of 21,33 x 109m
    
    #print(min_x,max_x,min_y,max_y)
    object_y=None
    object_x=None
    tries=0
    while object_y is None or object_x is None or is_colliding((object_x,object_y), object_positions):
        print("collision detected,regenerating position")
        object_y=camera_y+random.uniform(min_y, max_y)
        object_x=random.uniform(min_x, max_x)
        while object_x < limits_x[0] or object_x > limits_x[1]:
            object_x=random.uniform(min_x, max_x)
        tries+=1
        if tries>100:
            print("Could not find a valid position for the object, exiting")
            #delete object
            bpy.data.objects.remove(new_obj, do_unlink=True)
            return
    #print("From camera_Y: ",camera_y," got object_Y: ",object_y)
    object_positions.append((object_x,object_y))

    new_obj.location = (object_x, object_y , object_z)
    #slightly change scale
    object_scale=random.uniform(object_scale_range[0],object_scale_range[1])
    new_obj.scale = (object_scale, object_scale, 1)
    #randomly rotate the object
    new_obj.rotation_mode = 'XYZ'
    new_obj.rotation_euler = (0, 0, random.uniform(0, 360))

    #get the letter text object, place it above the object and change its body with a random letter
    letter = bpy.data.objects['Letter']
    letter_obj = letter.copy()
    letter_obj.data = letter.data.copy()
    scene.collection.objects.link(letter_obj)

    letter_obj.data.body = random_letter
    letter_obj.location = (new_obj.location[0], new_obj.location[1], new_obj.location[2])
    letter_obj.rotation_mode = 'XYZ'    
    letter_obj.rotation_euler = (0, 0, new_obj.rotation_euler[2])
    #name
    letter_obj.name = "LetterDataset." + str(i) + "." + letter_obj.data.body

    
    letter_obj.active_material = bpy.data.materials[letter_material]
    return obj.name

def write_annotation_file(annotation_path,scene,camera):
    one_valid=False #flag to check if at least one object  is in the image and can be annotated
    with open(annotation_path, "w+") as f:
        for obj in bpy.data.objects:
            if obj.name.startswith("Dataset"):
                print("Generating bounding box for object: {} at location: {}".format(obj.name,obj.location))
                box=camera_view_bounds_2d(scene, camera,obj )
                #if not zero
                if sum(box) == 0: 
                    continue
                else: 
                    one_valid=True
                print(box)
                #write the bounding box in yolo format
                #x_center, y_center, width, height
                x_center=box[0]+box[2]/2
                y_center=box[1]+box[3]/2
                width=box[2]
                height=box[3]
                #normalize the values
                x_center=x_center/scene.render.resolution_x
                y_center=y_center/scene.render.resolution_y
                width=width/scene.render.resolution_x
                height=height/scene.render.resolution_y

                shape = obj.name.split(".")[2]
                material = obj.name.split(".")[3]
                symbol = obj.name.split(".")[4]
                symbol_material = obj.name.split(".")[5] #not used

                shape_id=classes.index(shape.lower())
                material_id=classes.index(material.lower())
                symbol_id=classes.index(symbol)

                #for each object three lines in format: class_id center_x center_y width height
                f.write("{} {} {} {} {}\n".format(shape_id,x_center,y_center,width,height))
                f.write("{} {} {} {} {}\n".format(material_id,x_center,y_center,width,height))
                f.write("{} {} {} {} {}\n".format(symbol_id,x_center,y_center,width,height))

    return one_valid



#**Image tiling

#Get the path for the tile at a given index i,j and the original image path
def get_image_tile_path(image_path, i, j):
    return image_path.replace('.png', f'_tile_{i}_{j}.png')

#Get the path for the annotation tile at a given index i,j and the original annotation path
def get_annotation_tile_path(annotation_path, i, j):
    return annotation_path.replace('.txt', f'_tile_{i}_{j}.txt')

def split_image_into_tiles(image_path, n):
    """
    Generate n x n tiles from the image at the given path.
    """
    #n=4 -> 16 tiles
    # Open the image file
    img = Image.open(image_path)
    # Calculate the width and height of each tile
    width, height = img.size
    tile_width = width // n
    tile_height = height // n

    # Loop over the image and save each tile
    for i in range(n):
        for j in range(n):
            #path=image_path.replace('.png',f'_{index}.png')
            path=get_image_tile_path(image_path,i,j)
            left = i * tile_width
            upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height
            # Crop the tile out of the image
            tile_img = img.crop((left, upper, right, lower))
            # Save the tile to a file
            tile_img.save(path)
        
    return tile_width, tile_height

def create_annotations_for_tiles(annotation_path, tile_width, tile_height, n):
    """
    Given an annotation file for each image in the yolo format,
    Where for each target in the image I have three rows:
        shape_id x_center y_center width height
        material_id x_center y_center width height
        symbol_id x_center y_center width height
    Create the new annotations for each generated tile fo the image
    """
    # Read the annotations
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # create the annotations for each tile
    for l in range(0,len(lines), 3):
        #print("\nProcessing tile: ", l // 3, " of ", len(lines) // 3, " tiles.")
        line1 = lines[l]
        line2 = lines[l+1]
        line3 = lines[l+2]
        # Parse the annotation
        shape_id, x_center, y_center, width, height = map(float, line1.split())
        material_id, _,_,_,_ = map(float, line2.split())
        symbol_id, _, _, _, _ = map(float, line3.split())
        # Check in which tile the annotation is
        #print("Original center: ", x_center, y_center)

        i = int(x_center // (1 / n))
        j = int(y_center // (1 / n))

        #print("Found corresponding tile: ", i, j, " of size ", tile_width, tile_height)
        # Write the annotation for the tile
        new_center = (x_center - i * (1 / n), y_center - j * (1 / n))
        #multiply by n to get the new center, width and height
        new_center=(new_center[0]*n,new_center[1]*n)
        new_width = (width* n, height* n)

        #print("New center: ", new_center)
        #print("New width: ", new_width)
        tile_annotation = f'{shape_id} {new_center[0]} {new_center[1]} {new_width[0]} {new_width[1]}\n'
        tile_annotation += f'{material_id} {new_center[0]} {new_center[1]} {new_width[0]} {new_width[1]}\n'
        tile_annotation += f'{symbol_id} {new_center[0]} {new_center[1]} {new_width[0]} {new_width[1]}\n'

        # Save the annotation to a file
        tile_annotation_path = get_annotation_tile_path(annotation_path, i, j)
        #append to file and crete if it not exist
        with open(tile_annotation_path, 'a') as file:
            file.write(tile_annotation)

def clean_empty_tiles(image_path, annotation_path, n,clean_tiles):
    """
    Delete the tiles with no corresponding annotations
    """
    print("Clean empty tiles set to: ",clean_tiles)
    for i in range(n):
        for j in range(n):
            tile_annotation_path = get_annotation_tile_path(annotation_path, i, j)
            if not os.path.exists(tile_annotation_path):
                tile_image_path = get_image_tile_path(image_path, i, j)
                if clean_tiles:
                    os.remove(tile_image_path)
                    print("Deleted: ", tile_image_path)  
                else:
                    #Create empty annotation file
                    with open(tile_annotation_path, 'w') as file:
                        file.write("")


#Set up the scene
scene = bpy.context.scene
camera = scene.camera

num_classes=len(possible_targets)
class2idx={possible_targets[i]:i for i in range(num_classes)}
idx2class={i:possible_targets[i] for i in range(num_classes)}

#*YOLO need  fyle with some information about the dataset
#write starting yaml file in the format:
'''
train: ../train/images
val: ../valid/images

nc: 3
names: ['head', 'helmet', 'person']

'''
#create file if it does not exist
with open(os.path.join(root_path,"DatasetFlight.yaml"), "w+") as f:
    #local train path
    f.write("train: {}".format("train"))
    f.write("\n")
    #local valid path
    f.write("val: {}".format("valid"))
    f.write("\n")
    #number of classes
    f.write("nc: {}".format(num_classes))
    f.write("\n")
    #classes names
    f.write("names: {}".format(possible_targets))
    f.write("\n")
    

for i in range(num_images):
    object_positions=[] #reset the list of object positions

    #*** Set up paths
    img_name="img_"+str(i+img_offset)+".png"
    print("\n---------------------------\nGenerating image: ",img_name)
    #determine if test or train or valid
    dataset_chosen=random.choices([train_path,val_path,test_path],weights=[train_percentage,val_percentage,test_percentage])[0]
    print("chosen dataset: ",dataset_chosen)
    img_path=os.path.join(dataset_chosen,img_name)
    print("Image path: ",img_path)
    #if path does not exist, create it
    if not os.path.exists(dataset_chosen):
        os.makedirs(dataset_chosen)

    
    #*** Set up the camera
    if move_camera:
        camera_x=random.uniform(camera_x_range[0],camera_x_range[1])
        camera_y=random.uniform(camera_y_range[0],camera_y_range[1])
        camera_z=random.uniform(camera_altitude_range[0],camera_altitude_range[1])
        camera.rotation_mode = 'XYZ'
        #Rotate and move camera
        #reset camera rotation
        rot_x=random.uniform(camera_rot_x_range[0],camera_rot_x_range[1])
        rot_y=random.uniform(camera_rot_y_range[0],camera_rot_y_range[1])
        rot_z=random.uniform(camera_rot_z_range[0],camera_rot_z_range[1])
        camera.rotation_euler = (0, 0, 0)
        camera.location=(camera_x, camera_y , camera_z)
        camera.rotation_euler = (np.radians(rot_x), np.radians(rot_y), np.radians(rot_z))
    else:
        camera_y=camera.location[1]
    
    #*** Modify environment
    # get the material
    mat = bpy.data.materials['Asphalt']
    # get the nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Noise Texture"].inputs[1].default_value=random.uniform(1,1)
    #change light intensity between 0.5 and 1.2
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value=random.uniform(0.8, 1.2)
    #randomly rotate hdri
    bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2]=random.uniform(0, 360)
    

    #*** Create the objects
    num_shapes = random.randint(object_number_range[0],object_number_range[1])
    if generate_shapes:
        for j in range(num_shapes):
            create_object(i,camera_y)
    

    #*** Render the image
    if render_image:
        output_path = img_path
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print("Image rendered and saved at: ",output_path)
        
    #*** Create annotations
    #get the bounding box of the objects and write them in yolo format to file
    #one file per image, one line per object, file is named as the image
    annotation_path=img_path.replace(".png",".txt")
    one_valid=write_annotation_file(annotation_path,scene,camera)

    #* maintain image only if at least one annotation is valid
    if not one_valid:
        os.remove(img_path.replace(".png",".txt"))
        os.remove(img_path)
    else:
        #* Generate tiles
        if render_image and create_tiles:
            tile_width, tile_height = split_image_into_tiles(img_path, tile_num)
            #* Create annotations for the tiles
            create_annotations_for_tiles(annotation_path, tile_width, tile_height, tile_num)
            #* Clean empty tiles
            clean_empty_tiles(img_path, annotation_path, tile_num,do_clean_empty_tiles)

            #remove original image and annotation
            os.remove(img_path)
            os.remove(annotation_path)
        
    #* reset the scene
    if clean_after_render:
        reset(keyword="Dataset")
        reset(keyword="LetterDataset")
