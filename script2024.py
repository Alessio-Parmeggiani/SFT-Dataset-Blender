import bpy
import random
import os
import time
import numpy as np

#********* Parameters********
#Check all these aprameters before starting generation 

num_images=1 #How many images to create
object_number_range=(40,40) # (x,y)= create between x and y objects, if x=y create x objects

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
train_percentage=0.8
test_percentage=0
val_percentage=0.2

#*if in previous session you created 100 images, set img_offset=100
#!otherwise they will be overwritten
img_offset=0

#* parameters for generation
symbols="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
possible_materials = ['White', 'Black', 'Red',"Green","Blue", "Orange","Purple","Brown"]
classes_list=["Circle", "Semicircle", "QuarterCircle", "Triangle", 
    "Square", "Rectangle","Pentagon", "Star", "Cross"]

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
for i in range(len(classes_list)):
    shape_weights[classes_list[i]]=1
symbols_weight={}
for i in range(len(symbols)):
    symbols_weight[symbols[i]]=1




collection = bpy.data.collections['Shapes'] #Blender collection from which to take objects
#Check always that objects in collection and classes_list have matching names


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
area_y=14
min_x=-area_x/2
max_x=area_x/2
min_y=-area_y/2
max_y=area_y/2
object_z=0.1 #putting target slightly above the ground to avoid Z-fighting
object_scale_range=(1,1.1)
limits_x=(-1000,1000) #object should not be outside of this area


#used to avoid collisions between objects
object_positions=[]


#*Debug flags
generate_shapes=True
move_camera=True
render_image=True
clean_after_render=True


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
    print("checking collision of object at position: ",pos
          ," with objects at positions: ",object_positions)
    object_radius=0.3
    for obj in object_positions:
        distance=np.sqrt((pos[0]-obj[0])**2+(pos[1]-obj[1])**2)
        print("Distance: ",distance)
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
        shape = random.choices(classes_list, shape_weights.values())[0]
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


#Set up the scene
scene = bpy.context.scene
camera = scene.camera

num_classes=len(classes_list)
class2idx={classes_list[i]:i for i in range(num_classes)}
idx2class={i:classes_list[i] for i in range(num_classes)}

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
    f.write("names: {}".format(classes_list))
    f.write("\n")
    

for i in range(num_images):
    object_positions=[] #reset the list of object positions

    img_name="img_"+str(i+img_offset)+".png"
    print("\n---------------------------\nGenerating image: ",img_name)
    
    #* Set up the camera
    camera_x=random.uniform(camera_x_range[0],camera_x_range[1])
    camera_y=random.uniform(camera_y_range[0],camera_y_range[1])
    camera_z=random.uniform(camera_altitude_range[0],camera_altitude_range[1])
    camera.rotation_mode = 'XYZ'

    #Rotate and move camera
    if move_camera:
        #reset camera rotation
        rot_x=random.uniform(camera_rot_x_range[0],camera_rot_x_range[1])
        rot_y=random.uniform(camera_rot_y_range[0],camera_rot_y_range[1])
        rot_z=random.uniform(camera_rot_z_range[0],camera_rot_z_range[1])
        camera.rotation_euler = (0, 0, 0)
        camera.location=(camera_x, camera_y , camera_z)
        camera.rotation_euler = (np.radians(rot_x), np.radians(rot_y), np.radians(rot_z))

    
    #* change terrain
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
    #* Create the objects
    
    num_shapes = random.randint(object_number_range[0],object_number_range[1])

    # Create the objects
    if generate_shapes:
        for j in range(num_shapes):
            create_object(i,camera_y)


    #* Create annotations
    one_valid=False
    
    #determine if test or train or valid
    dataset_chosen=random.choices([train_path,val_path,test_path],weights=[train_percentage,val_percentage,test_percentage])[0]
    print("chosen dataset: ",dataset_chosen)
    img_path=os.path.join(dataset_chosen,img_name)
    print("Image path: ",img_path)
    #if path does not exist, create it
    if not os.path.exists(dataset_chosen):
        os.makedirs(dataset_chosen)

    if render_image:
        output_path = img_path
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print("Image rendered and saved at: ",output_path)
        
    #get the bounding box of the objects and write them in yolo format to file
    #one file per image, one line per object, file is named as the image
    with open(img_path.replace(".png",".txt"), "w+") as f:
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
                #write the values in the file
                #shape=obj.name.split(".")[2]
                #for each object one line in format: class_id center_x center_y width height
                #f.write("{} {} {} {} {}\n".format(class2idx[shape],x_center,y_center,width,height))
                #print("For shape: ",shape, " wrote: ",class2idx[shape],x_center,y_center,width,height)

                shape = obj.name.split(".")[2]
                material = obj.name.split(".")[3]
                symbol = obj.name.split(".")[4]
                symbol_material = obj.name.split(".")[5]
                #to get the class of the symbol just use its index number
                symbol_index = symbols.index(symbol)
                material_index = possible_materials.index(material)
                symbol_material_index = possible_materials.index(symbol_material)

                #for each object one line in format: class_id center_x center_y width height symbol_class
                f.write("{} {} {} {} {} {} {} {}\n".format(
                    class2idx[shape],x_center,y_center,width,height,
                    symbol_index,material_index,symbol_material_index))
    #* maintain image only if at least one annotation is valid
    if not one_valid:
        os.remove(img_path.replace(".png",".txt"))
        os.remove(img_path)
        
    #* reset the scene
    if clean_after_render:
        reset(keyword="Dataset")
        reset(keyword="LetterDataset")
