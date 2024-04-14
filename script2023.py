import bpy
import random
import os
import time
import numpy as np
num_images=1 #number of images to create
max_num_shapes=7 #maximum number of shapes per image
root_path='C:/Users/alessio/Desktop/tests/Flight/dataset'

train_path=os.path.join(root_path,'datasets/train')
test_path=os.path.join(root_path,'datasets/test')
val_path=os.path.join(root_path,'datasets/valid')

#percentage of images for train, test and validation
train_percentage=0.8
test_percentage=0
val_percentage=0.2

img_offset=0

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
        print("Invalid bounding box:", (round(min_x * dim_x), round(min_y * dim_y), round((max_x - min_x) * dim_x), round((max_y - min_y) * dim_y)))
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )
    
def reset(keyword):
    for obj in bpy.data.objects:
        if obj.name.startswith(keyword):
            bpy.data.objects.remove(obj, do_unlink=True)

def create_object(i,camera_y):
    symbols="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    collection = bpy.data.collections['Shapes'] # Replace 'Collection' with the name of your collection
    obj = random.choice(collection.objects)
    #print(obj.name)
    material = random.choice(possible_materials)
    obj.active_material = bpy.data.materials[material]

    new_obj = obj.copy()    
    
    random_symbol = random.choice(symbols)   
    #assign material different from the one of the object to the letter
    possible_symbol_materials = [x for x in possible_materials if x != material]
    symbol_material = random.choice(possible_symbol_materials)

    new_obj.name = "Dataset." + str(i) + "." + obj.name + "." + random_symbol + "." + material+ "." + symbol_material

    #new_obj.name = "Dataset." + str(i) + "." + obj.name
    new_obj.data = obj.data.copy()
    scene.collection.objects.link(new_obj)

    # Move the new object to a random position
    #objects must be in an area of 21,33 x 109m
    area_x=1400/100
    area_y=3800/100
    min_x=-area_x/2
    max_x=area_x/2
    min_y=-area_y/2
    max_y=area_y/2
    #print(min_x,max_x,min_y,max_y)
    object_y=camera_y+random.uniform(min_y, max_y)
    new_obj.location = (random.uniform(min_x, max_x), object_y , 0.1)  
    #slightly change scale
    new_obj.scale = (random.uniform(1, 1.2), random.uniform(1, 1.20),1)
    #randomly rotate the object
    new_obj.rotation_mode = 'XYZ'
    new_obj.rotation_euler = (0, 0, random.uniform(0, 360))

    #get the letter text object, place it above the object and change its body with a random letter
    letter = bpy.data.objects['Letter']
    letter_obj = letter.copy()
    letter_obj.data = letter.data.copy()
    scene.collection.objects.link(letter_obj)

    letter_obj.location = (new_obj.location[0], new_obj.location[1], new_obj.location[2])
    letter_obj.rotation_mode = 'XYZ'    
    letter_obj.rotation_euler = (0, 0, new_obj.rotation_euler[2])
    letter_obj.data.body = random_symbol
    #name
    letter_obj.name = "LetterDataset." + str(i) + "." + letter_obj.data.body

    
    
    letter_obj.active_material = bpy.data.materials[symbol_material]
    return obj.name


#create random objects from a collection,assign a random color and move them to a random position
# Set up the scene
scene = bpy.context.scene
camera = scene.camera
'''
classes:    circle, semicircle, quarter circle, triangle, 
    square, rectangle, trapezoid, pentagon, hexagon,
     heptagon, octagon, star, cross. 
'''
classes_list=["Circle", "Semicircle", "QuarterCircle", "Triangle", 
    "Square", "Rectangle", "Trapezoid", "Pentagon", "Hexagon",
     "Heptagon", "Octagon", "Star", "Cross"]
num_classes=len(classes_list)
class2idx={classes_list[i]:i for i in range(num_classes)}
idx2class={i:classes_list[i] for i in range(num_classes)}
possible_materials = ['White', 'Black', 'Gray', 'Red',"Green","Blue","Yellow","Orange","Purple","Brown"]

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
    img_name="img_"+str(i+img_offset)+".png"
    # Set up the camera
    camera_y=random.uniform(-30, -120)
    camera.location = (random.uniform(-2, 2), camera_y , random.uniform(27, 50))
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = (0, 0, random.uniform(-360, 360))

    
    #change asphalt value
    # get the material
    mat = bpy.data.materials['Asphalt']
    # get the nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Noise Texture"].inputs[1].default_value=random.uniform(1,1)


    #change light intensity between 0.5 and 1.2
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value=random.uniform(0.5, 1.2)
    #randomly rotate hdri
    bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2]=random.uniform(0, 360)

    #change texture 
    bpy.data.materials["Asphalt"].node_tree.nodes["Value.004"].outputs[0].default_value=random.uniform(1.3, 3)

    #number of shapes to create 
    num_shapes = random.randint(1, max_num_shapes)
    # Create the objects
    for j in range (num_shapes):
        create_object(i,camera_y)
        # Render the image

    one_valid=False
    
    #determine if test or train or valid
    dataset_chosen=random.choices([train_path,val_path,test_path],weights=[train_percentage,val_percentage,test_percentage])[0]
    print("chosen dataset: ",dataset_chosen)
    img_path=os.path.join(dataset_chosen,img_name)

    #get the bounding box of the objects and write them in yolo format to file
    #one file per image, one line per object, file is named as the image
    with open(img_path.replace(".png",".txt"), "w+") as f:
        for obj in bpy.data.objects:
            if obj.name.startswith("Dataset"):
                print(obj.name, obj.location)
                box=camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera,obj )
                #if not zero
                if sum(box) == 0: 
                    #delete this object
                    bpy.data.objects.remove(obj, do_unlink=True)
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
                shape = obj.name.split(".")[2]
                symbol = obj.name.split(".")[3]
                material = obj.name.split(".")[4]
                symbol_material = obj.name.split(".")[5]
                #to get the class of the symbol just use its index number 
                symbol_index = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890".index(symbol)
                

                material_index = possible_materials.index(material)
                symbol_material_index = possible_materials.index(symbol_material)
                #for each object one line in format: class_id center_x center_y width height symbol_class
                f.write("{} {} {} {} {} {} {} {}\n".format(class2idx[shape],x_center,y_center,width,height,symbol_index,material_index,symbol_material_index))
                print("printed to file: ",class2idx[shape],x_center,y_center,width,height)
    
    if one_valid:
        # Render the image
        output_path = img_path
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
    
    else:
        #delete the text file
        os.remove(img_path.replace(".png",".txt"))
        
    #reset the scene
    reset(keyword="Dataset")
    reset(keyword="LetterDataset")
