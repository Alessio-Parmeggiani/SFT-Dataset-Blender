from PIL import Image
import numpy as np
import os

n = 8 #Create NxN tiles from each image
clean_empty_tiles = True

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
        print("\nProcessing tile: ", l // 3, " of ", len(lines) // 3, " tiles.")
        line1 = lines[l]
        line2 = lines[l+1]
        line3 = lines[l+2]
        # Parse the annotation
        shape_id, x_center, y_center, width, height = map(float, line1.split())
        material_id, _,_,_,_ = map(float, line2.split())
        symbol_id, _, _, _, _ = map(float, line3.split())
        # Check in which tile the annotation is
        print("Original center: ", x_center, y_center)

        i = int(x_center // (1 / n))
        j = int(y_center // (1 / n))

        print("Found corresponding tile: ", i, j, " of size ", tile_width, tile_height)
        # Write the annotation for the tile
        new_center = (x_center - i * (1 / n), y_center - j * (1 / n))
        #multiply by n to get the new center, width and height
        new_center=(new_center[0]*n,new_center[1]*n)
        new_width = (width* n, height* n)

        print("New center: ", new_center)
        print("New width: ", new_width)
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

def main():
    image_path = r"C:\Users\alessio\Desktop\DatasetFlight\datasets\train\img_0.png"
    annotation_path = image_path.replace('png', 'txt')


    print("Generating {} x {} tiles...".format(n, n))
    tile_width, tile_height = split_image_into_tiles(image_path, n)
    print("creating annotations for tiles...")
    create_annotations_for_tiles(annotation_path, tile_width, tile_height, n)
    print("Deleting tiles with no targets...")
    clean_empty_tiles(image_path, annotation_path, n, clean_empty_tiles)

if __name__ == '__main__':
    main()