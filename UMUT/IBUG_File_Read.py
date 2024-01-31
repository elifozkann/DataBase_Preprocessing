import os
import scipy.io
import shutil

dbName = 'IBUG'

files = os.scandir('./'+dbName)

for file in files:
    #print(file_name)
    #example ->AFW_815038_1_12 , if our extension is jpg and XXX_XXX_XXX_0.jpg
    file_name = file.name
    
    file_name_split = file_name.split('_')
    file_name_withoutExtension = file_name.split('.')[0]
    #Result of split -> number.extension / example -> 12.jpg => 12 refers to the number of rotated version of the image
    iter_extension = file_name_split[-1]
    extension = iter_extension.split('.')[1] # jpg or mat
    iter_number = iter_extension.split('.')[0] # refer to the number of rotated version of the image
    file_id = file_name_split[-2] # id of the image

    #splitted elements of file_name except the last one
    except_last_split = file_name_split[:-1]
    
    #len of features of name can be 4 or 5
    if len(file_name_split) == 4:
        True
    else:
        file_id = file_name_split[-3]
        #print(str(file_id)+ " fileName:"+ str(file_name))
        False
    
    output_folder = './' + dbName + '_FOLDERED/' + file_id + '/'
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = output_folder + file_name_withoutExtension + '.' + extension
    shutil.copy('./' + dbName + '/' + file_name, output_file_path)  