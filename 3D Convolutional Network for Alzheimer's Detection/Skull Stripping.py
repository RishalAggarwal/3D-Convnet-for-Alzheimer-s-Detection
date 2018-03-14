import os
import nipype
import nipype.interfaces.fsl as fsl

data_dir='path_to_raw_data'                                                                                                                              #path to raw image directory
ssdata_dir='output_data_path'                                                                                                                            #path to skull stripped image directory

for file in os.listdir(data_dir):
    try:
        mybet = nipype.interfaces.fsl.BET(in_file=os.path.join(data_dir,file),out_file=os.path.join(ssdata_dir,file +'_2.nii'), frac=0.2)                #frac=0.2
        mybet.run()                                                                                                                                      #executing the brain extraction
        print(file+'is skull stripped')
    except:
        print(file+'is not skull stripped')
