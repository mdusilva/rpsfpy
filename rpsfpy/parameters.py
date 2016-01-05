import ConfigParser
import warnings
import numpy as np

config = ConfigParser.SafeConfigParser(allow_no_value=True)

#def new(filename, objpos=[], r0='None', cn2=[], h_profile=[], L0 = 'None', ngs = [], lgs = [], lgs_height='None', zm='None', D='None', pixels='None', separator=','):
def new(filename, r0='None', cn2=[], h_profile=[], L0 = 'None', ngs = [], lgs = [], lgs_height='None', zm='None', D='None', pixels='None', separator=','):
    """Create new parameter file"""
    cn2string = separator.join(map(str, cn2))
    hstring = separator.join(map(str, h_profile))
    ngsstring = separator.join(map(str, ngs))
    lgsstring = separator.join(map(str, lgs))
#    objpostring = separator.join(map(str, objpos))
    config.add_section('atmosphere')
    config.add_section('AO system')
    config.add_section('image')
    config.set('atmosphere', 'r0', r0)
    config.set('atmosphere','cn2 profile' , cn2string)
    config.set('atmosphere', 'h profile', hstring)
    config.set('atmosphere', 'L0', L0)
    config.set('AO system', 'ngs', ngsstring)
    config.set('AO system', 'lgs', lgsstring)
    config.set('AO system', 'lgs altitude', lgs_height)
    config.set('AO system', 'zernike number', zm)
    config.set('AO system', 'telescope diameter', D)
    config.set('image', 'pixel diameter', pixels)
#    config.set('image', 'PSF position', objpostring)
    header_string = "#Parameter initialization\n#---------------------------------------\n#This file contains the parameters needed to compute the PSF\n#for an AO system consisting of a number of LGS and NGS\n#\n#How to configure:\n#In the section [atmosphere] write a number for r0, values for the Cn2 profile,\n#one for each layer, separated by commas\n#the altitues for each layer separated by commas,\n#and a value for the outer scale l0\n#In the section [AO system] write the coordinates in arcsec for the NGS, \n#separated by commas in the format X1,Y1,X2,Y2,..., and the same for the LGS\n#Also provide the the altitude of the LGS, the number of Zernike modes corrected by the system\n#and the diameter of the primary mirror of the telescope (in m)\n#In the section [image] write the diameter of the aperture in pixels\n#---------------------------------------"
    
    with open(filename, 'w') as configfile:
        configfile.write(header_string+'\n')
        config.write(configfile)

def read(filename, separator=','):
    config.read(filename)
    r0 = config.get('atmosphere', 'r0')
    cn2 = config.get('atmosphere', 'cn2 profile')
    h = config.get('atmosphere', 'h profile')
    L0 = config.get('atmosphere', 'L0')
    ngs = config.get('AO system', 'ngs')
    lgs = config.get('AO system', 'lgs')
    lgs_height = config.get('AO system', 'lgs altitude')
    zm = config.get('AO system', 'zernike number')
    D = config.get('AO system', 'telescope diameter')
    pixels = config.get('image', 'pixel diameter')
#    objpos = config.get('image', 'PSF position')
    
    r0 = float(r0)
    L0 = float(L0)
    lgs_height =float(lgs_height)
    zm = int(zm)
    D = float(D)
    pixels = int(pixels)
    #Process Cn2 profile
    cn2_list = [float(i) for i in cn2.split(separator)]
    size_cn2 = len(cn2_list)
    cn2_array = np.array(cn2_list)
    #Process height profile
    h_list = [float(i) for i in h.split(separator)]
    size_h = len(h_list)
    h_array = np.array(h_list)
    if size_cn2 != size_h: raise ValueError('Cn2 and height profile must have same number of elements!')
    #Process NGS list
    ngs_list = [float(i) for i in ngs.split(separator)]
    size_ngs = len(ngs_list)
    ngs_array = np.array(ngs_list)
    ngs_array = ngs_array.reshape(size_ngs/2,2)
    #Process LGS list
    lgs_list = [float(i) for i in lgs.split(separator)]
    size_lgs = len(lgs_list)
    lgs_array = np.array(lgs_list)
    lgs_array = lgs_array.reshape(size_lgs/2,2)
    #Process list of objects
#    obj_list = [float(i) for i in objpos.split(separator)]
#    size_obj = len(obj_list)
#    obj_array = np.array(obj_list)
#    obj_array = obj_array.reshape(size_obj/2,2)
    atmosphere_dic = {'dr0': r0, 'cn2': cn2_array, 'h_profile': h_array, 'L0': L0}
#    return obj_array, atmosphere_dic, zm, D, pixels, ngs_array, lgs_array, lgs_height
    return atmosphere_dic, zm, D, pixels, ngs_array, lgs_array, lgs_height

if __name__ =="__main__":
    new("default.cfg", r0='90.3', cn2=[0.2,0.4,0.4], h_profile=[200.,5690., 9600.], L0='25', ngs=[-75.6,-75.6], lgs=[64.1,0.,0.,64.1,-64.1,0.,0.,-64.1], lgs_height = '90000.', zm='60', D='8', pixels='128', separator=',')
    read('default.cfg')
