import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import math

def compareVectors(control_filename, filtered_filename, num_frames, digits=4):
    '''
    A function which will, for each frame of a video clip analyzed in PIVlab and exported as an ASCII text file, give the percent of vectors that
    have been replaced/interpolated using validation filters. Assuming reasonable filters (that is up to the user), this will give the amount of
    usable vectors after removing noise or interpolating missing values.

    Ensure that all files for filtered/unfiltered data are in the same folder!
    THIS WILL ONLY WORK FOR .TXT FILES!
    Also, as much as it is the most annoying thing ever, if your path copies with backslashes (\) you will need to manually change them to
    forward slashes (/). I know. Python throws a whole fit about it.

    NOTE: Assumes the following settings on export...
    Delimiter: comma
    ONLY "add column headers" on

    Inputs:
    
    str control_filename: 
    TLDR: The file path of a .txt file from your PIVlab export WITHOUT the frame number and the .txt suffix.
    When exporting multiple frames, the files will be named something_0001, something_002, et cetera.
    This string will take the WHOLE PATH (as in, from the C:user/documents etc etc) UP UNTIL the numbers at the end. So, a file may look something like
    C:/user/downloads/folder/something_0001.txt

    str filtered_filename: Same as control_filename, but for your FILTERED data (after applying validation)

    int num_frames: How many frames are there?

    (optional) int digits: The string of numbers at the end of your file names should not be included in filename. How many 'digits' are there 
    of the string of numbers at the end? i.e, for file data_0001, digits=4. For file data_001, digits=3.

    Outputs:

    An array of float64 values representing the Valid Detection Probability (VDP) of the filtered vectors as a percent, 
    in chronological order of the frames.

    In other words, it calculates how many vectors (excluding vectors within the mask, which will always be NaN) remain unchanged after applying
    validation filters. This is useful for getting an idea of how noisy your data is or whether you're overfiltering it. Ideally, the VDP should be
    above 90% for a given frame. In other words, you shouldn't filter out/interpolate more than 10% of your vectors per frame.

    '''
    valid_vector_percent=np.zeros(num_frames)
    counter = np.arange(num_frames)

    #for each frame, calculate VDP
    for i in counter:
        numstring = str(i+1)
        while len(numstring)<digits:
            numstring = "0"+numstring
        control = pd.read_csv(control_filename+numstring+".txt")
        filtered = pd.read_csv(filtered_filename+numstring+".txt")
        num_vectors = len(control[control.columns[1]])
        isTheSame = [False]*num_vectors
        isMask = [False]*num_vectors

        #Has the vector been interpolated? Is it part of the mask?
        for j in np.arange(num_vectors):
            isTheSame[j] = (str(control[control.columns[2]][j]) == str(filtered[filtered.columns[2]][j])) and (str(control[control.columns[3]][j]) == str(filtered[filtered.columns[3]][j]))
            isMask[j] = np.isnan(filtered[filtered.columns[2]][j]) and np.isnan(filtered[filtered.columns[3]][j])
            
        numSame = np.cumsum(isTheSame)[len(np.cumsum(isTheSame))-1]
        numMasked = np.cumsum(isMask)[len(np.cumsum(isMask))-1]

        #How many of the non-mask vectors remain as their original values?
        valid_vector_percent[i] = (numSame-numMasked)/(num_vectors-numMasked)*100
    return valid_vector_percent

def histogramVDP(control_filename, filtered_filename, num_frames, digits=4):
    '''
    Plots a histogram of VDP (valid detection probability) values, along with a bar at 90% to show if the vectors meet the ideal conditions.
    Also prints the percent above 90%

    Takes the same inputs as compareVectors() in order to call it.
    '''
    #get data
    VDP_data = compareVectors(control_filename, filtered_filename, num_frames, digits)

    #greater than 90%?
    valid_data = np.where(VDP_data >= 90)
    print('Of your data,',(len(valid_data[0])/len(VDP_data))*100,"% is acceptable.")

    #worryingly low?
    concerning_data = np.where(VDP_data <= 70)
    print('Of your data,',(len(concerning_data[0])/len(VDP_data))*100,"% is really worrying.")

    #plot
    plt.figure(figsize=(6, 4), dpi=100)

    #Decide bins for histogram
    min_VDP = math.floor(VDP_data.min())
    max_VDP = math.ceil(VDP_data.max())+1
    binses = np.arange(min_VDP, max_VDP, 1.0)

    #plot for realsies
    plt.hist(VDP_data, bins = binses, color = 'teal')
    plt.title('VDP distribution across video clip')
    plt.xlabel('VDP (%)') 
    plt.ylabel('# of frames') 
    plt.vlines(ymin = 0, ymax = int(len(VDP_data)*(1/3)), x=90.0, color='red')
    plt.show()

def findWeirdos(control_filename, filtered_filename, num_frames, digits=4):
    '''
    Plots VDP of each frame in sequence, so that unusually low VDPs can be isolated to their source. 
    VDPs (valid detection probabilities) below 80%, and then below 70% will be printed with their frame number.

    Takes the same inputs as compareVectors() in order to call it.
    '''

    #plot the data
    xaxis = np.arange(1, num_frames+1)
    yaxis = compareVectors(control_filename, filtered_filename, num_frames, digits)
    plt.scatter(xaxis, yaxis, color = 'pink')
    plt.title('VDPs over time')
    plt.ylabel('VDP (%)') 
    plt.xlabel('Frame') 

    #indicators
    plt.hlines(xmin = 0, xmax = len(xaxis), y=90.0, color='green')
    plt.hlines(xmin = 0, xmax = len(xaxis), y=70.0, color='red')
    plt.show()

    #worryingly low?

    if (yaxis.min()<=80):
        concerning_data = np.where(yaxis <= 80)
        print('Caution! The following data is below 80!')
        for i in concerning_data[0]:
            print(yaxis[i],"%, frame",i)
        
    if (yaxis.min()<=70):
        bad_data = np.where(yaxis <= 70)
        print('WARNING! The following data is below 70!')
        for i in bad_data[0]:
            print(yaxis[i],"%, frame",i)