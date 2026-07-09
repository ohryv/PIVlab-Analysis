import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import seaborn as sns

def about():
    resp = '''A set of functions meant to calculate velocity attenuation from processed PIVlab data.

    Overview of the functions in this file~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    For data analysis by the user:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    velocity_attenuation_heatmap(String filename, float left, float right, float bott, float top)--- Returns a heatmap of the velocity attenuation 
    ratio for *one* frame of PIVlab data, as well as a list of the attenuation values for each pixel in the same order as the input data. Must define
    the free stream region.

    timeavg_velocity_attenuation_heatmap(String filename, int start_frame, int end_frame, float left, float right, float bott, float top, 
    int digits = 4)--- Returns a time-averaged heatmap of the velocity attenuation ratio from PIVlab data, as well as a 1D Numpy array of the 
    time-averaged attenuation values for each pixel in the same order as the input data. Must define a free stream region that is constant in time.
    Arguably the most useful function in this whole set of data.

    small_area_attenuation_rect(String filename, floats pl, pr, pb, pt, fl, fr, fb, ft)--- Returns the mean value of the velocity attenuation ratio
    over a RECTANGULAR area, given that a rectangular free stream region is also provided. Returns ONE value from ONE frame.

    small_area_attenuation_rad(String filename, floats px, py, fl, fr, fb, ft, radius)--- Returns the mean value of the velocity attenuation ratio
    over a RADIAL area, given that a rectangular free stream region is also provided. Returns ONE value from ONE frame.

    timeavg_small_area_attenuation_rect(String filename, int start_frame, int end_frame, floats pl, pr, pb, pt, fl, fr, fb, ft, int digits = 4)---
    Returns the mean value of the velocity attenuation ratio over a RECTANGULAR area as a time average over several frames, given that a rectangular
    free stream region is also provided. Returns ONE value for MULTIPLE frames.

    timeavg_small_area_attenuation_rad(String filename, int start_frame, int end_frame, floats px, py, fl, fr, fb, ft, radius, int digits = 4)
    Returns the mean value of the velocity attenuation ratio over a RADIAL area as a time average over several frames, given that a rectangular
    free stream region is also provided. Returns ONE value for MULTIPLE frames.

    Strictly called by other functions:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    break_down_data(String filename)--- Uses Pandas to extract data from a .txt ASCII file and returns a DataFrame with its contents, as well as
    other useful information.
    
    find_file(String filename, int frame, int digits)--- When looping through a sequence of files, returns the full file name for a given frame's data

    inrange_rect(DataFrame data, String xcol, String ycol, int col_len, floats left, right, bott, top)--- Returns an array of indices in the DataFrame
    whose x and y values are within the provided rectangle

    mean_mag_velocity_rect(DataFrame data, float left, float right, float bott, float top)--- Calculates the mean magnitude of the velocity over a 
    defined rectangular area

    draw_heatmap(2DArray array, String title, Boolean diverging = False, Boolean normalized = False)--- Draws a heatmap of data in a 2D array.
    '''
    print(resp)

def break_down_data(filename):
    '''
    Reads the csv file and derives some useful numbers. Used to save space when a function takes a filename as an input.

    Parameters:
    String filename: path to the file you would like to analyze

    Returns, in order:
    DataFrame data = the Pandas DataFrame with all data
    String xcol, ycol, ucol, vcol = the names of the columns with the x coordinate, y coordinate, u-component of velocity, and v-component of 
    velocity, respectively. Allows generalizability with calibrated and noncalibrated data.
    int col_len = the number of data points
    vec_dist = the distance between each vector data point
    int xmin, xmax = maximum and minimum x values in the ORIGINAL data
    Numpy Array xrange = A numpy arange array of all x data. Does not correspond to image coordinates or the max/min
    int ymin, ymax = maximum and minimum y values in the ORIGINAL data
    Numpy Array yrange = A numpy arange array of all y data. Does not correspond to image coordinates or the max/min
    Numpy 2D Array xy = An array that creates a grid with exactly as many entries as there will be data points
    '''

    data = pd.read_csv(filename)

    #Get column names for x, y, and u. This makes it generalizable between calibrated and uncalibrated data
    xcol = data.columns[0]
    ycol = data.columns[1]
    ucol = data.columns[2]
    vcol = data.columns[3]
    #define the amount of data points (assuming same number of data in each col)
    col_len = len(data[xcol])

    #finds distance between vectors (will be useful later)
    for i in np.arange(col_len):
        vec_dist = data[xcol][i]-data[xcol][0]
        if vec_dist != 0:
            break

    #Some useful size variables
    xmin = data[xcol].min()
    xmax = data[xcol].max()
    xrange = np.unique(data[xcol])
    ymin = data[ycol].min()
    ymax = data[ycol].max()
    yrange = np.unique(data[ycol])
    #Initialize an array 
    xy = np.zeros((len(yrange), len(xrange)))

    return data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, xy

def find_file(filename, frame, digits):
    '''
    Parameters:
    String filename = file path up to the frame count
    int frame = the frame you want the file for
    digits = the number of digits the frame "suffix" takes up
    
    Returns:
    String fullfile = a full filename for a frame in a chronological sequence
    '''
    numstring = str(frame)
    while len(numstring)<digits:
        numstring = "0"+numstring
    fullfile = filename + numstring + ".txt"
    return fullfile

def inrange_rect(data, xcol, ycol, col_len, left, right, bott, top):
    '''
    Parameters:
    DataFrame data = the data that needs to fit in the box
    String xcol = the name of the column with the x-vals
    String ycol = the name of the column with the y-vals
    int col_len = length of the DataFrame columns
    float l: Left bound (lowest x-val)
    float r: Top bound (highest y-val)
    float b: Bottom bound (lowest y-val)
    float t: Right bound (highest x-val)
    
    Returns:
    Numpy Array box = the indices of rows of data that fits in the box
    '''
    box = []
    for i in np.arange(col_len):
            if (data[xcol][i]>=left and data[xcol][i]<=right and data[ycol][i]<=top and data[ycol][i]>=bott):
                box.append(i)
    box = np.array(box)
    return box

def mean_mag_velocity_rect(data, left, right, bott, top):
    '''
    Returns the mean magnitude of the velocity over a specified RECTANGULAR area.

    Parameters:
    DataFrame data = your original dataset with coordinates
    float left, right, bott, top = the coordinates of the corners. left = min x, right = max x, bott = min y, top = max x
    '''
    #Get column names for x, y, and u. This makes it generalizable between calibrated and uncalibrated data
    xcol = data.columns[0]
    ycol = data.columns[1]
    ucol = data.columns[2]
    vcol = data.columns[3]
    #define the amount of data points (assuming same number of data in each col)
    col_len = len(data[xcol])
    
    counter = 0
    cumsum = 0
    
    for i in np.arange(col_len):
        if (data[xcol][i]>(right) and data[ycol][i]>(top)):
                break #to save unneccesary computation over the rest of the frame
        if (data[xcol][i]>=left and data[xcol][i]<=right and data[ycol][i]<=top and data[ycol][i]>=bott and not np.isnan(data[ucol][i])):
            counter+=1
            temp = (data[ucol][i])**2+(data[vcol][i])**2 # x^2+y^2
            cumsum+= math.sqrt(temp) #(x^2+y^2)^(1/2)
    mean_mag = cumsum/counter
    return mean_mag

def draw_heatmap(array, title, diverging = True, normalized = False):
    '''
    Graphs a given 2D array with a heatmap centered at 0, using a dark, diverging colormap that allows masks and NaNs to be seen.
    X and Y labels are turned off and assumed to be relative.

    Parameters:
    2D Array or Arraylike array: a NumPy array or Pandas DataFrame or whatever with the data you want plotted
    String title: Describes what's being plotted
    ''' 
    
    #Graphs the data using a seaborn heatmap
    if diverging:
        heatmap = sns.heatmap(array, center = 0, cmap = "berlin", xticklabels = False, yticklabels = False)
    elif normalized:
        heatmap = sns.heatmap(array, vmin = 0, vmax = 1, cmap = "viridis", xticklabels = False, yticklabels = False)
    else:
        heatmap = sns.heatmap(array, cmap = "viridis", xticklabels = False, yticklabels = False)
    #heatmap.xaxis.tick_top()
    plt.xlabel("x (relative)")
    plt.ylabel("y (relative)")
    plt.title(title)
    plt.tight_layout(pad=2.0)

def velocity_attenuation_heatmap(filename, left, right, bott, top, output_title = "Output/velocityattenuationheatmap.png"):
    '''
    Returns a heatmap of velocity attenuation compared to a defined free-stream area for an individual frame.
    Compares the *magnitude of the velocity.*

    Also draws a rectangle of the defined free stream area over the plot. If you are working with the very boundary of the image, add one pixel to 
    make sure the box shows up properly (i.e. image ends on pixel 800, put right as 801 for the right bound) 

    Parameters:
    String filename: path of file for analysis

    The following parameters are meant to give a rectangle of area considered to have "free stream" behavior, far away from the coral sample.
    These values are INCLUSIVE
    !!!!!NOTE!!!!! If using CALIBRATED data, be EXACT with the bounds! Otherwise the code will not select as much FS data as you intended, even if the
    box looks fine!
    float left: Left bound (lowest x-val)
    float top: Top bound (highest y-val)
    float right: Right bound (highest x-val)
    float bott: Bottom bound (lowest y-val)
    These assume x increasing to the right and y increasing toward the bottom

    (optional, but highly recommended) String output_title: How should the output be labeled? Default: "Output/velocityattenuationheatmap.png"

    Returns:
    The heatmaps, duh
    A 1D Numpy array mag_attenuation with the velocity attenuation ratios for each data point in the same order as the original data.

    Credit to Dr. Alison Weber for help debugging the box around the free stream
    '''

    #Unloads data
    data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, pointarray_mag = break_down_data(filename)
    
    #Average mag over the rectangle using a for loop to get the free stream velocity
    mean_mag = mean_mag_velocity_rect(data, left, right, bott, top)
    print("Free Stream Velocity:", mean_mag)
    
    #For the whole data set, find the velocity attenuation as the velocity at that point over the free stream velocity
    #Add the results to the original DataFrame

    #vel magnitude
    mag_attenuation = np.zeros(col_len)
    velmag = np.zeros(col_len)
    for i in np.arange(col_len):
        temp = (data[ucol][i])**2+(data[vcol][i])**2 # x^2+y^2
        temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
        velmag[i] = temp
        mag_attenuation[i] = temp/mean_mag

    #pass through the data to assign values to the array
    i=0
    for x in np.arange(len(xrange)):
        for y in np.arange(len(yrange)):
            pointarray_mag[y][x] = mag_attenuation[i]
            i+=1
    
    #Graph each using the draw_heatmap() function
    #Also box off free stream area using rect = mpatches.Rectangle((left, bottom), width, height, fill=False, edgecolor="color", linewidth=width)


    #define the coordinates of the box in the modified grid
    fs_box = inrange_rect(data, xcol, ycol, col_len, left, right, bott, top)
    x_in_box = []
    y_in_box = []
    for i in fs_box:
        x_in_box.append(data[xcol][i])
        y_in_box.append(data[ycol][i])
    x_in_box = np.array(x_in_box)
    y_in_box = np.array(y_in_box)

    fl = np.round((left-xmin)/vec_dist)
    fb = np.round((bott-ymin)/vec_dist)
    
    fswidth = len(np.unique(x_in_box))
    fsheight = len(np.unique(y_in_box))
    
    #Plot the graph
    fig = plt.figure(figsize=(5.5,5))
    draw_heatmap(pointarray_mag, "Velocity Attenuation Ratio for Total Velocity Magnitude", normalized = True, diverging = False)
    rect = mpatches.Rectangle((fl, fb), fswidth, fsheight, fill=False, edgecolor="red", linewidth=2)
    ax = plt.gca()
    ax.add_patch(rect)
    rect.set_zorder(10000)

    plt.savefig(output_title)
    plt.show()
    return mag_attenuation

def timeavg_velocity_attenuation_heatmap(filename, start_frame, end_frame, left, right, bott, top, digits = 4, output_title = "Output/velocityattenuationheatmap.png"):
    '''
    Finds a heatmap of the velocity attenuation ratio from PIVlab data using a defined free stream region. This function takes this as a time
    average over a set amount of frames, averaging the velocities before deriving the attenuation value at the end. In the case of missing data, 
    averages over the number of available data and returns a heatmap of the amount of available data for each pixel to display where more data may
    have been missing.

    Also draws a rectangle of the defined free stream area over the plot. If you are working with the very boundary of the image, add one pixel to 
    make sure the box shows up properly (i.e. image ends on pixel 800, put right as 801 for the right bound) 

    Parameters:
    String filename: path of file for analysis UNTIL THE STRING OF NUMBERS THAT DENOTES FRAME NUMBER

    int start_frame = the first frame you want analyzed
    int end_frame = the last frame you want analyzed (i.e. inclusive!)

    The following parameters are meant to give a rectangle of area considered to have "free stream" behavior, far away from the coral sample.
    These values are INCLUSIVE.
    !!!!!NOTE!!!!! If using CALIBRATED data, be EXACT with the bounds! Otherwise the code will not select as much FS data as you intended, even if the
    box looks fine!
    float left: Left bound (lowest x-val)
    float top: Top bound (highest y-val)
    float right: Right bound (highest x-val)
    float bott: Bottom bound (lowest y-val)
    These assume x increasing to the right and y increasing toward the bottom

    (optional) int digits: Default = 4. The string of numbers at the end of your file names should not be included in filename. How many 'digits' are there 
    of the string of numbers at the end? i.e, for file data_0001, digits=4. For file data_001, digits=3.
    (optional, but highly recommended) String output_title: How should the output be labeled? Default: "Output/velocityattenuationheatmap.png"

    Returns:
    The heatmaps, duh
    A 1D Numpy array mag_attenuation with the velocity attenuation ratios for each data point in the same order as the original data.
    '''

    # #Standard unpacking from other fns
    #Define variables
    framect = end_frame - start_frame + 1 #because inclusive
    #unpack some key values based off of the first frame. All of these, except for the dataframe, should be constant among all frames.
    fullfile = find_file(filename, start_frame, digits)
    data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, cumarray_mag = break_down_data(fullfile)
    val_pts_arr = cumarray_mag.copy()

    #other vars
    mag_avg = np.zeros(col_len)
    val_pts = framect + np.zeros(col_len, dtype = "int")
    fs_mag = 0
    ct_mag = 0
    
    #Define an array of valid indices in fs box (saves computational time)
    fs_box = inrange_rect(data, xcol, ycol, col_len, left, right, bott, top)
    

    #for each frame, 1) get the filename, 2) unpack the data, 3) find attenuation
    for k in np.arange(framect):
        fullfile = find_file(filename, start_frame+k, digits)
        #Unloads data
        data = pd.read_csv(fullfile)
    
        #Average u and mag over the rectangle using a for loop to get the free stream velocity

        #velocity magnitude
        for i in fs_box:
            if not np.isnan(data[ucol][i]) and not np.isnan(data[vcol][i]):
                ct_mag+=1
                temp = (data[ucol][i])**2+(data[vcol][i])**2
                temp = math.sqrt(temp)
                fs_mag+=temp
    
        #For each frame, add the velocity to the cumulative list

        #vel magnitude
        for i in np.arange(col_len):
            if (not np.isnan(data[ucol][i]) and not np.isnan(data[vcol][i])):
                temp = (data[ucol][i])**2+(data[vcol][i])**2 # x^2+y^2
                temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
                mag_avg[i] += temp
            else:
                val_pts[i]-=1

    #once the loop has ended, divide the cumulative values to avg
    fs_mag = fs_mag/ct_mag # averaged over the total number of vectors in the box
    mag_avg = mag_avg/val_pts #averaged over time

    #Define attenuation and pass through the data to assign values to the array
    i=0
    mag_attenuation = mag_avg/fs_mag
    for x in np.arange(len(xrange)):
        for y in np.arange(len(yrange)):
            cumarray_mag[y][x] = mag_attenuation[i]
            val_pts_arr[y][x] = val_pts[i]
            i+=1
    
    #Graph each using the draw_heatmap() function
    #Also box off free stream area using rect = mpatches.Rectangle((left, bottom), width, height, fill=False, edgecolor="color", linewidth=width)

    #First, create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(11,5), sharey=True)
    #define the coordinates of the box in the modified grid
    x_in_box = []
    y_in_box = []
    for i in fs_box:
        x_in_box.append(data[xcol][i])
        y_in_box.append(data[ycol][i])
    x_in_box = np.array(x_in_box)
    y_in_box = np.array(y_in_box)

    fl = np.round((left-xmin)/vec_dist)
    fb = np.round((bott-ymin)/vec_dist)
    
    fswidth = len(np.unique(x_in_box))
    fsheight = len(np.unique(y_in_box))
    
    #Plot the top graph

    plt.sca(axes[0])
    draw_heatmap(cumarray_mag, "Velocity Attenuation Ratio for Total Velocity Magnitude", normalized = True, diverging = False)
    rect = mpatches.Rectangle((fl, fb), fswidth, fsheight, fill=False, edgecolor="red", linewidth=2)
    axes[0].add_patch(rect)
    rect.set_zorder(10000)

    #plot how many valid data points there are for each
    i=0
    for x in np.arange(len(xrange)):
        for y in np.arange(len(yrange)):
            if (val_pts_arr[y][x] == 0):
                val_pts_arr[y][x] = float('NaN')
                i+=1
    plt.sca(axes[1])
    draw_heatmap(val_pts_arr, "Number of Valid (Non-Nan) Vectors For Each Pixel", normalized = False, diverging = False)
    plt.savefig(output_title)
    plt.show()
    
    return mag_attenuation






































































    

def small_area_attenuation_rect(filename, pl, pr, pb, pt, fl, fr, fb, ft):
    '''
    Returns the average velocity attenuation of some small RECTANGULAR area (such as the region around a polyp or other structure) compared with 
    some defined rectangular free stream region. This applies to ONE frame.

    Parameters:
    String filename: path of file for analysis

    float pl = literally "polyp left", left coordinate of the region of interest
    float pr = literally "polyp right", right coordinate of the region of interest
    float pb = literally "polyp bottom", bottom (lowest y value) coordinate of the region of interest
    float pt = literally "polyp top", top (highest y value) coordinate of the region of interest

    float fl = literally "free stream left", left coordinate of the free stream region
    float fr = literally "free stream right", right coordinate of the free stream region
    float fb = literally "free stream bottom", bottom (lowest y value) coordinate of the free stream region
    float ft = literally "free stream top", top (highest y value) coordinate of the free stream region

    Returns:
    float attenuation = the velocity attenuation ratio between the mean magnitude of velocity of the ROI and the mean magnitude of velocity of 
    the free stream region
    
    '''
    #Unpack the filename
    data = pd.read_csv(filename)
    #Recycle code to get free stream velocity
    mean_mag_FS = mean_mag_velocity_rect(data, fl, fr, fb, ft)
    print("Free Stream Velocity:", mean_mag_FS)
    #Recycle again to get ROI velocity
    mean_mag_ROI = mean_mag_velocity_rect(data, pl, pr, pb, pt)
    print("ROI Velocity:", mean_mag_ROI)
    attenuation = mean_mag_ROI/mean_mag_FS
    return attenuation

def timeavg_small_area_attenuation_rect(filename, start_frame, end_frame, pl, pr, pb, pt, fl, fr, fb, ft, digits = 4):
    '''
    Returns the average velocity attenuation of some small area (such as the region around a polyp or other structure) compared with some defined
    free stream region. This is TIME AVERAGED over multiple frames by finding the average velocity in each region through an aggregate of all available
    data in the region.

    Parameters:
    String filename: path of file for analysis UNTIL THE STRING OF NUMBERS THAT DENOTES FRAME NUMBER

    int start_frame = the first frame you want analyzed
    int end_frame = the last frame you want analyzed (i.e. inclusive!)

    float pl = literally "polyp left", left coordinate of the region of interest
    float pr = literally "polyp right", right coordinate of the region of interest
    float pb = literally "polyp bottom", bottom (lowest y value) coordinate of the region of interest
    float pt = literally "polyp top", top (highest y value) coordinate of the region of interest

    float fl = literally "free stream left", left coordinate of the free stream region
    float fr = literally "free stream right", right coordinate of the free stream region
    float fb = literally "free stream bottom", bottom (lowest y value) coordinate of the free stream region
    float ft = literally "free stream top", top (highest y value) coordinate of the free stream region

    (optional) int digits: Default = 4. The string of numbers at the end of your file names should not be included in filename. How many 'digits' are there 
    of the string of numbers at the end? i.e, for file data_0001, digits=4. For file data_001, digits=3.

    Returns:
    float attenuation = the velocity attenuation ratio between the mean magnitude of velocity of the ROI and the mean magnitude of velocity of 
    the free stream region
    
    '''
    
    #Define variables
    framect = end_frame - start_frame + 1 #because inclusive
    #Places to sum our stuff
    cum_ROI_mag = 0
    cum_FS_mag = 0
    ct_ROI = 0
    ct_FS = 0
    #Define what is in the "box" now so that the for loop doesn't go thru each point
    fullfile = find_file(filename, start_frame, digits)
    data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, pointarray_mag = break_down_data(fullfile)
    ROI_box = inrange_rect(data, xcol, ycol, col_len, pl, pr, pb, pt)
    fs_box = inrange_rect(data, xcol, ycol, col_len, fl, fr, fb, ft)

    x_in_box = []
    y_in_box = []
    for i in fs_box:
        x_in_box.append(data[xcol][i])
        y_in_box.append(data[ycol][i])
    x_in_box = np.array(x_in_box)
    y_in_box = np.array(y_in_box)

    print("FS xrange:", x_in_box[0], "to", x_in_box[len(x_in_box)-1])
    print("FS yrange:", y_in_box[0], "to", y_in_box[len(y_in_box)-1])

    x_in_box = []
    y_in_box = []
    for i in ROI_box:
        x_in_box.append(data[xcol][i])
        y_in_box.append(data[ycol][i])
    x_in_box = np.array(x_in_box)
    y_in_box = np.array(y_in_box)

    print("ROI xrange:", x_in_box[0], "to", x_in_box[len(x_in_box)-1])
    print("ROI yrange:", y_in_box[0], "to", y_in_box[len(y_in_box)-1])
    

    #for each frame, 1) get the filename, 2) unpack the data, 3) find attenuation
    for i in np.arange(framect):
        fullfile = find_file(filename, start_frame+i, digits)
        data = pd.read_csv(fullfile)

        #Add velocities to cumulative for both free stream and ROI
        for j in ROI_box:
            if not np.isnan(data[ucol][j]) and not np.isnan(data[vcol][j]):
                ct_ROI+=1
                temp = (data[ucol][j])**2+(data[vcol][j])**2 # x^2+y^2
                temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
                cum_ROI_mag+= temp
        for j in fs_box:
            if not np.isnan(data[ucol][j]) and not np.isnan(data[vcol][j]):
                ct_FS+=1
                temp = (data[ucol][j])**2+(data[vcol][j])**2 # x^2+y^2
                temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
                cum_FS_mag+= temp
    
    #Get avg velocity mag
    avg_ROI_mag = cum_ROI_mag/ct_ROI
    print("ROI Mean Velocity:", avg_ROI_mag)
    avg_FS_mag = cum_FS_mag/ct_FS
    print("Free Stream Velocity", avg_FS_mag)
    #Derive attenuation
    avg_att = avg_ROI_mag/avg_FS_mag
    return avg_att

def small_area_attenuation_rad(filename, px, py, fl, fr, fb, ft, rad):
    '''
    Returns the average velocity attenuation of some small area (such as the region around a polyp or other structure) compared with some defined
    free stream region. This applies to ONE frame.

    Parameters:
    String filename: path of file for analysis

    float px = literally "polyp x", x coordinate of the center of the region of interest
    float py = literally "polyp y", y coordinate of the center of the region of interest

    float fl = literally "free stream left", left coordinate of the free stream region
    float fr = literally "free stream right", right coordinate of the free stream region
    float fb = literally "free stream bottom", bottom (lowest y value) coordinate of the free stream region
    float ft = literally "free stream top", top (highest y value) coordinate of the free stream region

    int rad: How many vectors out from the center of the region of interest are we going? Make sure to not make this too large, and pls don't put 0.

    Returns:
    float attenuation = the velocity attenuation ratio between the mean magnitude of velocity of the ROI and the mean magnitude of velocity of 
    the free stream region
    
    '''

    #Unpack the filename
    data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, unneccesary_array = break_down_data(filename)
    #Convert radius to useable number
    radius = (float(rad)+0.1)*vec_dist
    #Recycle code to get free stream velocity
    mean_mag_FS = mean_mag_velocity_rect(data, fl, fr, fb, ft)
    print("Free Stream Velocity:", mean_mag_FS)

    counter = 0
    cumsum = 0
    circ_coords = []
    for i in np.arange(col_len):
        distance = (data[xcol][i]-px)**2+(data[ycol][i]-py)**2
        distance = math.sqrt(distance)
        if (distance<=radius and not np.isnan(data[ucol][i])):
            if (data[xcol][i]>(px+radius) and data[ycol][i]>(py+radius)):
                break #to save unneccesary computation over the rest of the frame
            counter+=1
            circ_coords.append([data[xcol][i], data[ycol][i]])
            temp = (data[ucol][i])**2+(data[vcol][i])**2 # x^2+y^2
            cumsum+= math.sqrt(temp) #(x^2+y^2)^(1/2)
    mean_mag_ROI = cumsum/counter
    print("ROI Mean Velocity:", mean_mag_ROI)
    print("Vectors in circle:", len(circ_coords))

    #divide and return
    attenuation = mean_mag_ROI/mean_mag_FS
    return attenuation

def timeavg_small_area_attenuation_rad(filename, start_frame, end_frame, px, py, fl, fr, fb, ft, rad, digits = 4):
    '''
    Returns the average velocity attenuation of some small area (such as the region around a polyp or other structure) compared with some defined
    free stream region. This is TIME AVERAGED over multiple frames by finding the average velocity in each region through an aggregate of all available
    data in the region.

    Parameters:
    String filename: path of file for analysis UNTIL THE STRING OF NUMBERS THAT DENOTES FRAME NUMBER

    int start_frame = the first frame you want analyzed
    int end_frame = the last frame you want analyzed (i.e. inclusive!)

    float px = literally "polyp x", x coordinate of the center of the region of interest
    float py = literally "polyp y", y coordinate of the center of the region of interest

    float fl = literally "free stream left", left coordinate of the free stream region
    float fr = literally "free stream right", right coordinate of the free stream region
    float fb = literally "free stream bottom", bottom (lowest y value) coordinate of the free stream region
    float ft = literally "free stream top", top (highest y value) coordinate of the free stream region

    int rad: How many vectors out from the center of the region of interest are we going? Make sure to not make this too large, and pls don't put 0.

    Returns:
    float attenuation = the velocity attenuation ratio between the mean magnitude of velocity of the ROI and the mean magnitude of velocity of 
    the free stream region
    
    '''
    #Define variables
    framect = end_frame - start_frame + 1 #because inclusive
    #Places to sum our stuff
    cum_ROI_mag = 0
    cum_FS_mag = 0
    ct_ROI = 0
    ct_FS = 0
    #Define what is in the "box" now so that the for loop doesn't go thru each point
    fullfile = find_file(filename, start_frame, digits)
    data, xcol, ycol, ucol, vcol, col_len, vec_dist, xmin, xmax, xrange, ymin, ymax, yrange, pointarray_mag = break_down_data(fullfile)
    #Convert radius to useable number
    radius = (float(rad)+0.1)*vec_dist
    fs_box = inrange_rect(data, xcol, ycol, col_len, fl, fr, fb, ft)

    ROI_box = []
    for i in np.arange(col_len):
        distance = (data[xcol][i]-px)**2+(data[ycol][i]-py)**2
        distance = math.sqrt(distance)
        if (distance<=radius and not np.isnan(data[ucol][i])):
            if (data[xcol][i]>(px+radius) and data[ycol][i]>(py+radius)):
                break #to save unneccesary computation over the rest of the frame
            ROI_box.append(i)
    ROI_box = np.array(ROI_box)
    print("Coords in Circle:", len(ROI_box))
    

    #for each frame, 1) get the filename, 2) unpack the data, 3) find attenuation
    for i in np.arange(framect):
        fullfile = find_file(filename, start_frame+i, digits)
        data = pd.read_csv(fullfile)

        #Add velocities to cumulative for both free stream and ROI
        for j in ROI_box:
            if (not np.isnan(data[ucol][j]) and not np.isnan(data[vcol][j])):
                ct_ROI+=1
                temp = (data[ucol][j])**2+(data[vcol][j])**2 # x^2+y^2
                temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
                cum_ROI_mag+= temp
        for j in fs_box:
            if (not np.isnan(data[ucol][j]) and not np.isnan(data[vcol][j])):
                ct_FS+=1
                temp = (data[ucol][j])**2+(data[vcol][j])**2 # x^2+y^2
                temp = math.sqrt(temp) #(x^2+y^2)^(1/2)
                cum_FS_mag+= temp
    
    #Get avg velocity mag
    avg_ROI_mag = cum_ROI_mag/ct_ROI
    print("ROI Mean Velocity:", avg_ROI_mag)
    avg_FS_mag = cum_FS_mag/ct_FS
    print("Free Stream Velocity:", avg_FS_mag)
    #Derive attenuation
    avg_att = avg_ROI_mag/avg_FS_mag
    return avg_att