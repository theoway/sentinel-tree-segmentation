import numpy as np
from math import sqrt
from scipy.ndimage import median_filter

def calc_indices(arr):
    
    '''
    Takes in an array of 10 and 20m sentinel-2 bands as input and 
    calculates 55 remote sensing indices. Returns indices 
    as a combined ndarray.
    '''
    
    # define bands
    blue = arr[...,0]
    green = arr[...,1] 
    red = arr[...,2]
    nir = arr[...,3] 
    red_edge1 = arr[...,4] 
    red_edge2 = arr[...,5]
    red_edge3 = arr[...,6] 
    narrow_nir = arr[...,7] 
    swir1 = arr[...,8]
    swir2 = arr[...,9]
    
    # calculate RS indices
    ndvi = (nir-red) / (nir+red)  
    
    atsavi = 1.22*((nir-1.22*red-0.03) / (1.22*nir+red-1.22*0.03+0.08*(1+1.22**2)))

    arvi = (nir-red-0.069*(red-blue)) / (nir+red-0.069*(red-blue)) 
     
    arvi2 = (-0.18+1.17)*ndvi
     
    bwdrvi = (0.1*nir-blue) / (0.1*nir+blue)
    
    ccci = ((nir-red_edge1) / (nir+red_edge1)) / ((nir-red_edge1) / (nir+red_edge1)) 
     
    chl_green = (red_edge3/green)**-1 
    
    ci_green = (nir/green)*-1 
    
    ci_rededge = (nir/red_edge1)*-1 
    
    chl_rededge = (red_edge3/red_edge1)**-1 
     
    cvi = nir*(red/green**2) 
    
    ci = (red-blue) / red 
    
    ctvi = ((ndvi+0.5) / np.abs((ndvi)+0.5))*np.sqrt(np.abs(ndvi+0.5))
     
    gdvi = nir-green 
    
    evi = 2.5*((nir-red) / ((nir+6*red-7.5*blue)+1)) 
    
    def global_env_mon_index(nir, red):
        n = (2*(nir**2-red**2)+1.5*nir+0.5*red) / (nir+red+0.5)
        gemi = (n*(1-0.25*n)-((red-0.125) / (1-red))) 
        return gemi
        
    gemi = global_env_mon_index(nir, red) 
    
    gli = (2*green-red-blue) / (2*green+red+blue) 
    
    gndvi = (nir-green) / (nir+green) 
    
    gosavi = (nir-green) / (nir+green+0.16) 
    
    gsavi = ((nir-green) / (nir+green+0.5))*(1+0.5) 
    
    gbndvi = (nir-(green+blue)) / (nir+(green+blue)) 
    
    grndvi = (nir-(green+red)) / (nir+(green+red)) 
    
    hue = np.arctan(((2*red-green-blue) / 30.5)*(green-blue)) 
    
    ivi = (nir-0.809) / (0.393*red) 
    
    ipvi = ((nir / nir+red)/2)*(ndvi+1) 
    
    intensity = (1/30.5)*(red+green+blue) 
    
    lwci = np.log(1.0-(nir-0.101)) / (-np.log(1.0-(nir-0.101)))
        
    msavi2 = (2*nir+1 - np.sqrt(np.abs((2*nir+1)**2-8*(nir-red)))) / 2 
        
    normg = green / (nir+red+green) 
    
    normnir = nir / (nir+red+green)
    
    normr = red / (nir+red+green)
    
    ndmi = (nir-swir1) / (nir+swir1) 
    
    ngrdi = (green-red) / (green+red)
    
    ndvi_ad = (swir2-nir) / (swir2+nir)  
    
    bndvi = (nir-blue) / (nir+blue) 
        
    mndvi = (nir-swir2) / (nir+swir2) 

    nbr = (nir-swir2) / (nir+swir2) 
    
    ri = (red-green) / (red+green) 
    
    ndvi690_710 = (nir-red_edge1) / (nir+red_edge1) 
    
    pndvi = (nir-(green+red+blue)) / (nir+(green+red+blue)) 
    
    pvi = (1 / np.sqrt(0.149**2+1)) * (nir-0.374-0.735) 
    
    rbndvi = (nir-(red+blue)) / (nir+(red+blue)) 
    
    rsr = (nir / red)*0.640-(swir2 / 0.640)-0.259 
        
    rdi = (swir2 / nir) 
    
    srnir = (nir / red_edge1)
    
    grvi = (nir / green) 
    
    dvi = (nir / red) 
    
    slavi = (nir / (red_edge1+swir2))
        
    gvi = (-0.2848*blue-0.2435*green-0.5436*red+0.7243*nir+0.0840*swir1-0.1800*swir2)
    
    wet = (0.1509*blue+0.1973*green+0.3279*red+0.3406*nir-0.7112*swir1-0.4572*swir2) 
    
    tsavi = (0.421*(nir-0.421*red-0.824)) / (red+0.421*(nir-0.824)+0.114*(1+0.421**2)) 
    
    tvi = np.sqrt(np.abs(ndvi+0.5)) 
    
    vari_rededge = (red_edge1-red) / (red_edge1+red)
    
    wdvi = (nir-0.752*red) 
    
    bsi = (swir1+red)-(nir+blue) / (swir1+red)+(nir+blue) 
        
    full_list = [ndvi, atsavi, arvi, arvi2, bwdrvi, ccci, chl_green, ci_green, 
               ci_rededge, chl_rededge, cvi, ci, ctvi, gdvi, evi, gemi, gli, 
               gndvi, gosavi, gsavi, gbndvi, grndvi, hue, ivi, ipvi, intensity, 
               lwci, msavi2, normg, normnir, normr, ndmi, ngrdi, ndvi_ad, bndvi, 
               mndvi, nbr, ri, ndvi690_710, pndvi, pvi, rbndvi, rsr, rdi, srnir, 
               grvi, dvi, slavi, gvi, wet, tsavi, tvi, vari_rededge, wdvi, bsi]
    
    gs_5 = [evi, msavi2, ndvi, ndmi, bsi] # RS indices for gridsearch
    
    rs_indices = np.empty((arr.shape[0], arr.shape[1], arr.shape[2], len(full_list)), dtype=np.float32)
    gs_indices = np.empty((arr.shape[0], arr.shape[1], arr.shape[2], len(gs_5)), dtype=np.float32)
    
    for i, v in enumerate(full_list):
        rs_indices[..., i] = v
    
    for i, v in enumerate(gs_5):
        gs_indices[..., i] = v
    
    return gs_indices, rs_indices

def process_dem(dem):
    dem =  median_filter(dem, size = 5)
    dem = calcSlope(dem.reshape((1, 512, 512)),
                      np.full((512, 512), 10),
                      np.full((512, 512), 10), 
                      zScale = 1, minSlope = 0.02)
    dem = dem / 90
    dem = dem.reshape((512, 512, 1))
    dem = dem[1:-1, 1:-1]
    dem = median_filter(dem, 5)[2:-2, 2:-2]
    return dem

def slopePython(inBlock, outBlock, inXSize, inYSize, zScale=1):
    """ Calculate slope using Python.
        If Numba is available will make use of autojit function
        to run at ~ 1/2 the speed of the Fortran module.
        If not will fall back to pure Python - which will be slow!
    """
    for x in range(1, inBlock.shape[2] - 1):
        for y in range(1, inBlock.shape[1] - 1):
            # Get window size
            dx = 2 * inXSize[y, x]
            dy = 2 * inYSize[y, x]

            # Calculate difference in elevation
            dzx = (inBlock[0, y, x - 1] - inBlock[0, y, x + 1]) * zScale
            dzy = (inBlock[0, y - 1, x] - inBlock[0, y + 1, x]) * zScale

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0, y, x] = slopeDeg

    return outBlock


def slopePythonPlane(inBlock,
                     outBlock,
                     inXSize,
                     inYSize,
                     A_mat,
                     z_vec,
                     winSize=3,
                     zScale=1):
    """ Calculate slope using Python.
        Algorithm fits plane to a window of data and calculated the slope
        from this - slope than the standard algorithm but can deal with
        noisy data batter.
        The matrix A_mat (winSize**2,3) and vector zScale (winSize**2) are allocated
        outside the function and passed in.
    """

    winOffset = int(winSize / 2)

    for x in range(winOffset - 1, inBlock.shape[2]):
        for y in range(winOffset - 1, inBlock.shape[1]):
            # Get window size
            dx = winSize * inXSize[y, x]
            dy = winSize * inYSize[y, x]

            # Calculate difference in elevation
            """
                Solve A b = x to give x
                Where A is a matrix of:
                    x_pos | y_pos | 1
                and b is elevation
                and x are the coefficents
            """

            # Form matrix
            index = 0
            for i in range(-1 * winOffset, winOffset + 1):
                for j in range(-1 * winOffset, winOffset + 1):

                    A_mat[index, 0] = 0 + (i * inXSize[y, x])
                    A_mat[index, 1] = 0 + (j * inYSize[y, x])
                    A_mat[index, 2] = 1

                    # Elevation
                    z_vec[index] = inBlock[0, y + j, x + i] * zScale

                    index += 1

            # Linear fit
            coeff_vec = np.linalg.lstsq(A_mat, z_vec)[0]

            # Calculate dzx and dzy
            dzx = coeff_vec[0] * dx
            dzy = coeff_vec[1] * dy

            # Find normal vector to the plane
            nx = -1 * dy * dzx
            ny = -1 * dx * dzy
            nz = dx * dy

            slopeRad = np.arccos(nz / sqrt(nx**2 + ny**2 + nz**2))
            slopeDeg = (180. / np.pi) * slopeRad

            outBlock[0, y, x] = slopeDeg

    return outBlock


def calcSlope(inBlock,
              inXSize,
              inYSize,
              fitPlane=False,
              zScale=1,
              winSize=3,
              minSlope=None):
    """ Calculates slope for a block of data
        Arrays are provided giving the size for each pixel.
        * inBlock - In elevation
        * inXSize - Array of pixel sizes (x)
        * inYSize - Array of pixel sizes (y)
        * fitPlane - Calculate slope by fitting a plane to elevation
                     data using least squares fitting.
        * zScale - Scaling factor between horizontal and vertical
        * winSize - Window size to fit plane over.
    """
    # If fortran class could be imported use this
    # Otherwise run through loop in python (which will be slower)
    # Setup output block
    outBlock = np.zeros_like(inBlock, dtype=np.float32)
    if fitPlane:
        # Setup matrix and vector required for least squares fitting.
        winOffset = int(winSize / 2)
        A_mat = np.zeros((winSize**2, 3))
        z_vec = np.zeros(winSize**2)

        slopePythonPlane(inBlock, outBlock, inXSize, inYSize, A_mat, z_vec,
                         zScale, winSize)
    else:
        slopePython(inBlock, outBlock, inXSize, inYSize, zScale)

    if minSlope is not None:
        # Set very low values to constant
        outBlock[0] = np.where(
            np.logical_and(outBlock[0] > 0, outBlock[0] < minSlope), minSlope,
            outBlock[0])
    return outBlock