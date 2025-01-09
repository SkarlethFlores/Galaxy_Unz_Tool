#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:43:14 2023

@author: smotinof
"""



from tqdm import tqdm 
from scipy.interpolate import RegularGridInterpolator
from math import log10, log1p
from matplotlib import cm
from scipy.interpolate import interp1d


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time 




# ------------------------------------------------------
#
# FUNCTION # 1
#
# ------------------------------------------------------

#execfile("Functions.ipynb")
#execfile("ReadDataTable.ipynb")
#from Functions.ipynb import GetDatatable

def GetDatatable(nameFile,  doPlots, saveFigure1, ArxName):
    '''
    Description: as
        
    
    Parameters
    ------------------
    nameFile : Name of csv file containing the galaxy list
    
    
    Returns
    ------------------
    Gal : TYPE
        DESCRIPTION.
    lg : TYPE
        DESCRIPTION.
    
    Example
    >>>
    
    ------------------

    '''
    
    
    Gal  = pd.read_csv(nameFile,  header = 0, engine='python')  #
    lg   = len(Gal)
    print (Gal.columns)
    Gal
    # Clean Data Table

    for i in range(lg): 
        m = str(Gal['Galaxy'][i]) == 'nan'# Gal_ratios['Galaxy'][j]
        if (m):
            Gal = Gal.drop(i)

    Gal  = Gal.reset_index()
    cols = ['Line','Wavelength(um)', 'Flux1', 'Flux1_Error', 'SNR1', 'Flux2', 'Flux2_Error', 'SNR2', 'Aperture']
    lg   = len(Gal)
    for i in range(len(cols)):
        Gal  = Gal.drop(cols[i], axis = 1)

    Cols = Gal.columns
    lcol = len(Cols)

    for i in range(lg): 
        Gal['index'][i] = i
        for j in range(lcol-2):
            jj = j+2
            m = (str(Gal[Cols[jj]][i])=='nan') or (str(Gal[Cols[jj]][i])=='#DIV/0!') or (str(Gal[Cols[jj]][i])=='0')# Gal_ratios['Galaxy'][j]
            if (m):
                Gal[Cols[jj]][i] = -10
            else:
                Gal[Cols[jj]][i] = float(Gal[Cols[jj]][i])
                
                
    # Plot galaxies data

    if doPlots:
        plt.figure(figsize = (24,12))
        plt.rc('text', usetex=True)
        rc("pdf", fonttype=3)
        rc('font',**{'family':'serif'})
        plt.rc('xtick', labelsize=18) 
        plt.rc('ytick', labelsize=24) 

        Title = ['[OIII]52/[OIII]88', '(2.2[OIII]88+[OIII]52)/[NIII]57', '([OIII]88)/[NIII]57', '[OIII]52/[NIII]57', '[OIII]88/[NIII]122', '[OIII]52/[NIII]122']


        ax1 = Doplot( 1, Title[0], np.array(Gal['index']), np.array(Gal['OIII52/OIII88']), Gal['OIII52/OIII88_ERR'], Gal['Galaxy'] )
        ax2 = Doplot( 2, Title[1], np.array(Gal['index']), np.array(Gal['(2.2OIII88+OIII52)/NIII57']), Gal['(2.2OIII88+OIII52)/NIII57_ERR'], Gal['Galaxy'] )
        ax3 = Doplot( 3, Title[2], np.array(Gal['index']), np.array(Gal['OIII88/NIII57']), Gal['OIII88/NIII57_ERR'], Gal['Galaxy'] )
        ax4 = Doplot( 4, Title[3], np.array(Gal['index']), np.array(Gal['OIII52/NIII57']), Gal['OIII52/NIII57_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 5, Title[4], np.array(Gal['index']), np.array(Gal['OIII88/NII122']), Gal['OIII88/NII122_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 6, Title[5], np.array(Gal['index']), np.array(Gal['OIII52/NII122']), Gal['OIII52/NII122_ERR'], Gal['Galaxy'] )


        ax3.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left', borderaxespad=0.05, fontsize= 28)

        plt.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.750, hspace=0.10, wspace=0.20)


        if (saveFigure1):

            print (ArxName)
            plt.savefig(ArxName)



    return Gal, lg


def GetFluxesTable(nameFile,  doPlots, saveFigure1, ArxName):
    '''
    Description: as
        
    
    Parameters
    ------------------
    nameFile : Name of csv file containing the galaxy list
    
    
    Returns
    ------------------
    Gal : TYPE
        DESCRIPTION.
    lg : TYPE
        DESCRIPTION.
    
    Example
    >>>
    
    ------------------

    '''
    
    Gal  = pd.read_csv(nameFile,  header = 0, engine='python')  #
    lg   = len(Gal)
    print (Gal.columns)
    
    # Clean Data Table

    for i in range(lg): 
        m = str(Gal['Galaxy'][i]) == 'nan'# Gal_ratios['Galaxy'][j]
        if (m):
            Gal = Gal.drop(i)

    Gal  = Gal.reset_index()
    Cols = ['id','Galaxy','date','OIII52','OIII52_unc','OIII88','OIII88_unc','NIII57','NIII57_unc','NII122','NII122_unc','reference']
    lg   = len(Gal) 
    lcol = len(Cols)

    for i in range(lg): 
        #Gal['index'][i] = i
        for j in range(lcol-4):
            jj = j+3
            m = (str(Gal[Cols[jj]][i])=='nan') or (str(Gal[Cols[jj]][i])=='#DIV/0!') or (str(Gal[Cols[jj]][i])=='0') 
            if (m):
                Gal[Cols[jj]][i] = -100.0
            else:
                Gal[Cols[jj]][i] = float(Gal[Cols[jj]][i])
    #
    print ('\n\nData Frame before adding 15 Cal to errors')
    print (Gal)
    
    # Adding Calibration error
    ###Y_sig = np.sqrt(np.power(Y_sig,2)+np.power(Y_obs*0.15,2))
    
    Gal['OIII52_unc'] = np.sqrt( np.power(Gal['OIII52_unc'],2) + np.power( Gal['OIII52']* 0.15,2) )
    Gal['OIII88_unc'] = np.sqrt( np.power(Gal['OIII88_unc'],2) + np.power( Gal['OIII88']* 0.15,2) )
    Gal['NIII57_unc'] = np.sqrt( np.power(Gal['NIII57_unc'],2) + np.power( Gal['NIII57']* 0.15,2) )
    Gal['NII122_unc'] = np.sqrt( np.power(Gal['NII122_unc'],2) + np.power( Gal['NII122']* 0.15,2) )
    
    print ('\n\nData Frame after adding errors')
    print (Gal)
    
    # Calculate ratios
    
    #Columns1    = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
    #               'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
    #               'OIII88/NII122','OIII88/NIII57',
    #               '(2.2OIII88+OIII52)/NIII57' ] 
        
    Gal['OIII52/OIII88'] =  Gal['OIII52'] / Gal['OIII88'] 
    Gal['NIII57/OIII88'] =  Gal['NIII57'] / Gal['OIII88'] 
    Gal['NII122/OIII88'] =  Gal['NII122'] / Gal['OIII88'] 
    Gal['OIII52/NIII57'] =  Gal['OIII52'] / Gal['NIII57']
    Gal['OIII52/NII122'] =  Gal['OIII52'] / Gal['NII122']
    Gal['NIII57/NII122'] =  Gal['NIII57'] / Gal['NII122'] 
    Gal['OIII88/NII122'] =  Gal['OIII88'] / Gal['NII122'] 
    Gal['OIII88/NIII57'] =  Gal['OIII88'] / Gal['NIII57']
    Gal['(2.2OIII88+OIII52)/NIII57'] = ( 2.2 * Gal['OIII88'] + Gal['OIII52'] ) / Gal['NIII57']
    
    # Uncertainties propagation
    
    # (A/B)_unc = (A/B)*sqrt(((A_unc/A))^2+(B_unc/B)^2)
    
    Gal['OIII52/OIII88_ERR'] =  Gal['OIII52/OIII88'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) )
            
        
    Gal['NIII57/OIII88_ERR'] =  Gal['NIII57/OIII88'] *  np.sqrt( np.power(Gal['NIII57_unc']/Gal['NIII57'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) )
    
    
    Gal['NII122/OIII88_ERR'] =  Gal['NII122/OIII88'] *  np.sqrt( np.power(Gal['NII122_unc']/Gal['NII122'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) ) 
    
    
    Gal['OIII52/NIII57_ERR'] =  Gal['OIII52/NIII57'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) )  
    
    
    Gal['OIII52/NII122_ERR'] =  Gal['OIII52/NII122'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) ) 
    
    
    Gal['NIII57/NII122_ERR'] =  Gal['NIII57/NII122'] *  np.sqrt( np.power(Gal['NIII57_unc']/Gal['NIII57'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) )
    
    
    Gal['OIII88/NII122_ERR'] =  Gal['OIII88/NII122'] *  np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) )
    
    
    Gal['OIII88/NIII57_ERR'] =  Gal['OIII88/NIII57']*   np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) )  
    
    
    Gal['(2.2OIII88+OIII52)/NIII57_ERR'] = Gal['(2.2OIII88+OIII52)/NIII57']*  np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) ) 
    
    # Plot galaxies data

    if doPlots:
        plt.figure(figsize = (24,12))
        plt.rc('text', usetex=True)
        rc("pdf", fonttype=3)
        rc('font',**{'family':'serif'})
        plt.rc('xtick', labelsize=18) 
        plt.rc('ytick', labelsize=24) 

        Title = ['[OIII]52/[OIII]88', '(2.2[OIII]88+[OIII]52)/[NIII]57', '[OIII]88/[NIII]57', '[OIII]52/[NIII]57', '[OIII]88/[NIII]122', '[OIII]52/[NIII]122']


        ax1 = Doplot( 1, Title[0], np.array(Gal['index']), np.array(Gal['OIII52/OIII88']), Gal['OIII52/OIII88_ERR'], Gal['Galaxy'] )
        ax2 = Doplot( 2, Title[1], np.array(Gal['index']), np.array(Gal['(2.2OIII88+OIII52)/NIII57']), Gal['(2.2OIII88+OIII52)/NIII57_ERR'], Gal['Galaxy'] )
        ax3 = Doplot( 3, Title[2], np.array(Gal['index']), np.array(Gal['OIII88/NIII57']), Gal['OIII88/NIII57_ERR'], Gal['Galaxy'] )
        ax4 = Doplot( 4, Title[3], np.array(Gal['index']), np.array(Gal['OIII52/NIII57']), Gal['OIII52/NIII57_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 5, Title[4], np.array(Gal['index']), np.array(Gal['OIII88/NII122']), Gal['OIII88/NII122_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 6, Title[5], np.array(Gal['index']), np.array(Gal['OIII52/NII122']), Gal['OIII52/NII122_ERR'], Gal['Galaxy'] )


        ax3.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left', borderaxespad=0.05, fontsize= 28)

        plt.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.750, hspace=0.10, wspace=0.20)


        if (saveFigure1):

            print ('\nPlot saved to: ',ArxName)
            plt.savefig(ArxName)



    return Gal, lg



def GetFluxesTable_tf(nameFile,  doPlots, saveFigure1, ArxName):
    '''
    Description: as
        
    
    Parameters
    ------------------
    nameFile : Name of csv file containing the galaxy list
    
    
    Returns
    ------------------
    Gal : TYPE
        DESCRIPTION.
    lg : TYPE
        DESCRIPTION.
    
    Example
    >>>
    
    ------------------

    '''
    
    Gal  = pd.read_csv(nameFile,  header = 13, sep='\s+', engine='python')  #
    lg   = len(Gal)
    print (Gal.columns)
    
    # Clean Data Table

    
    Cols = ['id','Galaxy','OIII52','OIII52_unc','NIII57','NIII57_unc','OIII88','OIII88_unc','NII122','NII122_unc','12+log(O/H) PT2005','12+log(O/H) I2006']
    Gal.columns = Cols
    lg   = len(Gal) 
    lcol = len(Cols)
    
    
    # Clean Data Table
    for i in range(lg): 
        m = str(Gal['Galaxy'][i]) == 'nan'# Gal_ratios['Galaxy'][j]
        if (m):
            Gal = Gal.drop(i)

    Gal  = Gal.reset_index()
    
    

    for i in range(lg): 
        #Gal['index'][i] = i
        for j in range(lcol-4):
            jj = j+3
            m = (str(Gal[Cols[jj]][i])=='nan') or (str(Gal[Cols[jj]][i])=='#DIV/0!') or (str(Gal[Cols[jj]][i])=='0') 
            if (m):
                Gal[Cols[jj]][i] = -100.0
            else:
                Gal[Cols[jj]][i] = float(Gal[Cols[jj]][i])
    #
    print ('\n\nData Frame before adding 15 Cal to errors')
    print (Gal)
    
    # Adding Calibration error
    ###Y_sig = np.sqrt(np.power(Y_sig,2)+np.power(Y_obs*0.15,2))
    
    Gal['OIII52_unc'] = np.sqrt( np.power(Gal['OIII52_unc'],2) + np.power( Gal['OIII52']* 0.15,2) )
    Gal['OIII88_unc'] = np.sqrt( np.power(Gal['OIII88_unc'],2) + np.power( Gal['OIII88']* 0.15,2) )
    Gal['NIII57_unc'] = np.sqrt( np.power(Gal['NIII57_unc'],2) + np.power( Gal['NIII57']* 0.15,2) )
    Gal['NII122_unc'] = np.sqrt( np.power(Gal['NII122_unc'],2) + np.power( Gal['NII122']* 0.15,2) )
    
    print ('\n\nData Frame after adding errors')
    print (Gal)
    
    # Calculate ratios
    
    #Columns1    = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
    #               'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
    #               'OIII88/NII122','OIII88/NIII57',
    #               '(2.2OIII88+OIII52)/NIII57' ] 
        
    Gal['OIII52/OIII88'] =  Gal['OIII52'] / Gal['OIII88'] 
    Gal['NIII57/OIII88'] =  Gal['NIII57'] / Gal['OIII88'] 
    Gal['NII122/OIII88'] =  Gal['NII122'] / Gal['OIII88'] 
    Gal['OIII52/NIII57'] =  Gal['OIII52'] / Gal['NIII57']
    Gal['OIII52/NII122'] =  Gal['OIII52'] / Gal['NII122']
    Gal['NIII57/NII122'] =  Gal['NIII57'] / Gal['NII122'] 
    Gal['OIII88/NII122'] =  Gal['OIII88'] / Gal['NII122'] 
    Gal['OIII88/NIII57'] =  Gal['OIII88'] / Gal['NIII57']
    Gal['(2.2OIII88+OIII52)/NIII57'] = ( 2.2 * Gal['OIII88'] + Gal['OIII52'] ) / Gal['NIII57']
    
    # Uncertainties propagation
    
    # (A/B)_unc = (A/B)*sqrt(((A_unc/A))^2+(B_unc/B)^2)
    
    Gal['OIII52/OIII88_ERR'] =  Gal['OIII52/OIII88'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) )
            
        
    Gal['NIII57/OIII88_ERR'] =  Gal['NIII57/OIII88'] *  np.sqrt( np.power(Gal['NIII57_unc']/Gal['NIII57'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) )
    
    
    Gal['NII122/OIII88_ERR'] =  Gal['NII122/OIII88'] *  np.sqrt( np.power(Gal['NII122_unc']/Gal['NII122'],2) + np.power( Gal['OIII88_unc']/Gal['OIII88'],2) ) 
    
    
    Gal['OIII52/NIII57_ERR'] =  Gal['OIII52/NIII57'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) )  
    
    
    Gal['OIII52/NII122_ERR'] =  Gal['OIII52/NII122'] *  np.sqrt( np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) ) 
    
    
    Gal['NIII57/NII122_ERR'] =  Gal['NIII57/NII122'] *  np.sqrt( np.power(Gal['NIII57_unc']/Gal['NIII57'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) )
    
    
    Gal['OIII88/NII122_ERR'] =  Gal['OIII88/NII122'] *  np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power( Gal['NII122_unc']/Gal['NII122'],2) )
    
    
    Gal['OIII88/NIII57_ERR'] =  Gal['OIII88/NIII57']*   np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) )  
    
    
    Gal['(2.2OIII88+OIII52)/NIII57_ERR'] = Gal['(2.2OIII88+OIII52)/NIII57']*  np.sqrt( np.power(Gal['OIII88_unc']/Gal['OIII88'],2) + np.power(Gal['OIII52_unc']/Gal['OIII52'],2) + np.power( Gal['NIII57_unc']/Gal['NIII57'],2) ) 
    
    # Plot galaxies data

    if doPlots:
        plt.figure(figsize = (24,12))
        plt.rc('text', usetex=True)
        rc("pdf", fonttype=3)
        rc('font',**{'family':'serif'})
        plt.rc('xtick', labelsize=18) 
        plt.rc('ytick', labelsize=24) 

        Title = ['[OIII]52/[OIII]88', '(2.2[OIII]88+[OIII]52)/[NIII]57', '[OIII]88/[NIII]57', '[OIII]52/[NIII]57', '[OIII]88/[NIII]122', '[OIII]52/[NIII]122']


        ax1 = Doplot( 1, Title[0], np.array(Gal['index']), np.array(Gal['OIII52/OIII88']), Gal['OIII52/OIII88_ERR'], Gal['Galaxy'] )
        ax2 = Doplot( 2, Title[1], np.array(Gal['index']), np.array(Gal['(2.2OIII88+OIII52)/NIII57']), Gal['(2.2OIII88+OIII52)/NIII57_ERR'], Gal['Galaxy'] )
        ax3 = Doplot( 3, Title[2], np.array(Gal['index']), np.array(Gal['OIII88/NIII57']), Gal['OIII88/NIII57_ERR'], Gal['Galaxy'] )
        ax4 = Doplot( 4, Title[3], np.array(Gal['index']), np.array(Gal['OIII52/NIII57']), Gal['OIII52/NIII57_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 5, Title[4], np.array(Gal['index']), np.array(Gal['OIII88/NII122']), Gal['OIII88/NII122_ERR'], Gal['Galaxy'] )
        ax5 = Doplot( 6, Title[5], np.array(Gal['index']), np.array(Gal['OIII52/NII122']), Gal['OIII52/NII122_ERR'], Gal['Galaxy'] )


        ax3.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left', borderaxespad=0.05, fontsize= 28)

        plt.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.750, hspace=0.10, wspace=0.20)


        if (saveFigure1):

            print ('\nPlot saved to: ',ArxName)
            plt.savefig(ArxName)



    return Gal, lg




# ------------------------------------------------------
#
# FUNCTION # 6
#
# ------------------------------------------------------


def LineDataGalaxy(gali, Gal, UseLines):
    
    Y_obs = np.array( [ Gal['OIII52/OIII88'][gali], 
                        Gal['OIII52/NIII57'][gali], 
                        Gal['OIII88/NIII57'][gali], 
                        Gal['OIII52/NII122'][gali], 
                        Gal['OIII88/NII122'][gali],  
                        Gal['(2.2OIII88+OIII52)/NIII57'][gali] ,
                        Gal['NIII57/NII122'][gali]
                     ])

    Y_sig = np.array([  Gal['OIII52/OIII88_ERR'][gali], 
                        Gal['OIII52/NIII57_ERR'][gali], 
                        Gal['OIII88/NIII57_ERR'][gali], 
                        Gal['OIII52/NII122_ERR'][gali], 
                        Gal['OIII88/NII122_ERR'][gali],  
                        Gal['(2.2OIII88+OIII52)/NIII57_ERR'][gali] ,
                        Gal['NIII57/NII122'][gali], 
                      ])
    print (' Y_obs',Y_obs,'\n Y_sig', Y_sig)
    
    
    
    # Data to Use:
    TxtLines  = '\n'
    Columns1    = ['[OIII]-52/[OIII]-88','[OIII]-52/[NIII]-57','[OIII]-88/[NIII]-57','OIII52/[NII]-122',
                   '\n [OIII]-88/[NII]-122','(2.2x[OIII]-88+[OIII]-52)/[NIII]-57' ,'NIII57/[NII]-122'] 
    
    TxtLinesShort =''  
    Columns2 = ['[OIII]-52_OIII88', '[OIII]-52_[NIII]-57', '[OIII]-88_NIII57', '[OIII]-52_[NII]-122',
           '\n [OIII]-88_[NII]-122', '(2.2X[OIII]-88+OIII52)/[NIII]-57','[NIII]-57_[NII]-122']
    # Not Used:
    # 'SIII19_SIII33','NII122_NII205', 'SIV11_NeII16', 'NeII13_NeIII16',  'OIII52_NII205',   'OIII88_NII205'
    
    for i in range(len(UseLines)):
        flag =  UseLines[i] 
        if (flag == 1): 
            TxtLines      = TxtLines+' - '+Columns1[i]
            TxtLinesShort = TxtLinesShort+'- '+Columns2[i]
        else:
            Y_sig[i] = 999999
    
    print ('\n\n Y_obs:',Y_obs,'\n Y_sig', Y_sig)
    print ('\nLines:   '+TxtLines,'\n '+TxtLinesShort )
    return (Y_obs, Y_sig, TxtLines , TxtLinesShort )



def LineDataGalaxy2(gali, Gal, UseLines):
    
    Y_obs = np.array( [ Gal['OIII52/OIII88'][gali], 
                        Gal['NIII57/OIII88'][gali], 
                        Gal['NII122/OIII88'][gali], 
                        
                        Gal['OIII52/NIII57'][gali], 
                        Gal['OIII52/NII122'][gali],  
                        Gal['NIII57/NII122'][gali],
                       
                        Gal['OIII88/NII122'][gali],  
                        Gal['OIII88/NIII57'][gali], 
                       
                        Gal['(2.2OIII88+OIII52)/NIII57'][gali] 
                     ])

    Y_sig = np.array([  Gal['OIII52/OIII88_ERR'][gali], 
                        Gal['NIII57/OIII88_ERR'][gali], 
                        Gal['NII122/OIII88_ERR'][gali],  
                      
                        Gal['OIII52/NIII57_ERR'][gali], 
                        Gal['OIII52/NII122_ERR'][gali],  
                        Gal['NIII57/NII122_ERR'][gali], 
                      
                        Gal['OIII88/NII122_ERR'][gali],  
                        Gal['OIII88/NIII57_ERR'][gali], 
                      
                        Gal['(2.2OIII88+OIII52)/NIII57_ERR'][gali]
                      ])
    print (' Y_obs',Y_obs,'\n Y_sig', Y_sig)
    
    
    
    # Data to Use:
    TxtLines  = '\n'
    Columns1    = ['[OIII]-52/[OIII]-88','[NIII]-57/[OIII]-88','[NII]-122/[OIII]-88', 
                   '[OIII]-52/[NIII]-57','[OIII]-52/[NII]-122','[NIII]-57/[NII]-122', 
                   '[OIII]-88/[NII]-122','[OIII]-88/[NIII]-57',    '(2.2x[OIII]-88+[OIII]-52)/[NIII]-57' ] 
    
    TxtLinesShort =''  
    Columns2     = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
                   'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
                   'OIII88/NII122','OIII88/NIII57',   '(2.2OIII88+OIII52)/NIII57' ]
    
    
    #Columns1    = ['[OIII]-52/[OIII]-88','[OIII]-52/[NIII]-57','[OIII]-88/[NIII]-57','OIII52/[NII]-122',
    #               ' [OIII]-88/[NII]-122','(2.2x[OIII]-88+[OIII]-52)/[NIII]-57' ,'NIII57/[NII]-122'] 
    
    
    #TxtLinesShort =''  
    #Columns2     = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
    #               'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
    #               'OIII88/NII122','OIII88/NIII57',  '(2.2OIII88+OIII52)/NIII57' ]
    #TxtLinesShort =''  
    #Columns2 = ['[OIII]-52_OIII88', '[OIII]-52_[NIII]-57', '[OIII]-88_NIII57', '[OIII]-52_[NII]-122',
    #       ' [OIII]-88_[NII]-122', '(2.2X[OIII]-88+OIII52)/[NIII]-57','[NIII]-57_[NII]-122']
    
    
    # Not Used:
    # 'SIII19_SIII33','NII122_NII205', 'SIV11_NeII16', 'NeII13_NeIII16',  'OIII52_NII205',   'OIII88_NII205'
    
    for i in range(len(UseLines)):
        flag =  UseLines[i] 
        if (flag == 1): 
            TxtLines      = TxtLines+' - '+Columns1[i]
            TxtLinesShort = TxtLinesShort+'- '+Columns2[i]
        else:
            Y_sig[i] = 999999
    
    # Adding Calibration error
    #Y_sig = np.sqrt(np.power(Y_sig,2)+np.power(Y_obs*0.15,2))
    print ('\n Selected lines: \n Y_obs:',Y_obs,'\n Y_sig', Y_sig)
    print ('\nLines:   '+TxtLines,'\n '+TxtLinesShort )
    return (Y_obs, Y_sig, TxtLines , TxtLinesShort )



# ------------------------------------------------------
#
# FUNCTION # 11
#
# ------------------------------------------------------


def selectlines(y_sig, UseLines):
    
    TxtLines  = '\n'
    #Columns1    = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
    #               'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
    #               'OIII88/NII122','OIII88/NIII57',   '(2.2OIII88+OIII52)/NIII57' ] 
    #TxtLines  = '\n'
    Columns1    = ['[OIII]-52/[OIII]-88','[OIII]-52/[NIII]-57','[OIII]-88/[NIII]-57','OIII52/[NII]-122',
                   ' [OIII]-88/[NII]-122','(2.2x[OIII]-88+[OIII]-52)/[NIII]-57' ,'NIII57/[NII]-122'] 
    
    
    TxtLinesShort =''  
    Columns2     = ['OIII52/OIII88','NIII57/OIII88','NII122/OIII88', 
                   'OIII52/NIII57','OIII52/NII122','NIII57/NII122', 
                   'OIII88/NII122','OIII88/NIII57',  '(2.2OIII88+OIII52)/NIII57' ]
    #TxtLinesShort =''  
    Columns2 = ['[OIII]-52_OIII88', '[OIII]-52_[NIII]-57', '[OIII]-88_NIII57', '[OIII]-52_[NII]-122',
           ' [OIII]-88_[NII]-122', '(2.2X[OIII]-88+OIII52)/[NIII]-57','[NIII]-57_[NII]-122']
    
    
    
    # Not Used:
    # 'SIII19_SIII33','NII122_NII205', 'SIV11_NeII16', 'NeII13_NeIII16',  'OIII52_NII205',   'OIII88_NII205'
    cl = 0
    for i in range(len(UseLines)):
        flag =  UseLines[i] 
        if (flag == 1): 
            if (cl == 2):
                TxtLines      = TxtLines+'\n'
            TxtLines      = TxtLines+' - '+Columns1[i]
            TxtLinesShort = TxtLinesShort+'-'+Columns2[i]
            cl = cl+1
            
        else:
            y_sig[i] = 999999
            
    return y_sig, TxtLines, TxtLinesShort












# ------------------------------------------------------
#
# FUNCTION # 2
#
# ------------------------------------------------------

def ReadModels(nameFile_Models):
    
    Models  = pd.read_csv(nameFile_Models, sep = ' & ', header = 0, engine='python')  #
    lmo   = len(Models)

    columns = ['log(U)','SIII19_SIII33', 'OIII52_OIII88',
           'NII122_NII205', 'SIV11_NeII16', 'NeII13_NeIII16', 'OIII52_NIII57',
           'OIII88_NIII57', 'OIII52_NII122', 'OIII52_NII205', 'OIII88_NII122',
           'OIII88_NII205']

    for i in range(lmo):

        Models['OIII88_NII205'][i] = Models['OIII88_NII205'][i].replace('\\','') 
        for col in columns: 
            
            Models[col][i] = float(str(Models[col][i]).replace('--','-') )


    columns = ['SIII19_SIII33', 'OIII52_OIII88',
           'NII122_NII205', 'SIV11_NeII16', 'NeII13_NeIII16', 'OIII52_NIII57',
           'OIII88_NIII57', 'OIII52_NII122', 'OIII52_NII205', 'OIII88_NII122',
           'OIII88_NII205']

    for col in columns: 
        Models[col ] = np.power( 10, Models[ col]) 


    Models['(2.2XOIII88+OIII52)/NIII57'] =(2.2 * Models['OIII88_NIII57']) + Models['OIII52_NIII57']
    Models['NIII57_NII122'] = Models['OIII88_NII122'] / Models['OIII88_NIII57']
    Models['U']        =  np.power( 10, Models['log(U)']) 
    
    
    
    Models['NIII57_OIII88'] = 1 / Models['OIII88_NIII57']
    Models['NII122_OIII88'] = 1 / Models['OIII88_NII122']
    
    
    
    return Models





# ------------------------------------------------------
#
# FUNCTION # 2
#
# ------------------------------------------------------

def PercentileXPx(X, Px, q, ShowPlot):
    #a      = Px
    #
    
    #Px_ave = (np.append(np.zeros(1),(Px[0:-1])) + np.append( (Px[1:-0]), np.zeros(1) )) / 2
    #aCum   = np.cumsum(Px_ave)
    
    
    a = np.append(np.zeros(1), (Px[0:-1]), axis = 0)
    b = np.append(np.zeros(1), (Px[1:]), axis =0)

    Px_ave = ( a + b ) / 2
    Px_ave = Px_ave/np.sum(Px_ave)

    aCum   = np.cumsum(Px_ave)
    if (ShowPlot == True):
        print ('Px:',Px)
        print ('P1:',a)
        print ('P2:',b)
        print ('Px_ave:',Px_ave)
        print ('Acum',aCum)
    
    Perc_q  = (q)/100    
    f = interp1d(aCum, X ) 
    Xval = f(Perc_q) 
    
    if (ShowPlot == True):
        print (' Psum = %2.3f'%aCum[-1])
        print (' Perc_q =', Perc_q,'          XVal =  ', Xval )
        fig = plt.figure(figsize = (8,4)) 
        plt.plot(X, aCum)
        plt.xlabel('x', labelpad=Xval)
        plt.ylabel('P(x)', labelpad=Perc_q)
        
        #xticks = plt.xticks()[0] 
        #xticks = list(xticks) + [Xval]
        #plt.xticks(xticks)
        
        plt.text(Xval*0.97, 1.07, str('%.1f'%Xval), { 'fontsize':16, 'color':'green'})
        plt.text(X[-1]*1.07, Perc_q, str('%.2f'%Perc_q), { 'fontsize':16, 'color':'red'})
        plt.axvline(Xval, linestyle='-', color='g', alpha=0.8,) 
        plt.axhline(Perc_q, linestyle='-', color='r', alpha=0.7,)
        plt.show()
    return Xval





# ------------------------------------------------------
#
# FUNCTION # 3
#
# ------------------------------------------------------

def CalcChi2(lugrid, lngrid, lzgrid ):
    #Chi[:,:,:,:]    = (( ModelsCI[:,:,:,:] - Y_obs) / Y_sig )**2 
    #
    Ny = len(Y_obs)
    Chi         = np.zeros((lugrid, lngrid, lzgrid,  Ny),     dtype = 'float128')
    Chi2        = np.zeros((lugrid, lngrid, lzgrid,  Ny),     dtype = 'float128')
    prob        = np.zeros((lugrid, lngrid, lzgrid),          dtype = 'float128')

    for i in range(Ny):
        Chi[:,:,:,i]    = (( ModelsCI[:,:,:,i] - Y_obs[i]) / Y_sig[i] )**2     

    chi2                   = Chi.sum(axis = 3) 
    prob                   = np.exp(-(chi2)/2.0 , dtype = 'float128')      # raw likelihood
    prob                   = prob/np.sum(prob)              # normalized likelihood, its sum over the whole parameter space is one.
    print ('Models cube shape:     ',ModelsCube.shape)
    print ('Models refined shape:  ', ModelsCI.shape)
    print ('Chi shape:   ', np.shape(Chi)) 
    print ('Chi^2 shape: ', chisqr_grid.shape, '\t Sum Chi^2 = %3.1E'%chisqr_grid.sum())
    print ('Prob Shape:  ', prob.shape, '\t Sum Prob = %3.1E'%prob.sum())
    print ('Chi^2_min = %3.2E'%chi2.min(), '\nChi^2_max = %3.2E'%chisqr_grid.max(),)

    return chi2, prob



def CalcChi2_N(lugrid, lngrid, lzgrid,  Y_mod_n, Y_sig_n, ModelsCI):
    #Chi[:,:,:,:]    = (( ModelsCI[:,:,:,:] - Y_obs) / Y_sig )**2 
    #
    Ny = len( Y_mod_n)
    Chi         = np.zeros((lugrid, lngrid, lzgrid,  Ny),     dtype = 'float128')
    Chi2        = np.zeros((lugrid, lngrid, lzgrid,  Ny),     dtype = 'float128')
    prob        = np.zeros((lugrid, lngrid, lzgrid),          dtype = 'float128')

    for i in range(Ny):
        Chi[:,:,:,i]    = (( ModelsCI[:,:,:,i] - Y_mod_n[i]) / Y_sig_n[i] )**2   

    chi2                   = Chi.sum(axis = 3) 
    prob                   = np.exp(-(chi2)/2.0 , dtype = 'float128')      # raw likelihood
    prob                   = prob/np.sum(prob)              # normalized likelihood, its sum over the whole parameter space is one.
    #print ('Models cube shape:     ',ModelsCube.shape)
    print ('Models refined shape:  ', ModelsCI.shape)
    print ('Chi shape:   ', np.shape(Chi)) 
    print ('Chi^2 shape: ', Chi2.shape, '\t Sum Chi^2 = %3.1E'%Chi2.sum())
    print ('Prob Shape:  ', prob.shape, '\t Sum Prob = %3.5E'%prob.sum())
    print ('Chi^2_min = %3.2E'%chi2.min(), '\nChi^2_max = %3.5E'%Chi2.max(),)

    return chi2, prob


# ------------------------------------------------------
#
# FUNCTION # 4
#
# ------------------------------------------------------

def CorrCoef(axis1, axis2, ave, prob, STD,  u_grid, n_grid, z_grid ):
    indx   = np.array([0,1,2])
    indx   = np.where( indx == axis1, -1, indx) 
    indx   = np.where( indx == axis2, -1, indx) 
    
    #axis4  = np.max(indx)
    #indx   = np.where( indx == axis4, -1, indx)
    print (indx)
    axis3  = np.max(indx) 
    #print (axis3, axis4)
    if (axis1 == 0): Array1 = u_grid
    if (axis1 == 1): Array1 = n_grid
    if (axis1 == 2): Array1 = z_grid 
        
    if (axis2 == 0): Array2 = u_grid
    if (axis2 == 1): Array2 = n_grid
    if (axis2 == 2): Array2 = z_grid 
        
    Ptable = prob.sum(axis = axis3 )
    #Ptable = Ptable.sum(axis = axis3 ) 
    Arr1Arr2_cov  = 0
    Arr1Arr2_corr = 0
    
    #COVARIANZA
    for i in range(len(Array1)):
        for j in range(len(Array2)):
            Arr1Arr2_cov = Arr1Arr2_cov + (Array1[i]-ave[axis1])*(Array2[j]-ave[axis2])*Ptable[i,j]
    #Correlation
    Arr1Arr2_corr = Arr1Arr2_cov / STD[axis1] / STD[axis2]
    return Arr1Arr2_corr







# ------------------------------------------------------
#
# FUNCTION # 5
#
# ------------------------------------------------------


def interpolate_data_cube(data_cube, new_shape): #new_x, new_y, new_z
    # Extract the dimensions of the original data cube
    dim1, dim2, dim3 , dim4 = data_cube.shape

    # Create coordinate arrays for the original data cube
    x = np.linspace(0, dim1-1, dim1)
    y = np.linspace(0, dim2-1, dim2)
    z = np.array( [0.05, 0.2, 0.40, 1.0, 2.0] ) # np.linspace(0, dim3-1, dim3)/2 # 
    N = np.linspace(0, dim4-1, dim4)
    print ('\n z:', z, z.shape)

    # Create the interpolating function
    interpolator = RegularGridInterpolator((x, y, z, N), data_cube)

    # Create coordinate arrays for the new grid
    new_x = np.linspace(0, dim1-1, new_shape[0])
    new_y = np.linspace(0, dim2-1, new_shape[1])
    new_z = np.round( np.arange( 0.05 , 2.01, 0.01), decimals=3 ) #  np.linspace(0, dim3-1, new_shape[2])/2 # # 
    new_N = np.linspace(0, dim4-1, new_shape[3])
    #print (new_x, new_y, new_z, new_N)
    print ('\n new z \n Shape:',new_z.shape) 
    #if (printgrids==True):
    #    print ('\n Values:',new_z)
    
    # Create a meshgrid for the new grid
    new_meshgrid = np.meshgrid(new_x, new_y, new_z, new_N, indexing='ij') 
    
    # Reshape the meshgrid into a 3-column array for the interpolator
    points = np.column_stack([new_meshgrid[0].ravel(), new_meshgrid[1].ravel(), 
                              new_meshgrid[2].ravel(), new_meshgrid[3].ravel()    ])

    # Interpolate the data at the new grid points
    interpolated_data = interpolator(points)

    # Reshape the interpolated data to match the new grid shape
    interpolated_data_cube = interpolated_data.reshape(new_shape)
    
    #print (interpolated_data_cube.shape)
    return interpolated_data_cube




# ------------------------------------------------------
#
# FUNCTION # 5.b 
#
# Zoom in the Grid
#
# ------------------------------------------------------


import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_data_cube_byRange(data_cube, range_x, range_y, range_z, new_shape):
    # Extract the dimensions of the original data cube
    dim1, dim2, dim3, dim4 = data_cube.shape

    # Create coordinate arrays for the original data cube
    x = np.linspace(range_x[0], range_x[1], dim1)
    y = np.linspace(range_y[0], range_y[1], dim2)
    z = np.linspace(range_z[0], range_z[1], dim3)
    N = np.linspace(0, dim4-1, dim4)

    # Create the interpolating function
    interpolator = RegularGridInterpolator((x, y, z, N), data_cube)

    # Create coordinate arrays for the new grid
    new_x = np.round(np.linspace(range_x[0], range_x[1], new_shape[0]) , decimals = 2)
    new_y = np.round(np.linspace(range_y[0], range_y[1], new_shape[1]) , decimals = 2)
    new_z = np.round(np.linspace(range_z[0], range_z[1], new_shape[2]) , decimals = 2)
    new_N = np.round(np.linspace(0, dim4-1, new_shape[3]) , decimals = 2)

    # Create a meshgrid for the new grid
    new_meshgrid = np.meshgrid(new_x, new_y, new_z, new_N, indexing='ij')

    # Reshape the meshgrid into a 3-column array for the interpolator
    points = np.column_stack([new_meshgrid[0].ravel(), new_meshgrid[1].ravel(), 
                              new_meshgrid[2].ravel(), new_meshgrid[3].ravel()])

    # Interpolate the data at the new grid points
    interpolated_data = interpolator(points)

    # Reshape the interpolated data to match the new grid shape
    interpolated_data_cube = interpolated_data.reshape(new_shape)

    return interpolated_data_cube, new_x, new_y, new_z






# ------------------------------------------------------
#
# FUNCTION # 7
#
# ------------------------------------------------------

import matplotlib.pyplot as plt
    
def plot_probs(x_grid, ave, x_16, x_84, x_min, x_max, y_min, y_max, ax, xlab= '', xtickslabs = False , show_legend = False):
    
    if (show_legend==False):
        ax.axvline(x_grid,   linestyle='--',   lw=2, color='limegreen', alpha=1)#
        ax.axvline(ave,      linestyle='--',   lw=2, color='r',     alpha=1)#
    
    if show_legend:
        ax.axvline(x_grid,   linestyle='--',   lw=2, color='limegreen', alpha=1,  label = 'Best Fit') 
        ax.axvline(ave,      linestyle='--',   lw=2, color='r',     alpha=1,  label = 'Mean') 
    
    
    #nPn.axvline(n,  linestyle='--', color='g', alpha=0.5) 
    plt.axvspan(x_16, x_84, color='r',  alpha=0.1) 
    
    plt.xlim([0.95*x_min, 1.05*x_max]) 
    plt.ylim([y_min, y_max])
    
    ax.set_xticks(np.arange(x_min, x_max, step=0.5)) #(x_max-x_min)/3.0))  # Adjust step as needed
    plt.setp(ax.get_yticklabels(), visible=False)
    #plt.setp(zPz.get_xticklabels(), visible=True)
    plt.xlabel(xlab, { 'fontsize':30, 'color':'black'})
    
    ax.tick_params(axis='y', labelsize=24, which='both',length=10.0, direction='in', right=True, labelright = True )
    ax.tick_params(axis='x', labelsize=24, which='both',length=10.0, direction='in', top=True,  labelbottom = xtickslabs )
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.10, 1.20), fontsize = 22 )
    # supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
    #'center left', 'center right', 'lower center', 'upper center', 'center'

from matplotlib.ticker import FuncFormatter
# Define a function to format tick labels
def format_ticks(x, pos):
    return "{:.1f}".format(x)

def plot_grids(x_grid, y_grid, xave, yave, x_min, x_max, y_min, y_max, ax, xlab='', ylab='',xtickslab = False, ytickslab = False ):
        # Set formatter for x and y axis ticks
        ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

        ax.plot( xave, yave, 'ro', markersize='12'  )
        #uz.plot( u, z, 'go' )
        ax.plot( x_grid, y_grid, 'o', color = 'limegreen', markersize='12')
        plt.xlabel(xlab, { 'fontsize':30, 'color':'black'})
        plt.ylabel(ylab, { 'fontsize':30, 'color':'black'})
        plt.xlim([0.95*x_min, 1.05*x_max]) 
        plt.ylim([0.95*y_min, 1.05*y_max])

        ax.set_xticks(np.arange(x_min, x_max, step=0.5)) #(x_max-x_min)/3.0))  # Adjust step as needed
        ax.set_yticks(np.arange(y_min, y_max, step=0.5)) #(y_max-y_min)/3.0))  # Adjust step as needed

        plt.setp(ax.get_xticklabels(), visible=xtickslab) 
        plt.setp(ax.get_yticklabels(), visible=ytickslab) 

        ax.tick_params(axis='y', labelsize=24, which='both',length=10.0, direction='in', right=True, labelright = False )
        ax.tick_params(axis='x', labelsize=24, which='both',length=10.0, direction='in', top=True) #,  labelbottom = False
        
        
        
def GetSolutions( prob, pu, pn, pz, ui,  ni, zi, u_grid, n_grid, z_grid,  text1, text2, title, SummaryFile, name, FluxData, TxtLinesShort, u_limits=(-4.0,-2.5), n_limits=(1,2.2), z_limits=(0.0,1.5), ):
    npar = 3
    Ave, STD   = [0.0]*npar, [0.0]*npar
    Ave[0], Ave[1] , Ave[2]   =  np.sum(u_grid*pu), np.sum(n_grid*pn),  np.double(np.sum(z_grid*pz))

    STD[0] = np.sqrt( np.sum( (u_grid - Ave[0])**2 *pu ))
    STD[1] = np.sqrt( np.sum( (n_grid - Ave[1])**2 *pn ))
    STD[2] = np.sqrt( np.sum( (z_grid - Ave[2])**2 *pz ))#np.sqrt(np.sum(z_grid**2*Pz)-np.sum(z_grid*Pz)**2)

    n_50 = PercentileXPx(n_grid, pn, 50, False)
    u_50 = PercentileXPx(u_grid, pu, 50, False) 
    z_50 = PercentileXPx(z_grid, pz, 50, False) 

    n_84 = PercentileXPx(n_grid, pn, 84, False)
    u_84 = PercentileXPx(u_grid, pu, 84, False) 
    z_84 = PercentileXPx(z_grid, pz, 84, False) 

    n_16 = PercentileXPx(n_grid, pn, 16, False)
    u_16 = PercentileXPx(u_grid, pu, 16, False)  
    z_16 = PercentileXPx(z_grid, pz, 16, False) 

    #uidx, nidx, zidx = np.where(chisqr_grid_N == chimin_N)
    #PrintResults()
    #print (nidx, uidx, zidx ,chimin_N)
    n_perc = np.array([n_16, n_50, n_84])
    u_perc = np.array([u_16, u_50, u_84])
    z_perc = np.array([z_16, z_50, z_84])    
    
    ###############################################################      Best Solution

    #print (T1_grid.shape, T2_grid.shape, A1_grid.shape, len(A1_grid))
    print ('Parameters Dimensions ',pn.shape,' ', pu.shape ,' ', pz.shape )
    print ('_______________________________________________________________________')
    print ("\nBest Fit ")
    print (' <n> = %3.2f'%n_grid[ni],'\t\t <u> = %3.4f'%u_grid[ui],'\t\t <Z/Z> = %3.2f'%z_grid[zi] )
    # print average b and its standard deviation
    print ('_______________________________________________________________________')
    print ('\nAverage Values, Standard Deviation')
    print ('\t <u> = %3.1f'%Ave[0],'+- %3.1f'%STD[0] ,
           ' <n> = %3.1f'%Ave[1],' +-  %3.1f'%STD[1],
           '\t <Z/Z> = %3.4f'%Ave[2],' +-  %3.4f'%STD[2] )
    print ('_______________________________________________________________________')
    print ('\nPercentiles (Mean)')
    print ('n: p_50 = %2.2f'%n_50 ,'\t\tp_16 = %2.2f'% n_16,'\t\tp_84 = %2.2f'% n_84)
    print ('U: p_50 = %2.1f'%u_50 ,'\t\tp_16 = %2.3f'% u_16,'\t\tp_84 = %2.3f'% u_84)
    print ('Z/Z: p_50 = %2.2f'%z_50 ,'\t\tp_16 = %2.2f'% z_16,'\t\tp_84 = %2.2f'% z_84) 
    print ('_______________________________________________________________________')

    
    
    ###############################################    Writing to file
    
    # Order:
    # BestSolution
    #', Mean+-STD '+\
    #', Percentiles '+\
    #LinesUsed:
    
    data = name+', '+str(FluxData)+\
    str(', %3.3f'%u_grid[ui])+\
    str(', %3.3f'%n_grid[ni])+\
    str(', %3.3f'%z_grid[zi])+\
    str(', %3.3f'%Ave[0])+str(', %3.3f'%STD[0])+\
    str(', %3.3f'%Ave[1])+str(', %3.3f'%STD[1])+\
    str(', %3.3f'%Ave[2])+str(', %3.3f'%STD[2])+\
    str(', %2.3f'%u_50+', %3.3f'%u_16+', %3.3f'%u_84)+\
    str(', %2.3f'%n_50+', %3.3f'%n_16+', %3.3f'%n_84)+\
    str(', %2.3f'%z_50+', %3.3f'%z_16+', %3.3f'%z_84)+\
    str(','+TxtLinesShort)

    #',  %3.2f'%u_grid[u_0]+',  %3.3f'%Delta_u_1+',  %3.3f'%Delta_u_2+\
    #',  %3.2f'%n_grid[n_0]+',  %3.3f'%Delta_n_1+',  %3.3f'%Delta_n_2+\
    #',  %3.2f'%z_grid[z_0]+',  %3.3f'%Delta_z_1+',  %3.3f'%Delta_z_2+\
    
    data =str(data)
    print (data)
    save = True
    if (save == True): 
        with open( SummaryFile, 'a') as f: # 'w' create new file, 'a' add to file
            f.write(data+'\n')
            f.close()
    print ('\n  --->   Summary Results File: ',SummaryFile)
    
    
    # Correlation Matrix
    C01 = CorrCoef(0, 1, Ave, prob, STD,  u_grid, n_grid, z_grid)
    C02 = CorrCoef(0, 2, Ave, prob, STD,  u_grid, n_grid, z_grid)
    C12 = CorrCoef(1, 2, Ave, prob, STD,  u_grid, n_grid, z_grid) 
    print (C01,'',C02,'', C12)
    
    
    
    ##############################################
    #
    #                                 Fig 1 : Prob Plots
    #
    ######DoProbPlots( saveFig, pltFileName1)       
    saveFig = True
    fig = plt.figure(figsize=(15,15))
    

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=22) 
    plt.rc('ytick', labelsize=18)  
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    #"xtick.labelsize":18,
    #"ytick.labelsize":18
    })

    fig.suptitle( title , fontsize=28)

    plt.subplot(311)
    ax1 = plt.plot( u_grid, pu , c='b', lw=0.9,  label ='Pu')
    plt.axvline(u_grid[ui], linestyle='--', color='c', alpha=0.6, label = 'Best Fit')
    plt.axvline(Ave[0],         linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(u_50,          linestyle='--', color='g', alpha=0.6, label = 'Perce 50')  
    plt.axvspan(u_16, u_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    plt.subplot(312)
    ax2 = plt.plot( n_grid, pn , c='b', lw=0.9,  label ='Pn') 
    plt.axvline(n_grid[ni],  linestyle='--', color='c', alpha=0.6, label = 'Best Fit')
    plt.axvline(Ave[1],      linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(n_50,        linestyle='--', color='g', alpha=0.6, label = 'Perce 50') 
    plt.axvspan(n_16, n_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    plt.subplot(313)
    ax3 = plt.plot( z_grid, pz , c='b', lw=0.9,  label ='Pz')
    plt.axvline(z_grid[zi], linestyle='--', color='c', alpha=0.6, label = 'Best Fit')
    plt.axvline(Ave[2],     linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(z_50,       linestyle='--', color='g', alpha=0.6, label = 'Perce 50')
    plt.axvspan(z_16, z_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    if (saveFig):   
        filename =  text1
        plt.savefig(filename+'.pdf') 
        print ("P-Plots save on: ",filename)
        
        
        
    
    #---- test
    z_min  =   z_limits[0] # z_grid.min()-0.1
    z_max  =   z_limits[1] # z_grid.max()
    n_min  =   n_limits[0] # 0.06*n_grid.min()
    n_max  =   n_limits[1] # n_grid.max()
    u_min  =   u_limits[0] # 1.05*u_grid.min()
    u_max  =   u_limits[1] # 0.95*u_grid.max()
    
    pz_min =  pz.min()
    pz_max =  1.10*pz.max()
    pn_min =  pn.min()
    pn_max =  1.10*pn.max()
    pu_min = pu.min()
    pu_max =  1.10*pu.max()

    ############################################################################################
    #
    #                                      FIG 2:   CORNER PLOTS
    #
    #
    ### DoTrianglePlots( saveFig, filename)
    filename =  text2  
    
    cm   =  plt.get_cmap("Blues") #Greys") # RdYlGn
    nl   = 10  
    fig    = plt.figure(figsize=(14,14), constrained_layout=False)
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    #"xtick.labelsize":14,
    #"ytick.labelsize":14 
    }) 
    
    gs1    = fig.add_gridspec(nrows=3, ncols=3, 
                    top=0.85, bottom=0.10, left=0.10, right=0.90,  wspace=0.00, hspace=0.00)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=24) 
    plt.rc('ytick', labelsize=24)   
    fig.suptitle( title, fontsize=32)
    
    ################################################# 1rst LINE
    ########################.      uPu
    nPn =   fig.add_subplot(gs1[0, 0]) #plt.subplot(3,3,1) 
    nPn.plot(  n_grid, pn , c='b', lw=2.5,  label ='P(n)') 
    plot_probs(n_grid[ni],Ave[1],n_16, n_84, n_min, n_max, pn_min, pn_max, nPn,  '', False  ) #'P(n)'
    
    
    ################################################# 2ND LINE
    ########################.      nu
    Ptable = prob.sum(axis = 2 ) 
    col  =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev = np.arange(1.2*np.min(Ptable), 1.00*np.max(Ptable), np.max(Ptable)/nl)  
    nu = fig.add_subplot(gs1[1, 0]) #plt.subplot(3,3,4)
    nu.contourf(n_grid, u_grid,  Ptable, levels=hlev, colors = col , extend='both')   
    xtext = 1.05*n_min
    ytext = -2.8 #1.4*(u_min-u_max)
    nu.text( xtext,  ytext, '$C_{01}$: %2.2f'%C01, fontsize=24, )  
    plot_grids(n_grid[ni], u_grid[ui], Ave[1], Ave[0], n_min, n_max, u_min, u_max, nu, '', 'log$(U)$', False, True )

    
    ########################.      u Pu
    uPu = fig.add_subplot(gs1[1, 1]) #plt.subplot(3,3,5)
    uPu.plot(  u_grid, pu , c='b', lw=2.5,  label ='P(U)') 
    #plot_probs(u_grid[ui],Ave[0],  u_16 ,  u_84 , u_min, u_max, pu_min, pu_max, uPu, '', False  )
    plot_probs(u_grid[ui],Ave[0],  u_16 ,  u_84 , u_min, u_max, pu_min, pu_max, uPu, '', False  ) #'P(U)'
    
    
    
    ################################################# 3RD LINE
    ########################.      nz
    Ptable = prob.sum(axis = 0 ) 
    col    =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev   = np.arange(0.0000*np.min(Ptable), 1.00*np.max(Ptable), np.max(Ptable)/nl) #### <------------------------------ Test2
    nz     = fig.add_subplot(gs1[2, 0]) #plt.subplot(3,3,7)
    nz.contourf( n_grid, z_grid, np.transpose(Ptable), levels=hlev, colors = col , extend='both') 
    ytext  = 0.85*(z_max-z_min)
    nz.text( xtext,  ytext, '$C_{02}$: %2.2f'%C02, fontsize=22)
    plot_grids(n_grid[ni],z_grid[zi], Ave[1], Ave[2], n_min, n_max,  z_min, z_max, nz,'log$(n_{H}$ cm$^{-3}$)', 'Z/Z$_{\odot}$',  True, True )

    ########################.      uz
    Ptable = prob.sum(axis = 1 ) 
    col  =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev = np.arange(1.8*np.min(Ptable), 1.0*np.max(Ptable), np.max(Ptable)/nl)
    uz = fig.add_subplot(gs1[2, 1]) #plt.subplot(3,3,8)
    uz.contourf( u_grid, z_grid, np.transpose(Ptable), levels=hlev, colors = col , extend='both') 
    #### <----------------- Test3
    xtext = 0.95*u_min
    uz.text( xtext, ytext, '$C_{12}$: %2.2f'%C12, fontsize=22)
    plot_grids(u_grid[ui], z_grid[zi], Ave[0], Ave[2], u_min, u_max,  z_min, z_max, uz, 'log$(U)$', '', True , False)
    
    ########################.      zPz
    zPz =fig.add_subplot(gs1[2, 2]) #plt.subplot(3,3,9)
    zPz.plot(z_grid, pz, c='b', lw=2.5,  label ='P(Z/Z$_{\odot}$)')
    plot_probs(z_grid[zi],Ave[2],  z_16 ,  z_84, z_min, z_max, pz_min, pz_max, zPz, 'Z/Z$_{\odot}$', True )
    
    ########################.   
    for axis in ['top','bottom','left','right']:
            nPn.spines[axis].set_linewidth(3.00) 
            nu.spines[axis].set_linewidth(3.00) 
            uPu.spines[axis].set_linewidth(3.00) 
            nz.spines[axis].set_linewidth(3.00) 
            uz.spines[axis].set_linewidth(3.00) 
            zPz.spines[axis].set_linewidth(3.00)   
    
    #
    # Write Results on TrianglePlots
    #
    ResultsTXT = ''
    ResultsTXT = ResultsTXT+'\n-----------------------------'
    
    ResultsTXT = '----------------------------- \n    Best Fit:'+' \n'
    ResultsTXT = ResultsTXT+'\t $log(U)$ = %3.1f'%u_grid[ui]+'\n\t'
    ResultsTXT = ResultsTXT+'\t $log(n_{H})$ = %3.1f'%n_grid[ni]+'\n\t'
    ResultsTXT = ResultsTXT+'\t Z/Z$_{\odot}$ = %3.1f'%z_grid[zi]
    ResultsTXT = ResultsTXT+'\n----------------------------- \n   Average Values and Error'
    ResultsTXT = ResultsTXT+'\n\t $log(U)$ = %3.1f'%Ave[0]+' +-  %3.1f'%STD[0]
    ResultsTXT = ResultsTXT+'\n\t $log(n_{H})$ = %3.1f'%Ave[1]+' +-  %3.1f'%STD[1]
    ResultsTXT = ResultsTXT+'\n\t Z/Z$_{\odot}$ = %3.1f'%Ave[2]+' +-  %3.1f'%STD[2]
    
    plotresults = True
    if (plotresults):
        #nPn.text(n_max*1.2,-0.0001, ResultsTXT, fontsize=22)
        #
        zPz.text(z_grid.min()+0.15, 2.3*pz.max(), ResultsTXT, fontsize=22)
        
    if (saveFig):   
        plt.savefig(filename+'.pdf') 
        print ("Corner-Plots save on: ",filename)
        
        
        
    return ( Ave, STD, n_perc, u_perc, z_perc )









def GetSolutions_2( prob, pu, pn, pz, ui,  ni, zi, u_grid, n_grid, z_grid, text1, text2, title, SummaryFile, name, FluxData, TxtLinesShort ):
    npar = 3
    Ave, STD   = [0.0]*npar, [0.0]*npar
    Ave[0], Ave[1] , Ave[2]   =  np.sum(u_grid*pu), np.sum(n_grid*pn),  np.double(np.sum(z_grid*pz))

    STD[0] = np.sqrt( np.sum( (u_grid - Ave[0])**2 *pu ))
    STD[1] = np.sqrt( np.sum( (n_grid - Ave[1])**2 *pn ))
    STD[2] = np.sqrt( np.sum( (z_grid - Ave[2])**2 *pz ))#np.sqrt(np.sum(z_grid**2*Pz)-np.sum(z_grid*Pz)**2)

    n_50 = PercentileXPx(n_grid, pn, 50, False)
    u_50 = PercentileXPx(u_grid, pu, 50, False) 
    z_50 = PercentileXPx(z_grid, pz, 50, False) 

    n_84 = PercentileXPx(n_grid, pn, 84, False)
    u_84 = PercentileXPx(u_grid, pu, 84, False) 
    z_84 = PercentileXPx(z_grid, pz, 84, False) 

    n_16 = PercentileXPx(n_grid, pn, 16, False)
    u_16 = PercentileXPx(u_grid, pu, 16, False)  
    z_16 = PercentileXPx(z_grid, pz, 16, False) 

    #uidx, nidx, zidx = np.where(chisqr_grid_N == chimin_N)
    #PrintResults()
    #print (nidx, uidx, zidx ,chimin_N)
    n_perc = np.array([n_16, n_50, n_84])
    u_perc = np.array([u_16, u_50, u_84])
    z_perc = np.array([z_16, z_50, z_84])    
    
    ###############################################################      Best Solution

    #print (T1_grid.shape, T2_grid.shape, A1_grid.shape, len(A1_grid))
    print ('Parameters Dimensions ',pn.shape,' ', pu.shape ,' ', pz.shape )
    print ('_______________________________________________________________________')
    print ("\nBest Fit ")
    print (' <n> = %3.2f'%n_grid[ni],'\t\t <u> = %3.4f'%u_grid[ui],'\t\t <Z/Z> = %3.2f'%z_grid[zi] )
    # print average b and its standard deviation
    print ('_______________________________________________________________________')
    print ('\nAverage Values, Standard Deviation')
    print ('\t <u> = %3.1f'%Ave[0],'+- %3.1f'%STD[0] ,
           ' <n> = %3.1f'%Ave[1],' +-  %3.1f'%STD[1],
           '\t <Z/Z> = %3.4f'%Ave[2],' +-  %3.4f'%STD[2] )
    print ('_______________________________________________________________________')
    print ('\nPercentiles (Mean)')
    print ('n: p_50 = %2.2f'%n_50 ,'\t\tp_16 = %2.2f'% n_16,'\t\tp_84 = %2.2f'% n_84)
    print ('U: p_50 = %2.1f'%u_50 ,'\t\tp_16 = %2.3f'% u_16,'\t\tp_84 = %2.3f'% u_84)
    print ('Z/Z: p_50 = %2.2f'%z_50 ,'\t\tp_16 = %2.2f'% z_16,'\t\tp_84 = %2.2f'% z_84) 
    print ('_______________________________________________________________________')

    
    
    ###############################################    Writing to file
    
    # Order:
    # BestSolution
    #', Mean+-STD '+\
    #', Percentiles '+\
    #LinesUsed:
    
    data = name+', '+str(FluxData)+\
    str(', %3.3f'%u_grid[ui])+\
    str(', %3.3f'%n_grid[ni])+\
    str(', %3.3f'%z_grid[zi])+\
    str(', %3.3f'%Ave[0])+str(', %3.3f'%STD[0])+\
    str(', %3.3f'%Ave[1])+str(', %3.3f'%STD[1])+\
    str(', %3.3f'%Ave[2])+str(', %3.3f'%STD[2])+\
    str(', %2.3f'%u_50+', %3.3f'%u_16+', %3.3f'%u_84)+\
    str(', %2.3f'%n_50+', %3.3f'%n_16+', %3.3f'%n_84)+\
    str(', %2.3f'%z_50+', %3.3f'%z_16+', %3.3f'%z_84)+\
    str(','+TxtLinesShort)

    #',  %3.2f'%u_grid[u_0]+',  %3.3f'%Delta_u_1+',  %3.3f'%Delta_u_2+\
    #',  %3.2f'%n_grid[n_0]+',  %3.3f'%Delta_n_1+',  %3.3f'%Delta_n_2+\
    #',  %3.2f'%z_grid[z_0]+',  %3.3f'%Delta_z_1+',  %3.3f'%Delta_z_2+\
    
    data =str(data)
    print (data)
    save = True
    if (save == True): 
        with open( SummaryFile, 'a') as f: # 'w' create new file, 'a' add to file
            f.write(data+'\n')
            f.close()
    print ('\n  --->   Summary Results File: ',SummaryFile)
    
    
    # Correlation Matrix
    C01 = CorrCoef(0, 1, Ave, prob, STD,  u_grid, n_grid, z_grid)
    C02 = CorrCoef(0, 2, Ave, prob, STD,  u_grid, n_grid, z_grid)
    C12 = CorrCoef(1, 2, Ave, prob, STD,  u_grid, n_grid, z_grid) 
    print (C01,'',C02,'', C12)
    
    
    
    ##############################################
    #
    #                                 Fig 1 : Prob Plots
    #
    ######DoProbPlots( saveFig, pltFileName1)       
    saveFig = True
    fig = plt.figure(figsize=(15,15))
    

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.rc('xtick', labelsize=14) 
    #plt.rc('ytick', labelsize=14)  
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

    fig.suptitle( title , fontsize=28)

    plt.subplot(311)
    ax1 = plt.plot( u_grid, pu , c='b', lw=0.9,  label ='Pu')
    plt.axvline(u_grid[ui], linestyle='--', color='c', alpha=0.6, label = 'Best Fit')
    plt.axvline(Ave[0],         linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(u_50,          linestyle='--', color='g', alpha=0.6, label = 'Perce 50')  
    plt.axvspan(u_16, u_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    plt.subplot(312)
    ax2 = plt.plot( n_grid, pn , c='b', lw=0.9,  label ='Pn') 
    plt.axvline(n_grid[ni],  linestyle='--', color='c', alpha=0.6 , label = 'Best Fit')
    plt.axvline(Ave[1],      linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(n_50,        linestyle='--', color='g', alpha=0.6, label = 'Perce 50') 
    plt.axvspan(n_16, n_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    plt.subplot(313)
    ax3 = plt.plot( z_grid, pz , c='b', lw=0.9,  label ='Pz')
    plt.axvline(z_grid[zi], linestyle='--', color='c', alpha=0.6, label = 'Best Fit')
    plt.axvline(Ave[2],     linestyle='--', color='r', alpha=0.6, label = 'Average')
    plt.axvline(z_50,       linestyle='--', color='g', alpha=0.6, label = 'Perce 50')
    plt.axvspan(z_16, z_84, color='r',  alpha=0.1)
    plt.legend(loc=1, fontsize = 18 )

    if (saveFig):   
        filename =  text1
        plt.savefig(filename+'.pdf') 
        print ("P-Plots save on: ",filename)
        
        
        
        
    ############################################################################################
    #
    #                                      FIG 2:   CORNER PLOTS
    #
    #
    ### DoTrianglePlots( saveFig, filename)
    filename =  text2  
    
    cm   =  plt.get_cmap("Blues") #Greys") # RdYlGn
    nl   = 10

    #fig = plt.figure(figsize=(15,15))
    
    fig    = plt.figure(figsize=(14,12), constrained_layout=False)
    gs1    = fig.add_gridspec(nrows=3, ncols=3, 
                            top=0.90, bottom=0.10, left=0.10, right=0.90,
                            wspace=0.00, hspace=0.00)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14)  

    fig.suptitle( title, fontsize=28)


    nPn =   fig.add_subplot(gs1[0, 0]) #plt.subplot(3,3,1)
    #plt.xscale('log') 
    nPn.plot(  n_grid, pn , c='b', lw=2.5,  label ='P(n)')
    nPn.axvline(n_grid[ni],  linestyle='--',   lw=2, color='limegreen', alpha=1,  label = 'Best Fit') 
    nPn.axvline(Ave[1],      linestyle='--',   lw=2, color='r',     alpha=1,  label = 'Mean') 
    #nPn.axvline(n,  linestyle='--', color='g', alpha=0.5) 
    plt.axvspan(n_16, n_84, color='r',  alpha=0.1) 

    #plt.ylabel('P(n)', { 'fontsize':28, 'color':'black'})
    
    #nPn.legend(loc=1, fontsize = 18 )
    
    n_min =  0.6*n_grid.min()
    n_max =  0.9*n_grid.max()
    
    
    
    plt.xlim([n_min, n_max])
    plt.ylim([-0.05*pn.max(), 1.1*pn.max()])
    plt.setp(nPn.get_yticklabels(), visible=False)
    nPn.tick_params(axis='y',which='both',length=10.0, direction='in', right=True, labelright = True )
    nPn.tick_params(axis='x',which='both',length=10.0, direction='in', top=True,  labelbottom = False )
    
    nPn.legend(loc='upper left', bbox_to_anchor=(0.45, 1.0), fontsize = 18 )
    
    
    
    
    
    
    
    
    ################################################# 2ND LINE
    ########################.      nu
    Ptable = prob.sum(axis = 2 )
    #Ptable = Ptable.sum(axis = 1 ) 
    col  =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev = np.arange(1.2*np.min(Ptable), 1.00*np.max(Ptable), np.max(Ptable)/nl) #.00001  #### <------ Test1
    
    nu = fig.add_subplot(gs1[1, 0]) #plt.subplot(3,3,4)
    nu.contourf(n_grid, u_grid,  Ptable, levels=hlev, colors = col , extend='both')  #np.transpose(Ptable)
    nu.plot( Ave[1], Ave[0], 'ro', markersize='12'  )
    nu.plot( n_grid[ni], u_grid[ui], 'o', color = 'limegreen', markersize='12' )
    
    xtext = 0.60*n_grid.max() #-n_grid.min()) + n_grid.min()
    ytext = 1.20*u_grid.max()
    nu.text( xtext,  ytext, '$C_{01}$: %2.2f'%C01, fontsize=22, )
    print( ' u_grid.max() ',u_grid.max(), '1.1*u_grid.max()',1.1*u_grid.max())
    
    #nu.plot( n, u, 'go' )
    #plt.xscale('log') 
    plt.xlim([n_min, n_max])
    plt.ylabel('$log$(U)', { 'fontsize':24, 'color':'black'})
    
    nu.tick_params(axis='y',which='both',length=10.0, direction='in', right=True, labelright = False )
    nu.tick_params(axis='x',which='both',length=10.0, direction='in', top=True, labelbottom = False)
    plt.setp(nu.get_yticklabels(), visible=True)
    #plt.setp(nu.get_yticklabels(), visible=False) 
    
    ########################.      u Pu
    
    uPu = fig.add_subplot(gs1[1, 1]) #plt.subplot(3,3,5)
    uPu.plot(  u_grid, pu , c='b', lw=2.5,  label ='P(U)')
    uPu.axvline( u_grid[ui],  linestyle='--',   lw=2, color='limegreen', alpha=1)
    uPu.axvline(Ave[0],       linestyle='--',   lw=2, color='r',         alpha=1)
    #uPu.axvline(u,  linestyle='--', color='g', alpha=0.5)
    plt.axvspan( u_16 ,  u_84 , color='r',  alpha=0.1)
    #plt.ylabel('P(U)', { 'fontsize':28, 'color':'black'})

    plt.ylim([-0.05*pu.max(), 1.1*pu.max()])
    u_max =  0.97*u_grid.max()
    u_min =  0.98*u_grid.min()
    plt.xlim([u_min, u_max])
    
    plt.legend(loc=1, fontsize = 18 )
    
    
    # Hide the y-axis labels on the left side
    uPu.tick_params(axis='y', labelleft=False)
    # Set the y-axis label on the right side
    uPu.tick_params(axis='y', labelright=True)
    # Hide the x-axis labels on the bottom side
    #uPu.tick_params(axis='x', bottom=False)
    
    # Hide labels
    #uz.set_yticks([])
    uPu.set_xticklabels([])
    
    #uPu.set_ylabel('Your Y-axis Label', fontsize=12)
    
    
    
    
    ################################################# 3RD LINE
    ########################.      nz
    Ptable = prob.sum(axis = 0 )
    #Ptable = Ptable.sum(axis = 1 ) 
    col  =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev = np.arange(0.0000*np.min(Ptable), 1.00*np.max(Ptable), np.max(Ptable)/nl) #### <------------------------------ Test2
    
    nz = fig.add_subplot(gs1[2, 0]) #plt.subplot(3,3,7)
    nz.contourf( n_grid, z_grid, np.transpose(Ptable), levels=hlev, colors = col , extend='both') 
    nz.plot( Ave[1], Ave[2], 'o', color = 'r', markersize='12'  )
    nz.plot( n_grid[ni],z_grid[zi], 'o', color = 'limegreen', markersize='12' )
    
    #xtext = 0.5*(n_grid.max()-n_grid.min()) + n_grid.min()
    ytext = 0.80*z_grid.max()
    nz.text( xtext,  ytext, '$C_{02}$: %2.2f'%C02, fontsize=22)
    
    plt.ylabel('Z/Z$_{\odot}$', { 'fontsize':26, 'color':'black'})
    plt.xlabel('$log$(n$_{H}$ cm$^-3$)', { 'fontsize':26, 'color':'black'})
    #nz.plot( n , z , 'go' )
    #plt.title( 'nz' )
    #plt.xscale('log') 
    # Hide labels
    #uz.set_yticks([])
    #uz.set_yticklabels([])
    plt.xlim([n_min, n_max])
    nz.tick_params(axis='y',which='both',length=10.0, direction='in', right=True, labelright = False )
    nz.tick_params(axis='x',which='both',length=10.0, direction='in', top=True)
    plt.setp(nz.get_yticklabels(), visible=True)
    #plt.setp(nz.get_yticklabels(), visible=False) 

    ########################.      uz
    Ptable = prob.sum(axis = 1 )
    #Ptable = Ptable.sum(axis = 0 ) 
    col  =  [cm(float(i)/(nl)) for i in range(nl)] 
    hlev = np.arange(1.8*np.min(Ptable), 1.0*np.max(Ptable), np.max(Ptable)/nl) #### <----------------- Test3
    uz = fig.add_subplot(gs1[2, 1]) #plt.subplot(3,3,8)
    uz.contourf( u_grid, z_grid, np.transpose(Ptable), levels=hlev, colors = col , extend='both') 
    uz.plot( Ave[0], Ave[2], 'ro', markersize='12'  )
    #uz.plot( u, z, 'go' )
    uz.plot( u_grid[ui], z_grid[zi], 'o', color = 'limegreen', markersize='12')
    plt.xlabel('$log$(U)', { 'fontsize':26, 'color':'black'})
    
    
    xtext = 0.20*u_grid.min() #-u_grid.min()) + u_grid.min()
    #ytext = 0.85*z_grid.max()
    uz.text( xtext, ytext, '$C_{12}$: %2.2f'%C12, fontsize=22)
    
    
    plt.xlim([u_min, u_max])
    # Hide labels
    #uz.set_yticks([])
    #uz.set_yticklabels([])
    uz.tick_params(axis='y',which='both',length=10.0, direction='in', right=True, labelright = False )
    uz.tick_params(axis='x',which='both',length=10.0, direction='in', top=True)
    #plt.setp(uz.get_yticklabels(), visible=True)
    plt.setp(uz.get_yticklabels(), visible=False) 


    ########################.      zPz
    zPz =fig.add_subplot(gs1[2, 2]) #plt.subplot(3,3,9)
    zPz.plot(z_grid, pz, c='b', lw=2.5,  label ='P(Z/Z$_{\odot}$)')
    zPz.axvline(z_grid[zi],  linestyle='--',   lw=2.5, color='limegreen', alpha=1)
    zPz.axvline(Ave[2],      linestyle='--',   lw=2.5, color='r',         alpha=1)
    #zPz.axvline(z,  linestyle='--', color='green', alpha=0.5)
    plt.axvspan(  z_16 ,  z_84, color='r',  alpha=0.1)
    plt.setp(zPz.get_xticklabels(), visible=True)
    plt.xlabel('Z/Z$_{\odot}$', { 'fontsize':26, 'color':'black'})
    
    plt.ylim([-0.05*pz.max() ,1.1*pz.max()])
    plt.legend(loc=1, fontsize = 22 )
    
    # Hide the y-axis labels on the left side
    zPz.tick_params(axis='y', labelleft=False) 
     
    # Set the y-axis label on the right side
    zPz.tick_params(axis='y', labelright=True)
    # Hide the x-axis labels on the bottom side
    #uPu.tick_params(axis='x', bottom=False)
    
    
    
    ########################.   
    
    for axis in ['top','bottom','left','right']:
        
            nPn.spines[axis].set_linewidth(3.00) 
            nu.spines[axis].set_linewidth(3.00) 
            uPu.spines[axis].set_linewidth(3.00) 
            
            nz.spines[axis].set_linewidth(3.00) 
            uz.spines[axis].set_linewidth(3.00) 
            zPz.spines[axis].set_linewidth(3.00)   
    
    #plt.subplots_adjust(top=0.85, bottom=0.15, left=0.12, right=0.88, hspace=0.01, wspace=0.01)

    
    #
    # Write Results on TrianglePlots
    #
    ResultsTXT = ''
    ResultsTXT = ResultsTXT+'\n-----------------------------'
    
    ResultsTXT = '----------------------------- \n    Best Fit:'+' \n'
    ResultsTXT = ResultsTXT+'\t $log(U)$ = %3.1f'%u_grid[ui]+'\n\t'
    ResultsTXT = ResultsTXT+'\t $log(n_{H})$ = %3.1f'%n_grid[ni]+'\n\t'
    ResultsTXT = ResultsTXT+'\t Z/Z$_{\odot}$ = %3.1f'%z_grid[zi]
    ResultsTXT = ResultsTXT+'\n----------------------------- \n   Average Values and Error'
    ResultsTXT = ResultsTXT+'\n\t $log(U)$ = %3.1f'%Ave[0]+' +-  %3.1f'%STD[0]
    ResultsTXT = ResultsTXT+'\n\t $log(n_{H})$ = %3.1f'%Ave[1]+' +-  %3.1f'%STD[1]
    ResultsTXT = ResultsTXT+'\n\t Z/Z$_{\odot}$ = %3.1f'%Ave[2]+' +-  %3.1f'%STD[2]
    
    plotresults = True
    if (plotresults):
        #nPn.text(n_max*1.2,-0.0001, ResultsTXT, fontsize=22)
        #
        zPz.text(z_grid.min()+0.1, 2.3*pz.max(), ResultsTXT, fontsize=22)
        
    if (saveFig):   
        plt.savefig(filename+'.pdf') 
        print ("Corner-Plots save on: ",filename)
        
        
        
    return ( Ave, STD, n_perc, u_perc, z_perc )













# ------------------------------------------------------
#
# FUNCTION # 8
#
# ------------------------------------------------------




def EraseNAN(df1, ldf):
    cols = df1.columns
    for i in range(ldf):
        if (np.isnan(df1[cols[0]][i])): 
            df1 = df1.drop(i)
    df1 = df1.reset_index()
    return df1






# ------------------------------------------------------
#
# FUNCTION # 10
#
# ------------------------------------------------------

from matplotlib import rc
#rc("pdf", fonttype=3)
#rc('font',**{'family':'serif'})
#rc('text', usetex=True)




def Doplot(Num, title, Xcol, Ycol,  Yerror, Labels):
    ax = plt.subplot(2,3,Num)
    plt.title(  title ,  {'color': 'black', 'fontsize': 28} )    #fontsize=16, fontweight='bold'
    plt.xlabel('Galaxy', { 'fontsize':20, 'color':'black'})
    #plt.ylabel('$[OIII]-52/[OIII]-88$', { 'fontsize':24, 'color':'black'}) 
    plt.xticks( np.arange(5) , ['0','1','2','3','5'])#, { 'fontsize':24, 'color':'black'}) 
    lgals = len(Xcol)
    
    
    cm       = plt.get_cmap('rainbow')
    clrs1    = [cm(float(i)/(lgals )) for i in range(lgals)]
    mrks     = ['o','o','o','o','o','o','o','o','o','o','o','o',] 

    
    for i in range(lgals):
        plt.scatter( Xcol[i],   Ycol[i], marker ='o',s =150, color = clrs1[i], 
                  edgecolors = 'black', linewidths=2.0, label= str(i)+'-'+Labels[i])
        
    ax.errorbar( Xcol,  Ycol, yerr = (Yerror), 
            fmt = 'o', markersize = 1,  color = 'r',     ecolor='k', 
            linewidth =1.0, capsize=3.0, capthick=1.0, alpha = 0.99 , uplims=False, zorder = 3)

    
    ax.set_xlim( -0.5, 1.0+Xcol.max()  )
    ax.set_ylim(  0.1, 1.5*Ycol.max() ) 
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0) 
    
    return ax




# ------------------------------------------------------
#
# FUNCTION # 12
#
# ------------------------------------------------------

#def AddNoiseGaussian(y, y_sig, mean, std, factor):
#    print ('Noyse factor: ',factor)
#    ly = len(y) 
#    o_noise =  np.random.normal(mean, std, ly) 
#    y_mod_noise = np.array( y + ( y * factor * o_noise ))
#    print ('Noyse Vectos:\n',o_noise) 
#    return y_mod_noise



# ------------------------------------------------------
#       
#      Update function   
#
# ------------------------------------------------------

def AddNoiseGaussian(y,  mean, std, factor, PrintVectors):
    ly = len(y) 
    o_noise =  np.random.normal(mean, std, ly) 
    y_mod_noise = np.array( y + ( y * factor * o_noise ))
    if PrintVectors:
        print ('Noise factor: ',factor)
        print ('Noise Vectors:\n',o_noise) 
        print ('Model + Noise Vectors:\n',y_mod_noise) 
    return y_mod_noise




# ------------------------------------------------------
#
# FUNCTION # 10
#
# ------------------------------------------------------