# -*- coding: utf-8 -*-
import json
import datetime
import time
import gzip
import numpy as np



##### Uncomment if you want to see how long it takes to run the script.
###tic = time.time()

### These offsets are:
###     Element 0: The beginning day of the forecast validation period.
###     Element 1: The centroid day of the forecast validation period.
###     Element 2: The end day of the forecast validation period.
###     The forecast period is then evaluated for days 16-29, which includes 
###         a one-day latency of the real-time dynamical model forecasts.
dayOffsets = [16,23,29]


###############################################################################
###############################################################################
##### Section 1: Read in the GEFSv12, ECMWF reforecasts and their weeks 3-4 predictions 
#####   of ACE in the Atlantic and the West Atlantic
###############################################################################
###############################################################################

### Read in all of the GEFSv12 reforecasts over the period 2000-2019. 
### The GEFSv12 reforecast data have a temporal resolution of six hours.
### This is a json file that is structured with the highest-order dictionary 
###     containing the dates of the forecasts. 
### Then, for each date there are the following elements:
###     ['windSpeed', 'latitude', 'longitude', 'outlookTime', 'ensembleNumber', 'stormNumber']
###     All of these six arrays are of the same size. For each date, all of the 
###     wind speeds are flattened into one long array, 
###     where each element of the windSpeed array has a corresponding latitude, 
###     longitude, outlook time, ensemble number, and storm number.

f1 = open('gefsv12_6hr_forecast_data.json','r')
gefsv12 = json.load(f1)
f1.close()

### Read all of the reforecast dates.
keys_dates = sorted(list(gefsv12.keys()))
dateValsGEFS = np.zeros((len(keys_dates),3),dtype=np.int16)
for itval1 in range(0,len(keys_dates)):
    dateValsGEFS[itval1,0] = int(keys_dates[itval1][0:4])
    dateValsGEFS[itval1,1] = int(keys_dates[itval1][5:7])
    dateValsGEFS[itval1,2] = int(keys_dates[itval1][8:10])

### Find the ordinal dates of the reforecast dates. 
### This will be used to locate the most recently available ECMWF reforecast prior to these dates.
dateValsGEFSord = np.zeros((dateValsGEFS.shape[0]),dtype=np.int32)
for dt in range(0,dateValsGEFS.shape[0]):
    dateValsGEFSord[dt] = datetime.date(dateValsGEFS[dt,0],dateValsGEFS[dt,1],dateValsGEFS[dt,2]).toordinal()  

### Find the time-centroid our 16-29 day validation period corresponding to the reforecast dates. 
dateValsGEFSValidationCentroid = np.zeros((1043,3),dtype=np.int16)
for day in range(0,dateValsGEFS.shape[0]):
    dtnow = datetime.date(*dateValsGEFS[day,:])+datetime.timedelta(days=dayOffsets[1])
    dateValsGEFSValidationCentroid[day,:] = [dtnow.year,dtnow.month,dtnow.day]

### Calculate the raw predicted ACE values from GEFSv12 over our validation period 16-29 days
###     and within the Atlantic and West Atlantic spatial domains.
predicted_ACE_Atlantic_GEFS = np.zeros((len(keys_dates),11)) ### This is the predicted ACE in the Atlantic domain. 
predicted_ACE_WestAtlantic_GEFS = np.zeros((len(keys_dates),11)) ### This is the predicted ACE in the Atlantic domain. 
for itval1 in range(0,len(keys_dates)):
    latNow1 = np.array(gefsv12[keys_dates[itval1]]['latitude'])
    lonNow1 = np.array(gefsv12[keys_dates[itval1]]['longitude'])
    windNow1 = np.array(gefsv12[keys_dates[itval1]]['windSpeed'])
    outlookNow1 = np.array(gefsv12[keys_dates[itval1]]['outlookTime'])
    ensembleNow1 = np.array(gefsv12[keys_dates[itval1]]['ensembleNumber'])
    uniqueEnsembles1 = np.unique(ensembleNow1)
    for itval2 in range(0,len(uniqueEnsembles1)):
        fndEns1 = np.nonzero(ensembleNow1 == uniqueEnsembles1[itval2])[0]
        fndNA1 = np.nonzero(windNow1[fndEns1]>=1)[0]
        fndOutlook1 = np.nonzero(outlookNow1[fndEns1[fndNA1]]>=dayOffsets[0])[0]
        fndOutlook1b = np.nonzero(outlookNow1[fndEns1[fndNA1[fndOutlook1]]]<=dayOffsets[2])[0]
        predicted_ACE_Atlantic_GEFS[itval1,uniqueEnsembles1[itval2]] = np.sum(windNow1[fndEns1[fndNA1[fndOutlook1[fndOutlook1b]]]]**2)/10000
        fndLoc1 = np.nonzero(lonNow1[fndEns1[fndNA1[fndOutlook1[fndOutlook1b]]]]>=-105)[0]
        fndLoc1b = np.nonzero(lonNow1[fndEns1[fndNA1[fndOutlook1[fndOutlook1b[fndLoc1]]]]]<=-60)[0]
        predicted_ACE_WestAtlantic_GEFS[itval1,uniqueEnsembles1[itval2]] = \
            np.sum(windNow1[fndEns1[fndNA1[fndOutlook1[fndOutlook1b[fndLoc1[fndLoc1b]]]]]]**2)/10000

### Read in all of the ECMWF reforecasts over the period 2000-2019. This dictionary 
###     has the same structure as the GEFSv12 reforecast data set. 
### The ECMWF reforecast data have a temporal resolution of 12 hours.
f1 = open('ecmwf_12hr_forecast_data.json','r')
ecmwf = json.load(f1)
f1.close()

keys_dates = sorted(list(ecmwf.keys()))
dateValsECMWF = np.zeros((len(keys_dates),3),dtype=np.int16)
for itval1 in range(0,len(keys_dates)):
    dateValsECMWF[itval1,0] = int(keys_dates[itval1][0:4])
    dateValsECMWF[itval1,1] = int(keys_dates[itval1][5:7])
    dateValsECMWF[itval1,2] = int(keys_dates[itval1][8:10])

dateValsECMWFord = np.zeros((dateValsECMWF.shape[0]),dtype=np.int32)
for dt in range(0,dateValsECMWF.shape[0]):
    dateValsECMWFord[dt] = datetime.date(dateValsECMWF[dt,0],dateValsECMWF[dt,1],dateValsECMWF[dt,2]).toordinal()  

dateValsECMWFValidationCentroid = np.zeros((len(keys_dates),3),dtype=np.int16)
for day in range(0,dateValsECMWF.shape[0]):
    dtnow = datetime.date(*dateValsECMWF[day,:])+datetime.timedelta(days=dayOffsets[1])
    dateValsECMWFValidationCentroid[day,:] = [dtnow.year,dtnow.month,dtnow.day]

predicted_ACE_Atlantic_ECMWF = np.zeros((len(keys_dates),11))
predicted_ACE_WestAtlantic_ECMWF = np.zeros((len(keys_dates),11))
for itval1 in range(0,len(keys_dates)):
    latNow2 = np.array(ecmwf[keys_dates[itval1]]['latitude'])
    lonNow2 = np.array(ecmwf[keys_dates[itval1]]['longitude'])
    windNow2 = np.array(ecmwf[keys_dates[itval1]]['windSpeed'])
    outlookNow2 = np.array(ecmwf[keys_dates[itval1]]['outlookTime'])
    ensembleNow2 = np.array(ecmwf[keys_dates[itval1]]['ensembleNumber'])
    uniqueEnsembles2 = np.unique(ensembleNow2)
    for itval2 in range(0,len(uniqueEnsembles2)):
        fndEns2 = np.nonzero(ensembleNow2 == uniqueEnsembles2[itval2])[0]
        fndNA2 = np.nonzero(windNow2[fndEns2]>=1)[0]
        fndOutlook2 = np.nonzero(outlookNow2[fndEns2[fndNA2]]>=dayOffsets[0])[0]
        fndOutlook2b = np.nonzero(outlookNow2[fndEns2[fndNA2[fndOutlook2]]]<=dayOffsets[2])[0]
        predicted_ACE_Atlantic_ECMWF[itval1,uniqueEnsembles2[itval2]] = np.sum(windNow2[fndEns2[fndNA2[fndOutlook2[fndOutlook2b]]]]**2)*2/10000
        fndLoc2 = np.nonzero(lonNow2[fndEns2[fndNA2[fndOutlook2[fndOutlook2b]]]]>=-105)[0]
        fndLoc2b = np.nonzero(lonNow2[fndEns2[fndNA2[fndOutlook2[fndOutlook2b[fndLoc2]]]]]<=-60)[0]
        predicted_ACE_WestAtlantic_ECMWF[itval1,uniqueEnsembles2[itval2]] = \
            np.sum(windNow2[fndEns2[fndNA2[fndOutlook2[fndOutlook2b[fndLoc2[fndLoc2b]]]]]]**2)*2/10000

### Here, we locate the most recently available ECMWF reforecast prior to the GEFSv12 reforecast dates.
nearestIndices = np.zeros((dateValsGEFS.shape[0]),dtype=np.int16)
nearestIndices2 = np.zeros((dateValsGEFS.shape[0]),dtype=np.int16)
for v1 in range(0,dateValsGEFS.shape[0]):
    dtDiff = dateValsGEFSord[v1]-dateValsECMWFord
    dtDiff[dtDiff<0] = 9999
    nearestIndices[v1]  = np.argsort(dtDiff)[0]    
    nearestIndices2[v1]  = np.sort(dtDiff)[0]    



###############################################################################
###############################################################################
##### Section 2: Read observed ACE in the Atlantic and the West Atlantic
###############################################################################
###############################################################################

### The vertices of the Atlantic and West Atlantic are as follows:
###     entire_atlantic_polygon = [[-5,10],[-83,10],[-100,20],[-105,30],[-105,50],[-5,50],[-5,10]]
###     west_atlantic_polygon = [[-60,10],[-83,10],[-100,20],[-105,30],[-105,50],[-60,50],[-60,10]]
### Daily ACE was calculated using the IBTrACS data set for these domains.
### The time series between Jan 1, 1901 - Dec 31, 2021 was computed and saved as numpy arrays.
### Read in these daily ACE values for the Atlantic.
f1 = np.load('aceVals_daily_entireAtlantic_19010101-20211231.npz')
aceValsDailyEA = f1['aceVals']
aceDateValsDailyEA = f1['dtvals']
f1.close()

### Read in these daily ACE values for the West Atlantic.
f1 = np.load('aceVals_daily_westAtlantic_19010101-20211231.npz')
aceValsDailyWA = f1['aceVals']
aceDateValsDailyWA = f1['dtvals']
f1.close()

### Calculate running biweekly ACE for the Atlantic.
aceValsBiweeklyEA = np.zeros((aceValsDailyEA.shape[0]-13))
for dt in range(0,len(aceValsBiweeklyEA)):
    aceValsBiweeklyEA[dt] = np.sum(aceValsDailyEA[dt:dt+14])
dtValsBiweeklyEA = np.array(aceDateValsDailyEA[:-13])

### Calculate running biweekly ACE for the West Atlantic.
aceValsBiweeklyWA = np.zeros((aceValsDailyWA.shape[0]-13))
for dt in range(0,len(aceValsBiweeklyWA)):
    aceValsBiweeklyWA[dt] = np.sum(aceValsDailyWA[dt:dt+14])
dtValsBiweeklyWA = np.array(aceDateValsDailyWA[:-13])


###############################################################################
###############################################################################
##### Section 3: Read in the daily OISST data set at a 5 degree resolution.
###############################################################################
###############################################################################

### Read in a multidimensional array of daily SSTs. It has the shape (days,lat,lon)
f1 = np.load("oisst_5deg_0N-50N_19820101-20201231.npz")
sst = f1['sst']
lat_sst = f1['lat']
lon_sst = f1['lon']
f1.close()

### Create a time series of the dates for the SST data.
d1 = datetime.date(1982,1,1).toordinal()
d2 = datetime.date(2021,1,1).toordinal()
sstDates = np.zeros((d2-d1,3),dtype=np.int16)
for day in range(d1,d2):
    currday = datetime.date.fromordinal(day)
    sstDates[day-d1,:] = [currday.year,currday.month,currday.day]

### Locate where there is the SST data. Then we can reshape the array 
###     to simply be (time,space)
fndNonNan = np.nonzero(sst[0,:,:]>0)
sstR = np.array(sst[:,fndNonNan[0],fndNonNan[1]])

### Here, we remove Feb 29, and reshape the sstR array to have the shape (year,day,gridCell)
fnd1 = np.nonzero(sstDates[:,1] == 2)[0]
fnd2 = np.nonzero(sstDates[fnd1,2] == 29)[0]
sstR = np.delete(sstR,fnd1[fnd2],axis=0)
sstR = np.reshape(sstR,(39,365,sstR.shape[1]))
sstD = np.delete(sstDates,fnd1[fnd2],axis=0)
sstD = np.reshape(sstD,(39,365,3))

### Here, set which domain to calculate the forecast skill. 
###     Either for the entire north Atlantic domain, or the subregion of the West Atlantic.
domainOptions = ["EntireAtlantic","WestAtlantic"]

for domain in domainOptions:


    ### Given the domain of interest, obtain the current set of biweekly ACE values and associated dates 
    if domain == "EntireAtlantic":
        dtValsBiweekly = np.array(dtValsBiweeklyEA)
        aceValsBiweekly = np.array(aceValsBiweeklyEA)
    elif domain == "WestAtlantic":
        dtValsBiweekly = np.array(dtValsBiweeklyWA)
        aceValsBiweekly = np.array(aceValsBiweeklyWA)
    print(domain)
    
    ### Create an array of the biweekly ACE that overlaps with the GEFSv12 reforecasts. 
    ###     This is subset of the entire time series, and this is the array that 
    ###     is used for validation.
    aceOverlapWithGEFS = np.zeros((dateValsGEFS.shape[0]))
    for day in range(0,dateValsGEFS.shape[0]):
        curryr = dateValsGEFS[day,0]
        currmon = dateValsGEFS[day,1]
        currday = dateValsGEFS[day,2]
        fnddt1 = np.nonzero(dtValsBiweekly[:,0]==curryr)[0]
        fnddt2 = np.nonzero(dtValsBiweekly[fnddt1,1]==currmon)[0]
        fnddt3 = np.nonzero(dtValsBiweekly[fnddt1[fnddt2],2]==currday)[0]    
        aceOverlapWithGEFS[day] = aceValsBiweekly[fnddt1[fnddt2[fnddt3]]+dayOffsets[0]]

    ###############################################################################
    ###############################################################################
    ##### Use the OISST data to obtain aggregated SST anomalies.
    ###############################################################################
    ###############################################################################

    ### Obtain a subset of the biweekly ACE data between the years 1982-2020. 
    ###     This will be used to develop and cross-validate the SST-derived 
    ###     forecasts of ACE.
    aceSub = np.array(aceValsBiweekly[29585:43830])
    aceDatesSub = np.array(aceDateValsDailyEA[29585:43830,:])
    fnd1 = np.nonzero(aceDatesSub[:,1] == 2)[0]
    fnd2 = np.nonzero(aceDatesSub[fnd1,2] == 29)[0]
    aceSub = np.delete(aceSub,fnd1[fnd2],axis=0)
    ### February is not of interest in this study, so remove Feb 29, and 
    ###     reshape our biweekly ACE to have the shape (years,days)
    aceSubR = np.reshape(aceSub,(39,365))
    aceDatesSub = np.delete(aceDatesSub,fnd1[fnd2],axis=0)
    aceDatesSubR = np.reshape(aceDatesSub,(39,365,3))

    ### Below we apply the following offsets in days. 
    ### Our validation dates are the centroids or day 8 of the biweekly forecast periods.
    ### For example, the validation date, August 25, then corresponds to observed 
    ###     ACE values accumulated over the period August 18 - August 31. The GEFSv12 
    ###     reforecasts would have been on August 3. And the OISST reforecasts 
    ###     would have been made on August 1. 
    ### Therefore, we need to offset the biweekly ACE amounts by minus 7 days, so 
    ###     that SST dates that we have actually correspond to the 8th day of 
    ###     the two week period.
    ### Similarly, our SST forecast are made 17 days prior to the beginning 
    ###     of the biweekly period of observed ACE (which is minus 24 days).
    aceOffset = -7
    sstOffset = -24

    ### Next, we build the aggregated standardized SST anomalies
    ### These are dummy arrays to be filled. Their shape 
    ###     is (year,day over period July 1 - November 30,31-day window)
    aggregatedSST_anomalies = np.zeros((39,153,31))
    observedACE = np.zeros((39,153,31))

    ### The correlation coefficient values about 31-day windows over the 
    ###     period July 1 - November 30. These are the correlation coefficients 
    ###     between the aggregated standardized SST anomalies and ACE 17-30 days later.
    rvals = np.zeros((153))

    ### Compute how many grid cells comprise 10% of the total number of SST 
    ###     grid cells that contain data.
    gridCells2use = int(np.round(sstR.shape[2]/10))

    ### Loop over the period July 1 - November 30 to find the 
    ###     aggregated standardized SST anomalies using leave-one-year-out 
    ###     cross validation.
    for day in range(181,334):      
        ### Get the SSTs and ACE values at a fixed forecast time step, but over 
        ###     a 31-day window.
        sstRsub = np.array(sstR[:,day-15+sstOffset:day+16+sstOffset,:])
        aceRsub = np.array(aceSubR[:,day-15+aceOffset:day+16+aceOffset])
        observedACE[:,day-181,:] = np.array(aceRsub)
        for yr in range(0,39):
            valid = yr 
            calib = np.delete(np.arange(0,39),yr)
            ### Obtain the SSTs and ACE values in the calibration period and 
            ###     reshape these arrays to have the shape (time,space).
            sstSubCalib = np.reshape(sstRsub[calib,:,:],(-1,sstRsub.shape[2]))
            aceSubCalib = np.reshape(observedACE[calib,day-181,:],(-1,1))
            ### Standardize the SSTs as a function of the calibration period.
            sstRsubZ = (sstRsub-np.mean(sstSubCalib,axis=0))/np.std(sstSubCalib,axis=0)
            concatenatedMatrix = np.append(aceSubCalib,sstSubCalib,axis=1)
            ### Compute the correlation coefficients as a function of the calibration period.
            rr = np.corrcoef(concatenatedMatrix.T)[0,1:]
            ### Sort the correlation coefficients from lowest to highest.
            rsrt = np.argsort(rr)
            ### Compute the aggregated standardized SST anomalies for this day and year.
            aggregatedSST_anomalies[valid,day-181,:] = np.mean(sstRsubZ[valid,:,:][:,rsrt[-gridCells2use:]],axis=(1))

        ### Compute the correlation coefficients between the aggregated 
        ###     standardized SST anomalies and ACE 17-30 days later.
        rvals[day-181] = np.corrcoef(np.ravel(aggregatedSST_anomalies[:,day-181,:]),np.ravel(observedACE[:,day-181,:]))[0,1]

    ### Compute a weighted average anomaly correlation over the season 
    ###     between the statistical model forecasts and observed ACE.
    print("Average July-November Anomaly Correlation (AggregatedSST):"\
        ,"%.3f"%(np.sum(rvals*np.mean(observedACE,axis=(0,2)))/np.sum(np.mean(observedACE,axis=(0,2)))))

    ###############################################################################
    ###############################################################################
    ##### Reshape GEFSv12 and ECMWF reforecasts to be on the same shape as the SST 
    #####        reforecasts and observed ACE values. This is (year,day,ensemble).
    ###############################################################################
    ###############################################################################
    
    ### Build arrays of GEFSv12 and ECMWF forecasts with the same moving-window structure 
    ###     as the SST data above.
    gefsDatesRange = np.array(aceDatesSubR[18:38,:,:])
    gefsDatesRangeR = np.reshape(gefsDatesRange,(-1,3))
    predicted_ACE_GEFS_Range = np.zeros((gefsDatesRangeR.shape[0],11),dtype=np.float32)*np.nan
    predicted_ACE_ECMWF_Range = np.zeros((gefsDatesRangeR.shape[0],11),dtype=np.float32)*np.nan
    aceAtGEFStimes = np.zeros((gefsDatesRangeR.shape[0]),dtype=np.float32)*np.nan
    for itval1 in range(0,len(predicted_ACE_Atlantic_GEFS)):
        if dateValsGEFSValidationCentroid[itval1,1] >= 6 and dateValsGEFSValidationCentroid[itval1,1] <= 12:
            fnd1 = np.nonzero(dateValsGEFSValidationCentroid[itval1,0] == gefsDatesRangeR[:,0])[0]
            fnd2 = np.nonzero(dateValsGEFSValidationCentroid[itval1,1] == gefsDatesRangeR[fnd1,1])[0]
            fnd3 = np.nonzero(dateValsGEFSValidationCentroid[itval1,2] == gefsDatesRangeR[fnd1[fnd2],2])[0][0]
            if domain == "EntireAtlantic":
                predicted_ACE_GEFS_Range[fnd1[fnd2[fnd3]],:] = predicted_ACE_Atlantic_GEFS[itval1,:]
                predicted_ACE_ECMWF_Range[fnd1[fnd2[fnd3]],:] = predicted_ACE_Atlantic_ECMWF[nearestIndices[itval1],:]
            elif domain == "WestAtlantic":
                predicted_ACE_GEFS_Range[fnd1[fnd2[fnd3]],:] = predicted_ACE_WestAtlantic_GEFS[itval1,:]
                predicted_ACE_ECMWF_Range[fnd1[fnd2[fnd3]],:] = predicted_ACE_WestAtlantic_ECMWF[nearestIndices[itval1],:]
            aceAtGEFStimes[fnd1[fnd2[fnd3]]] = float(aceOverlapWithGEFS[itval1])            
    predicted_ACE_GEFS_RangeR = np.reshape(predicted_ACE_GEFS_Range,(20,365,11))
    predicted_ACE_ECMWF_RangeR = np.reshape(predicted_ACE_ECMWF_Range,(20,365,11))
    aceAtGEFStimesR = np.reshape(aceAtGEFStimes,(20,365))


    ###############################################################################
    ###############################################################################
    ##### Compute the skill scores of the different models.
    ###############################################################################
    ###############################################################################

    ### Values along the ACE axis to compute CRPSS.
    xx = np.arange(-.1,200,.1)
    ### Create an array to fill climatological CRPSS for each time window over the period July-November.
    crpssClimVals = np.zeros((153))
    ### Create an array to fill different modeled CRPSS values for each time window over the period July-November.
    crpssModVals = np.zeros((153,5))

    ### Iterate over the days in July-November.
    for day in range(0,153):
        
        ### Current observed ACE for this time window. This is used to construct 
        ###     a more robust empirical CDF than the 20 year reforecast period.
        currObsACE1 = np.array(observedACE[:,day,:])
        ### Current observed ACE for this time window in the GEFSv12 reforecast 
        ###     period of record.
        currObsACE2 = np.array(observedACE[18:38,day,:])

        ### Current aggregated SST anomalies for this time window. 
        currPred1 = np.array(aggregatedSST_anomalies[:,day,:])
        ### Current aggregated SST anomalies for this time window in the reforecast period.
        currPred2 = np.array(aggregatedSST_anomalies[18:38,day,:])

        ### Current GEFSv12 and ECMWF reforecasts for this time window. 
        currGefs = np.array(predicted_ACE_GEFS_RangeR[:,181+day-15:181+day+16,:])
        currEcmwf = np.array(predicted_ACE_ECMWF_RangeR[:,181+day-15:181+day+16,:])
        ### Current observed ACE at the reforecast time steps for this time window. 
        currACE = np.array(aceAtGEFStimesR[:,181+day-15:181+day+16])
        
        ### Create a list to fill with climatological and modeled CRPS values. 
        crpsClimNow = []
        crpsModNow1 = []
        crpsModNow2 = []
        crpsModNow3 = []
        crpsModEnsNow12 = []
        crpsModEnsNow123 = []
        
        ### These are the time steps within the time window that have reforecast data. 
        ###     These are the times when the GEFSv12 made reforecasts. 
        ###     This is used to validate the model performance for this time window.
        fndValsAll = np.nonzero(currACE >= 0)

        ### Iterate over the 20 reforecast years.
        for yr in range(0,20):

            ### Remove the current year's observed ACE values to construct a 
            ###     climatological distribution in a cross-validated manner.
            currObsCalib1 = np.ravel(np.delete(currObsACE1,yr+18,axis=0))
            ### Remove the current year's ACE values to construct a 
            ###     climatological distribution in a cross-validated manner.
            ###     This is done using only the reforecast times of the GEFSv12.
            fndVals = np.nonzero(np.delete(currACE,yr,axis=0) >= 0)
            currObsCalib2 = np.delete(currObsACE2,yr,axis=0)[fndVals]
            
            ### This is the constructed climatological distribution.
            ecdfClim1 = np.arange(0,currObsCalib2.shape[0])/(currObsCalib2.shape[0]-1)
            cdfClim1 = np.interp(xx,np.sort(currObsCalib2),ecdfClim1)            
            
            ### These are the constructed reforecast distributions for the calibraiton years.
            currPredCalib = np.ravel(np.delete(currPred1,yr+18,axis=0))
            currGefsCalib = np.ravel(np.delete(currGefs,yr,axis=0)[fndVals[0],fndVals[1],:])
            currEcmwfCalib = np.ravel(np.delete(currEcmwf,yr,axis=0)[fndVals[0],fndVals[1],:])

            ### Obtain moment statistics of the transformed forecasts and observations.
            obs_mean = np.mean(currObsCalib2**.5)
            obs_std = np.std(currObsCalib2**.5)
            gefs_mean = np.mean(currGefsCalib**.5)
            gefs_std = np.std(currGefsCalib**.5)
            ecmwf_mean = np.mean(currEcmwfCalib**.5)
            ecmwf_std = np.std(currEcmwfCalib**.5)

            ### Find the days in the window in the current year in which 
            ###     reforecasts were made. 
            fndValsInYear = np.nonzero(fndValsAll[0]==yr)[0]

            ### Iterate over the days when reforecasts were made.
            for itvalDay in np.arange(0,fndValsInYear.shape[0]):
                
                ### Find the index location of the current reforecast date.
                d2 = fndValsAll[1][fndValsInYear][itvalDay]
                ### The current observed ACE for this reforecast date.
                currObsVal = currObsACE2[yr,d2]
                ### The current aggregated SST anomaly for this reforecast date.
                currPredVal = currPred1[yr+18,d2]

                ### Bias correct the current set of model ensembles.
                currGefsVals = currGefs[fndValsAll[0][fndValsInYear],fndValsAll[1][fndValsInYear],:][itvalDay,:]
                currEcmwfVals = currEcmwf[fndValsAll[0][fndValsInYear],fndValsAll[1][fndValsInYear],:][itvalDay,:]
                currGefsValsBC = (((currGefsVals**.5)-gefs_mean)*(obs_std/gefs_std)+obs_mean)**2
                currGefsValsBC[currGefsValsBC<0] = 0.
                currEcmwfValsBC = (((currEcmwfVals**.5)-ecmwf_mean)*(obs_std/ecmwf_std)+obs_mean)**2
                currEcmwfValsBC[currEcmwfValsBC<0] = 0.

                ### Find where along the x-axis (ACE), the current observed value is 
                ###     located. This is the observed ECDF.
                cdfObs = np.zeros(xx.shape)
                fndNearest = np.nonzero((xx-currObsVal)>=0)[0][0]
                cdfObs[:fndNearest] = 0
                cdfObs[fndNearest:] = 1

                ### The bias-corrected model reforecasted ensemble distribution for GEFSv12.
                ecdfMod1 = np.arange(0,int(currGefsValsBC.shape[0]))/(int(currGefsValsBC.shape[0])-1)
                cdfMod1 = np.interp(xx,np.sort(currGefsValsBC),ecdfMod1)          
                cdfMod1[:1] = 0

                ### The bias-corrected model reforecasted ensemble distribution for ECMWF.
                ecdfMod2 = np.arange(0,int(currEcmwfValsBC.shape[0]))/(int(currEcmwfValsBC.shape[0])-1)
                cdfMod2 = np.interp(xx,np.sort(currEcmwfValsBC),ecdfMod2)          
                cdfMod2[:1] = 0

                ### The model reforecasted distribution for the SST model.
                fndN1 = np.argsort(np.abs(currPredVal-currPredCalib))
                ecdfMod3 = np.arange(0,int(currPredCalib.shape[0]/3))/(int(currPredCalib.shape[0]/3)-1)
                cdfMod3 = np.interp(xx,np.sort(currObsCalib1[fndN1[:int(currObsCalib1.shape[0]/3)]]),ecdfMod3)
                cdfMod3[:1] = 0
                
                ### Average of the GEFSv12 and ECMWF CDFs.
                cdfModEns12 = (cdfMod1+cdfMod2)/2
                ### Average of the GEFSv12, ECMWF, and SST CDFs.
                cdfModEns123 = (cdfMod1+cdfMod2+cdfMod3)/3

                ### Compute the CRPS vlaues.
                crpsClimNow.append(int(np.trapz((cdfClim1-cdfObs)**2,xx)*100))
                crpsModNow1.append(int(np.trapz((cdfMod1-cdfObs)**2,xx)*100))
                crpsModNow2.append(int(np.trapz((cdfMod2-cdfObs)**2,xx)*100))
                crpsModNow3.append(int(np.trapz((cdfMod3-cdfObs)**2,xx)*100))
                crpsModEnsNow12.append(int(np.trapz((cdfModEns12-cdfObs)**2,xx)*100))
                crpsModEnsNow123.append(int(np.trapz((cdfModEns123-cdfObs)**2,xx)*100))
                
        ### Compute the CRPSS vlaues for the current time window.
        crpssClimVals[day] = np.sum(np.array(crpsClimNow))
        crpssModVals[day,0] = np.sum(np.array(crpsModNow1))
        crpssModVals[day,1] = np.sum(np.array(crpsModNow2))
        crpssModVals[day,2] = np.sum(np.array(crpsModNow3))
        crpssModVals[day,3] = np.sum(np.array(crpsModEnsNow12))
        crpssModVals[day,4] = np.sum(np.array(crpsModEnsNow123))

    ### Print the seasonal CRPSS values for the different individual models and model combinations.
    print("GEFSv12 CRPSS:","%.3f"%(1-np.sum(crpssModVals[:,0])/np.sum(crpssClimVals)))
    print("ECMWF CRPSS:","%.3f"%(1-np.sum(crpssModVals[:,1])/np.sum(crpssClimVals)))
    print("OISST CRPSS:","%.3f"%(1-np.sum(crpssModVals[:,2])/np.sum(crpssClimVals)))
    print("GEFSv12+ECMWF CRPSS:","%.3f"%(1-np.sum(crpssModVals[:,3])/np.sum(crpssClimVals)))
    print("GEFSv12+ECMWF+OISST CRPSS:","%.3f"%(1-np.sum(crpssModVals[:,4])/np.sum(crpssClimVals)))
    print('')

##### Uncomment if you want to see how long it takes to run the script.
###toc = time.time()
###print(toc-tic)

