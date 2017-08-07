#These haven't been cleaned up yet but the functions work with the notebooks
#import numpy as np
#maybe change all of these to n or np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as py
import cluster_tools.tools as tools
#import matplotlib.pylab as py
#import mask as ma
import pandas as pd
import sys
#from matplotlib.nxutils import points_inside_poly #see edit in polyfit.py with path
#from cluster_tools.CAT import dstest

#pandas analysis and plotting functions


def strcid(cid):
    #get string name for int cid
    #reg=strcid( params[params['cid']==0] )
    #reg = strcid( params.cid ) for already in func_reg()
    if cid==0:
        reg='North'
    if cid==1:
        reg='South'
    if cid==2:
        reg='NorthEast'
    if cid==3:
        reg='Remainder'
    if cid==4:
        reg='All'

    #print 'region: ', reg
    return reg

def plotfit(xlims, params, color='m'):
    #NOTE CHANGED CONVENTION IN FILES SO SLOPE IS LITERAL, + or - isn RedSeq catalog
    #if color==None: color='m'
    #else: color=color
    #t.plotfit(py.xlim, params[params['cid']==x])
    #plotfit(py.xlim, params[params['cid']==0],color='k')
    #currently does for single region, must input cid
    #correction is in old version
    xmin, xmax = xlims
    #    print 'xlims',xmin,xmax
    #NOTE CHANGED TO VALUES HERE SO INDEX ISN'T PRINTED
    y0,m,sig = params['y0'].values, params['m'].values, params['sigma'].values #could change so arrays

    y0,m,sig = params['y0'].values, params['m'].values, params['sigma'].values #could change so arrays
   
    #if (slopemap!=False):
    #    print 'plotfit mapping CSM<->CMD so m=-m: m=',m
    #    m=-m
    #    print 'now m=',m
    #else: color=color

    #delV=V-Vdered, delR=R-Rdered
    #take max value for ( delV - delR )
    #delR, delV = 0.062, 0.077
    #note there is an actual, propagated V-R error value .015,...
    #wait shouldnt dust correction make brighter? like reduce mag?
    #alternative from IDL program, create int array within range (or not)
    #this may extend the plot range but maybe not if lims already set
    #map all these points rather than just endpts
#	fakezband = indgen(30)
#	fakecolor = result[0] + result[1]*fakezband
#	soplot, fakezband, fakecolor, thick=2, color=4
    #y value predicted at edge of plot
    #adds up to (y - m * x + 3*sig ) but for single pt
#    py.plot(xlims(), map(lambda y: ( y0 + m * (xmin) ) + 3*sig, y_cent) , 'm--')
    ymin= y0 + m * (xmin)
    ymax= y0 + m * (xmax)
    y_cent= ymin, ymax #tuple
    #need to use mapping for this plot since just two pts
    #print params, 'x,x, y,y:',xmin,xmax, ymin,ymax
    #print 'mapped +3sig:',map(lambda y: y + 3*sig, y_cent),
    #print 'mapped -3sig:',map(lambda y: y - 3*sig, y_cent)
    #print type(xlims), type(y_cent)
    py.plot(xlims, map(lambda y: y + 3*sig, y_cent) , color=color, linestyle='dashed')
    py.plot(xlims, map(lambda y: y - 3*sig, y_cent) ,  color=color, linestyle='dashed')
    #py.plot(np.asarray(xlims), np.asarray( map(lambda y: y + 3*sig, y_cent) ), color=color, linestyle='dashed')
    #print [map(lambda y: y - 3*sig, y_cent) for a in y_cent][0]
    #print [xmin,xmax]
    #py.plot([xmin,xmax],[map(lambda y: y - 3*sig, y_cent) for a in y_cent][0],  color=color, linestyle='dashed')
    #py.plot(xlims, map(lambda y: y, y_cent) ,  color=color, linestyle='solid')

def rscut(params, data_rs,x='R'):
    print 'running rscut() to calculate bounds using: ',x,' for ',params['cid']
    #This calculates theoretical bounds for each values of R (or SM_K) or x
    #returns dataframe with upper lower bounds for each point to then be sorted
    #default is R, but can change to any other column, so call with rscut(params, data_rs, 'logsmk')
    #should probably test this out without the shift
    #returns dataframe with upper and lower boundaries for each object to sort/map
    #call sep for each region bounds0=rscut(params[params['cid']==0], cat['R'])
    #keep as individual for now so can plot whichever, can put in frame outside of funct
    #could aso do this by applying elementwise to dataframe applymap
    y0, m, sig = params['y0'], params['m'], params['sigma']  #these are cid=_ element from columns of 4
    #print 'cid', params['cid']
    #print  'params', y0, m, sig
    #print 'IN RSCUT',data_rs.info()
    #print len(data_rs) , 'preerror',pd.Series(np.repeat(np.array([y0]),len(data_rs)), index=data_rs.index)
    #pd.Series(np.repeat(np.array([y0]),len(data_rs)), index=data_rs.index)
    #replacing already made y0
    #NEW PANDAS REQUIRED REMOVAL OF [] from y0 numpy array!!!! AHHHH!
    data_rs['y0']=pd.Series(np.repeat(np.array(y0),len(data_rs)), index=data_rs.index)

#        data_rs['y0']=pd.Series(np.repeat(np.array([y0]),len(data_rs)), index=data_rs.index)
    #print pd.Series(np.repeat(np.array(y0),len(data_rs)), index=data_rs.index)[:5]
    #print data_rs.index.names,len(data_rs)
    #data_rs['y0']=y0
    #data_rs['y0']=pd.Series(np.repeat(np.array([y0]),len(data_rs)) )
    #data_rs['y0']=y0
    #print 'Sample data_rs.y0\n',data_rs['y0'][:5]
    #delV, delR = data_rs.V-data_rs.Vdered, data_rs.R-data_rs.Rdered
    #these are Series or dataframe objects, doesn't create field
    #data_rs['delV'], data_rs['delR'] = data_rs['Vnd']- data_rs['V'], data_rs['Rnd'] - data_rs['R']
    #y0 = y0 + ( delV - delR ) #correction shifts y-intercept (up bc y0>0)
    #data_rs['y0d'] = data_rs['y0'] - ( data_rs['delV'] - data_rs['delR'] ) #correction shifts y-intercept (up bc y0>0)
    #add dered corrections for V and R 0.062 for R, 0.077 for V, 0.102 for B
    fyh = lambda x,y: float(y + m * x + 3*sig ) #theory V-R
    fyl = lambda x,y: float(y + m * x - 3*sig ) #theory V-R
    fyc = lambda x,y: float(y + m * x ) #central line, don't need this for sort
    #I could probably skip this step if don't need initialize before mapping
#    if (x=None):
#        data_rs['VRh'], data_rs['VRl'], data_rs['VRc'] = data_rs['R'], data_rs['R'], data_rs['R']
#    if (x!=None):
#        data_rs['VRh'], data_rs['VRl'], data_rs['VRc'] = data_rs[x], data_rs[x], data_rs[x]
    #data_rs['color'].loc['North']= map(func_rs,  data_rs['VR'].loc['North'], bounds0['Rl'], bounds0['Rh'])
    #print "data_rs['VRh']:", data_rs['VRh']
    #high low limits for each pt
    #note the input value of VRh isn't VR, it's R or x, y0 is uncorrected
#    data_rs['VRh']=map(fyh, data_rs['VRh'], data_rs['y0'])
#    data_rs['VRl']=map(fyl, data_rs['VRl'], data_rs['y0'])
#    data_rs['VRc']=map(fyc, data_rs['VRc'], data_rs['y0'])
    #print x,data_rs[x][:5]
    data_rs['VRh']=map(fyh, data_rs[x], data_rs['y0'])
    data_rs['VRl']=map(fyl, data_rs[x], data_rs['y0'])
    data_rs['VRc']=map(fyc, data_rs[x], data_rs['y0'])

        #print "data_rs['VRh']:", data_rs['VRh']
    #returns array with TRUE where null, returns True if any are True (if this returns True then there is a null)
    #or check for nulls in each column (df>0).any(), (df_params==np.nan).any() for all rows
    #print 'if this is True then there is a np.nan : ', df_params['Rh'].isnull().any()
    #print df_params.values[0]
    return data_rs

def func_rs(VR,  y_temp_l, y_temp_h):
    #print 'sorting with func_rs'
    #func_rs(cat['V-R'], bound0['Rl'], bound0['Rh'])
    #returns array of strings (or mask) corresponding to each object mapped
    #boarder points assigned to blue
    #if (VR==y_temp_l or VR==y_temp_h): print 'border point at VR=', VR
    if (VR > y_temp_l and VR <= y_temp_h):
        return 'Red'
    #return True
    elif (VR <= y_temp_l):
        return 'Blue'
    #return False
    elif (VR >= y_temp_h):
        return 'Above' #above line >=yt_h
    else:
        return np.nan
                
def rscut_mask(data_rs, filename, x='R'):
    #If I did this by removing the slope from the 3sig bounds, then sort by constants VRmin<<VRmax
    #or wait then I would need to remove slope from points TOO to rotate them into the frame of the line
    #re-visit RSfit program, does Gaussian cut by removing slope from pts then using y0 from line fit
    #I would cut out the need to calculate individual bounds for each line
    
    #NEED TO DO THIS AFTER REGION ASSIGNED AND OUTER INDEX
    color='color_'+x
    #print 'type',type(color)
    print 'running rscut using: ',x, 'new column: ',color,'\nread in params from: ',filename
    #sorts all red sequence and blue cloud iterates over all regions, outputs array or masks (if spec in func)
#    file='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/cats/RedSeqFits.tab'
    params=pd.read_table(filename, skiprows=1)
    #later, instead of loop, make 3D arrays
    #each bounds is a dataframe with Rh, Rl, Rc only for each obj in order (I think if no null)
    data_rs[color]=pd.Series(np.repeat(np.array([np.nan]),len(data_rs)), index=data_rs.index)
    
    #mapping an entire column also works: df1['e'] = df1['a'].map(lambda x: np.random.random())
    #with mapping don't need to pass index
    #data_rs=data_rs.set_index(['region'], drop=False)
    #already multiindexed in code main body
    #    data_rs.loc['North']
    #data_rs.loc['South']
    #data_rs.loc['Remainder']
    #grouped=data_rs.groupby(level='region')
    #bounds0=rscut(params[params['cid']==0], data_rs['R']) 
    #bounds1=rscut(params[params['cid']==1], data_rs['R'])
    #bounds2=rscut(params[params['cid']==2], data_rs['R'])
    #bounds3=rscut(params[params['cid']==3], data_rs['R'])
    #I didn't drop any values so don't need to reindex
    #two dataframes have the same indices so should match still
    #creates entire dataframe with params for each region
    #boundaries are then evaluated for objects selected by region in data_rs
    #returns full data_rs df, but into bounds, doesn't permanently alter
    #I could probably shorted this to speed it up by only outputting df of bounds
    #and only inputting 
    #bounds0=rscut(params[params['cid']==0], data_rs) 
#    data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds['VRl'].loc[reg], bounds['VRh'].loc[reg])
    #bounds1=rscut(params[params['cid']==1], data_rs)
    #bounds2=rscut(params[params['cid']==2], data_rs,x='R')
    #bounds3=rscut(params[params['cid']==3], data_rs,x='R')
    #reg=strcid(0)
    #print reg
    #data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds0['VRl'].loc[reg], bounds0['VRh'].loc[reg])
    #reg=strcid(1)
    #print reg
    #data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds1['VRl'].loc[reg], bounds1['VRh'].loc[reg])
    #reg=strcid(2)
    #data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds2['VRl'].loc[reg], bounds2['VRh'].loc[reg])
    #reg=strcid(3)
    #data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds3['VRl'].loc[reg], bounds3['VRh'].loc[reg])
    
#bounds=[]
    for num in [a for a in params['cid']]:
        #temporarily cut out 'All'
        #        if num!=4:
        for num in [a for a in [0,1,2,3]]:
            #bounds=rscut(params[params['cid']==num], data_rs,x='R')
            print 'reg in rsmask', num,strcid(num)
            reg=strcid(num)
            #puts in entire data_rs df to calculate bounds
            #does this for entire df, creating new color_R every time
            #then only reassigns bounds with mapping
            bounds=rscut(params[params['cid']==num], data_rs,x=x)
            #OK maybe should do all of this with data_rs to match indices
#           data_rs['color'].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds['VRl'].loc[reg], bounds['VRh'].loc[reg])
            #print data_rs[color].loc[reg]
            #puts in entire df
            data_rs[color].loc[reg]= map(func_rs,  data_rs['VR'].loc[reg], bounds['VRl'].loc[reg], bounds['VRh'].loc[reg])
            #print 'new',color,reg,len(data_rs[color].loc[reg]),data_rs[color].loc[reg]
            #print 'some not mapped',data_rs[[color,'region']]
            #print 'Done',reg, len(data_rs), 'note data_rs is not replaced bf moving to next reg'
            #print data_rs.info()
            #but only using data_rs to calculate new region, now has partially mapped
     
    #ok maybe I have to pass all the bounds in data_rs so it maps the right values to indices
    #NOTE this only works for NSNER, if do NEa or All will need to add...
    #data_rs['color'].loc['North']= map(func_rs,  data_rs['VR'].loc['North'], bounds0['VRl'].loc['North'], bounds0['VRh'].loc['North'])
    #data_rs['color'].loc['South']= map(func_rs,  data_rs['VR'].loc['South'], bounds1['VRl'].loc['South'], bounds1['VRh'].loc['South'])
    #keep NE because will exclude structure from remainder
    #data_rs['color'].loc['NorthEast']= map(func_rs,  data_rs['VR'].loc['NorthEast'], bounds2['VRl'].loc['NorthEast'], bounds2['VRh'].loc['NorthEast'])
    #data_rs['color'].loc['Remainder']= map(func_rs,  data_rs['VR'].loc['Remainder'], bounds3['VRl'].loc['Remainder'], bounds3['VRh'].loc['Remainder'])
    #        mask_rs =map(func_rs, cat[:, key_rs['R']], y_temp_l, y_temp_h, mask=True)
    #maybe i should return a dic instead, map tuple to dic
    #return dic_rs={'RSBC':array_rs[0], 'mask':array_rs[1]}
    #    print 'returning array_rs from rscut_mask()'
    #print 'new size',color, data_rs.groupby(color).size()
    #print data_rs.groupby(['region', 'color', 'quality']).size()
    return data_rs

def ppoly(group,X,zgroup=None):
    '''
    Takes groupby object grouped returned by make_cont()
    Makes df of of contours with their content objects and N obj
    Iso_weight_all=grouped.apply(t.ppoly,X) #ALL
    Iso_weight_HST=groupedHST.apply(t.ppoly,X)
    Iso_weight_pz=grouped.apply(t.ppoly,X)
    Iso_weight_sz=grouped.apply(t.ppoly,X)
    Iso_weight_all.to_csv('Iso_weight_all.tab',sep='\t',index=False)
    Iso_weight_HST.to_csv('Iso_weight_HST.tab',sep='\t',index=False)
    
    Iso_weight_all_rev2=groupedR2.apply(t.ppoly,X) #ALL plus new region
    Iso_weight_all_rev2.to_csv('Iso_weight_all_rev2.tab',sep='\t',index=False)
    
    NB: For memory sake I turned off "show" option in plot
    Read into region_select func
    '''
    from matplotlib.path import Path
    #py.plot(cat_all.xs(['high','cz'],level=['q_group','z_group']).alpha,cat_all.xs(['high','cz'],level=['q_group','z_group']).delta,'bo')
    #REPLACED with path after nxutls depreciated in MPL upgrade
    #P=group[['alpha','delta']].values
    #mask= points_inside_poly(X1, P)
    #make contours with polytest program, groupby (will add those in)
    #X=cat_all[['objid','alpha','delta']] #all objects not just pz
    #X=cat_all[['objid','alpha','delta']].xs(['pz'],level=['z_group_DLS']) #all objects not just pz
    #blah=grouped.apply(pt_in_poly, X, 'Remnew_' )
    #X=X.reset_index().set_index('objid',drop=False)
    path = Path(group[['alpha','delta']])
    mask = path.contains_points(X[['alpha','delta']])
    Y=X[mask]
    Y['N']=len(X[mask])
    Y['contour']=group.name
    print 'contour N=',len(group),', #',group.name,'contains:',len(X[mask]),'gals out of',len(X)
    if zgroup!=None:
        fig = py.figure()
        py.plot(X[mask].alpha,X[mask].delta,'ro',ms=2,alpha=0.3)
        py.plot(group.alpha,group.delta,'b-',label=str(group.name))
        py.gca().invert_xaxis()
        py.xlabel('RA')
        py.ylabel('DEC')
        py.tick_params(axis='both', which='major', labelsize=10)
        py.ticklabel_format(useOffset=False)
        title='Contour '+str(group.name)+' contains: '+str(len(X[mask]))+' galaxies, Photometric Objects z=z$_{cluster} \pm \sigma_{photoz}=[0.43,0.63]$'
        #        py.legend(numpoints=1,loc='lower left')
        py.title(title)
        #py.show()
        filename = 'contour_'+str(group.name)+zgroup
        py.savefig(filename)
    return Y

#def make_cont(file1, file2=None):
def make_cont():
    '''
    grouped=t.make_cont(file1,file2=file2)
    grouped=t.make_cont()
    file2 is second contour file to append onto end, regardless of file
    file1 format alpha delta, from DS9 contour file (current issue with nans) 
    file2 CSV (easy from DS9 region file, change to .con delete everything but middle comma)
    files should have NO HEADER
    If from DS9 ploygon remember to repeat first point at the bottom to close polygon
    option to pickle
    Outputs grouped contours ready for plotting or point in poly function
    '''
    import pickle
    #must be either ds9 .con format and pd format file2
    #make df and group for use in ppoly
    #for now adds Remainder file2 onto end
    #file1='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/MBszpz600_weight_norm_rev2_numberdensity.con'
    #file2='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/region_cont/contR_rev2.con'
    #file1='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/region_cont/Apj_regions2.con'
        
    #file1='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/region_cont/HSTregions_rev2_wcs_calc.con'
    #file2='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/region_cont/contRforstats_rev3_polyunrotated.con'
    #file2=None
    #conts=pd.read_table('/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/MBszpz600_weight_norm_rev2_numberdensity.con', sep=r'\s',names=['alpha','delta'])
    #contR=pd.read_csv('/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/region_cont/contR_rev2.con',names=['alpha','delta'])
    #conts=pd.read_table(file1,names=['alpha','delta'])
    #Update: had to add skip_blank_lines bc removed feature in update
    conts=pd.read_table(file1, sep=r'\s',names=['alpha','delta'],skip_blank_lines=False) #this excludes nan rows
    #conts=pd.read_table(file1, sep=',',names=['alpha','delta'])
    cutnan=np.where(conts.alpha.isnull())[0]
    #conts.loc[np.split(conts.index, conts.alpha.loc[cutnan].index)[3] ]
    nconts=len(np.split(conts.index, conts.alpha.loc[cutnan].index))
    dfs=[conts.loc[np.split(conts.index, conts.alpha.loc[cutnan].index)[i]] for i in range(nconts)] 
    cont=[c.dropna() for c in dfs if not isinstance(c, np.ndarray)]
    cont.sort(key=lambda c: len(c))
    cont=pd.concat(cont, names=['contour'],keys=np.arange(len(cont)) )
    #n=con.index.levels[0].max()+1
    #print cont.index.max(), cont.index.max()[0]
    n=cont.index.max()[0]+1 #need this 'contour' starts at 1, then +1
    cont=cont.reset_index().drop(['level_1'],axis=1).set_index('contour',drop=False)
    #quit()
    if (file2!=None):
        #I'm not really sure yet why header has to be false
        #but otherwise sticks in alpha delta column
        #this may cut out header first row if header is false
        contR=pd.read_csv(file2, names=['alpha','delta'])       
        #        contR=pd.read_csv(file2, names=['alpha','delta'],header=False)       
        #        contR=pd.read_table(file2, names=['alpha','delta'])
        A=np.repeat(n,len(contR)) #adds nth contour as remainder, check ptinpoly works first
        contR.index=A
        contR.index.names=['contour']
        contR=contR.reset_index().set_index('contour',drop=False)
        #print contR
        cont=pd.merge(cont,contR,how='outer',on=['alpha','delta','contour'],right_index=True,left_index=True)
    #        cont=pd.concat([cont,contR])
    #print cont.groupby(level=0).size()
    grouped=cont.groupby(level=0)
    #print 'pickling off last: contHST.pickle'
    print cont.head()
    #pickle new contours
    #filename = 'contHST.pickle'
    filename = 'contApJ.pickle'
    f = open('/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename,'wb')
    pickle.dump(cont,f)
    f.close()
    return grouped
    #makes contour dfs then returns grouped for reg_contour

def cont_plot(grouped,annotate=True):
    '''
    Plots groupby objects of contours returned by make_cont
    t.cont_plot(grouped,annotate=False)
    t.cont_plot(grouped)
    annotate numbers contours for further grouping by region in t.reg_contour function
    Or to read from pickled conts
    import pickle
    filename = 'contApJ.pickle'
    f = '/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename
    cont=pd.read_pickle(f)
    grouped=cont.groupby(level=0)
    t.cont_plot(grouped,False)
    '''
    for a,b in grouped:
        py.plot(b.alpha,b.delta,'k-',label=str(a))
        if (annotate==True):
            py.annotate(str(a), xy=(np.array(b.alpha.values)[0], np.array(b.delta.values)[0]), size=20)
    py.gca().invert_xaxis()
    py.xlabel('RA',fontsize=18)
    py.ylabel('Dec',fontsize=18)
    py.tick_params(axis='both', which='major', labelsize=16)
    py.ticklabel_format(useOffset=False)
    #xmin,xmax=xlim()
    #ymin,ymax=ylim()
    
def reg_contour(data_rs,Iso_weight):
    '''
    must have pre-processed cat with (prolly all) contours running pt in poly
    I think I could use pz for data_rs too, might have to when call cat_all
    Iso_weight_all=Iso_weight_all.reset_index(['contour'],drop=False).set_index(['objid'],drop=False)
    t.reg_contour(data_rs,Iso_weight_all)
    t.reg_contour(DLS,Iso_weight_all)
    cat_all=t.reg_contour(cat_all,Iso_weight_all)
    I shifted each contour here by one after new functions skipped 0
    '''
    #    HST=[0]
    #AllNE=[8,28,36,45,3,25,35,43]
    #AllN=[7,13,21,27,31,33,37,46,48]
    #ALLS=[1,4,20,29,38,40,42,47]
    #    ALL=[8,28,36,45,3,25,35,43,7,13,21,27,31,33,37,46,48,1,4,20,29,38,40,42,47,49]
    #output to df or append and output in append mode
#    for a in ALL: py.plot(grouped.get_group(a).alpha,grouped.get_group(a).delta)
#    for a in ALL: py.plot(grouped['alpha'].get_group(a),grouped['delta'].get_group(a))
        #df.to_csv('my_csv.csv', mode='a', header=False)
        #might have to store in df temp
    '''
for a in ALL:
        blah=pd.concat([grouped.get_group(a),grouped.get_group(a).head(1),pd.DataFrame({'alpha':'','delta':'','contour':''},index=[a])])
        blah[['alpha','delta']].to_csv('Apj_regions.con', sep=' ', mode='a', header=False,index=False)
        #grouped[['alpha','delta']].get_group(a).to_csv('Apj_regions.con', sep=' ', mode='a', header=False,index=False)

        blah=pd.concat([grouped.get_group(a),grouped.get_group(a).head(1),pd.DataFrame({'alpha':'','delta':'','contour':''},index=[a])])

        df = "\n"
        df = pd.DataFrame(df.split("\n"))
    '''
    R=[49]
    N1c=[48]
    N2c=[33,46,27]
    S1c=[47]
    S2c=[42,40]
    NEa1c=[45]
    NEa2c=[36]
    NEb1c=[43]
    NEb2c=[35]
    NE=[45,43]
    
    data_rs['region_c']=np.nan
    #data_rs['region_c']='Remainder' #initialize with biggest catch-all contour
    list_all=[R]
    list_reg=['Remainder']
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_c'][data_rs.objid.isin(reg_df.objid) ]=name
    #data_rs['region_c']='Remainder'
    list_all=[N1c, S1c, NE]
    list_reg=['North', 'South', 'NorthEast']
    #list_all=[North, South, NorthEasta,NorthEastb]
    #list_reg=['North', 'South', 'NorthEasta','NorthEastb']
    #Note this changes cat_all output into ipython to overwrite NEa&b for region_c2
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_c'][data_rs.objid.isin(reg_df.objid) ]=name

    return data_rs

def reg_contour_r2(data_rs,Iso_weight):
    '''
    NB: OVERALL CLIP of all regions into statistical square of max slitmask coverage
    py.plot(blah.xs([cid],level=['region_r2']).alpha,blah.xs([cid],level=['region_r2']).delta,'yo',ms=4)
    Must have pre-processed cat with (prolly all) contours running pt in poly
    I think I could use pz for data_rs too, might have to when call cat_all
    Iso_weight_all_rev2=Iso_weight_all_rev2.reset_index(['contour'],drop=False).set_index(['objid'],drop=False)
    t.reg_contour_r2(DLS,Iso_weight_all)
    cat_all=t.reg_contour_r2(cat_all,Iso_weight_all)
    I shifted each contour here by one after new functions skipped 0
    '''
    R=[49] #from make cont this is the same assignment for reduced
    N1c=[48]
    N2c=[33,46,27]
    S1c=[47]
    S2c=[42,40]
    NEa1c=[45]
    NEa2c=[36]
    NEb1c=[43]
    NEb2c=[35]
    NE=[45,43]
    
    data_rs['region_r2']=np.nan
    #data_rs['region_r2']='Remainder' #initialize with biggest catch-all contour
    list_all=[R]
    list_reg=['Remainder']
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_r2'][data_rs.objid.isin(reg_df.objid) ]=name
    #data_rs['region_r2']='Remainder'
    list_all=[N1c, S1c, NE]
    list_reg=['North', 'South', 'NorthEast']
    #list_all=[North, South, NorthEasta,NorthEastb]
    #list_reg=['North', 'South', 'NorthEasta','NorthEastb']
    #Note this changes cat_all output into ipython to overwrite NEa&b for region_c2
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_r2'][data_rs.objid.isin(reg_df.objid) ]=name
    #For stats purposes repeat to set everything outside R to nan (some NE+N)
    #assumes ALL objects inside R have objids in contour group ()N=49 has 1211 gals
    list_all=[R]
    list_reg=['Remainder']
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_r2'][~data_rs.objid.isin(reg_df.objid) ]=np.nan

    return data_rs
#Do this outside
#When done replace for good with new assignments for plotting...
#data_rs=data_rs.reset_index(['region'],drop=True)
#data_rs['region_r']=data_rs['region']
#data_rs['region']=data_rs['region_c']
#reset region index for plotting, now with new assignment
#don't need to do this for cat_all bc not plotting
#data_rs=data_rs.set_index(['region'],drop=False, append=True)

def reg_contour_HST(data_rs,Iso_weight):
    '''
    must have pre-processed cat with (prolly all) contours running pt in poly
    I think I could use pz for data_rs too, might have to when call cat_all
    Iso_weight_all=Iso_weight_all.reset_index(['contour'],drop=False).set_index(['objid'],drop=False)
    t.reg_contour(data_rs,Iso_weight_all)
    t.reg_contour(DLS,Iso_weight_all)
    cat_all=t.reg_contour(cat_all,Iso_weight_all)
    I shifted each contour here by one after new functions skipped 0
    '''
    HST=[0]
    data_rs['region_HST']=np.nan #initialize with biggest catch-all contour
    list_all=[HST]
    list_reg=['HST']
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_HST'][data_rs.objid.isin(reg_df.objid) ]=name
    return data_rs

def reg_contour_PG(data_rs,Iso_weight):
    '''
    cat_all=t.reg_contour_PG(cat_all,PG_slitmasks) see pynbk for details
    must have pre-processed cat with (prolly all) contours running pt in poly
    I think I could use pz for data_rs too, might have to when call cat_all
    Iso_weight_all=Iso_weight_all.reset_index(['contour'],drop=False).set_index(['objid'],drop=False)
    t.reg_contour(data_rs,Iso_weight_all)
    t.reg_contour(DLS,Iso_weight_all)
    cat_all=t.reg_contour(cat_all,Iso_weight_all)
    I shifted each contour here by one after new functions skipped 0
    '''
    PG=[0]
    data_rs['region_PG']=np.nan #initialize with biggest catch-all contour
    list_all=[PG]
    list_reg=['PG']
    for reg,name in zip(list_all,list_reg):
        list_dfs=[Iso_weight.iloc[Iso_weight.index.get_level_values('contour')==a] for a in reg]
        reg_df=pd.concat(list_dfs)
        data_rs['region_PG'][data_rs.id.isin(reg_df.id) ]=name
    return data_rs


def func_reg(data_rs, params):

        #only does box no rotation, see contour function
    #returns new DataFrame with specified region to assign to initiated remainder array
    #passed in dataframe cause need alpha and delta
    #data_rs = func_reg(data_rs, params[params['cid']==0])
    #params.cid should be single value, then multipy
    reg = strcid( params.cid )
    #print 'cutting region:', reg
    alpha_min=pd.Series(np.repeat(np.array([params.alpha_min]),len(data_rs)))
    alpha_max=pd.Series(np.repeat(np.array([params.alpha_max]),len(data_rs)))
    delta_min=pd.Series(np.repeat(np.array([params.delta_min]),len(data_rs)))
    delta_max=pd.Series(np.repeat(np.array([params.delta_max]),len(data_rs)))
    #    params = pd.DataFrame( { 'Rh': R, 'Rl': R, 'Rc': R } ) or potentially som mapping...
    #params=pd.DataFrame([alpha_min, alpha_max, delta_min, delta_max])
    condition = ( ( (data_rs.alpha >= alpha_min)&(data_rs.alpha <= alpha_max) ) &  ( (data_rs.delta >= delta_min)&(data_rs.delta <= delta_max) ) )
    #    condition = ( ( (data_rs.alpha >= params.alpha_min)&(data_rs.alpha <= params.alpha_max) ) &  ( (data_rs.delta >= params.delta_min)&(data_rs.delta <= params.delta_max) ) )
    #data_rs.loc[conditionN,'region']= 'North'
    data_rs.loc[condition,'region'] = reg
    #print 'applied condition to data_rs, returning dataframe'
    return data_rs


#def region_mask(data_rs, verbose=True):
def region_mask(data_rs):
#with pandas now do this with mapping/sort function? should prevent duplicates, or at least bin to first
#function should return by 4 groups of all objects in a range, so can check for duplicates within slice
#maybe groupby/slice to remove region index
    print 'reading in RA&DEC params'
    file='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/cats/AlphaDeltaMB.tab'
    params=pd.read_table(file, skiprows=1)
    #Initial array all remainder, so auto assigned to objects not in region
    data_rs['region']=pd.Series(np.repeat(np.array(['Remainder']),len(data_rs)))
    print 'none should be NULL:',data_rs[ pd.isnull(data_rs['region']) ].region
    #data_rs.region='Remainder'
    #replaces new data_frame every time...figure out why this is worth the inefficiency
    #enables running of one region at a time (perhaps duplicate testing?)
    #alternative is to pass only column in and out of function, then reattach it each time
    data_rs = func_reg(data_rs, params[params['cid']==0])
    data_rs = func_reg(data_rs, params[params['cid']==1])
    data_rs = func_reg(data_rs, params[params['cid']==2])
    #data_rs = func_reg(data_rs, params[params['cid']==3]) Don't need bc initialize all at Rem.
    #in this case doing duplicate by objid will only identify objects with same objid which are in two diff regions
    #could also check withing same region
    #this will not test for objects which are in the same region that
    #print DLS[ pd.isnull(DLS['region']) ].color
    print data_rs.groupby(['region']).size()
    return data_rs

def duplicate_info(data_rs):
    #stats about duplicates in catalogue
    #by region, quality, not yet by color tho...so run only after have regions
    #will be useful later to generalize so can do this before region, or after color
    #data_rs.groupby(['objid','region']).size()
    #blah=data_rs.groupby(['objid','region']).size()
    #print 'where data_rs has two:' , blah[blah==2], 'three+:' , blah[blah>2]
    condition_dup=(data_rs.duplicated(['objid'])==True)
    print 'Current duplicates in each', data_rs[condition_dup].groupby(['region']).size()
    print 'Duplicates sorted by', data_rs[data_rs.duplicated(['objid'])==True].groupby(['quality']).size()
    #data_rs[data_rs.duplicated(['objid'])==True].groupby(['objid','region'],sort=True).size()
    print 'Verbose, sorted by',data_rs[data_rs.duplicated(['objid'])==True].groupby(['region', 'quality','objid']).size()
    print data_rs[data_rs.duplicated(['objid'])==True].groupby(['region','q_group','z_group','m_group','color','objid']).size()
    print 'See duplicate function in test2_functs.py for verbose print options'
    #data_rs[data_rs.objid==223085581].sort('quality').quality
    #can sort then take last to get higher quality
    dupcat=data_rs[condition_dup]
    dupcat['int']=pd.Series( np.arange(len( dupcat['redshift'] ) ), index=dupcat.index )
    dupcat=dupcat.set_index(['int'], drop=False)
    
    return dupcat
#dupcat.xs([cid,'high','cz','in'],level=['region','q_group','z_group','m_group']).comment
#blah1=data_rs.sort('quality').drop_duplicates(cols=['objid','R']) gives 443 q4, 66 q3,41 q2
#blah=data_rs.sort('quality').drop_duplicates(cols=['objid','R'],take_last=True) gives 462 q4,62q3, 38q2
#data_rs=data_rs.sort('quality').drop_duplicates(cols=['objid'],take_last=True)

#we would expect this because by sorting and selecting last, will take higher q dup
# drop on objid R eliminates offsers/serends with same R (takes higher q), drop on objid eliminates repeat obs when drop lowq. R gets eliminated anyway cause already know all same mag ones
#for objects with both high q, same mag, zpair, doesn't matter which, unless re-extract
'''
Most dups had lowq counterparts-so either redone 
-maybe check out spectra of offsers? z-pair: Either a supser (superimposed spatially (i.e., the y-direction in z-spec)) or offser (offset from the target spatially) that is at the same redshift as the target
Brian: All z-pairs have exact same mag...is this unavoidable?
-for now just take lesser
-is this post-re-extraction? I could look at redshift, if one was redone second time (q check absolves this)
region     q_group  z_group  m_group  color  objid    
NORTH      high     cz       in       Red    223070079* (same R mag)    1 Red, Red (HeI,_H&K,_marg_Hd,_Hg/H&K,_marg_Gband,_marg_Hb). Everything is the same but RA&DEC
                                             223073817    1 Other, Red ("y=56, z=0.52690, q=4, H&K, Gband, Hb")
                                             223074349    1 Red, Red-nz,lowq
SOUTH      high     cz       in       Red    223068001    1 Red (offser), Red-nz, lowq
                                             223068103*    1 Red, Red (z-pair,_H&K,_Gbandk,_Mgb/H&K,_Gband,_Hb,_marg_Mgb)
                                             223069083*    1 Red, Red (H&K,_Hd,_Gband,_marg_Hg,_Hb,_Mgb/z-pair,_H&K,_Gband,_Hb,_marg_Mgb)
                                             223071495*    1 Red, Red (marg_H&K,_marg_Hb,_marg_Mgb/H&K,_Hd,_Gband,_marg_Hd,_Hb,_Mgb)
REMAINDER  high     cz       above    Red    223069839    1 Red-nz,above, Red, above (OIII,_offser,_y=29,_z=0.30648,_q=4)
                             in       Blue   223067448*    1 Blue, Blue (H&K,_marg_Hd,_Gband,_Hg,_marg_Hb,_Mgb,_offser)
                                             223076361    1 Blue (pz, low, fill_gap_failed), Blue ("bcol[r]!, OII, H&K, marg Hd, Hg, Hb")
                                             223082188    1 Blue (low), Blue ("bcol[b]!, OII, marg H&K, marg Hd, Hg, Hb, OII...)
                                      Red    223060251    1 Red (low), Red ("OII, H&K, Gband, Hg, Hb, offser1, y=56, z= 0....)
'''

    #print 'Indices of dups', np.where(data_rs.duplicated(['objid'])==True)
    #http://pandas.pydata.org/pandas-docs/dev/indexing.html
    #For duplicate checking, dropping first or second entry, maybe w condition?
    #a.duplicated(['objid'], __)
    #returns mask
    #a.drop_duplicates(['a','b'])
#could check for duplicates by index
#data_rs.groupby(['region']).region.index
#data_rs.groupby(['region']).objid.values
#data_rs[data_rs.duplicated(['objid'])==True].objid
#data_rs[data_rs.duplicated(['objid'])==True].groupby(['objid']).objid.size()
#data_rs[data_rs.duplicated(['objid'])==True].groupby(['objid']).objid.values
#data_rs[data_rs.duplicated(['objid'])==True].groupby(['region','objid']).size()
#data_rs[data_rs.objid==223078316].objid


def comp_bar(comp_df):
    #input dataframe with completeness and mag ranges for hist
    #comp_df=pd.DataFrame(complist, columns=['cid','magrange','completeness','span' ])
    py.figure(1)
    num=[0,1,3]
    col=['red','blue','green']
    arrays=[num,col]
    tups = list(zip(*arrays))
    for x,col in tups:
        scid=strcid(x)
        north=comp_df[comp_df.cid==scid]
        north['int']=pd.Series( np.arange(len( north ) ), index=north.index )
        north=north.set_index(['int'])
        dat=north.completeness
        LABELS=north.magrange
        place=north.index
        py.bar(place,dat,alpha=0.5,color=col,label=scid)
       
        py.xticks(place,LABELS,rotation=70)
        #will need to change this based on params of bar graph
        #        py.title('Completeness,cz+pz DLS, high+cz+star Keck')
        py.title('Completeness,all DLS, highq(+star) Keck')
        #pz DLS, all targeted as pz (don't know)
    #py.legend()
    #py.savefig('Hq_DLS')
    py.show()
    py.close()
    #py.figure(2)
    #py.plot(place,north.span)
    #py.xticks(place,LABELS,rotation=70)
    #py.show()
    
    #########Redshift sorting functions#########
def func_z(z, cz_min, cz_max, pz_min, pz_max):
    #func_rs(cat['V-R'], bound0['Rl'], bound0['Rh'])
    #returns array of strings (or mask) corresponding to each object mapped
    if (z >= cz_min and z <= cz_max):
        return 'cz'
    #return True
    #    elif ( (z > cz_min and z < cz_max) and (z >= pz_min and z <= pz_max) ):
    #elif (z >= pz_min and z <= pz_max ):
        #       print 'pz'
    #    return 'pz'
    #return False
    elif (z < cz_min or z > cz_max):
        #all 19 nan are 'nz' data_rs[isnull(data_rs.redshift)].redshift
        return 'nz' #above line >=yt_h
    else:
#        print 'returning z np.nan for ', z
        return np.nan

def func_pznz(z, cz_min, cz_max, pz_min, pz_max):
    #THIS IS FOR DLS/photoz only, just nz and pz, for photoz objects don't care about cz
    #func_rs(cat['V-R'], bound0['Rl'], bound0['Rh'])
    #returns array of strings (or mask) corresponding to each object mapped
    if (z >= pz_min and z <= pz_max ):
        #       print 'pz'
        return 'pz'
    #return False
    elif ((z <= pz_min) or (z >= pz_max )):
        return 'nz'
    else:
        return np.nan

                 
def func_azlz_rs(z, cz_min, cz_max, pz_min, pz_max):
    #func_rs(cat['V-R'], bound0['Rl'], bound0['Rh'])
    #returns array of strings (or mask) corresponding to each object mapped
    #here (to help spec w/wo error confirmation range) azlz outside cz range
    #note scatter will be objects with DLSpz in that were confirmed out (a or l)
    #scatter, objects in DLSpz (dls error does limit to cz) confirmed as pz (or pz)
    #no for nonz must add pz and az and lz
    if (z >= cz_min and z <= cz_max):
        return 'cz'
    elif ((z >= pz_min and z <= pz_max )&(z < cz_min and z > cz_max)):
        #    elif ((z >= pz_min and z <= pz_max ):
        return 'pz'
    elif (z <= cz_min):
        return 'lz'
    #    else (z >= cz_max):
    elif (z >= cz_min):
        return 'az'
    else:
        return np.nan

def func_azlz_DLS(z, cz_min, cz_max, pz_min, pz_max):
    #func_rs(cat['V-R'], bound0['Rl'], bound0['Rh'])
#    if (z >= cz_min and z <= cz_max):
#        return 'cz'
#    elif ((z >= pz_min and z <= pz_max )&(z < cz_min and z > cz_max):
    if (z >= pz_min and z <= pz_max ):
        return 'pz'
    elif (z <= pz_min):
        return 'lz'
    elif (z >= pz_max):
    #else:
        return 'az'
    else: #if nan for whatever reason (shouldn't be for DLS)
        return np.nan            

def zcut(data_rs, cz_min, cz_max, pz_min, pz_max, zcol='redshift'):
    #outputs z_group column based on zcol
    #This currently does all three cuts: specs data_rs, DLS pz nz and detailed azlz
    #print 'z_cut params',cz_min, cz_max, pz_min, pz_max
    cz_min=pd.Series(np.repeat(np.array([cz_min]),len(data_rs)))
    cz_max=pd.Series(np.repeat(np.array([cz_max]),len(data_rs)))
    pz_min=pd.Series(np.repeat(np.array([pz_min]),len(data_rs)))
    pz_max=pd.Series(np.repeat(np.array([pz_max]),len(data_rs)))
    data_rs['z_group']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_azlz_rs']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_pz']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)

    data_rs['z_group'] = map(func_z, data_rs[zcol], cz_min, cz_max, pz_min, pz_max)
    data_rs['z_group_azlz_rs'] = map(func_azlz_rs, data_rs[zcol], cz_min, cz_max, pz_min, pz_max)
    data_rs['z_group_pz'] = pd.Series( map(func_pznz, data_rs['z_peak'], cz_min, cz_max, pz_min, pz_max),index=data_rs.index)
#    data_rs['z_group_azlz'] = pd.Series( map(func_z_azlz, data_rs['redshift'], cz_min, cz_max, pz_min, pz_max),index=data_rs.index)
    print data_rs.groupby(['z_group']).size()
    print data_rs.groupby(['z_group_azlz_rs']).size()
    print data_rs.groupby(['z_group_pz']).size()

    return data_rs

def zcut_PG(data_rs, cz_min, cz_max, pz_min, pz_max,pz_min2, pz_max2, zcol='redshift'):
    #outputs z_group column based on zcol
    #DOES 2 SIGMA CUT ALSO stored as z_group_pz2, defined in rotate PG notebook
    #This currently does all three cuts: specs data_rs, DLS pz nz and detailed azlz
    print 'z_cut params',cz_min, cz_max, pz_min, pz_max,pz_min2, pz_max2
    cz_min=pd.Series(np.repeat(np.array([cz_min]),len(data_rs)))
    cz_max=pd.Series(np.repeat(np.array([cz_max]),len(data_rs)))
    pz_min=pd.Series(np.repeat(np.array([pz_min]),len(data_rs)))
    pz_max=pd.Series(np.repeat(np.array([pz_max]),len(data_rs)))
    pz_min2=pd.Series(np.repeat(np.array([pz_min2]),len(data_rs)))
    pz_max2=pd.Series(np.repeat(np.array([pz_max2]),len(data_rs)))
    data_rs['z_group']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    #data_rs['z_group_azlz_rs']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_pz']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_pz2']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)

    data_rs['z_group'] = map(func_z, data_rs[zcol], cz_min, cz_max, pz_min, pz_max)
    #data_rs['z_group_azlz_rs'] = map(func_azlz_rs, data_rs[zcol], cz_min, cz_max, pz_min, pz_max)
    data_rs['z_group_pz'] = pd.Series( map(func_pznz, data_rs['z_peak'], cz_min, cz_max, pz_min, pz_max),index=data_rs.index)
    data_rs['z_group_pz2'] = pd.Series( map(func_pznz, data_rs['z_peak'], cz_min, cz_max, pz_min2, pz_max2),index=data_rs.index)
    #    data_rs['z_group_azlz'] = pd.Series( map(func_z_azlz, data_rs['redshift'], cz_min, cz_max, pz_min, pz_max),index=data_rs.index)
    print data_rs.groupby(['z_group']).size()
    #print data_rs.groupby(['z_group_azlz_rs']).size()
    print data_rs.groupby(['z_group_pz2']).size()
    print data_rs.groupby(['z_group_pz']).size()

    return data_rs


def zcut_DLS(data_rs, cz_min, cz_max, pz_min, pz_max,zcol='redshift'):
    #outputs cols z_group_DLS, z_group_azlz
    #For DLS cuts on photoz values may be stored in variable different from redshift or redshift_rs
    #This currently does all three cuts: specs data_rs, DLS pz nz and detailed azlz
    #print 'z_cut params',cz_min, cz_max, pz_min, pz_max
    #print type(data_rs),data_rs.info(),len(data_rs)
    cz_min=pd.Series(np.repeat(np.array([cz_min]),len(data_rs)))
    cz_max=pd.Series(np.repeat(np.array([cz_max]),len(data_rs)))
    pz_min=pd.Series(np.repeat(np.array([pz_min]),len(data_rs)))
    pz_max=pd.Series(np.repeat(np.array([pz_max]),len(data_rs)))
    #I'm pretty sure I don't need to put these into a dataframe like in RS sort
    #Needed that so could apply .map function (then pass vars, but also do that here)
#    data_rs['z_group_rs'] = pd.Series( map(func_z, data_rs['redshift'], cz_min, cz_max, pz_min, pz_max),index=data_rs.index)
#    data_rs['z_group_DLS'] = pd.Series( map(func_pznz, data_rs[zcol], cz_min, cz_max, pz_min, pz_max),index=data_rs.index) #note for plotting in cat6 I tranfer this to z_group
    data_rs['z_group_DLS']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_azlz_DLS']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(data_rs)), index=data_rs.index)
    data_rs['z_group_DLS'] = map(func_pznz, data_rs[zcol], cz_min, cz_max, pz_min, pz_max) #note for plotting in cat6 I tranfer this to z_group
    data_rs['z_group_azlz_DLS'] = map(func_azlz_DLS, data_rs[zcol], cz_min, cz_max, pz_min, pz_max)
    #array_z = pd.Series( map(func_z, data_rs['redshift'], cz_min, cz_max, pz_min, pz_max) )
    print data_rs.groupby(['z_group_DLS']).size()
    print data_rs.groupby(['z_group_azlz_DLS']).size()
    return data_rs



###QUALITY?

def func_q(q):
    #func_rs(data_rs['quality'])
    #returns array of strings (or mask) corresponding to each object mapped
    #note:high when called should add stars data_rs.ix[ ['high','star'] ]
    if ( q > 2 ):
        return 'high'
    elif ( q == -1 ):
        return 'star'
    elif (q < 3 and q > -1): #0,1,2
        return 'low'
    else:
        return np.nan #above line >=yt_h
    
def qcut(data_rs):
    data_rs['q_group'] = pd.Series( map(func_q, data_rs['quality']) , index=data_rs.index )
    print data_rs.groupby(['q_group']).size()
    return data_rs

#########MAG CUT
def func_mag(R, magh, magl):
    #func_mag(data_rs['R'], magh, magl)
    #mag low is bright end, low number. mag high is faint end, high #apparent mag)
    #returns array of strings (or mask) corresponding to each object mapped
#make sure this is working correctly for integers, I think it would since it worked for z
    if (R <= magh and R >= magl):
        return 'in'
    elif (R > magh):	
        return 'faint'
    elif (R < magl):
        return 'bright' #above line >=yt_h
    else:
        return np.nan
        
def func_mag2(R, mag):
    #func_mag2(data_rs['R'], mag)
    #should I put a bright end cut here?
    #R<= low R is bright, high R faint, bimodal for use w sampling 
    #mag low is bright end, low number. mag high is faint end, high #apparent mag)
    #returns array of strings (or mask) corresponding to each object mapped
#make sure this is working correctly for integers, I think it would since it worked for z
    if (R <= mag):
        return 'bright'
    elif (R > mag):
        return 'faint' #above line >=yt_h
    else:
        return np.nan
        
def func_SM(logsmk, smcut):
    #func_mag(data_rs['R'], magh, magl)
    #mag low is bright end, low number. mag high is faint end, high #apparent mag)
    #returns array of strings (or mask) corresponding to each object mapped
#make sure this is working correctly for integers, I think it would since it worked for z
    if (logsmk >= smcut):
        return 'high'
    elif (logsmk < smcut):	
        return 'low'
    else:
        #print 'returning logsmk bc np.nan:', logsmk
        return np.nan #this should return np.nan?
        #return np.nan
        
def func_c(VR, ch, cl):
    '''
    t.func_c(data_rs, ch, cl)
    ch, cl = 1.4, 0.2
    mag low is bright end, low number. mag high is faint end, high #apparent mag)
    returns array of strings (or mask) corresponding to each object mapped
    make sure this is working correctly for integers, I think it would since it worked for z
    '''
    if (VR <= ch and VR >= cl):
        return 'in'
    elif (VR > ch):	
        return 'bluer'
    elif (VR < cl):
        return 'redder' #above line >=yt_h
    else:
        return np.nan #there should be no nan values, VR not null
    
def ccut(data_rs, ch, cl):
    #t.ccut(data_rs,ch,cl)
    #maps VR based on high and low color (VR) values, in, out higher and out lower
    #print 'color lims: [', cl, ch, '] spans',(ch-cl)
    ch=pd.Series(np.repeat(np.array([ch]),len(data_rs)))
    cl=pd.Series(np.repeat(np.array([cl]),len(data_rs)))
    data_rs['c_group'] = pd.Series( map(func_c, data_rs['VR'], ch, cl) , index=data_rs.index )
    #print data_rs.groupby(['m_group']).size()
    return data_rs

def func_smk(VR, ch, cl):
    '''
    t.func_smk(data_rs, ch, cl)
    ch, cl = 12.5, 9
    mag low is bright end, low number. mag high is faint end, high #apparent mag)
    returns array of strings (or mask) corresponding to each object mapped
    make sure this is working correctly for integers, I think it would since it worked for z
    '''
    if (VR <= ch and VR >= cl):
        return 'in'
    elif (VR > ch):	
        return 'higher'
    elif (VR < cl):
        return 'lower' #above line >=yt_h
    else:
        return np.nan #there should be no nan values, VR not null
    
def smcut2(data_rs, ch, cl):
    #t.smcut2(data_rs,smh,sml) smh,sml=12.5,9
    #maps logsmk based on high and low SM_K values, in, out higher and out lower
    #print 'color lims: [', cl, ch, '] spans',(ch-cl)
    ch=pd.Series(np.repeat(np.array([ch]),len(data_rs)))
    cl=pd.Series(np.repeat(np.array([cl]),len(data_rs)))
    data_rs['sm_group2'] = pd.Series( map(func_smk, data_rs['logsmk'], ch, cl) , index=data_rs.index )
    #print data_rs.groupby(['m_group']).size()
    return data_rs


def mcut(data_rs, magh, magl):
    #maps R based on high and low mag values, in, out higher and out lower
    #print 'mag lims: [', magl, magh, '] spans',(magh-magl)
    print 'Changed to Z band cut for PG mag lims: [', magl, magh, '] spans',(magh-magl)

    magh=pd.Series(np.repeat(np.array([magh]),len(data_rs)))
    magl=pd.Series(np.repeat(np.array([magl]),len(data_rs)))
#    data_rs['m_group'] = pd.Series( map(func_mag, data_rs['R'], magh, magl) , index=data_rs.index )
    data_rs['m_group'] = pd.Series( map(func_mag, data_rs['Z'], magh, magl) , index=data_rs.index )
    #print data_rs.groupby(['m_group']).size()
    return data_rs

def mcut2(data_rs, mag):
#bimodal mapping R based on bright or faint cut (Brian 21.5) used for sampling analysis
#print 'mag lims: [', magl, magh, '] spans',(magh-magl)
    maghl=pd.Series(np.repeat(np.array([mag]),len(data_rs)))
    data_rs['m_group2'] = pd.Series( map(func_mag2, data_rs['R'], maghl) , index=data_rs.index )
    print data_rs.groupby(['m_group2']).size()
    return data_rs
 
def smcut(data_rs, smcut):
#maps R based on high and low mag values, in, out higher and out lower
#print 'mag lims: [', magl, magh, '] spans',(magh-magl)
    smcut=pd.Series(np.repeat(np.array([smcut]),len(data_rs)))
    #magl=pd.Series(np.repeat(np.array([magl]),len(data_rs)))
    data_rs['sm_group'] = pd.Series( map(func_SM, data_rs['logsmk'], smcut) , index=data_rs.index )
    #print data_rs.groupby(['m_group']).size()
    return data_rs

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mpl.colors.LinearSegmentedColormap('CustomMap', cdict)

def dstest(ra,dec,z,N_neighbors='SqrtTotal',prefix=None):
    '''
    Temp adde contour plot inside this function
    grouped=t.make_cont()
    import pickle
    filename = 'contApJ.pickle'
    f = '/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename
    cont=pd.read_pickle(f)
    grouped=cont.groupby(level=0)
    t.cont_plot(grouped,False)
    O=cat_all[['alpha','delta','redshift']]
    t.dstest( O.T.values[0],O.T.values[1],O.T.values[2] )

    Creates a DS-test plot and returns a delta array, with delta value for each
    object in the input catalogs. Much of this code is borrowed from an ipython
    notebook by Nate Golovich, but with modifications.
    
    Input:
    ra = [1D array; units:degrees] RA of each object
    dec = [1D array; units:degrees] Dec of each object
    z = [1D array] Redshift of each object
    N_neighbors = ['SqrtTotal' or int] Number of nearest neighbors to use when 
        calculating the local velocity and dispersion parameters. 'SqrtTotal'
        uses the sqrt of the total number of galaxies (rounded up).
    prefix = [string] prefix of output DS-plot. If None then no figure is saved.
    '''
    import numpy
    import matplotlib.pylab as pylab
    import matplotlib.colors

    # Number of galaxies in the catalog
    N_gal = numpy.size(ra)
    
    # determine the number of local galaxies to use
    if N_neighbors == 'SqrtTotal':
        N_neighbors = int(numpy.ceil(numpy.sqrt(N_gal)))
    elif isinstance(N_neighbors,int):
        # then a int was input and everything is fine
        pass
    else:
        print "Error: invalid N_neighbor input. Please input an integer or 'SqrtTotal'. Exiting."
        sys.exit()
    z=numpy.asarray(z)
    print type(z)
    # Convert redshifts into velocities
    v = ((z+1)**2-1)/((z+1)**2+1)*299792.458
    # Calculate the global system properties
    v_global_mean = numpy.mean(v)
    v_global_std = numpy.std(v)
    
    # create a blank array to keep track of the distance between objects
    dist = numpy.zeros(N_gal)
    # create a blank delta array
    delta = numpy.zeros(N_gal)
    
    # Loop through the list of galaxies calculating a delta for each
    for i in range(N_gal):
        for j in range(N_gal):
            if i == j:
                dist[j] = 0
            else:
                dist[j] = tools.angdist(ra[i],dec[i],ra[j],dec[j])
        # select the N_neighbor nearest objects to object i, including object i
        index = numpy.argsort(dist)
        local_index = index[:N_neighbors]
        # calculate the average velocity of these nearest objects
        v_local_mean = numpy.mean(v[local_index])
        v_local_std = numpy.std(v[local_index])
        # calcualte the delta parameter
        delta[i] = numpy.sqrt((N_neighbors/v_global_std**2)*
                              ((v_local_mean-v_global_mean)**2 +
                               (v_local_std-v_global_std)**2))
    big_delta=numpy.sum(delta)
    print 'Big Delta: ', big_delta
    #Uncomment all of this plotting stuff when do MC reshuffle calc of delta
    #Maybe I should just make a new function or put this in the code to save time...
    # Create a DS Plot
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    '''
    #optional, temporary read from apj musketball contours pickle
    import pickle
    filename = 'contApJ.pickle'
    f = '/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename
    cont=pd.read_pickle(f)
    grouped=cont.groupby(level=0)
    cont_plot(grouped,False)
    '''

    #Nate suggests changing radius to clean up diagram
    #N = z1/z1.max()  # normalie 0..1
    #surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(N), linewidth=0, antialiased=False, shade=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #    pylab.scatter(ra,dec,s=10**delta,facecolors='none',c=z,color=cmap(c**2))
    #m = pylab.cm.ScalarMappable(cmap=pylab.cm.hot)
    #m.set_array(z)
    #cmap is only used if c is an array of floats
#    cmap = pylab.cm.coolwarm
#    cmap = pylab.colors.ListedColormap(C/255)
    c = matplotlib.colors.ColorConverter().to_rgb
    cmap = make_colormap(
#	    	  [ c('blue'), 0.30,c('violet'), 0.33, c('violet'), 0.66, c('violet'),c('red')])

        	    	  [ c('blue'), c('violet'),c('red')])
	  #[ c('blue'), c('green'),c('red')])

#    pylab.scatter(ra,dec,s=4*(4**delta),facecolors='none',linewidth='1.2',c=z,cmap=cmap)
    pylab.scatter(ra,dec,s=5*(3**delta),facecolors='none',linewidth='1.2',c=z,cmap=cmap)

    #m = pylab.cm.ScalarMappable(cmap=pylab.cm.coolwarm)
    m = pylab.cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    pylab.colorbar(m)
    #cbar = pylab.colorbar()
    #cbar.set_label('Redshift')
    pylab.gca().invert_xaxis() #TEMPORARILY REMOVE WHEN CONT PLOT
    pylab.xlabel('RA')
    pylab.ylabel('Dec')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.ticklabel_format(useOffset=False)
#    zmax=str(z.max)
    title='DS plot z=[{0:0.3f}, {1:0.3f}]'.format(z.min(),z.max())
    pylab.title(title, fontsize='14')
#    py.title('DS plot of Spectroscopically Confirmed Galaxies', fontsize='15')
    pylab.tight_layout()
    if prefix != None:
        filename = prefix+'_DSplot'
        pylab.savefig(filename,bbox_inches='tight')
    #return delta

def dstest_mc(ra,dec,z,N_neighbors='SqrtTotal'):
    '''
    Sept 19th: made for MC reshuffling of DS values outside of this function
    still calc and shuffle outside before input with shuffle()
    Temp adde contour plot inside this function
    grouped=t.make_cont()
    import pickle
    filename = 'contApJ.pickle'
    f = '/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename
    cont=pd.read_pickle(f)
    grouped=cont.groupby(level=0)
    t.cont_plot(grouped,False)
    O=cat_all[['alpha','delta','redshift']]
    t.dstest( O.T.values[0],O.T.values[1],O.T.values[2] )

    Creates a DS-test plot and returns a delta array, with delta value for each
    object in the input catalogs. Much of this code is borrowed from an ipython
    notebook by Nate Golovich, but with modifications.
    
    Input:
    ra = [1D array; units:degrees] RA of each object
    dec = [1D array; units:degrees] Dec of each object
    z = [1D array] Redshift of each object
    N_neighbors = ['SqrtTotal' or int] Number of nearest neighbors to use when 
        calculating the local velocity and dispersion parameters. 'SqrtTotal'
        uses the sqrt of the total number of galaxies (rounded up).
    prefix = [string] prefix of output DS-plot. If None then no figure is saved.
    '''
    import numpy
    import matplotlib.pylab as pylab
    import matplotlib.colors

    # Number of galaxies in the catalog
    N_gal = numpy.size(ra)
    
    # determine the number of local galaxies to use
    if N_neighbors == 'SqrtTotal':
        N_neighbors = int(numpy.ceil(numpy.sqrt(N_gal)))
    elif isinstance(N_neighbors,int):
        # then a int was input and everything is fine
        pass
    else:
        print "Error: invalid N_neighbor input. Please input an integer or 'SqrtTotal'. Exiting."
        sys.exit()
    z=numpy.asarray(z)
    #print type(z)
    # Convert redshifts into velocities
    v = ((z+1)**2-1)/((z+1)**2+1)*299792.458
    # Calculate the global system properties
    v_global_mean = numpy.mean(v)
    v_global_std = numpy.std(v)
    
    # create a blank array to keep track of the distance between objects
    dist = numpy.zeros(N_gal)
    # create a blank delta array
    delta = numpy.zeros(N_gal)
    
    # Loop through the list of galaxies calculating a delta for each
    for i in range(N_gal):
        for j in range(N_gal):
            if i == j:
                dist[j] = 0
            else:
                dist[j] = tools.angdist(ra[i],dec[i],ra[j],dec[j])
        # select the N_neighbor nearest objects to object i, including object i
        index = numpy.argsort(dist)
        local_index = index[:N_neighbors]
        # calculate the average velocity of these nearest objects
        v_local_mean = numpy.mean(v[local_index])
        v_local_std = numpy.std(v[local_index])
        # calcualte the delta parameter
        delta[i] = numpy.sqrt((N_neighbors/v_global_std**2)*
                              ((v_local_mean-v_global_mean)**2 +
                               (v_local_std-v_global_std)**2))
    big_delta=numpy.sum(delta)
    #print 'Big Delta: ', big_delta, ' Now returning little delta array'
    '''
    #Uncomment all of this plotting stuff when do MC reshuffle calc of delta
    #Maybe I should just make a new function or put this in the code to save time...
    # Create a DS Plot
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    #optional, temporary read from apj musketball contours pickle
    import pickle
    filename = 'contApJ.pickle'
    f = '/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/'+filename
    cont=pd.read_pickle(f)
    grouped=cont.groupby(level=0)
    cont_plot(grouped,False)

    #Nate suggests changing radius to clean up diagram
    #N = z1/z1.max()  # normalie 0..1
    #surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(N), linewidth=0, antialiased=False, shade=False)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #    pylab.scatter(ra,dec,s=10**delta,facecolors='none',c=z,color=cmap(c**2))
    #m = pylab.cm.ScalarMappable(cmap=pylab.cm.hot)
    #m.set_array(z)
    #cmap is only used if c is an array of floats
#    cmap = pylab.cm.coolwarm
#    cmap = pylab.colors.ListedColormap(C/255)
    c = matplotlib.colors.ColorConverter().to_rgb
    cmap = make_colormap(
#	    	  [ c('blue'), 0.30,c('violet'), 0.33, c('violet'), 0.66, c('violet'),c('red')])

        	    	  [ c('blue'), c('violet'),c('red')])
	  #[ c('blue'), c('green'),c('red')])

#    pylab.scatter(ra,dec,s=4*(4**delta),facecolors='none',linewidth='1.2',c=z,cmap=cmap)
    pylab.scatter(ra,dec,s=5*(3**delta),facecolors='none',linewidth='1.2',c=z,cmap=cmap)

        #m = pylab.cm.ScalarMappable(cmap=pylab.cm.coolwarm)
    m = pylab.cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    pylab.colorbar(m)
    #cbar = pylab.colorbar()
    #cbar.set_label('Redshift')
    #pylab.gca().invert_xaxis() #TEMPORARILY REMOVE WHEN CONT PLOT
    pylab.xlabel('RA')
    pylab.ylabel('Dec')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.ticklabel_format(useOffset=False)
#    zmax=str(z.max)
    title='DS plot z=[{0:0.3f}, {1:0.3f}]'.format(z.min(),z.max())
    pylab.title(title, fontsize='14')
#    py.title('DS plot of Spectroscopically Confirmed Galaxies', fontsize='15')
    pylab.tight_layout()
    if prefix != None:
        filename = prefix+'_DSplot'
        pylab.savefig(filename,bbox_inches='tight')
    '''
    return delta

#Ignore this DS test stuff is was a fail at getting group dv values using pandas. Use CAT dstest I edited.

def dstest_mod(group,N_neighbors='SqrtTotal',prefix=None):
#def dstest_mod(ra,dec,z,N_neighbors='SqrtTotal',prefix=None):
    '''
    Ali edit: For pandas df, changed scaling and colorbar
    Changed output so will give vdisp values nearest neighbor for each object
    Should do this by region/cluster/redshift/q, inputc
    for ones in CMD?
    Maybe change so include bootstrap
    O=cat_all[['alpha','delta','redshift_rs','objid']]
    dstest( O.T[0],O.T[1],O.T[2] )
    catsm.set_index('region',drop=False,inplace=True)
    catsm.swaplevel('region',0)
    Keep it simple, pass in whole thing
    catsm=t.dstest_mask(catsm)
    Edit: Aug 10, 2015: simple input
    O=cat_all[['alpha','delta','redshift']]
    blah=t.dstest_mod(O)
    


        
    Creates a DS-test plot and returns a delta array, with delta value for each
    object in the input catalogs. Much of this code is borrowed from an ipython
    notebook by Nate Golovich, but with modifications.
    
    Input:
    ra = [1D array; units:degrees] RA of each object
    dec = [1D array; units:degrees] Dec of each object
    z = [1D array] Redshift of each object
    N_neighbors = ['SqrtTotal' or int] Number of nearest neighbors to use when 
        calculating the local velocity and dispersion parameters. 'SqrtTotal'
        uses the sqrt of the total number of galaxies (rounded up).
    prefix = [string] prefix of output DS-plot. If None then no figure is saved.
    '''
#    ra,dec,z=group.T[0],group.T[1],group.T[2]
    ra,dec,z=group.T.values[0],group.T.values[1],group.T.values[2]
    #group['vdelta']=
    #group.set_index(['objid'],drop=False,inplace=True)
    #ra=group.alpha
    #dec=group.delta
    #z=group[x]
    #print type(z)
    # Number of galaxies in the catalog
    N_gal = np.size(ra)
    # determine the number of local galaxies to use
    if N_neighbors == 'SqrtTotal':
        N_neighbors = int(np.ceil(np.sqrt(N_gal)))
    elif isinstance(N_neighbors,int):
        # then a int was input and everything is fine
        pass
    else:
        print "Error: invalid N_neighbor input. Please input an integer or 'SqrtTotal'. Exiting."
        sys.exit()
    # Convert redshifts into velocities
    group['v'] = pd.Series( ((z+1)**2-1)/((z+1)**2+1)*299792.458 , index=group.index )
    #print group.v
    # Calculate the global system properties
    v_global_mean = group.v.mean()
    v_global_std = group.v.std()    
    # create a blank array to keep track of the distance between objects
    group['dist'] = pd.Series(np.repeat(np.array([np.nan]),N_gal), index=group.index )
    #dist = np.zeros(N_gal)
    # create a blank delta array to return
    group['vdelta'] = pd.Series(np.repeat(np.array([np.nan]),N_gal), index=group.index )
    # Loop through the list of galaxies calculating a delta for each
    for i in range(N_gal):
        for j in range(N_gal):
            if i == j:
                group.dist.iloc[j] = 0
            else:
                #print i,j, group.alpha.iloc[i]
                group.dist.iloc[j] = tools.angdist(group.alpha.iloc[i],group.delta.iloc[i],group.alpha.iloc[j],group.delta.iloc[j])
        #print group.dist.iloc[i]
        # select the N_neighbor nearest objects to object i, including object i
        #catsm.iloc[np.argsort(catsm.z)]
        index=group.dist.argsort()
        #index = np.argsort(group.dist) #this will get messy unless index is just objid
        local_index = index[:N_neighbors]
        # calculate the average velocity of these nearest objects
        v_local_mean = np.mean(group.v.iloc[local_index])
        v_local_std = np.std(group.v.iloc[local_index])
        #print 'local index, v',v_local_mean
         
        # calculate the delta parameter need iloc for integer index
        group['vdelta'].iloc[i] = np.sqrt((N_neighbors/v_global_std**2)*
                              ((v_local_mean-v_global_mean)**2 +
                               (v_local_std-v_global_std)**2))
    # Create a DS Plot
    
    fig = py.figure()
    ax = fig.add_subplot(111)
    c = mpl.colors.ColorConverter().to_rgb
    cmap = make_colormap(
        	    	  [ c('blue'), c('violet'),c('red')])
    #print group.delta
    py.scatter(ra,dec,s=5*(3**group.vdelta),facecolors='none',linewidth='1.2',c=z,cmap=cmap)
    m = py.cm.ScalarMappable(cmap=cmap)
    m.set_array(z)
    py.colorbar(m)
    py.gca().invert_xaxis()
    py.xlabel('RA')
    py.ylabel('Dec')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.ticklabel_format(useOffset=False)
    if prefix != None:
        filename = prefix+'_DSplot'
        py.savefig(filename)
    #group['vdelta']=pd.Series(np.repeat(np.array([np.nan]),len(group)),index=group.index)
    #print group.vdelta
    return group['vdelta']
## Debug
#a = numpy.sqrt(numpy.arange(20))*100
#b = numpy.arange(20)
#z = numpy.arange(20)/20.
#deltout = dstest(a,b,z,prefix='temp')
#print deltout
#pylab.show()
#print 'finished'


###Modified version of Nate's DS plot to output nearest neightbor vdisp for cat/scaling
#I think I started making this to try to assign individual vdisp to each object but not worth it
#I think I wanna do this by region (need region in index)
#model after RS cut, have outside function that calls DStest
def dstest_mask(data_rs, x='redshift'):
    #data_rs.swaplevel('region',0)
    #data_rs=t.dstest_mask(data_rs.swaplevel('region',0))
    #NEED TO DO THIS AFTER REGION ASSIGNED AND OUTER INDEX
#    vdisp='vdelta_'+x
    vdisp='vdelta'
    #print 'type',type(color)
    print 'running rscut using: ',x, 'new column: ',vdisp
    data_rs[vdisp]=pd.Series(np.repeat(np.array([np.nan]),len(data_rs)), index=data_rs.index)
    grouped=data_rs[['alpha','delta','objid',x]].swaplevel('region',0).groupby(level=0)
    #    data_rs[vdisp].loc['']= grouped.apply(dstest_mod,x)
    blah = grouped.apply(dstest_mod,x)
    print 'printing blah',blah
    #need to match by objid too. blah should have region, objid
    #why are there duplicates in objid?????? I GIVE UP. Also need to merge to insert.
    for i in blah.index.levels[0]:
#        blah[blah.index[0]]
#        data_rs[vdisp].loc[i]= blah[blah[i]]
        data_rs[vdisp].loc[i]= blah[i]

        
    '''
    for num in [a for a in [0,1,2,3]]:
            #bounds=rscut(params[params['cid']==num], data_rs,x='R')
            reg=strcid(num)
            print 'vdisp for reg: ',strcid(num)
            #puts in entire data_rs df to calculate bounds
            #does this for entire df, creating new color_R every time
            #then only reassigns bounds with mapping
            #bounds=rscut(params[params['cid']==num], data_rs,x=x)
            #This puts in one variable at a time
            grouped=data_rs[['alpha','delta',x]].loc[reg].groupby(level=0)
            data_rs[vdisp].loc[reg]= grouped.apply(dstest_mod)
            #data_rs[vdisp].loc[reg]= map(dstest_mod,  data_rs['alpha'].loc[reg], data_rs['delta'].loc[reg], data_rs[x].loc[reg])
    '''
    return data_rs


def boxreg_qcut(df,file,color='green',angle=0,size=5,null=False):
    #def circlereg(objid,ra,dec,radius,file,color='green'):
    #t.boxreg(df,'blah.tab',null=True) t.boxreg(Jelly_all,'Jelly.reg')
    color='Red'
    #file=pops[0]+color+'.tab'
    #t.boxreg(cat_all,file,null=True,color='cyan')
    color='Blue'
    #file=pops[0]+color+'.tab'
    #t.boxreg(cat_all,file,null=True,color='magenta')

        #t.boxreg(DLS,'blah_DLS.reg',color='white',null=True) #note no quality
    '''
    Accepts df with [['alpha','delta','objid','quality']] 
    Creates a ds9.reg file where each object input is represented by a circle.
    color = ['black', 'white', 'red' , 'green', 'blue', 'cyan', 'magenta', 
       'yellow'] color of the point
    I think 5 is " arcsecs now from my output, could adjust l&w for rectangle/slit
    '''
    #    outfile = prefix+'_circles.reg'
    size=np.repeat(size,len(df)) #5"x5"
    angle=np.repeat(angle,len(df))
    if null==True:
        nullcat='null_hq_'+file
        print 'printing null objids to ',nullcat
        #df[(df.objid.isnull())&(df.quality>2)].comment
        print df[(df.objid.isnull())&(df.quality>2)].count(), ' null high q values'
        print df[df.objid.isnull()].count(), ' null values'
        dfn=df[df.objid.isnull()&(df.quality>2)]
        size=np.repeat(size,len(dfn)) #5"x5"
        angle=np.repeat(angle,len(dfn))
        with open(nullcat,'a') as F:
            F.write('global color=red dashlist=8 3 width=2 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
            F.write('fk5'+'\n')
            for a,b,c,d,e,f,g,h,i,j in zip(dfn.alpha,dfn.delta,size,size,angle,dfn.maskname,dfn.slit,dfn.quality,dfn.alpha,dfn.delta):
                F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5} {6} {7} {8} {9}}}\n'.format(a,b,c,d,e,f,g,h,i,j) )
        F.close()
    print 'dropping null objid rows'
    df=df.dropna(subset=['objid']) #get rid of null values for int objid
    print 'highq+lowq' ,df.objid.count()
    with open(file,'a') as F:
        F.write('global color=white dashlist=8 3 width=2 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
        for a,b,c,d,e,f in zip(df.alpha,df.delta,size,size,angle,df.objid.astype('int64')):
            F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5}}}\n'.format(a,b,c,d,e,f ) ) #box x y width height angle text
    F.close()
    #quit() #add for DLS no quality column
    df=df[(df.quality<3)&(df.quality>-1)]
    size=np.repeat(size,len(df)) #5"x5"
    angle=np.repeat(angle,len(df))
    print 'lowq does not include stars' ,df[(df.quality<3)&(df.quality>-1)].objid.count()
    with open('lowq_'+file,'a') as F:
        F.write('global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
        for a,b,c,d,e,f in zip(df.alpha,df.delta,size,size,angle,df.objid.astype('int64')):
            F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5}}}\n'.format(a,b,c,d,e,f ) )
    F.close()


def boxreg(df,file,color='cyan',angle=0,size=10,null=False):
    #def circlereg(objid,ra,dec,radius,file,color='green'):
    #t.boxreg(df,'blah.tab',null=True) t.boxreg(Jelly_all,'Jelly.reg')
        #t.boxreg(DLS,'blah_DLS.reg',color='white',null=True) #note no quality
    '''
    Accepts df with [['alpha','delta','objid','quality']] 
    Creates a ds9.reg file where each object input is represented by a circle.
    color = ['black', 'white', 'red' , 'green', 'blue', 'cyan', 'magenta', 
       'yellow'] color of the point
    I think 5 is " arcsecs now from my output, could adjust l&w for rectangle/slit
    '''
    #    outfile = prefix+'_circles.reg'
    size=np.repeat(size,len(df)) #5"x5"
    angle=np.repeat(angle,len(df))
    print 'Writing all input objects to file, no quality cut: ' ,df.ID.count()
    with open(file,'a') as F:
        F.write('global color='+color+' dashlist=8 3 width=2 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
#        for a,b,c,d,e,f,g,h in zip(df.alpha,df.delta,size,size,angle,df.objid.astype('int64'),df.comment_f814,df.comment_f606):
#            F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5}, {6}, {7}}}\n'.format(a,b,c,d,e,f,g,h) ) #box x y width height angle text
        for a,b,c,d,e,f in zip(df.LFC_RA,df.LFC_DEC,size,size,angle,df.ID):
#        for a,b,c,d,e,f in zip(df.alpha,df.delta,size,size,angle,df.objid.astype('int64')):
            F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5}}}\n'.format(a,b,c,d,e,f ) ) #box x y width height angle text
    F.close()



def quickbox(df,filename):
    '''
    t.quickbox(df.values[0],'temp_slit.reg')
    
    '''
    print 'number objects writing to' ,filename,'test'
    with open(filename,'a') as F:
        F.write('global color=white dashlist=8 3 width=2 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
        print 'test'
        for a,b,c,d,e,f,g,h,i in zip(df.SLITRA,df.SLITDEC,df.SLITLEN,df.SLITWID,df.SLITWPA,df.objid,df.maskname,df.SLITNAME,df.redshift):
            F.write('box({0},{1},{2}",{3}",{4}) # text = {{{5} {6} {7} {8}}}\n'.format(a,b,c,d,e,f,g,h,i ) ) #box x y width height angle text
    F.close()
    
def circlereg(df,file,color='white',size=5,null=False):
    #def circlereg(objid,ra,dec,radius,file,color='green'):
    #t.boxreg(df,'blah.tab',null=True)
    '''
    Accepts df with [['alpha','delta','objid','quality']] 
    Creates a ds9.reg file where each object input is represented by a circle.
    null=True creates separate cat with null values
    Can edit so just gives yellow lowq
    color = ['black', 'white', 'red' , 'green', 'blue', 'cyan', 'magenta', 
       'yellow'] color of the point
    I think 5 is " arcsecs now from my output, could adjust l&w for rectangle/slit
    '''
    #    outfile = prefix+'_circles.reg'
    size=np.repeat(size,len(df)) #5"x5"
    #angle=np.repeat(angle,len(df))
    if null==True:
        nullcat='null_'+file
        print 'printing null objids to ',nullcat
        #df[(df.objid.isnull())&(df.quality>2)].comment
        print df[(df.objid.isnull())&(df.quality>2)].count(), ' null high q values'
        print df[df.objid.isnull()].count(), ' null values'
        dfn=df[df.objid.isnull()]
        size=np.repeat(size,len(dfn)) #5"x5"
        with open(nullcat,'a') as F:
            F.write('global color=red dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
            F.write('fk5'+'\n')
            for a,b,c,d in zip(dfn.alpha,dfn.delta,size,df.objid.astype('int64')):
                F.write('box({0},{1},{2}",{3})\n'.format(a,b,c,d) )
        F.close()
    print 'dropping null objid rows'
    #df=df.dropna(subset=['objid']) #get rid of null values for int objid
    df=df.dropna(subset=['id_spec2']) #get rid of null values for int objid
    print 'highq+lowq' ,df.count()

    with open(file,'a') as F:
        F.write('global color=black dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
        for a,b,c,d in zip(df.LFC_RA,df.LFC_DEC,size,df.id_spec2):
#        for a,b,c,d in zip(df.alpha,df.delta,size,df.objid.astype('int64')):
            F.write('circle({0},{1},{2}") # text = {{{3}}}\n'.format(a,b,c,d) )
            #box x y width height angle text
    F.close()
    df=df[(df.quality<3)&(df.quality>-1)]
    size=np.repeat(size,len(dfn)) #5"x5"
    angle=np.repeat(angle,len(dfn))
    print 'lowq does not include stars' ,df[(df.quality<3)&(df.quality>-1)].count()
    with open('lowq_'+file,'a') as F:
        F.write('global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1'+'\n')
        F.write('fk5'+'\n')
        for a,b,c,d in zip(df.alpha,df.delta,size,df.objid.astype('int64')):
            F.write('circle({0},{1},{2}",{3}) # text = {{{5}}}\n'.format(a,b,c,d) )
    F.close()


#####SIGNIFICANCE FUNCTIONS#

def sig_func(cat_all):
    '''
    sc = t.sig_func(cat_all)
    Input: specific population, cuts already applied: m_group, c_group, q_group (sm_group2, m_group2)
    No quality cut! Done inside program
    Option: maybe input without quality cut so can do missing within
    Must be formatted with multiindices: z_group, z_group_DLS
    Output: contamination, completeness rates
    '''  
    print 'DID YOU DO R2 STORE?????'
    print cat_all[(cat_all.q_group=='low')].xs(['pz'],level=['z_group_DLS']).objid.count(),' targeted objects photoz in pz yielded low quality spectra (trend, blue?)'
    print cat_all[(cat_all.q_group=='low')].objid.count(),' targeted objects total yielded low quality spectra (total outside pz is this minus inside pz above)'
    print cat_all[(cat_all.q_group=='star')].objid.count(),' targeted objects were stars, high q'
    a=cat_all.xs(['high'],level=['q_group']).groupby(['z_group','z_group_DLS']).size()
    #doing groupby for z_group with high quality automatically cuts out null speczs
    #a=cat_all.groupby(['z_group','z_group_DLS']).size()
    print a
    a=a.astype('float64')
    print a.sum(),' objects with high quality spectra (cut should already be made, missing stars)'
    print a.xs(['cz','pz'],level=['z_group','z_group_DLS']).values[0], ' objects with photozs in cluster (pz) range confirmed to be in cluster (cz) range '
    print a.xs(['pz'],level=['z_group_DLS']).sum(),'photoz pz confirmed as cz or nz'
    print (a.xs(['cz','pz'],level=['z_group','z_group_DLS']).values[0] / a.xs(['pz'],level=['z_group_DLS']).sum() ), ' success rate, objects (cz) confirmed out of all DLS pz objects confirmed '
    print a.xs(['nz'],level=['z_group_DLS']).sum(), 'photoz nz confirmed as cz or nz'
    print (a.xs(['cz','nz'],level=['z_group','z_group_DLS']).values[0] / a.xs(['nz'],level=['z_group_DLS']).sum() ), ' photoz nz confirmed as cz (contamination rate)'

    #to select use df.rates.loc['s']
    return pd.DataFrame([(a.xs(['cz','pz'],level=['z_group','z_group_DLS']).values[0] / a.xs(['pz'],level=['z_group_DLS']).sum() ), (a.xs(['cz','nz'],level=['z_group','z_group_DLS']).values[0] / a.xs(['nz'],level=['z_group_DLS']).sum() )],index=['s','c'],columns=['rates'])

def sig_calc(cat_all,df):
    '''
    Will need to recalc sig for normalized function
    m=t.sig_calc(cat_all,sc)
    Info on potential missing members (pz or nz no hq spectra) and contamination
    Input: sc df with rates, cat without q pr z cuts\
    Return DataFrame/Series with column 'missing' of integer max missing/error
    To be fed into significance and sig error calculation
    NO QUALITY CUT (so can store variable and put into sig_func, sig_err)
    '''

    #cat_all[(cat_all.q_group!='high')].xs(['pz','in','in','in'],level=['z_group_DLS','m_group','c_group','sm_group2']).groupby('region').size()
    mpz=cat_all[(cat_all.q_group!='high')].xs(['pz'],level=['z_group_DLS']).groupby('region').size()
    mnz=cat_all[(cat_all.q_group!='high')].xs(['nz'],level=['z_group_DLS']).groupby('region').size()
    print 'missing pz',mpz*df.rates.loc['s']
    print 'missing nz',mnz*df.rates.loc['c']
    m= (mpz*df.rates.loc['s']).add(mnz*df.rates.loc['c'], fill_value=0 )
    m=pd.DataFrame(m,columns=['missing'])
    print 'Missing upper limit:', m.missing.apply(round),
    return m.missing.apply(round)

def sig_err(cat_all, Nerr):
    '''
    Jan 18th, 2016: reformat cat to add normalized version, normalized by given array of the total population, no significance needed
    Nnorm is a Series with the same format as Nerr with region as the index
    Need to propagate errors, so insert error array for full pop too, which is sc from full r2 pop:
    
    Dec 15th, 2015: returns cat of significances
    cat_all=cat_all[cat_all.region_r2.not_null()]
    cat_all['region']=cat_all['region_r2']
    
    blah=cat_all.xs(['in','in','in'],level=['m_group','sm_group2','c_group']) #Narrow down pop to subpop, sigfunc filters out quality so don't do q cut here. Need numbers from total pop so make this an input option
    sc = t.sig_func(blah)
    m=t.sig_calc(blah,sc)
    t.sig_err(blah, m)
    
    Dec 8, 2015: Now returns table object to use for .to_latex:
    a=t.sig_err(blah, m)
    a.to_csv('blarg.tab',sep='\t',header=False,float_format='%.2f',mode='a') #quick and dirty with added latex symbols amp, \pm and \\
    Jan 18th: Normalize by N total galaxy population by region. I think I already do this manually with input catalog
    #Need to input this Series for full pop in r2
    cat_all=cat_all[cat_all.region_r2.not_null()]
    cat_all['region']=cat_all['region_r2']
    N=cat_all.xs(['high','cz'],level=['q_group','z_group']).groupby(['region']).size()
a.Galaxies/ Nnorm

    a.to_latex('blarg.tab',float_format=lambda x:"{0:.2f}".format(x)) #didn't work with $\pm$ so don't use
    
    cat_all has all cuts except for quality and redshift
    sc = t.sig_func(cat_all)
    m=t.sig_calc(cat_all,sc)
    t.sig_err(cat_all, m)
    Possible problem if Nerr null for index
    Need number of actual galaxy pop all cuts from cat_all
    Error df for missing galaxies by region
    Area df for density calculation for significance, should be same as before
    All should have region as multiindex
    Maybe make loop for both colors later (or all anyway and block out "above")
    
    '''
    filenm='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/sigcalc_rev2.tab'
    print 'using areas from reduced Remainder',filenm
    df=pd.read_table(filenm,sep='\t',index_col=[0])
    N=cat_all.xs(['high','cz'],level=['q_group','z_group']).groupby(['region']).size()
    #N=cat_all.xs(['cz','in','in','high','in','Blue'],level=['z_group','m_group','c_group','q_group','sm_group2','color_logsmk']).groupby(['region']).size()
    #Keep in mind Nerr here is a dataframe
    #Will missing regions (NE) probagate through as null? maybe use operations...
    #alternative is to artificially add index if missing 
    SD=N/df.area_arcmin
    SDerr=Nerr**2/df.area_arcmin**2
    Signorm=SD/SD.Remainder
    #Signormerr=np.sqrt( ( SDerr**2 * SD.Remainder**(-2.) ) + SDerr.Remainder**2 * (-SD / SD.Remainder**2)**2 )
    #EDIT June 27th 2015
    #np.sqrt( ( Berr**2 * R**(-2) ) + Rerr**2 * (B**2 / R**4.) )
    print 'correction added to significance error'
    Signormerr=np.sqrt( ( SDerr**2 * SD.Remainder**(-2.) ) + SDerr.Remainder**2 * (SD**2 / SD.Remainder**4.) )
    #"{0:.2f}".format((Signorm+Signormerr))
    #Siglims=[zip('{0:.2f}'.format((Signorm+Signormerr)),'{0:.2f}'.format((Signorm+Signormerr)) for )
    #print Siglims
    flim=lambda w,x:'[{0:.2f},{1:.2f}]'.format(w,x)
    df['Siglims']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(df)), index=df.index)
    df['Siglims']= map(flim, Signorm+Signormerr,Signorm-Signormerr)
    #Sigerr is propagated based on normalized function f=A/B
    #Sigerrnorm=Sigerr/Sigerr.Remainder
    #print pd.concat([N,Nerr,SD,SDerr,Signorm,Signormerr,SDerr/SDerr.Remainder],axis=1,keys=['galaxies','comp+contm_err','Surface Density (#/arcmin^2)','SD_err','Significance (SD/SD_R)','Signorm_err','SDerr/SDRerr'])
    #print pd.concat([N,Nerr,SD,SDerr,Signorm,Signormerr,df.Siglims],axis=1,keys=['#Galaxies','#err (compl+contam)','SD (#/arcmin^2)','SD_err','Significance (SD/SD_Rem)','Sig_err','Min/Max'])
    #TEMP to_csv option, with separator as ampersand or use to_latex
    amp=pd.Series(np.repeat(np.array(['&']),len(df)), index=df.index)
    pm=pd.Series(np.repeat(np.array(['$\pm$']),len(df)), index=df.index)
    endr=pd.Series(np.repeat(np.array([r'\\']),len(df)), index=df.index)
    print pd.concat([N,Nerr,SD,SDerr,Signorm,Signormerr,df.Siglims],axis=1,keys=['#Galaxies','#err (compl+contam)','SD (#/arcmin^2)','SD_err','Significance (SD/SD_Rem)','Sig_err','Min/Max'])
    #    return pd.concat([N,pm,Nerr,amp,SD,pm,SDerr,amp,Signorm,pm,Signormerr],axis=1,keys=['Galaxies','Sampling error','Surface Density (Galaxies/arcmin^2)','SD_err','Significance (SD/SD_Rem)','Sig_err'])
    return pd.concat([N,pm,Nerr,amp,SD,pm,SDerr,amp,Signorm,pm,Signormerr],axis=1)

    #a.to_csv('blarg.tab',sep='\t',header=False,float_format='%.2f',mode='a')
    #From in terminal, manually insert headers for ApJ table
    #with open('blarg.tab','a')as blah: blah.write('\\cutinhead{Blue Bright \n')
    #with open('blarg.tab','a')as blah: blah.write('\\sidehead{Blue} \n')
    #with open('blarg.tab','a')as blah: blah.write('\\sidehead{Red} \n')
    #I think signormerror is huge because we are comparing to Remainder error which is huge

    
def sig_err_norm(cat_all,cat_all_full, Nerr):
    '''
    We use this to find the relative overdensity of a subpopulation of galaxies between regions. So, the overdensty of galaxies relative to the whole population in a region, relative to that overdensity relative to the whole in other regions. 
    Input the full population values for sc and m, so run twice. Then process them the same way as sig but without surface density
    Must be for full pop of galaxies in r2, same as go into sig full pop
    Calc everything inside this same function, just input both. so do entire loop twice
    cat_all=cat_all.xs(['in','in','in'],level=['m_group','sm_group2','c_group'])
    m=t.sig_calc(cat_all, t.sig_func(cat_all)) #for full region function
    
    blah=cat_all.xs(['Blue'],level=['color_logsmk'])
    m=t.sig_calc(blah, t.sig_func(blah))
    sig_err_norm(blah, cat_all, m)
    Nnorm=cat_all_full.xs(['high','cz'],level=['q_group','z_group']).groupby(['region']).size()
    Jan 18th, 2016: Switched to makign entirely new function that duplicates and adds on to old sig_err, since do kind of the same thing but don't want to overcomplicate the one.
    reformat cat to add normalized version, normalized by given array of the total population, no significance needed
    Nnorm is a Series with the same format as Nerr with region as the index
    Need to propagate errors, so insert error array for full pop too, which is sc from full r2 pop:
    
    Dec 15th, 2015: returns cat of significances
    cat_all=cat_all[cat_all.region_r2.not_null()]
    cat_all['region']=cat_all['region_r2']
    
    blah=cat_all.xs(['in','in','in'],level=['m_group','sm_group2','c_group']) #Narrow down pop to subpop, sigfunc filters out quality so don't do q cut here. Need numbers from total pop so make this an input option
    sc = t.sig_func(blah)
    m=t.sig_calc(blah,sc)
    t.sig_err(blah, m)
    
    Dec 8, 2015: Now returns table object to use for .to_latex:
    a=t.sig_err(blah, m)
    a.to_csv('blarg.tab',sep='\t',header=False,float_format='%.2f',mode='a') #quick and dirty with added latex symbols amp, \pm and \\
    Jan 18th: Normalize by N total galaxy population by region. I think I already do this manually with input catalog
    #Need to input this Series for full pop in r2
    cat_all=cat_all[cat_all.region_r2.not_null()]
    cat_all['region']=cat_all['region_r2']    
    cat_all has all cuts except for quality and redshift
    sc = t.sig_func(cat_all)
    m=t.sig_calc(cat_all,sc)
    t.sig_err(cat_all, m)
    Possible problem if Nerr null for index
    Need number of actual galaxy pop all cuts from cat_all
    Error df for missing galaxies by region
    Area df for density calculation for significance, should be same as before
    All should have region as multiindex
    Maybe make loop for both colors later (or all anyway and block out "above")
    
    '''
    import cluster_tools.Cluster_functs as t
    
    filenm='/Users/alison/DLS/deimosspectracatalog/ApJ_MB/membership/numberdensity/weighted/Area/sigcalc_rev2.tab'
    print 'using areas from reduced Remainder',filenm
    df=pd.read_table(filenm,sep='\t',index_col=[0])
    N=cat_all.xs(['high','cz'],level=['q_group','z_group']).groupby(['region']).size()
    #N=cat_all.xs(['cz','in','in','high','in','Blue'],level=['z_group','m_group','c_group','q_group','sm_group2','color_logsmk']).groupby(['region']).size()
    #Keep in mind Nerr here is a dataframe
    #Will missing regions (NE) probagate through as null? maybe use operations...
    #alternative is to artificially add index if missing
    #cat_all_full=cat_all_full[cat_all_full.region_r2.not_null()]
    #cat_all_full['region']=cat_all_full['region_r2']
    Nreg=cat_all_full.xs(['high','cz'],level=['q_group','z_group']).groupby(['region']).size()
    #blah=cat_all_full
    #sc = t.sig_func(blah)
    #m=t.sig_calc(blah, t.sig_func(blah))
    Nregerr=t.sig_calc(cat_all_full, t.sig_func(cat_all_full))
    Nregpoisson=np.sqrt(Nreg)
    Npoisson=np.sqrt(N)
    #Now that have N and Nerr for full galaxy pop in each region, normalize sig for subpop and do error
    #(# North Blue Massive galaxies)/(# North All galaxies)
    Sigreg= N/Nreg #Are these both ints so does it round incrrectly? cast one as float?
    #N, Nreg, Nerr, Nregerr...however Brian says Poisson errors which are just sqrt(N)
    #BRerr=np.sqrt( ( Berr**2 * R**(-2.) ) + Rerr**2 * (B**2 / R**4.) )
    #BRfracttoterr=np.sqrt( ( Berr**2 * BRtot**(-2.) ) + BRtoterr**2 * (B**2 / BRtot**4.) )
    #ELfracttoterr=np.sqrt( ( Eerr**2 * ELtot**(-2.) ) + ELtoterr**2 * (E**2 / ELtot**4.) )
    Sigregerr=np.sqrt( ( Nerr**2 * Nreg**(-2.) ) + Nregerr**2 * (N**2 / Nreg**4.) )
    Sigregpoisson=np.sqrt( ( Npoisson**2 * Nreg**(-2.) ) + Nregpoisson**2 * (N**2 / Nreg**4.) )
    SD=N/df.area_arcmin
    SDpoisson=Nregpoisson/df.area_arcmin
    
    ###########EDIT CORRECTION?
    #SDerr=Nerr**2/df.area_arcmin**2 #why am I squaring both...err(n/dr.)=sig_n^2/alpha sq? or take sqrt?
    SDerr=np.sqrt(Nerr**2/df.area_arcmin**2) #why am I squaring both...err(n/dr.)=sig_n^2/alpha sq? or take sqrt?
    SDpoisson=np.sqrt(Npoisson**2/df.area_arcmin**2) #why am I squaring both...err(n/dr.)=sig_n^2/alpha sq? or take sqrt?
    Signorm=SD/SD.Remainder
    Signormerr=np.sqrt( ( SDerr**2 * SD.Remainder**(-2.) ) + SDerr.Remainder**2 * (-SD / SD.Remainder**2)**2 )
    #EDIT June 27th 2015
    #np.sqrt( ( Berr**2 * R**(-2) ) + Rerr**2 * (B**2 / R**4.) )
    #    print 'correction added to significance error'
    Signormpoisson=np.sqrt( ( SDpoisson**2 * SD.Remainder**(-2.) ) + SDpoisson.Remainder**2 * (SD**2 / SD.Remainder**4.) )
    #"{0:.2f}".format((Signorm+Signormerr))
    #Siglims=[zip('{0:.2f}'.format((Signorm+Signormerr)),'{0:.2f}'.format((Signorm+Signormerr)) for )
    #print Siglims
    #quit()
    flim=lambda w,x:'[{0:.2f},{1:.2f}]'.format(w,x)
    #df['Siglims']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(df)), index=df.index)
    #df['Siglims']= map(flim, Signorm+Signormerr,Signorm-Signormerr)
    df['Siglims']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(df)), index=df.index)
    df['Siglims']= map(flim, Signorm+Signormpoisson,Signorm-Signormpoisson)

    #Sigerr is propagated based on normalized function f=A/B
    #Sigerrnorm=Sigerr/Sigerr.Remainder
    #print pd.concat([N,Nerr,SD,SDerr,Signorm,Signormerr,SDerr/SDerr.Remainder],axis=1,keys=['galaxies','comp+contm_err','Surface Density (#/arcmin^2)','SD_err','Significance (SD/SD_R)','Signorm_err','SDerr/SDRerr'])
    #print pd.concat([N,Nerr,SD,SDerr,Signorm,Signormerr,df.Siglims],axis=1,keys=['#Galaxies','#err (compl+contam)','SD (#/arcmin^2)','SD_err','Significance (SD/SD_Rem)','Sig_err','Min/Max'])
    #TEMP to_csv option, with separator as ampersand or use to_latex
    amp=pd.Series(np.repeat(np.array(['&']),len(df)), index=df.index)
    pm=pd.Series(np.repeat(np.array(['$\pm$']),len(df)), index=df.index)
    endr=pd.Series(np.repeat(np.array([r'\\']),len(df)), index=df.index)
    #flim=lambda x:'{0:0f}'.format(x)
    #df['Nerr']=pd.Series(np.repeat(np.array(['Not_Mapped']),len(df)), index=df.index)
    #quit()
    Nerr=Nerr.fillna(0)
    Nerr= Nerr.astype('int64') #Note this doesn't work for NaN, inf...thought they fixed
    
    print '\n',pd.concat([N,Nerr,Npoisson, Sigreg, Sigregpoisson,SD,SDerr,SDpoisson, Signorm,Signormerr,Signormpoisson,df.Siglims],axis=1,keys=['#Galaxies','#err (compl+contam)','Npoisson','N/Ntot','N/Ntotpoisson','SD (#/arcmin^2)','SD_err','SDerrpoisson','Significance (SD/SD_Rem)','Sig_err','Sigerrpoisson','Min/Max'])
    #    return pd.concat([N,pm,Nerr,amp,SD,pm,SDerr,amp,Signorm,pm,Signormerr],axis=1,keys=['Galaxies','Sampling error','Surface Density (Galaxies/arcmin^2)','SD_err','Significance (SD/SD_Rem)','Sig_err'])
    #Klugey format for pretty latex output table
#    return pd.concat([amp,N,pm,Nerr,pm,Npoisson,amp,np.round(Sigreg,2),pm,np.round(Sigregerr,2),amp,np.round(SD,2),pm,np.round(SDerr,2),amp,np.round(Signorm,2),pm,np.round(Signormerr,2),endr],axis=1)
    return pd.concat([amp,N,amp,np.round(Sigreg,2),pm,np.round(Sigregpoisson,2),amp,np.round(SD,2),pm,np.round(SDpoisson,2),amp,np.round(Signorm,2),pm,np.round(Signormpoisson,2),endr],axis=1)

    #b.to_csv('blarg.tab',sep='\t',header=False,na_rep='NaN',mode='a') #problem floatifies ints
    #a.to_csv('blarg.tab',sep='\t',header=False,float_format='%.2f',mode='a') #problem floatifies ints
    #From in terminal, manually insert headers for ApJ table
    '''
    m=t.sig_calc(blah, t.sig_func(blah))
    b=t.sig_err_norm(blah,cat_all,m)
    b.to_csv('blarg.tab',sep='\t',header=False,mode='a') #problem floatifies ints
    with open('blarg.tab','a')as blahh: blahh.write('\\cutinhead{Bright Massive} \n')
    with open('blarg.tab','a')as blahh: blahh.write('\\cutinhead{Full Color} \n')
    with open('blarg.tab','a')as blahh: blahh.write('\\cutinhead{Massive} \n')
    with open('blarg.tab','a')as blahh: blahh.write('\\cutinhead{Bright} \n')
    
    with open('blarg.tab','a')as blahh: blahh.write('\\sidehead{Blue} \n')
    with open('blarg.tab','a')as blahh: blahh.write('\\sidehead{Red} \n')
    
    color='Blue'
    color='Red'
    #BBM
    blah=cat_all.xs([color,'high','bright'],level=['color_logsmk','sm_group','m_group2'])
    #BM
    #pop='BlueMassive'
    blah=cat_all.xs([color,'high'],level=['color_logsmk','sm_group'])
    #BB
    #pop='BlueBright'
    blah=cat_all.xs([color,'bright'],level=['color_logsmk','m_group2'])
    #Blue
    blah=cat_all.xs([color],level=['color_logsmk'])
    
    '''
