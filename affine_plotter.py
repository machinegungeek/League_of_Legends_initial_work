from csv_handler import CSV_Handler
import sys
sys.path[0] = '/nfs/slac/g/suncatfs/aidank/software/lib/python2.7/site-packages/matplotlib-1.5.1-py2.7-linux-x86_64.egg'
import numpy as np
import matplotlib as mpi
mpi.use('Agg')
import pylab as plt
import os
from Fitting_code.fitlib import *
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from marker_dict import *

#Class written to make many affine plots. Currently a fucking mess
class Affine_Plotter:
    #Should axe the filename segment
    def __init__(self, shortname=''):
        self.shortname = shortname
    
    def init_file(self,myfile):
        assert os.path.isfile(myfile)
        self.filename = myfile
        if self.shortname == '':
            self.get_shortname()

    def get_shortname(self):
        assert self.shortname == ''
        self.shortname = self.filename.split('/')[-1].split('.')[0]
    #Reads in heading and data from csv files
    def get_data(self):
        CSV = CSV_Handler()
        res = CSV.read(self.filename)
        data = res[5]
        headings = res[2]
        return headings, data

    def make_dir(self, dirname):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
    #Creates the affine fits and gets errors for the 3 varieties 
    def regression(self,x,y):
        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        m,b = np.polyfit(x,y,1)
        plty = np.reshape(np.poly1d((m,b))(x),(-1,))
        #errors = y-x
        errors = x-y
        MAE,MSE,RMS,STD = self.get_errors(errors)        

        new_errors = plty-y
        NMAE, NMSE, NRMS, NSTD = self.get_errors(new_errors)

        trans_x = x - MSE
        trans_errors = trans_x-y
        TMAE, TMSE, TRMS, TSTD = self.get_errors(trans_errors)

        return m,b,errors, new_errors,plty,MAE,NMAE, MSE, NMSE, RMS, NRMS, STD, NSTD, TMAE,TMSE,TRMS,TSTD
    #Gives error stats given a vector of error values
    def get_errors(self, errors):
        MAE = np.sum(np.abs(errors))/errors.shape[0]
        MSE = np.sum(errors)/errors.shape[0]
        RMS = np.sqrt(np.sum((errors)**2)/errors.shape[0])
        STD = np.std(errors)    
        return MAE, MSE, RMS, STD
    #Pretty useless now. Makes histograms of the error distribution for both
    #affine corrected and original numbers
    def error_plot(self,destination,err,nerr,nbins,errlab1,errlab2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(err,nbins, facecolor='blue', label=errlab1, alpha=0.4)
        ax.hist(nerr,nbins, facecolor='orange', label=errlab2, alpha=0.4)
        ax.legend(loc='best')
        ax.set_xlabel('Errors')
        ax.set_ylabel('Number of Systems')
        fig.savefig(destination)
    
    #Method for creating histrogram matrices for error_bar_plot
    def create_bins(self,i,errors,width,offset):
        #number of bars in cluster
        ncluster = errors.shape[1] -1
        #number of functionals/clusters
        nfunc = errors.shape[0]
        left = np.zeros((nfunc,))
        height = np.zeros((nfunc,))
        for j in range(nfunc):
            left[j] = 0 +i*width + j*(offset+ncluster*width)
            height[j] = errors[j][i+1]
        return left, height

    #Actually plot plots for error_bar_plot
    def make_bar_plot(self,ax,i,left,height,width,color,legend_labels,yerr):
         if legend_labels:
            if yerr is not None and i==ncluster-1:
                ax.bar(left,height,width, color = color, label = legend_labels[i],yerr=yerr)
            else:
                ax.bar(left,height,width, color = color, label = legend_labels[i])
         else:
            if yerr is not None and i==ncluster-1:
               ax.bar(left,height,width, color=color)
            else:
               ax.bar(left,height,width, color=color)

    #make labels for error_bar_plot
    def set_labels(self,ax,errors,offset,width,ylabel):
        #number of bars in cluster
        ncluster = errors.shape[1] -1
        #number of functionals/clusters
        nfunc = errors.shape[0]
        if ylabel:
            ylabel=ylabel.replace('_',r'\_')
            ax.set_ylabel(ylabel)
        xtls = list(np.take(errors,[0],axis=1))
        for i,tl in enumerate(xtls):
            xtls[i] = tl[0].replace('_',r'\_')
        ax.set_xticklabels(xtls, size=10.5, rotation=45)
        ax.set_xticks(np.arange(0,nfunc*(offset+ncluster*width),(offset+ncluster*width))+(ncluster/2.)*width)
        ax.legend(loc='best')

    #messy method. makes colored/labeled clustered bar plots
    #can have error bars on them
    def error_bar_plot(self,destination,errors, width = 0.45, offset = 1.4, legend_labels=None, ylabel=None, yerr=None):
        #number of bars in cluster
        ncluster = errors.shape[1] -1
        #number of functionals/clusters
        nfunc = errors.shape[0]
        cvalues = np.arange(ncluster)/float(ncluster)
        if legend_labels:
            assert len(legend_labels)==ncluster
            for i, ll in enumerate(legend_labels):
                legend_labels[i] = ll.replace('_',r'\_')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #creates the matrices for the plotting
        for i in range(ncluster):
            left,height=self.create_bins(i,errors,width,offset) 
            color = cm.jet(cvalues[i])
            self.make_bar_plot(ax,i,left,height,width,color,legend_labels,yerr)
        self.set_labels(ax,errors,offset,width,ylabel)
        fig.savefig(destination)
    
    #Makes the individual line plots w/ error label    
    def reg_plot(self,destination,x,y,xlabel = '', ylabel = '',int='on'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        m,b,errors,new_errors,plty,MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD,TMAE,TMSE,TRMS,TSTD = self.regression(x,y)
        tstring1 = r'Experimental and Calculated Data'
        if int=='on':
            tstring2 = 'y = %.3f * x + %.3f' % (m,b)
            if not xlabel =='':
                if not xlabel == '':
                    xlabel=xlabel.replace('_',r'\_')
                    #tstring2=tstring2
                    #tstring2 = xlabel + ' = %.3f * ' % (1/m) + ylabel + ' + %.3f' % (-b/m)
                    #tstring2 = ylabel + ' = %.3f * ' % (m) + xlabel + ' + %.3f' % (b)
                    tstring2 = r'$\widetilde{\textrm{E}}' + r'$ = %.3f $\cdot$  ' % (m) + xlabel + ' + %.3f' % (b)
            tstring3 = r'\noindent MAE: %.3f \newline MAE$_{\textrm{fit}}$: %.3f \newline \newline MSE: %.3f \newline MSE$_{\textrm{fit}}$: %.3f \newline \newline RMS: %.3f \newline RMS$_{\textrm{fit}}$: %.3f \newline \newline STD: %.3f \newline STD$_{\textrm{fit}}$: %.3f' % (MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD)
            #tstring3 =r'blah: \newline\newline blah %.3f' % (MAE)
            ax.plot(x,y,'ro', label=tstring1)
            ax.plot(x,plty,'--k', label = tstring2)
            leg = ax.legend(loc='upper left')
            for labels in leg.get_texts():
                labels.set_fontsize('medium')
            ax.text(0.02,0.807,tstring3,transform=ax.transAxes,verticalalignment='top',fontsize='medium')
            if not xlabel == '':
                ax.set_xlabel(xlabel)
            if not ylabel == '':
                ylabel=ylabel.replace('_',r'\_')
                #print ylabel
                ax.set_ylabel(ylabel)
        else:
            co,NMAE,NMSE,NRMS,NSTD =self.ref_reg(np.reshape(x,(-1,1)),y,np.reshape(x,(-1,1)),y)
            plty = x*co[0]
            tstring2 = 'y = %.3f * x ' % co[0]
            if not xlabel =='':
                if not xlabel == '':
                    #tstring2 = xlabel + ' = %.3f * ' % (1/m) + ylabel + ' + %.3f' % (-b/m)
                    #tstring2 = ylabel + ' = %.3f * ' % (co[0]) + xlabel 
                    tstring2 = r'$\widetilde{\mathrm{E}}' + ' = %.3f \cdot \mathrm{' % (co[0]) + xlabel +'}$'
            tstring3 = r'\noindent MAE: %.3f\newline MAE$_{\textrm{fit}}$: %.3f \newline\newline MSE: %.3f\newline MSE$_{\textrm{fit}}$: %.3f\newline\newline RMS: %.3f\newline RMS$_{\textrm{fit}}$: %.3f\newline\newline STD: %.3f\newline STD$_{\textrm{fit}}$: %.3f' % (MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD)
            #tstring3=''
            ax.plot(x,y,'ro', label=tstring1)
            ax.plot(x,plty,'-k', label = tstring2)
            leg = ax.legend(loc='upper left')
            for labels in leg.get_texts():
                labels.set_fontsize('medium')
            ax.text(0.02,0.83,tstring3,transform=ax.transAxes,verticalalignment='top',fontsize='medium')
            if not xlabel == '':
                ax.set_xlabel(xlabel)
            if not ylabel == '':
                ax.set_ylabel(ylabel)

        fig.savefig(destination)
    
    #Makes the individual line plots w/ error label. Adds in color lines    
    def new_reg_plot(self,destination,x,y,xlabel = '', ylabel = '',int='on'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cvalues = np.arange(4)/4.0
        m,b,errors,new_errors,plty,MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD,TMAE,TMSE,TRMS,TSTD = self.regression(x,y)
        tstring1 = r'$\textrm{Experimental \ and \ Calculated \ Data}$'
        sy,ty,ay,slope,trans = self.get_plty(x,y)
        if int=='on':
            tstring2 = r'y = %.3f $\cdot$ x + %.3f' % (m,b)
            if not xlabel =='':
                if not ylabel == '':
                    xlabel2 = '\_'.join(xlabel.split('_'))
                    ylabel = ylabel.replace('_',r'\_')
                    #tstring2 = xlabel + ' = %.3f * ' % (1/m) + ylabel + ' + %.3f' % (-b/m)
                    tstring2 = r'$\widetilde{\textrm{E}}$' + r' = %.3f $\cdot$ ' % (m) + xlabel2 + ' + %.3f' % (b)
                    ostring = r'$\widetilde{\textrm{E}}$' + r' = ' + xlabel2
                    trstring = r'$\widetilde{\textrm{E}}$' + r' = ' + xlabel2 + ' + %.3f' % (-1*trans)
                    sstring = r'$\widetilde{\textrm{E}}$' + r' = %.3f $\cdot$ ' % (1-slope) + xlabel2
            tstring3 = r'\noindent MAE: %.3f\newline MAE$_{\textrm{fit}}$: %.3f\newline\newline MSE: %.3f\newline MSE$_{\textrm{fit}}$: %.3f\newline\newline RMS: %.3f\newline RMS$_{\textrm{fit}}$: %.3f\newline\newline STD: %.3f\newline STD$_{\textrm{fit}}$: %.3f' % (MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD)
            ax.plot(x,y,'ro', label=tstring1)
            ax.plot(x,x,'-',color=cm.jet(cvalues[0]), label = ostring)
            ax.plot(x,sy,'-',color=cm.jet(cvalues[1]), label = sstring)
            ax.plot(x,ty,'-',color=cm.jet(cvalues[2]), label = trstring)
            ax.plot(x,plty,'-',color=cm.jet(cvalues[3]), label = tstring2)
            leg = ax.legend(loc='upper left')
            for labels in leg.get_texts():
                labels.set_fontsize('small')
            ax.text(0.02,0.71,tstring3,transform=ax.transAxes,verticalalignment='top',fontsize='small')
            if not xlabel == '':
                ax.set_xlabel(xlabel2)
            if not ylabel == '':
                ax.set_ylabel(ylabel)
        else:
            co,NMAE,NMSE,NRMS,NSTD =self.ref_reg(np.reshape(x,(-1,1)),y,np.reshape(x,(-1,1)),y)
            plty = x*co[0]
            tstring2 = r'y = %.3f $\cdot$ x ' % co[0]
            if not xlabel =='':
                if not xlabel == '':
                    #tstring2 = xlabel + ' = %.3f * ' % (1/m) + ylabel + ' + %.3f' % (-b/m)
                    tstring2 = r'$\tilde{E}$' + ' = %.3f * ' % (co[0]) + xlabel 
            tstring3 = r'\noindent MAE: %.3f\newline MAE$_{\textrm{fit}}$: %.3f\newline\newline MSE: %.3f\newline MSE$_{\textrm{fit}}$: %.3f\newline\newline RMS: %.3f\newline RMS$_{\textrm{fit}}$: %.3f\newline\newline STD: %.3f\newline STD$_{\textrm{fit}}$: %.3f' % (MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD)
            ax.plot(x,y,'ro', label=tstring1)
            ax.plot(x,plty,'--k', label = tstring2)
            leg = ax.legend(loc='upper left')
            for labels in leg.get_texts():
                labels.set_fontsize('medium')
            ax.text(0.02,0.83,tstring3,transform=ax.transAxes,verticalalignment='top',fontsize='medium')
            if not xlabel == '':
                ax.set_xlabel(xlabel)
            if not ylabel == '':
                ax.set_ylabel(ylabel)

        fig.savefig(destination)

    #Gets plot data for new reg plot
    def get_plty(self,x,y,custom_x=None):
        exp = np.reshape((x-y),(-1,1))
        res1 = self.ref_reg(np.ones((x.shape[0],1)),exp,np.ones((x.shape[0],1)),exp)
        res2 = self.ref_reg(np.reshape(x,(-1,1)),exp,np.reshape(x,(-1,1)),exp)
        fit = np.hstack((np.ones((x.shape[0],1)),np.reshape(x,(-1,1))))
        res3 = self.ref_reg(fit,exp,fit,exp)
        #print res1[0],res2[0],res3[0]
        print res1[-2],res2[-2],res3[-2]
        if custom_x is not None:
            fit = np.hstack((np.ones((custom_x.shape[0],1)),np.reshape(custom_x,(-1,1))))
            ty = np.reshape(custom_x,(-1,))-np.dot(np.ones((custom_x.shape[0],1)),res1[0].T)
            sy = np.reshape(custom_x,(-1,))-np.dot(np.reshape(custom_x,(-1,1)),res2[0].T)
            ay = np.reshape(custom_x,(-1,))-np.dot(fit,res3[0].T)
        else:
            ty = np.reshape(x,(-1,))-np.dot(np.ones((x.shape[0],1)),res1[0].T)
            sy = np.reshape(x,(-1,))-np.dot(np.reshape(x,(-1,1)),res2[0].T)
            ay = np.reshape(x,(-1,))-np.dot(fit,res3[0].T)

        return sy,ty,ay,res2[0][0],res1[0][0]#,res3[1]

    #Makes the set of two error line plots
    def make_error_plots(self,destination,calc,exp,xlabel='',ylabel=''):
        m,b,errors,new_errors,plty,MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD,TMAE,TMSE,TRMS,TSTD = self.regression(calc,exp)
        args = np.argsort(exp)
        exp = exp[args]
        errors = errors[args]
        new_errors = new_errors[args]
        ymax = np.max([np.max(errors),np.max(new_errors)])
        ymax *=1.1
        ymin = np.min([np.min(errors),np.min(new_errors)])
        ymin *=1.1
        self.graph_error_plot(destination +'_uncorr.png',exp,errors,xlabel=xlabel,ylabel=ylabel,ymin=ymin,ymax=ymax)
        self.graph_error_plot(destination +'_corr.png',exp,new_errors,xlabel=xlabel,ylabel=ylabel,ymin=ymin,ymax=ymax)

    #Actually plot the error plots
    def graph_error_plot(self,destination,exp,errors,xlabel='',ylabel='',ymin = None,ymax =None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(exp,errors,'bo',markersize=8)
        #ax.plot(exp,errors,'b-')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        #ax.spines['left'].set_position('zero')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if not xlabel == '':
            ax.set_xlabel(xlabel)
        if not ylabel == '':
            ax.set_ylabel(ylabel)
        if ymin:
            ax.set_ylim(bottom=ymin)
        if ymax:
            ax.set_ylim(top=ymax)
        fig.savefig(destination)

    #Makes the individual line plots for more generalized regression    
    def gen_reg_plot(self,destination,x,corr_x,y,xlabel = '', ylabel = ''):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tstring1 = 'Original Calculated Data'
        tstring2 = 'Corrected Calculated Data'
        MAE,MSE,RMS,STD = self.get_errors(y-x)
        NMAE,NMSE,NRMS,NSTD = self.get_errors(y-corr_x)
        tstring3 = r'\noindent MAE: %.3f\newline MAE$_{\textrm{fit}}$: %.3f\newline\newline MSE: %.3f\newline MSE$_{\textrm{fit}}$: %.3f\newline\newline RMS: %.3f\newline RMS$_{\textrm{fit}}$: %.3f\newline\newline STD: %.3f\newline STD$_{\textrm{fit}}$: %.3f' % (MAE,NMAE,MSE,NMSE,RMS,NRMS,STD,NSTD)
        ax.plot(x,y,'ro', label=tstring1)
        ax.plot(corr_x,y,'bD', label = tstring2)
        ax.plot(y,y,'k-')
        leg = ax.legend(loc='upper left')
        for labels in leg.get_texts():
            labels.set_fontsize('medium')
        ax.text(0.02,0.83,tstring3,transform=ax.transAxes,verticalalignment='top',fontsize='medium')
        if not xlabel == '':
            ax.set_xlabel(xlabel)
        if not ylabel == '':
            ax.set_ylabel(ylabel)

        fig.savefig(destination)
    
    def multi_reg_plot(self,destination,data,xlabel='',ylabel='',labels=[],lines = False,displayslope=False,slope_lines = False):
        #data should be a list of x/y numpy arrays
        for d in data:
            assert len(d.shape)==2
            assert 2==d.shape[1]
        if labels:
            #print labels
            assert len(labels) == len(data)
        fig =plt.figure()
        ax=fig.add_subplot(111)
        cvalues = np.arange(len(data))/float(len(data))
        for i,d in enumerate(data):
            #ax.plot(d[:,0],d[:,1],linestyle = 'None',marker = get_marker_symbol(i),color=cm.jet(cvalues[i]))
            ax.plot(d[:,1],d[:,0],linestyle = 'None', marker = 'o', color = cm.jet(cvalues[i]),label = labels[i])
            #print d.shape
            #d = np.reshape(d,(-1,2))
            if i==0:
                full_mat =d
            else:
                #bads=[]
                #for j, dat in enumerate(d):
                #    if dat in full_mat:
                #        bads.append(j)
                        #print i,j
                        #print dat
                #print bads
                #d_ = np.delete(d,bads,axis=0)
                full_mat = np.vstack((full_mat,d))
            #print full_mat.shape
        
        #regression line
        fit = np.hstack((np.ones((full_mat.shape[0],1)),np.reshape(full_mat[:,1],(-1,1))))
        expt = np.reshape(full_mat[:,0],(-1,1))
        co,mae,mse,rms,std = self.ref_reg(fit,expt,fit,expt)
        #plty = np.dot(fit,co.T)
        stuff = self.get_plty(full_mat[:,1],full_mat[:,0])
        rms = np.sqrt(np.mean((full_mat[:,0]-full_mat[:,1])**2))
        if slope_lines:
            cvalues2 = np.arange(4.)/4.
            print full_mat.shape
            lstring1 = 'Base Model, RMSE: %.3f' % np.sqrt(np.mean((full_mat[:,0]-full_mat[:,1])**2))
            lstring2 = 'Slope Correction, RMSE: %.3f' % np.sqrt(np.mean((full_mat[:,0]-stuff[0])**2))
            lstring3 = 'Translation Correction, RMSE: %.3f' % np.sqrt(np.mean((full_mat[:,0]-stuff[1])**2))
            lstring4 = 'Affine Correction, RMSE: %.3f' % np.sqrt(np.mean((full_mat[:,0]-stuff[2])**2))
            #FIX THIS NOW
            ax.plot(full_mat[:,0],full_mat[:,0],linestyle='-',color=cm.jet(cvalues2[0]),label = lstring1)
            ax.plot(full_mat[:,1],stuff[0],linestyle='-',color=cm.jet(cvalues2[1]),label = lstring2)
            ax.plot(full_mat[:,1],stuff[1],linestyle='-',color=cm.jet(cvalues2[2]),label = lstring3)
            ax.plot(full_mat[:,1],stuff[2],linestyle='-',color=cm.jet(cvalues2[3]),label = lstring4)
        else:
            ax.plot(full_mat[:,1],stuff[2],'-k')

        if labels:
            #labels[:] = [r'$\textrm{' + r' \ '.join(l.split())+r'}$' for l in labels]
            #labels.append(r'$\widetilde{\textrm{E}}$' + r' = %.3f $\cdot$ ' % (co[1]) + xlabel + ' + %.3f' % (co[0]))
            if not slope_lines and not displayslope:
                labels.append('RMSE: %.3f eV/bond' % rms)
            if displayslope:
                labels.append('Slope: %.3f' % co[1])
            if slope_lines:
                leg = ax.legend(loc='upper left',markerscale=0.6,labelspacing=.1, fontsize='small')
            else:
                leg=ax.legend(labels,loc='upper left',markerscale=0.75,labelspacing=.25)
            for l in leg.get_texts():
                l.set_fontsize(7)
        if not xlabel == '':
            ax.set_xlabel(xlabel)
        if not ylabel == '':
            ax.set_ylabel(ylabel)
        if lines:
            for i,d in enumerate(data):
                ax.plot(full_mat[:,0],self.get_plty(d[:,0],d[:,1],custom_x=full_mat[:,0])[2],'-',color=cm.jet(cvalues[i]))

        fig.savefig(destination,dpi=160)
       
    #Plotter for bivariate correlation scatter plots. Currently hardcoded for two sets of scatter
    def bivariate_plot(self,destination,data1,data2,xlabel='',ylabel='',pt_labels=[],leg_labels=[], origin = False, arrows=False):
        #assertions
        assert data1.shape[0] == data2.shape[0]
        assert data1.shape[1] ==2
        assert data2.shape[1] ==2
        if leg_labels:
            assert len(leg_labels) ==2
        if pt_labels:
            assert len(pt_labels) == data1.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if leg_labels:
            ax.scatter(data1[:,0],data1[:,1],c='b',marker='o',s=35,label=leg_labels[0])
            ax.scatter(data2[:,0],data2[:,1],c='g', marker='o',s=35,label=leg_labels[1])
            ax.legend()
        else:
            ax.scatter(data1[:,0],data1[:,1],c='b',marker='o',s=35)
            ax.scatter(data2[:,0],data2[:,1],c='g', marker='o',s=35)
        if pt_labels:
            for label,x,y in zip(pt_labels,data1[:,0],data1[:,1]):
                ax.annotate(label,xy=(x,y),textcoords='offset points',xytext=(-15,8),fontsize=6.5)
            for label,x,y in zip(pt_labels,data2[:,0],data2[:,1]):
                ax.annotate(label,xy=(x,y),textcoords='offset points',xytext=(-15,-10),fontsize=6.5)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        #Origo on plot?
        if origin:
            #ax.spines['left'].set_position('zero')
            #ax.spines['bottom'].set_position('zero')
            ymax = ax.get_ylim()[1]
            xmax = ax.get_xlim()[1]
            ax.set_ylim([0.0,ymax])
            ax.set_xlim([0.0,xmax])
        #Arrows between sets?
        if arrows:
            for x1,y1,x2,y2 in zip(data1[:,0],data1[:,1],data2[:,0],data2[:,1]):
                ax.annotate('',xy=(x2,y2),xytext=(x1,y1),textcoords='data',arrowprops=dict(headwidth=3.5,width = 0.5, facecolor='black',shrink=0.05))
        fig.savefig(destination)

    #Plotter for bivariate correlation scatter plots. Currently hardcoded for three sets of scatter each with a set of arrows
    def bivariate_plot2(self,destination,data1,data2,data3,leg_labels,xlabel='',ylabel='',pt_labels=[], origin = False,rms=[]):
        import matplotlib.patches as mpatches
        #assertions
        assert data1.shape[1] ==4
        assert data1.shape[1] == data2.shape[1]
        assert data1.shape[1] == data3.shape[1]
        assert len(leg_labels) ==3
        if rms:
            assert len(rms) ==3
        if pt_labels:
            assert len(pt_labels) == data1.shape[0]+data2.shape[0]+data3.shape[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(data1.shape[0]):
            cvalues = np.linspace(0.5,.85,data1.shape[0])
            ax.scatter(data1[i,0],data1[i,1],c=cm.Reds(cvalues[i]),marker='o',s=35)
            #if i ==0:
                #ax.quiver(data1[i,0],data1[i,1],data1[i,2]-data1[i,0],data1[i,3]-data1[i,1],label=leg_labels[0],color=cm.Reds(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
                #ax.legend()
            #else:
            ax.quiver(data1[i,0],data1[i,1],data1[i,2]-data1[i,0],data1[i,3]-data1[i,1],color=cm.Reds(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
            if rms:
                ax.quiver(data1[i,2],data1[i,3],rms[0][i,0]-data1[i,2],rms[0][i,1]-data1[i,3],color=cm.Reds(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
        for i in range(data2.shape[0]):
            cvalues = np.linspace(0.5,.85,data2.shape[0])
            ax.scatter(data2[i,0],data2[i,1],c=cm.Blues(cvalues[i]),marker='o',s=35)
            #if i ==0:
                #ax.quiver(data2[i,0],data2[i,1],data2[i,2]-data2[i,0],data2[i,3]-data2[i,1],label=leg_labels[1],color=cm.Blues(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
                #ax.legend()
            #else:
            ax.quiver(data2[i,0],data2[i,1],data2[i,2]-data2[i,0],data2[i,3]-data2[i,1],color=cm.Blues(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
            if rms:
                ax.quiver(data2[i,2],data2[i,3],rms[1][i,0]-data2[i,2],rms[1][i,1]-data2[i,3],color=cm.Blues(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
        for i in range(data3.shape[0]):
            cvalues = np.linspace(0.5,.85,data3.shape[0])
            ax.scatter(data3[i,0],data3[i,1],c=cm.Greens(cvalues[i]),marker='o',s=35)
            #if i ==0:
                #ax.quiver(data3[i,0],data3[i,1],data3[i,2]-data3[i,0],data3[i,3]-data3[i,1],label=leg_labels[2],color=cm.Greens(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
                #ax.legend()
            #else:
            ax.quiver(data3[i,0],data3[i,1],data3[i,2]-data3[i,0],data3[i,3]-data3[i,1],color=cm.Greens(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
            if rms:
                ax.quiver(data3[i,2],data3[i,3],rms[2][i,0]-data3[i,2],rms[2][i,1]-data3[i,3],color=cm.Greens(cvalues[i]),scale_units='xy',angles='xy',scale=1,width=.007,headwidth=2)
        datax = np.vstack((np.vstack((np.reshape(data1[:,0],(-1,1)),np.reshape(data2[:,0],(-1,1)))),np.reshape(data3[:,0],(-1,1))))
        datay = np.vstack((np.vstack((np.reshape(data1[:,1],(-1,1)),np.reshape(data2[:,1],(-1,1)))),np.reshape(data3[:,1],(-1,1))))
        if pt_labels:
            for label,x,y in zip(pt_labels,datax,datay):
                ax.annotate(label,xy=(x,y),textcoords='offset points',xytext=(-15,8),fontsize=10.5)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        #Origo on plot?
        if origin:
            #ax.spines['left'].set_position('zero')
            #ax.spines['bottom'].set_position('zero')
            ymax = ax.get_ylim()[1]
            xmax = ax.get_xlim()[1]
            ax.set_ylim([0.0,ymax])
            ax.set_xlim([0.0,xmax])
        #ax2 = fig.add_subplot(111)
        #rp=mpatches.Patch(color='red',label=leg_labels[0])
        #bp=mpatches.Patch(color='blue',label=leg_labels[1])
        #gp=mpatches.Patch(color='green',label=leg_labels[2])
        rp = plt.Rectangle((0,0),1,1,fc='r')
        bp = plt.Rectangle((0,0),1,1,fc='b')
        gp = plt.Rectangle((0,0),1,1,fc='g')
        ax.legend([rp,bp,gp],leg_labels)#,prop={'size':'small'})
        #ax.legend()
        fig.savefig(destination)

    #A quick qr solver for rectangular underdetermined systems
    def qr_solver(self,x,y):
        assert x.shape[0] < x.shape[1]
        assert x.shape[0] == y.shape[0]
        q,r = np.linalg.qr(x.T)
        sol = np.dot(q,np.dot(np.linalg.inv(r.T),y))
        return sol

    #A simple data decorrelation method (shrinks nxm x to mxm) 
    def decorrelate_cost(self, X, Y):
        s = np.cov(X)                    # empirical covariance
        v1, eigs, v2 = np.linalg.svd(s)  # spectral decomposition via SVD
        x = np.dot(v2, X)
        y = np.dot(v2, Y)
        (sh1, sh2) = np.shape(x)
        assert len(y) == sh1
        if sh1 > sh2:
            x = x[:sh2,:]
            y = y[:sh2]
            (sh1, sh2) = np.shape(x)
            assert sh1 == sh2
            assert sh1 == len(y)
        return x, y

    #Method that should be in another file which deals with correlation coefficient BS
    def corrcoef(self,fname):
        assert os.path.isfile(fname)
        mat = np.load(fname)['ref_mat']
        mat = np.hstack((np.identity(mat.shape[0]),mat))
        corr = np.corrcoef(mat,rowvar=0)
        sum = np.sum(np.abs(corr),axis=0)
        sorted = np.sort(sum)
        args = np.argsort(sum)

        return sorted, args

    def corr_graph(self,fname,name):
        sorted,args = self.corrcoef(fname)
        min = self.get_base_corr(fname)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = np.load(fname)['ref_mat'].shape[1]
        x = range(ind)[::-1]
        y = sorted[-ind:]
        ax.plot(x,y,'bo-')
        val = [min]*ind
        ax.plot(x,val,'b--',label='Example Minimum')
        val = [min*np.e]*ind
        ax.plot(x,val,'k--', label='Proposed Cutoff')
        ax.set_ylabel('Correlation Sum')
        plt.gca().set_ylim(bottom=0)
        ax.legend()
        plt.savefig(name)

    def get_base_corr(self,fname):
        mat = np.load(fname)['ref_mat']
        mat = np.hstack((np.identity(mat.shape[0]),mat))
        mat = np.vstack((mat,np.zeros((1,mat.shape[1]))))
        mat = np.hstack((mat,np.zeros((mat.shape[0],1))))
        mat[-1][-1] = 1
        corr = np.corrcoef(mat,rowvar=0)
        sum = np.sum(np.abs(corr),axis=0)
        min=sum[-1]
        
        return min

    #Removes columns of zeros. Hopefully improves fits.
    def trim(self,x):
        bad = []
        for i in range(x.shape[1]):
            col = np.take(x,[i],axis=1)
            if not col.any():
                bad.append(i)
        nx = np.delete(x,bad,axis=1)
        return nx,bad

    #Perform bf regression. ox/oy = orignal (non-decorrelated)
    def ref_reg(self,x,y,ox,oy):
        #hopefully makes x better conditioned
        x,bad = self.trim(x)
        #print x.shape
        #u,s,v =np.linalg.svd(x)
        #print np.sum(s>1e-1)
        y = np.reshape(y,(-1,))
        oy = np.reshape(oy,(-1,))
        if not x.size == 0:
            co,blah = RR(x,y,np.ones(x.shape[1]),0)
            bad.sort()
            for b in bad:
                co = np.insert(co,[b],[0])
            y_hat = np.dot(ox,co)
            errors = y_hat-oy
        else:
            errors = oy
            co = np.zeros(ox.shape[1])
        mae,mse,rms,std = self.get_errors(errors)
        return co,mae,mse,rms,std
   
    #Sort of counts how dependent columns/references are to each other
    def calc_ind(self,data):
        assert data.shape[0]<data.shape[1]
        assert np.linalg.matrix_rank(data) == data.shape[0]
        counts = np.zeros(data.shape[1])
        eps = 1e-12
        for i in range(data.shape[1]):
            b = np.take(data,[i],axis=1)
            x = np.hstack((data[:,:i],data[:,i+1:]))
            if np.linalg.matrix_rank(x) < data.shape[0]:
                counts[i] = 0
            else:
                sol = self.qr_solver(x,b)   
                count = 0
                for s in sol:
                    if s <eps:
                        count+=1
                counts[i] =count
        return counts         
 
    #10-fold CV implementation. X is already fitted, dur
    def TENCV(self,x,y,seed = 23,num=5):
        seeds = np.linspace(seed, seed+4, num=num)
        ferrors = np.zeros(num)
        for k in range(num):
            #create our random divisions
            prng = np.random.RandomState(int(seeds[k]))
            indexes = np.floor(x.shape[0]*prng.rand(x.shape[0]))
            #Create a list of ten bins
            bins = [[],[],[],[],[],[],[],[],[],[]]
            #Deal with bin sizes for datasets not divisible by ten
            excess = x.shape[0] % 10
            counter = 0
            #Fill the bins with the appropriate indices
            for i in range(10):
                if i < excess:
                    size = int(np.ceil(x.shape[0]/10.0))
                else: 
                    size = int(np.floor(x.shape[0]/10.0))
                stuff = range(counter,counter + size)
                counter += size
                for j in stuff:
                    bins[i].append(indexes[j])
            errors = np.zeros(10)
            #Run the CV process ten times
            for i in range(10):
                #Create/fill the test and training matrices
                test = bins[i]
                test_x = np.zeros((len(test),x.shape[1]))
                test_y = np.zeros((len(test),))
                for j in range(len(test)):
                    test_x[j] = x[test[j]]
                    test_y[j] = y[test[j]]
                train = [] 
                for j in range(10):
                    if not j==i:
                        train += bins[j]
                #print train
                train_x = np.zeros((len(train),x.shape[1]))
                train_y = np.zeros((len(train),))
                for j in range(len(train)):
                    train_x[j] = x[train[j]]
                    train_y[j] = y[train[j]]
                co_,rmae,rmse,rrms,rstd = self.ref_reg(train_x,train_y,train_x,train_y) 
                #print co_
                #print test_x.shape
                #print co_.shape
                sys.stdout.flush()
                test_y_hat = np.dot(test_x,co_)
                mae,mse,rms,std = self.get_errors(test_y - test_y_hat)
                errors[i] = rms
            ferrors[k] = np.mean(errors)
        return np.mean(ferrors)

    #Bootstrap using bf regression (so w/ atomic ref corrections)
    def ref_BS(self,x,y,limit=200,seed=23):
       #create our random seed
        prng = np.random.RandomState(int(seed))
        err_mat = np.zeros((limit,2))
        fails=[]        

        #Run through `limit' number of BS runs
        for i in range(limit):
            #Create a matrix of indices for the test data set
            tindexes = np.array([])
            while not tindexes.any():
                #Sample from the input data with replacement
                indexes = np.floor(x.shape[0]*prng.rand(x.shape[0]))
                indexes = np.array(indexes,dtype=int)
                rand_x = np.zeros((x.shape[0],x.shape[1]))
                rand_y = np.zeros((y.shape[0],))
                #Get random x,y matrices
                for j in range(x.shape[0]):
                    #print j
                    #print indexes[j]
                    rand_x[j] = x[indexes[j]]
                    rand_y[j] = y[indexes[j]]
                tindexes = []
                #make the test data matrices by putting in stuff not in the sample matrices
                for j in range(x.shape[0]):
                    if j not in indexes:
                        tindexes.append(j)
                tindexes = np.array(tindexes)
                #Create test x and y matrices, w/ size dependant on the data not sampled
                test_x = np.zeros((tindexes.shape[0],x.shape[1]))
                test_y = np.zeros((tindexes.shape[0],))
                #and populate the matrices
                for j in range(tindexes.shape[0]):
                    test_x[j] = x[tindexes[j]]
                    test_y[j] = y[tindexes[j]]
            #Creates a model fitted on the sampled training set
            co_,rmae,rmse,rrms,rstd = self.ref_reg(rand_x,rand_y,rand_x,rand_y)
            #This is the rms associated with fitting to the training data
            training_set_rms = rrms

            #gets error from fitting to the test set
            test_y_hat = np.dot(test_x,co_)
            rmae,rmse,rrms,rstd = self.get_errors(test_y_hat - test_y)
            test_set_rms = rrms            

            if not test_set_rms>0:
                #print tindexes,indexes
                fails.append(i)        

            #Maybe? This would be error of the f' model over the full data set
            y_hat = np.dot(x,co_)
            rmae,rmse,rrms,rstd = self.get_errors(y_hat-y)
            full_rms = rrms
    
            #co_,rmae,rmse,rrms,rstd = self.ref_reg(test_x,test_y,test_x,test_y)
            #err_mat[i][1] = np.exp(-1) * bsarms + (1-np.exp(-1)) * arms
            #err_mat[i][0] = np.exp(-1) *bstrms + (1-np.exp(-1)) * trms
            
            #Fill up the error matrices. Should have one column be .632 BS and on be 1 BS
            #May be a typo here or in the lines above. 
            #Also, not sure if full_rms or test_set_rms should be used
            err_mat[i][0] = np.exp(-1)* full_rms + (1-np.exp(-1)) * test_set_rms
            err_mat[i][1] = test_set_rms
        #Mean the 200 (or limit) runs
        if fails:
            err_mat = np.delete(err_mat,fails,axis=0)
        means = np.mean(err_mat,axis=0)
        return means

    def LOOCV(self,X,Y):
        X,bad_ = self.trim(X)
        if X.size > 0:
            # Implementation of http://www.anc.ed.ac.uk/rbf/intro/node43.html
            omega2 = 0
            Y_ = Y
            XtX = np.dot(X.T,X)
            Ainv = np.linalg.inv(XtX + np.diag(np.ones(np.shape(X)[1]))*omega2)
            P = np.diag(np.ones(len(Y_))) - np.dot(X,np.dot(Ainv,X.T))

            LOOCV_EPE = len(Y_)**-1 * \
                        np.dot(
                            np.dot(
                                np.dot(
                                        np.dot(Y_.T, P),
                                    np.diag(np.diag(P)**-2)
                                    ),
                                P),
                            Y_)
            return np.sqrt(LOOCV_EPE)
        else:
            mae,mase,rms,std = self.get_errors(Y)
            return rms*(Y.size -1)/ (Y.size)

    #Makes a ref mat from a list of names
    def make_ref_mat(self, names):
        import re
        ref_names = []
        ref_mat = np.zeros((len(names),0))
        for i,n in enumerate(names):
            elements = re.findall('[A-Z][^A-Z]*',n)
            for e in elements:
                nam_num = re.match('([^0-9]+)([0-9]*)',e)
                ref_nam = nam_num.group(1)
                ref_num = nam_num.group(2) 
                if ref_num == '':
                    ref_num=1
                else:
                    ref_num = int(ref_num)
                if ref_nam not in ref_names:
                    ref_names.append(ref_nam)
                    ref_mat = np.append(ref_mat,np.zeros((len(names),1)),axis=1)
                ind = ref_names.index(ref_nam)
                ref_mat[i][ind] += ref_num

        return ref_mat,ref_names

    #Runs some of the reference fitting routines
    def ref_run(self,mat1,mat2,index=0,decorr = False):
        res = np.zeros((mat1.shape[1]-1,5))
        coeffs = np.zeros((mat1.shape[1]-1, mat2.shape[1]+2))
        y = np.reshape(np.take(mat1,[index],axis=1),(-1,))
        I = np.zeros(mat2.shape[1]+2)
        lcount = range(mat1.shape[1])
        lcount.remove(index)
        for i,cou in enumerate(lcount):
            x = np.take(mat1,[cou],axis=1)
            x = np.hstack((np.ones(x.shape),x))
            x = np.hstack((x,mat2))
            if decorr:
                x_,y_ = self.decorrelate_cost(x,y)
            else:
                x_=x
                y_=y
            #for testing
            #y = np.reshape(np.take(x,[1],axis=1),(-1,))-y
            #y_ = np.reshape(np.take(x_,[1],axis=1),(-1,))-y_
            co,mae,mse,rms,std = self.ref_reg(x_,y_,x,y)
            epe = self.ref_BS(x_,y_)[0]
            res[i] = np.array([mae,mse,rms,std,epe])
            coeffs[i] = co
        return res,coeffs

    #Abbreviated method for finding best model. Runs in n^2 time
    def ref_short_combo(self,mat1,mat2,index=0,key ='632BS',mod=1.0,powers=[1]):
        nvar = mat2.shape[1] + len(powers)
        nfunc = mat1.shape[1]-1
        vmat = range(nvar)
        umat = []
        y=np.reshape(np.take(mat1,[index],axis=1),(-1,))
        lcount = range(mat1.shape[1])
        lcount.remove(index)
        res = np.zeros((mat1.shape[1]-1,6))
        coeffs = np.zeros((mat1.shape[1]-1,nvar))

        for i,cou in enumerate(lcount):
            x1 = np.take(mat1,[cou],axis=1)
            y_ = x1[:,0]-y
            ox = mat2
            if powers:
                for p in powers:
                    ox = np.hstack((x1**p,ox))
            x_ = np.zeros((y.shape[0],1))
            if key=='632BS':
                base_epe = self.ref_BS(x_,y_)[0]
            if key =='BS':
                base_epe = self.ref_BS(x_,y_)[1]
            if key =='TENCV':
                base_epe = self.TENCV(x_,y_,num=5)
            if key =='LOOCV':
                base_epe = self.LOOCV(x_,y_)
            succeed = True
            min_epe = base_epe
            while succeed:
                #print umat
                min_epe,umat,vmat,succeed = self.check_for_best(ox,y_,umat,vmat,min_epe,key,mod)
            umat.sort()
            vmat.sort()
            x_ = np.take(ox,umat,axis=1)
            co,mae,mse,rms,std = self.ref_reg(x_,y_,x_,y_)
            for v in vmat:
                co = np.insert(co,[v],[0])
            indexes = 0
            for u in umat:
                indexes += 10**u
            res[i] = [indexes,mae,mse,rms,std,min_epe]
            coeffs[i] = co

        return res,coeffs

    def check_for_best(self,x,y,umat,vmat,min,key,mod=1.0):
        succ = False
        print vmat
        for v in vmat:
            #print v
            umat_ = list(umat)
            umat_.append(v)
            x_ = np.take(x,umat_,axis=1)
            if key == '632BS':
                epe_ = self.ref_BS(x_,y)[0]
            if key == 'BS':
                epe_ = self.ref_BS(x_,y)[1]
            if key == 'LOOCV':
                epe_ = self.LOOCV(x_,y)
            if key == 'TENCV':
                epe_ = self.TENCV(x_,y)
            if epe_ < mod*min:
                succ = True
                min = epe_
                min_v = v

        if succ:
            umat.append(min_v)
            vmat.remove(min_v)
            return min,umat,vmat,succ
        else:
            return min,umat,vmat,succ 

    #Abbreviated method for finding best model. Used to make plots
    #I switched x and mat, so watch out if errors pop up
    def mod_ref_short_combo(self,mat1,mat2,index=0,key ='632BS',powers=[0,1],mod_x=None):
        #Again, mat1 should just be exp. and DFT data, mat2 should be references
        """mat1 is a blah
        mat2 is a blah2
        """
        nfunc = mat1.shape[1] -1
        #nvar = mat.shape[1]
        nvar = mat2.shape[1]+len(powers)
        #print nfunc
        #assert nfunc == 1
        vmat = range(nvar)
        umat = []
        y=np.reshape(np.take(mat1,[index],axis=1),(-1,))
        lcount = range(mat1.shape[1])
        lcount.remove(index)
        #res = np.zeros((nfunc,6))
        #coeffs = np.zeros((mat1.shape[1]-1,nvar))
        res = np.zeros((nvar+1,3))
        detailed_results=[]
        #print lcount
        for i,cou in enumerate(lcount):
            x1 = np.take(mat1,[cou],axis=1)
            ox = mat2
            if powers:
                for p in powers:
                    ox = np.hstack((x1**p,ox))
            x_ = np.zeros((y.shape[0],1))
            if not mod_x==None:
                y = np.reshape(mod_x,(-1,))-y
            else:
                y = np.reshape(x1,(-1,)) - y
            if key=='632BS':
                #print x_.shape, y.shape
                base_epe = self.ref_BS(x_,y)[0]
            if key=='BS':
                base_epe = self.ref_BS(x_,y)[1]
            if key =='TENCV':
                base_epe = self.TENCV(x_,y)
            if key =='LOOCV':
                base_epe = self.LOOCV(x_,y)
            co,mae,mse,rms,std = self.ref_reg(x_,y,x_,y)
            res[0] = [-1,rms,base_epe]
            succeed = True
            min_epe = base_epe
            count = 1
            while succeed:
                min_epe,umat,vmat,min_v,succeed,all_res = self.mod_check_for_best(ox,y,umat,vmat,min_epe,key)
                #print min_epe
                x_ = np.take(ox,umat,axis=1)
                co,mae,mse,rms,std = self.ref_reg(x_,y,x_,y)
                res[count] = [min_v,rms,min_epe]
                count +=1
                detailed_results.append(all_res)
        
        return res,co,detailed_results
    
    
    def mod_check_for_best(self,x,y,umat,vmat,min,key):
        succ = True
        min =  2555555555
        all_res = np.zeros(len(vmat))
        print vmat
        for i,v in enumerate(vmat):
            umat_ = list(umat)
            umat_.append(v)
            x_ = np.take(x,umat_,axis=1)
            if key == '632BS':
                epe_ = self.ref_BS(x_,y)[0]
            if key == 'BS':
                epe_ = self.ref_BS(x_,y)[1]
            if key == 'LOOCV':
                epe_ = self.LOOCV(x_,y)
            if key == 'TENCV':
                epe_ = self.TENCV(x_,y)
            all_res[i] = epe_
            #if i == 0:
                #print epe_
                #print x_
            if epe_ < min:
                succ = True
                min = epe_
                min_v = v
        #if min ==2555555555:
            #print epe_
            #print vmat,umat
            #print y
        #print epe_
        umat.append(min_v)
        vmat.remove(min_v)
        if len(vmat) == 0:
            succ = False
        return min,umat,vmat,min_v,succ,all_res
    
    def get_atomic_errors1(self,base,expt,refs):
        base_err = self.ref_reg(base,expt,base,expt)[-2]
        ref_num = refs.shape[1]
        errs = np.zeros((ref_num,1))
        for i in range(ref_num):
            ref = np.take(refs,[i],axis=1)
            fit = np.hstack((base,ref))
            err = self.ref_reg(fit,expt,fit,expt)[-2]
            err -= base_err
            err = np.abs(err)
            err/= np.sum(ref)
            errs[i] = err

        return errs

    def get_atomic_errors2(self,base,expt,refs):
        fit = np.hstack((base,refs))
        base_err = self.ref_reg(fit,expt,fit,expt)[-2]
        ref_num = refs.shape[1]
        errs = np.zeros((ref_num,1))
        for i in range(ref_num):
            ref = np.delete(refs,[i],axis=1)
            fit = np.hstack((base,ref))
            err = self.ref_reg(fit,expt,fit,expt)[-2]
            err -= base_err
            err = np.abs(err)
            err/= np.sum(ref)
            errs[i] = err

        return errs
    
    #Makes nvar vs. epe/rms graphs
    def nvar_graph(self,destination,res,names,displaymin=True):
        gnames =['null']
        for i in range(len(names)):
            gnames.append(names[int(res[i+1][0])])
        fig = plt.figure(figsize=(10,8),tight_layout=True)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(np.arange(res.shape[0]),np.take(res,[1],axis=1), '+',mew=2,markersize=16, label='RMSE')
        ax1.plot(np.arange(res.shape[0]),np.take(res,[2],axis=1),'x', mew=2,markersize=16, label = 'EPE')
        if displaymin:
            min = np.argsort(res[:,2])[0]
            ax1.axvline(min,color='r',linestyle='--',label='Ideal Model Size')
        ax1.set_xlabel('nvar',fontsize=16)
        tick_locations = np.linspace(1/(2*float(len(gnames))),((2*len(gnames))-1)/(2*float(len(gnames))),num=len(names)+1)
        ax1.set_xticks(np.linspace(0,len(names),num=len(names)+1))
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(gnames,fontsize=12)
        for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
        ax2.set_xlabel('Variable Added',fontsize=16)
        ax1.set_xlim(-0.5,len(names)+.5)
        ax1.set_ylim(0.0, np.max(np.take(res,[1],axis=1))*1.3)
        ax1.legend(loc='best')
        fig.savefig(destination)

    #Makes a histogram for nvar used
    def nvar_hist(self,destination,data,bins):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stuff = ax.hist(data,bins)
        b_width = bins[1] - bins[0]
        ax.set_xlim((1,bins[-1] + b_width))
        ax.set_xlabel('nvar')
        ax.set_ylabel('Number of Functionals')
        labels = []
        ticks = []
        for b in bins:
            labels.append(str(b))
            ticks.append(b)
        labels.append(str(bins[-1]+b_width))
        ticks.append(bins[-1]+b_width)
        ax.set_xticklabels(labels)
        ax.set_xticks(ticks)
        plt.savefig(destination)
        

    #Runs all combos
    def ref_combo_run(self,mat1,mat2,index=0,key = 'BS'):
        nvar = mat2.shape[1] + 2
        nfunc = mat1.shape[1]-1
        lcount = range(mat1.shape[1])
        lcount.remove(index)
        y = np.reshape(np.take(mat1,[index],axis=1),(-1,))
        res = np.zeros((mat1.shape[1]-1,6))
        coeffs = np.zeros((mat1.shape[1]-1, nvar))
        
        for i,cou in enumerate(lcount):
            min_epe = 1000000
            ox = np.take(mat1,[cou],axis=1)
            ox = np.hstack((np.ones(ox.shape),ox))
            ox = np.hstack((ox,mat2))
            #Run all the combos
            for j in range(2**nvar):
                if j % 1000 == 0:
                    #print j
                    #if not j==0:
                        #print min_epe
                        #print min_index
                    sys.stdout.flush()
                inds = list(("{0:" + str(nvar) + "b}").format(j))
                bads = []
                for k,num in enumerate(inds):
                    if num == '0' or num == ' ':
                        bads.append(k)
                x_ = np.copy(ox)
                for b in bads:
                    x_[:,b] = 0
                if key == 'BS':
                    epe = self.ref_BS(x_,y)[0]
                elif key == 'LOOCV':
                    epe = self.LOOCV(x_,y)
                elif key == 'TENCV':
                    epe = self.TENCV(x_,y)
                if epe < min_epe:
                    min_epe = epe
                    min_index = j
            inds = list(("{0:" + str(nvar) + "b}").format(min_index))
            bads = []
            for k,num in enumerate(inds):
                if num == '0' or num ==' ':
                    bads.append(k)
            for b in bads:
                ox[:,b] = 0 
            co,mae,mse,rms,std = self.ref_reg(ox,y,ox,y)
            res[i] = [int(min_index),mae,mse,rms,std,min_epe]
            coeffs[i] = co
        return res, co

    #Make matrices that will be turned into heat plots
    def make_heat_matrices(self,db_list,key='632BS'):
        csv = CSV_Handler()
        rmss = np.zeros((len(db_list),0,4))
        epes = np.zeros((len(db_list),0,4))
        names = np.zeros(0,dtype=object)
        for i, db in enumerate(db_list):
            blah = csv.read(db)
            for j in range(len(blah[2])-1):
                if blah[2][j+1] not in names:
                    names = np.insert(names,j,np.zeros(1))
                    rmss= np.insert(rmss,j,0,axis=1)
                    epes = np.insert(epes,j,0,axis=1)
                    names[j] = blah[2][j+1]
                ind = list(names).index(blah[2][j+1])
                x_ = np.take(blah[5], [j+1], axis=1)
                y_ = np.reshape(x_ - np.take(blah[5],[0],axis=1),(-1,))
                if key=='632BS':
                    rmss[i][ind][0] = self.ref_reg(np.zeros((y_.shape[0],1)),y_,np.zeros((y_.shape[0],1)),y_)[3]
                    epes[i][ind][0] = self.ref_reg(np.zeros((y_.shape[0],1)),y_,np.zeros((y_.shape[0],1)),y_)[3]
                    rmss[i][ind][1] = self.ref_reg(np.ones((y_.shape[0],1)),y_,np.ones((y_.shape[0],1)),y_)[3]
                    epes[i][ind][1] = self.ref_BS(np.ones((y_.shape[0],1)),y_)[0]
                    rmss[i][ind][2] = self.ref_reg(x_,y_,x_,y_)[3]
                    epes[i][ind][2] = self.ref_BS(x_,y_)[0]
                    rmss[i][ind][3] = self.ref_reg(np.hstack((x_,np.ones((y_.shape[0],1)))),y_,np.hstack((x_,np.ones((y_.shape[0],1)))),y_)[3]
                    epes[i][ind][3] = self.ref_BS(np.hstack((x_,np.ones((y_.shape[0],1)))),y_)[0]                           
                elif key == 'BS':
                    rmss[i][ind][0] = self.ref_reg(np.zeros((y_.shape[0],1)),y_,np.zeros((y_.shape[0],1)),y_)[3]
                    epes[i][ind][0] = self.ref_reg(np.zeros((y_.shape[0],1)),y_,np.zeros((y_.shape[0],1)),y_)[3]
                    rmss[i][ind][1] = self.ref_reg(np.ones((y_.shape[0],1)),y_,np.ones((y_.shape[0],1)),y_)[3]
                    epes[i][ind][1] = self.ref_BS(np.ones((y_.shape[0],1)),y_)[1]
                    rmss[i][ind][2] = self.ref_reg(x_,y_,x_,y_)[3]
                    epes[i][ind][2] = self.ref_BS(x_,y_)[1]
                    rmss[i][ind][3] = self.ref_reg(np.hstack((x_,np.ones((y_.shape[0],1)))),y_,np.hstack((x_,np.ones((y_.shape[0],1)))),y_)[3]
                    epes[i][ind][3] = self.ref_BS(np.hstack((x_,np.ones((y_.shape[0],1)))),y_)[1]                           

        #now we need to 'post-process' our data
        bads = []
        for i, name in enumerate(names):
            test = rmss[:,i,0]
            if 0 in test:
                bads.append(i)
        rmss = np.delete(rmss,bads,axis=1)
        epes = np.delete(epes,bads,axis=1)
        names = np.delete(names,bads)
        names=list(names)
        #bads.sort(reverse=True)
        #for b in bads:
        #    del names[b]

        return rmss,epes,names

    #Make epe matrices for 2D epe plots
    def make_epe_mat(self,model_list,ref_mat,target_data,x,key='632BS'):
        #target_data needs to be in x-y format
        epe_mat = np.zeros((len(model_list),ref_mat.shape[1]+1))
        rms_mat = np.zeros((len(model_list),ref_mat.shape[1]+1))
        for i,m in enumerate(model_list):
            print i
            co = self.ref_reg(m,x-target_data,m,x-target_data)[0]
            new_x = x-np.reshape(np.dot(m,co),(-1,1))
            mat = np.hstack((target_data,new_x))
            res = self.mod_ref_short_combo(mat,ref_mat,key=key,powers=[],mod_x=new_x)
            epe_mat[i] = res[0][:,-1]
            rms_mat[i] = res[0][:,-2]

        return epe_mat,rms_mat

    #Make a heat plot
    #rms =blue heat plot, relative rms
    #epe = blue/red heat plot, relative epe
    #epe2 = blue/red version of nvar plots
    #epe3 = new rmse/epe or w/e (in progress)
    def make_heat_plot(self,destination,data,type='rms',xticks=None,yticks=None,ylabel=None,xlabel=None,subset=[],tick_labels = [],minor_labels=True, norm_index = None,vscale=None):
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedFormatter, LinearLocator, NullFormatter
        import matplotlib
        #Actually, first we take the appropiate subset
        if subset:
            assert np.min(np.array(subset)) >=0
            assert np.max(np.array(subset)) <=data.shape[2]
            data = np.take(data,subset,axis=2)
            if len(subset) ==1:
                data = np.reshape(data,(data.shape[0],data.shape[1],1))
        if tick_labels:
            assert data.shape[2]==len(tick_labels)/2+1
        if type =='epe' or type=='rms' or type=='epe3':
            model_num = data.shape[2]-1
        if type =='epe2' or type=='epe4':
            model_num = data.shape[2]
        grids = data.shape[1]+1
        mgrids = data.shape[1]*model_num+1
        #First we need to condition our matrix
        if type =='rms' or type =='epe':
            if len(data.shape) == 3:
                ndata = np.zeros(data.shape)
                sh1 = data.shape[0]
                #base = np.take(data,[0],axis=2)
                for i in range(data.shape[2]-1):
                    #data[:,:,i+1]/=np.reshape(base,(sh1,-1))
                    ndata[:,:,i+1] = data[:,:,i+1]/data[:,:,0]
                ndata = ndata[:,:,1:]
        
        if type =='epe2':
            assert len(data.shape)==3
                #ndata = np.zeros(data.shape)
            mins = []
            sh1 = data.shape[0]
            for i in range(data.shape[0]):
                mins.append(np.min(data[i]))
            bmin=np.max(np.array(mins))
            #ndata=data/bmin
            if not norm_index:
                norm_index = (-1,0)
            ndata=data/(2*data[norm_index[0]][norm_index[1]])

        if type == 'epe3' or type=='epe4':
            sh1 = data.shape[0]
            assert len(data.shape)==3
            if type == 'epe3':
                ndata = data[:,:,1:] 
            else:
                ndata = data       
    
        ndata = np.reshape(ndata, (sh1,-1))
        ndata = np.repeat(ndata,2,axis=1)

        majorLocator = MultipleLocator(model_num)
        majorLocator2 = LinearLocator(grids)
        minorLocator = MultipleLocator(1)
        minorLocator2 = MultipleLocator(1)
        minorLocator3 = AutoMinorLocator(2)
        minorLocator4 = LinearLocator(mgrids)
        #minorFormatter = FixedFormatter(['Int','Slope','Affine'])
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.15,left=0.2)
        ax = fig.add_subplot(111)
        if type == 'rms':
            img=ax.imshow(1-ndata,cmap=cm.Blues,extent=(0,ndata.shape[1],0,ndata.shape[0]),aspect='auto',interpolation = 'none')
        elif type =='epe' or type =='epe2':
            scale = np.max(np.abs(ndata))
            if type =='epe':
                ndata = ndata/scale+ndata*(ndata-scale)*((0.5-1/scale)/(1-scale))
            elif type =='epe2':
                min = np.min(ndata)
                #val = (scale*min -min*min-(min/scale))/(scale-min)
                val = (scale-2)/(2*scale)
                #ndata = ndata/scale +(ndata-val)*(ndata-scale)*((0.5-1/scale)/((1-scale)*(1-min)))
                #a = -(((min/scale)+val*(1-min/(scale-1)+1/(scale-1)))/(min*min-1-((1-scale*scale)/(scale-1))+((min-scale*scale*min)/(scale-1))))
                #b = (a*(1-scale*scale)-val)/(scale-1)
                #c = val-a-b
                #ndata = ndata/scale +c +a*ndata*ndata+b*ndata
                m1 = (scale-1)/(scale-.5)
                m2 = (min-1)/(min-.5)
                for i in range(ndata.shape[0]):
                    for j in range(ndata.shape[1]):
                        if ndata[i][j] > 0.5:
                            ndata[i][j]/=(m1*ndata[i][j] - m1*scale+scale)
                        elif ndata[i][j]<0.5:
                            ndata[i][j] = 1 - ndata[i][j]/((m2*ndata[i][j] - m2*min+min))
            #data = 1-data
            #scale = np.max(np.abs(data))
            #data = data*(0.5/scale)+0.5
            #data = 1-data
            img=ax.imshow(ndata,norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0),cmap=cm.seismic,extent=(0,ndata.shape[1],0,ndata.shape[0]),aspect='auto',interpolation='none')
        elif type =='epe3' or type == 'epe4':
            if vscale:
                img = ax.imshow(ndata,norm=matplotlib.colors.Normalize(vmin=vscale[0],vmax=vscale[1]),cmap=cm.Oranges,extent=(0,ndata.shape[1],0,ndata.shape[0]),aspect='auto',interpolation='none')
            else:
                img = ax.imshow(ndata,cmap=cm.Reds,extent=(0,ndata.shape[1],0,ndata.shape[0]),aspect='auto',interpolation='none')
        else:
            raise NameError('No such type')
        if type =='rms':
            fac = np.max(1-ndata)
            cbar = fig.colorbar(img,ticks=[0,0.1*fac,0.2*fac,0.3*fac,0.4*fac,0.5*fac,0.6*fac,0.7*fac,0.8*fac,0.9*fac,np.max(1-ndata)])
            cbar.ax.set_yticklabels(['0.0','0.1','0.2','0.3', '0.4','0.5','0.6','0.7','0.8','0.9', '1.0'],size=9)
            cbar.ax.set_ylabel(r'$\rho$',rotation=0,size=14)
        elif type=='epe' or type =='epe2':
            #cbar = fig.colorbar(img,ticks=[0,0.5,1])
            #cbar.ax.set_yticklabels(['Helpful', 'No change', 'Harmful'],size=9)
            cbar = fig.colorbar(img,ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            cbar.ax.set_yticklabels(['0.0','0.1','0.2','0.3', '0.4','0.5','0.6','0.7','0.8','0.9', '1.0'],size=9)
            cbar.ax.set_ylabel(r'$\delta$',rotation=0,size=14)
            if type =='epe2':
                cbar.ax.set_ylabel(r'$\lambda$',rotation=0,size=14)
        elif type =='epe3' or type=='epe4':
            if vscale:
                min = vscale[0]
                max = vscale[1]
            else:
                min = np.min(ndata)
                max = np.max(ndata)
            fac = max-min
            cbar = fig.colorbar(img,ticks=[min,min+0.1*fac,min+0.2*fac,min+0.3*fac,min+0.4*fac,min+0.5*fac,min+0.6*fac,min+0.7*fac,min+0.8*fac,min+0.9*fac,max])
            cbar.ax.set_ylabel(r'$\kappa$',rotation=0,size=14)
        #if type =='rms' or type=='epe':
        ax2=ax.twiny()
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(minorLocator2)
        ax.yaxis.set_minor_locator(minorLocator3)
        ax.yaxis.set_major_formatter(NullFormatter())
        ax2.xaxis.set_major_locator(majorLocator2)
        ax2.xaxis.set_minor_locator(minorLocator4)
        ax.xaxis.set_tick_params(which='major',bottom='off')
        ax.xaxis.set_tick_params(which='minor',bottom='off')
        ax2.xaxis.set_tick_params(which='major',labeltop='off',labelbottom='off')
        ax2.xaxis.set_tick_params(which='minor',labeltop='off',labelbottom='off')
        ax.grid(b=True,which='major',linewidth=2.5,linestyle='-',axis='y')
        ax2.grid(b=True,which='major',linewidth=2.5,linestyle='-',axis='x')
        ax2.grid(b=True,which='minor',linewidth=1,linestyle='-')
        ax2.yaxis.grid(b=False,which='minor')
        #ax2= ax.twiny()
        ax.xaxis.set_tick_params(which='minor',labeltop='on',labelbottom='off',labelsize='6')
        #ax2.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_minor_formatter(minorFormatter)
        #if type =='epe2':
        #    ax.xaxis.set_major_locator(majorLocator)
        #    ax.yaxis.set_major_locator(minorLocator2)
        #    ax.yaxis.set_minor_locator(minorlocator3)
        #    ax.yaxis.set_major_formatter(NullFormatter())
        if xticks is not None:
            if type =='epe' or type=='rms' or type=='epe3':
                ax.set_xticklabels(xticks,size='x-small',rotation='vertical')
            elif type=='epe2' or type=='epe4':
                ax.set_xticklabels(xticks,size='small')
        if yticks is not None:
            for tick in ax.yaxis.get_minor_ticks():
                tick.tick1line.set_markersize(0)
                tick.tick2line.set_markersize(0)
            ax.yaxis.set_ticklabels(yticks,size='x-small',minor=True)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if minor_labels:
            if not subset:
                if not tick_labels and not type=='epe2' and not type=='epe4':
                    ax.xaxis.set_ticklabels(['','Int','','Slope','','Affine']*(data.shape[1]/(2*model_num)),minor=True,rotation=90)
            if tick_labels:
                ax.xaxis.set_ticklabels(['']+(tick_labels*(ndata.shape[1]/model_num)),minor=True,rotation=90)
        #minor_labels = [tick.label1 for tick in ax.xaxis.get_minor_ticks()]
        #print minor_labels
        #for label in minor_labels:
        #    label.set_rotation(120)
        plt.savefig(destination)                      

    #Make a heat plot for abs rms (for now...)
    def make_mod_heat_plot(self,destination,data,type='rms',xticks=None,yticks=None,ylabel=None,xlabel=None):
        from matplotlib.ticker import MultipleLocator, FixedFormatter
        import matplotlib
        #First we need to condition our matrix
        if len(data.shape) == 3:
            ndata = np.zeros((data.shape[0]*2,data.shape[1]))
            for i in range(data.shape[0]):
                ndata[2*i] = np.reshape(data[i,:,0],(1,-1))
                ndata[2*i+1] = np.reshape(data[i,:,-1],(1,-1))
                ndata[2*i] /= np.max(data[i,:,0])
                ndata[2*i+1] /= np.max(data[i,:,0])
        xmajorLocator = MultipleLocator(1)
        minorLocator = MultipleLocator(0.5)
        ymajorLocator = MultipleLocator(1)
        #minorFormatter = FixedFormatter(['Int','Slope','Affine'])
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.15,left=0.15)
        ax = fig.add_subplot(111)
        if type == 'rms':
            img=ax.imshow(ndata,cmap=cm.Blues,extent=(0,data.shape[1],0,data.shape[0]),aspect='auto',interpolation = 'none')
        else:
            raise NameError('No such type')
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.grid(b=True,which='major',linewidth=2.5,linestyle='-')
        ax.grid(b=True,which='minor',linewidth=1,linestyle='-')
        ax.yaxis.set_tick_params(which='minor',labelright='on',labelleft='off',labelsize='6')
        if xticks is not None:
            ax.set_xticklabels(xticks,size='xx-small',rotation='vertical')
        if yticks is not None:
            ax.set_yticklabels(yticks,size='xx-small')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_ticklabels(['Affine','Original']*(ndata.shape[0]/2),minor=True)
        if type =='rms':
            cbar = fig.colorbar(img,ticks=[np.min(ndata),np.max(ndata)],pad=0.08)
            cbar.ax.set_yticklabels(['Low RMSE', 'High RMSE'],size=9)
        plt.savefig(destination)                      
    
    #Make a heat plot
    def make_mod2_heat_plot(self,destination,data,type='rms',norm_index = 1,norm_label='PBE',xticks=None,yticks=None,ylabel=None,xlabel=None,subset=[],tick_labels=[]):
        from matplotlib.ticker import LinearLocator,AutoMinorLocator, MultipleLocator, FixedFormatter, NullFormatter
        import matplotlib
        from matplotlib.colors import LinearSegmentedColormap
        #Actually, we first take the appropriate subset
        if subset:
            if subset:
                assert np.min(np.array(subset)) >=0
                assert np.max(np.array(subset)) <=data.shape[2]
                data = np.take(data,subset,axis=2)
                if len(subset) ==1:
                    data = np.reshape(data,(data.shape[0],data.shape[1],1))
        model_num = data.shape[2]
        grids = data.shape[1]+1
        mgrids = data.shape[1]*data.shape[2]+1
        if tick_labels:
            assert len(tick_labels)==2*model_num
        #First we need to condition our matrix
        if len(data.shape) == 3:
            ndata = np.zeros(data.shape)
            sh1 = data.shape[0]
            for i in range(sh1):
                ndata[i,:,:] =data[i,:,:]/data[i,norm_index,0]
                ndata[i,:,:] = np.log(ndata[i,:,:])
                maxi = np.max(ndata[i,:,:])
                mini = np.min(ndata[i,:,:])
                scale = maxi-mini
                ndata[i,:,:] /= scale
                ndata[i,:,:] *= 0.5/np.max(np.abs(ndata[i,:,:]))
                ndata[i,:,:] += 0.5
                #nmin = np.min(ndata[i,:,:])
                #a = (0.5-nmin)/(nmin*(nmin-1))
                #ndata +=np.abs(nmin)
                #ndata[i,:,:] += ndata[i,:,:]*(ndata[i,:,:]-1)*a
                #ndata[i,:,:] *=-1
                #ndata[i,:,:] = ndata[i,:,:]/scale+(ndata[i,:,:]-mini)*(ndata[i,:,:]-scale)*((0.5-1/scale)/((1-mini)*(1-scale)))
            data = np.reshape(ndata, (sh1,-1))
            data = np.repeat(data,2,axis=1)
        
        majorLocator = MultipleLocator(model_num)
        majorLocator2 = LinearLocator(grids)
        minorLocator = MultipleLocator(1)
        minorLocator2 = MultipleLocator(1)
        minorLocator3 = AutoMinorLocator(2)
        minorLocator4 = LinearLocator(mgrids)
        #minorFormatter = FixedFormatter(['Int','Slope','Affine'])
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.15,left=0.19)
        ax = fig.add_subplot(111)
        
        #Make our custom color map
        cdict = {'red':     ((0.0,0.0,0.0),
                            (0.25,0.0,0.0),
                            (0.5,1.0,1.0),
                            (0.75,1.0,1.0),
                            (1.0,1.0,1.0)),

                 'green':   ((0.0,0.0,0.0),
                            (0.25,0.75,0.75),
                            (0.5,1.0,1.0),
                            (0.75,1.0,1.0),
                            (1.0,0.0,0.0)),

                'blue':     ((0.0,1.0,1.0),
                            (0.25,0.25,0.25),
                            (0.5,1.0,1.0),
                            (0.75,0.0,0.0),
                            (1.0,0.0,0.0))
                }
        rywgb = LinearSegmentedColormap('Spectral2',cdict)
        #ax2=ax.twiny()                   
        if type == 'rms':
            #scale = np.max(np.abs(data))
            #data = data/scale+data*(data-scale)*((0.5-1/scale)/(1-scale))
            #data = 1-data
            #scale = np.max(np.abs(data))
            #data = data*(0.5/scale)+0.5
            #data = 1-data
            #img=ax2.imshow(data,norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0),cmap=rywgb,extent=(0,data.shape[1],0,data.shape[0]),aspect='auto',interpolation='none')
            img=ax.imshow(data,norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0),cmap=rywgb,extent=(0,data.shape[1],0,data.shape[0]),aspect='auto',interpolation='none')
        else:
            raise NameError('No such type')
        if type =='rms':
            #cbar = fig.colorbar(img,ticks=[0,0.5,1])
            #cbar.ax.set_yticklabels(['Lower RMSE',norm_label, 'Higher RMSE'],size=9)
            cbar = fig.colorbar(img,ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            cbar.ax.set_yticklabels(['-1.0','-0.8','-0.6','-0.4', '-0.2','0.0','0.2','0.4','0.6','0.8', '1.0'],size=9)
            cbar.ax.set_ylabel(r'$\sigma$',rotation=0,size=14)
        ax2=ax.twiny()
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(minorLocator2)
        ax.yaxis.set_minor_locator(minorLocator3)
        ax.yaxis.set_major_formatter(NullFormatter())
        ax2.xaxis.set_major_locator(majorLocator2)
        ax2.xaxis.set_minor_locator(minorLocator4)
        #ax2.yaxis.set_major_locator(minorLocator2)
        #ax2.yaxis.set_minor_locator(minorLocator3)
        #ax2.yaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_tick_params(which='major',bottom='off')
        ax.xaxis.set_tick_params(which='minor',bottom='off')
        ax2.xaxis.set_tick_params(which='major',labeltop='off',labelbottom='off')
        ax2.xaxis.set_tick_params(which='minor',labeltop='off',labelbottom='off')
        ax.grid(b=True,which='major',linewidth=2.5,linestyle='-',axis='y')
        ax2.grid(b=True,which='major',linewidth=2.5,linestyle='-',axis='x')
        ax2.grid(b=True,which='minor',linewidth=1,linestyle='-')
        ax.yaxis.grid(b=False,which='minor')
        #ax2= ax.twiny()
        ax.xaxis.set_tick_params(which='minor',labeltop='on',labelbottom='off',labelsize='6')
        #ax2.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_minor_formatter(minorFormatter)
        if xticks is not None:
            ax.set_xticklabels(xticks,size='x-small',rotation='vertical')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if not subset:
            if not tick_labels:
                ax.xaxis.set_ticklabels(['','Original','','Affine']*(data.shape[1]/model_num),minor=True,rotation=90)
        if tick_labels:
            ax.xaxis.set_ticklabels(['']+(tick_labels*(data.shape[1]/(2*model_num))),minor=True,rotation=90)
            #ax2.xaxis.set_ticklabels(tick_labels*(data.shape[1]/model_num),minor=True,rotation=90)
        #minor_labels = [tick.label1 for tick in ax.xaxis.get_minor_ticks()]
        #print minor_labels
        #for label in minor_labels:
        #    label.set_rotation(120)
        #if type =='rms':
            #cbar = fig.colorbar(img,ticks=[0,0.5,1])
            #cbar.ax.set_yticklabels(['Lower RMSE',norm_label, 'Higher RMSE'],size=9)
        #    cbar = fig.colorbar(img,ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        #    cbar.ax.set_yticklabels(['-1.0','-0.8','-0.6','-0.4', '-0.2','0.0','0.2','0.4','0.6','0.8', '1.0'],size=9)
        #    cbar.ax.set_ylabel(r'$\sigma$',rotation=0,size=14)
        if yticks is not None:
            for tick in ax.yaxis.get_minor_ticks():
                tick.tick1line.set_markersize(0)
                tick.tick2line.set_markersize(0)
            ax.yaxis.set_ticklabels(yticks,size='xx-small',minor=True)
            #ax2.yaxis.set_ticklabels(yticks,size='xx-small',minor=True)
        #img=ax2.imshow(data,norm=matplotlib.colors.Normalize(vmin=0.0,vmax=1.0),cmap=rywgb,extent=(0,data.shape[1],0,data.shape[0]),aspect='auto',interpolation='none')
        #ax2.xaxis.set_major_locator(majorLocator2)
        #ax2.grid(b=True,which='major',linewidth=2.5,linestyle='-',axis='x')
        plt.savefig(destination)                      
    
    #All of the messy cross-validation BS. Taken w/ replacement
    def error_test(self,destination,x,y,limit=1000):
        slopes = np.zeros((limit,))
        ints = np.zeros((limit,))
        rel_rmss = np.zeros((limit,))
        rel_trmss = np.zeros((limit,))
        err_mat = np.zeros((limit,6))
        #Get the original x,y errors
        tm,tb,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(y,x)
        base_rms = nrms
        base_trms = trms
        tindexes = np.array([])
        #print x,y
        #run the CV
        for i in range(limit):
            #make sure we have test data left
            while not tindexes.any():
                indexes = np.floor(x.shape[0]*np.random.rand((x.shape[0])))
                rand_x = np.zeros((x.shape[0],))
                rand_y = np.zeros((y.shape[0],))
                #Get random x,y matrices
                for j in range(x.shape[0]):
                    #print j, indexes.shape, x.shape[0], rand_x.shape, rand_y.shape
                    rand_x[j] = x[indexes[j]]
                    rand_y[j] = y[indexes[j]]
                tindexes = []
                #make the test data matrices
                for j in range(x.shape[0]):
                    if j not in indexes:
                        tindexes.append(j)
                tindexes = np.array(tindexes)
                test_x = np.zeros((tindexes.shape[0],))
                test_y = np.zeros((tindexes.shape[0],))
                for j in range(tindexes.shape[0]):
                    test_x[j] = x[tindexes[j]]
                    test_y[j] = y[tindexes[j]]
                #print rand_x,rand_y
                #print test_x,test_y
            #Get regression data for our random stuff
            m,b,errs,nerrs,plty,mae,nmae,bsmse,nmse,rms,bsnrms,std,nstd, tmae,tmse,bstrms,tstd = self.regression(rand_y,rand_x)
            slopes[i] = m
            ints[i] = b
            pltx = np.reshape(np.poly1d((m,b))(y),(-1,))
            errors = pltx-x
            terrors = y -bsmse-x
            mae,mse,rms,std = self.get_errors(errors)
            tmae,tmse,trms,tstd = self.get_errors(terrors)
            err_mat[i][0] = rms
            err_mat[i][1] = trms
            rel_rmss[i] = (rms-base_rms)*100/base_rms
            rel_trmss[i] = (trms - base_trms)*100/base_trms
            
            #bootstrap  
            pltx = np.reshape(np.poly1d((m,b))(test_y),(-1,))
            errors = pltx-test_x
            terrors =test_y -bsmse-test_x
            mae,mse,rms,std = self.get_errors(errors)
            tmae,tmse,trms,tstd = self.get_errors(terrors)
            err_mat[i][2] = np.exp(-1) * bsnrms + (1-np.exp(-1)) * rms
            err_mat[i][3] = np.exp(-1) *bstrms + (1-np.exp(-1)) * trms
            err_mat[i][4] = base_rms
            err_mat[i][5] = base_trms        

        #make output stuff
        m_mean = np.mean(slopes)
        m_std = np.std(slopes)
        b_mean = np.mean(ints)
        b_std = np.std(ints)
        rel_rms_mean = np.mean(rel_rmss)
        rel_trms_mean = np.mean(rel_trmss)
        mat = np.concatenate((slopes,ints), axis=1)
        mat = np.concatenate((mat,rel_rmss),axis=1)
        mat = np.concatenate((mat,rel_trmss),axis=1)
        means = np.mean(err_mat,axis=0)
        mat = np.concatenate((mat,means),axis=1)
        mat = np.concatenate((mat,means),axis=1)
        
        #make a plot of the rms of the actual/bootstrap error for the two models
        gmat = np.array([['Actual',base_rms,base_trms],['Bootstrap',means[2],means[3]]],dtype=object)
        self.error_bar_plot(destination+'_bootstrap_plot',gmat,legend_labels=['Affine Corrected','Translation Corrected'], ylabel='RMS')

        return m_mean, m_std, b_mean, b_std, tm, tb, rel_rms_mean, rel_trms_mean,means, mat
    #should be deprecated lololol w/o replacement/half the size used. No bootstrap
    def error_test2(self,x,y,limit=1000):
        slopes = np.zeros((limit,))
        ints = np.zeros((limit,))
        rel_rmss = np.zeros((limit,))
        rel_trmss = np.zeros((limit,))
        tm,tb,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(y,x)
        base_rms = nrms
        base_trms = trms
        #key difference here
        size = int(x.shape[0]/2)
        for i in range(limit):
            indexes = np.random.permutation(np.arange(x.shape[0]))[:size]
            rand_x = np.zeros((size,))
            rand_y = np.zeros((size,))
            for j in range(size):
                #print j, indexes.shape, x.shape[0], rand_x.shape, rand_y.shape
                rand_x[j] = x[indexes[j]]
                rand_y[j] = y[indexes[j]]
            m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(rand_y,rand_x)
            slopes[i] = m
            ints[i] = b
            pltx = np.reshape(np.poly1d((m,b))(y),(-1,))
            errors = pltx-x
            terrors = y-mse-x
            mae,mse,rms,std = self.get_errors(errors)
            tmae,tmse,trms,tstd = self.get_errors(terrors)
            rel_rmss[i] = (rms-base_rms)*100/base_rms
            rel_trmss[i] = (trms-base_trms)*100/base_trms
        
        m_mean = np.mean(slopes)
        m_std = np.std(slopes)
        b_mean = np.mean(ints)
        b_std = np.std(ints)
        rms_mean = np.mean(rel_rmss)
        trms_mean = np.mean(rel_trmss)
        mat = np.concatenate((slopes,ints), axis=1)
        mat = np.concatenate((mat,rel_rmss),axis=1)
        mat = np.concatenate((mat,rel_trmss),axis=1)
        return m_mean, m_std, b_mean, b_std, tm, tb, rms_mean,trms_mean, mat
                
    #Make two plots. The first is a two-axed slope/intercept 
    #error plot. The second is just an error bar graph
    def cross_val_bar(self,destination1, destination2, mat,width =1, offset =3, xlabels=None,ylabels=None,bar_labels=None, leg_labels = ['Affine Corrected', 'Bias Corrected']):
        nclusters = mat.shape[0]
        nfuncs = mat.shape[1]
        #Stupid but necessary hard coding
        #assert nclusters == 3 
        cvalues = np.arange(nfuncs)/float(nfuncs)
        if bar_labels is not None:
            assert len(bar_labels)==nfuncs
        #First make the double bar plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #ax2 = ax1.twinx()
        for i in range(nfuncs):
            ax1.bar(i*width,mat[0][i],width,color=cm.jet(cvalues[i]), label= bar_labels[i])
        full_ticks = np.concatenate((np.arange(0,nfuncs*width,width)+width/2., np.arange(0,nfuncs*width,width)+(width/2.)+nfuncs*width+offset))
        #print full_ticks
        ax1.set_xticks(full_ticks)
        ax2 = ax1.twinx()
        for i in range(nfuncs): 
           ax2.bar((i+nfuncs)*width + offset,mat[1][i],width, color = cm.jet(cvalues[i]))
        ax1.set_xticks(full_ticks)
        ax2.set_xticks(full_ticks)
        #for i in range(nfuncs):
        #    left = np.zeros((nclusters,))
        #    height = np.zeros((nclusters,))
        #    for j in range(nclusters):    
        #        left[j] = i*width + j*(nfuncs*width+offset)
        #        height[j] = mat[j][i]
        #    if bar_labels is not None:
        #        ax.bar(left,height,width,color=cm.jet(cvalues[i]), label = bar_labels[i])
        #    else:
        #        ax.bar(leftheight,width,color=cm.jet(cvalues[i]))
        if ylabels is not None:
            ax1.set_ylabel(ylabels[0])
            ax2.set_ylabel(ylabels[1])
        if xlabels is not None:
            ax1.set_xlabel(xlabels[0], size='small')
            ax2.set_xlabel(xlabels[1], size='small')
        if bar_labels is not None:
            ax1.set_xticklabels(np.concatenate((bar_labels,bar_labels)),size='xx-small', rotation='vertical')
        #    ax2.set_xticklabels(bar_labels,size='xx-small', rotation='vertical')
        #ax1.set_xticks(np.arange(0,nclusters*(offset+nfuncs*width),(offset+nfuncs*width))+(nfuncs/2.)*width)
        #ax1.set_xticks(np.arange(0, nfuncs*width, width) +(nfuncs/2.)*width)
        #ax2.set_xticks(np.arange(0, nfuncs*width,width)+(nfuncs/2.)*width + nfuncs*width+offset)
        fontp = FontProperties()
        fontp.set_size('xx-small')
        #ax1.legend(loc='best',prop=fontp)
        fig.savefig(destination1)

        #make the RMS plot
        max = mat.shape[0] 
        errors = np.take(mat, range(2,max),axis=0)
        #print bar_labels.shape, errors.shape
        errors = np.concatenate((np.reshape(bar_labels,(1,bar_labels.shape[0])),errors), axis=0)
        print errors.shape
        #print errors.T
        self.error_bar_plot(destination2,errors.T,legend_labels = leg_labels,ylabel='RMS')

        #fig2 = plt.figure()
        #ax = fig2.add_subplot(111)
        #for i in range(nfuncs):
        #    ax.bar(i*width,mat[2][i],width,color=cm.jet(cvalues[i]),label=bar_labels[i])
        #if ylabels is not None:
        #    ax.set_ylabel(ylabels[2])
        #if xlabels is not None: 
        #    ax.set_xlabel(xlabels[-1])
        #if bar_labels is not None:
        #    ax.set_xticklabels(bar_labels, size = 'x-small',rotation='vertical')
        #ax.set_xticks(np.arange(0,nfuncs*width, width)+width/2.)
        #ax.legend(loc='best')
        #fig2.savefig(destination2)

    def cross_val_line(self,destination,mat,stds,xlabel=None,ylabels=None, tick_labels=None):
        assert mat.shape[0] == 2
        assert stds.shape[0] == 2
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = np.arange(mat.shape[1])
        ax1.errorbar(x,mat[0], yerr=stds[0], fmt='co-')
        if xlabel:
            ax1.set_xlabel(xlabel)
        if ylabels is not None:
            ax1.set_ylabel(ylabels[0])
        for t1 in ax1.get_yticklabels():
            t1.set_color('c')

        ax2 = ax1.twinx()
        ax2.errorbar(x,mat[1],yerr=stds[1], fmt='mo-')
        if ylabels is not None:
            ax2.set_ylabel(ylabels[1])
        for t1 in ax2.get_yticklabels():
            t1.set_color('m')

        ax1.set_xticks(np.arange(mat.shape[1]))
        if tick_labels is not None:
            ax1.set_xticklabels(tick_labels, size='x-small', rotation='vertical')
        
        fig.savefig(destination)
    #runs both error test and fitting/graphing
    def full_run(self, filename, parent='.', limit=1000):
        #should change this. get_data should take fname or it is set by a new method
        self.init_file(filename)
        headings, data = self.get_data()
        #for i in range(len(headings)-1):
        #WTF is this loop? Oh was to run all permutations. Hard-coded not to
        for i in range(1):
            #set shit up
            error_mat = np.zeros((data.shape[1]-(1+i),13), dtype=object)
            m_means = np.zeros((data.shape[1]-(1+i),2))
            m_stds = np.zeros((data.shape[1]-(i+1),))
            b_means = np.zeros((data.shape[1]-(1+i),2))
            b_stds = np.zeros((data.shape[1]-(i+1),))
            rel_rmss = np.zeros((data.shape[1]-(i+1),))
            rel_trmss = np.zeros((data.shape[1]-(i+1),))
            meanss = np.zeros((data.shape[1]-(i+1),6))
            
            #is this for error_test2? Garbage
            m2_means = np.zeros((data.shape[1]-(1+i),2))
            m2_stds = np.zeros((data.shape[1]-(i+1),))
            b2_means = np.zeros((data.shape[1]-(1+i),2))
            b2_stds = np.zeros((data.shape[1]-(i+1),))
            rel2_rmss = np.zeros((data.shape[1]-(i+1),))
            rel2_trmss = np.zeros((data.shape[1]-(i+1),))
            labs = np.zeros((data.shape[1]-(i+1),), dtype=object)
            #when doing x v. y plots, this loops the y (x = exp.)
            for j in range(len(headings))[i+1:]:
                xlab = headings[i]
                ylab = headings[j]
                labs[j-(i+1)] = ylab
                gname = self.shortname + '_' + xlab + '_' + ylab
                #make dirs
                if i == 0:
                    self.make_dir(parent + '/' + self.shortname.encode('string_escape') + '_vs_exp')
                    destination = parent + '/' + self.shortname.encode('string_escape') + '_vs_exp'
                elif i == 1:
                    self.make_dir(parent + '/' + self.shortname.encode('string_escape') + '_vs_funcs')
                    destination = parent + '/' + self.shortname.encode('string_escape') + '_vs_funcs'
                
                #organize data
                x = np.take(data, [i], axis=1)
                x = np.reshape(x,(-1,))
                y = np.take(data, [j], axis=1)
                y = np.reshape(y, (-1,))

                #get errors/fill mats
                m_mean,m_std, b_mean, b_std, m, b, rel_rms,  rel_trms,means,mat = self.error_test(destination+'/'+gname,x,y, limit=limit)
                #m_mean,m_std, b_mean, b_std, m, b, rel_rms,  mat = self.error_test2(x,y, limit=limit)
                m_means[j-(i+1)][0] = m_mean
                m_means[j-(i+1)][1] = m
                m_stds[j-(i+1)] = m_std
                b_means[j-(i+1)][0] = b_mean
                b_means[j-(i+1)][1] = b
                m2_means[j-(i+1)][0] = m2_mean
                m2_means[j-(i+1)][1] = m2
                m2_stds[j-(i+1)] = m2_std
                b2_means[j-(i+1)][0] = b2_mean
                b2_means[j-(i+1)][1] = b2
                b2_stds[j-(i+1)] = b2_std
                rel2_rmss[j-(i+1)] = rel2_rms
                np.savez(destination + '/' + gname + '_cross_val2', Results = mat)
                
                m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(y,x)
                self.reg_plot(destination+'/'+gname,y,x,xlabel=ylab, ylabel=xlab)
                self.error_plot(destination+'/'+'error_' +gname, errs,nerrs,errs.shape[0]/2,'Original Error', 'Fitted Error')
                err_list = [mae,nmae,tmae,mse,nmse,tmse,rms,nrms,trms,std,nstd,tstd]
                error_mat[j-(i+1)][0] = ylab
                for k, err in enumerate(err_list):
                    error_mat[j-(i+1)][k+1] = err
                
                #np.savez(destination+'/'+gname,x=x, y=y, new_y=plty, slope=m, intercept=b, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
                np.savez(destination+'/'+gname,x=y, y=x, new_y=plty, slope=m, intercept=b, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
            np.savez(destination+'/'+'Error_file_for_vs_' + headings[i],Errors = error_mat)
            #print np.take(error_mat,[0,7,9,8],axis=1).shape, (np.take(error_mat,[8],axis=1)*rel_rmss).shape, rel_rmss.shape, np.take(error_mat,[8],axis=1).shape
            self.error_bar_plot(destination + '/' + 'Error_plot_rms_for_vs_' + headings[i], np.take(error_mat,[0,7,9,8],axis=1),legend_labels = ['Standard','Translation Corrected', 'Affine Corrected'], ylabel = 'RMSE', yerr=np.reshape(np.take(error_mat,[8],axis=1),(rel_rmss.shape[0],))*rel_rmss/100.)
            self.error_bar_plot(destination + '/' + 'Error_plot_mae_for_vs_' + headings[i], np.take(error_mat,[0,1,3,2],axis=1),legend_labels = ['Standard','Translation Corrected', 'Affine Corrected'], ylabel = 'MAE')

            np.savez(destination + '/' + 'Cross_val_results', Slope_Means = m_means, Slope_Stds = m_stds, Int_Means = b_means, Int_Stds = b_stds, Relative_RMS = rel_rmss, Labels = labs, Errors = meanss)

            cv_mat = np.vstack((np.vstack((np.vstack((m_stds,b_stds)),rel_rmss)),rel_trmss))
            self.cross_val_bar(destination +'/'+'Cross_val_dblbar',destination+'/'+ 'Cross_val_rmsbar', cv_mat, xlabels=['Slope', 'Intercept', 'RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs)
            bs_mat = np.vstack((np.vstack((np.vstack((m_stds,b_stds)),np.reshape(np.take(meanss,[2],axis=1),rel_rmss.shape))),np.reshape(np.take(meanss,[3],axis=1),rel_trmss.shape)))
            self.cross_val_bar(destination +'/'+'Cross_val_dblbar',destination+'/'+ 'Cross_val_bsbar', bs_mat, xlabels=['Slope', 'Intercept', 'BS RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs)
            cvl_mat = np.vstack((np.reshape(np.take(m_means,[0],axis=1),(-1,)), np.reshape(np.take(b_means,[0],axis=1),(-1,))))
            cvl_std = np.vstack((m_stds,b_stds))
            self.cross_val_line(destination+'/'+'Cross_val_line',cvl_mat,cvl_std, xlabel = 'Functional', ylabels = ['Slope', 'Intercept'], tick_labels=labs)
            
            np.savez(destination + '/' + 'Cross_val2_results', Slope_Means = m2_means, Slope_Stds = m2_stds, Int_Means = b2_means, Int_Stds = b2_stds, Relative_RMS = rel2_rmss, Labels = labs)

            cv_mat = np.vstack((np.vstack((np.vstack((m2_stds,b2_stds)),rel2_rmss)),rel_trmss))
            self.cross_val_bar(destination +'/'+'Cross_val2_dblbar',destination+'/'+ 'Cross_val2_rmsbar', cv_mat, xlabels=['Slope', 'Intercept', 'RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs)
            cvl_mat = np.vstack((np.reshape(np.take(m2_means,[0],axis=1),(-1,)), np.reshape(np.take(b2_means,[0],axis=1),(-1,))))
            cvl_std = np.vstack((m2_stds,b2_stds))
            self.cross_val_line(destination+'/'+'Cross_val2_line',cvl_mat,cvl_std, xlabel = 'Functional', ylabels = ['Slope', 'Intercept'], tick_labels=labs)
    
    def run_error_test(self, filename,parent='.', limit=1000):
        self.init_file(filename)
        headings, data = self.get_data()
        #for i in range(len(headings)-1):
        for i in range(1):
            m_means = np.zeros((data.shape[1]-(1+i),2))
            m_stds = np.zeros((data.shape[1]-(i+1),))
            b_means = np.zeros((data.shape[1]-(1+i),2))
            b_stds = np.zeros((data.shape[1]-(i+1),))
            rel_rmss = np.zeros((data.shape[1]-(i+1),))
            rel_trmss = np.zeros((data.shape[1]-(i+1),))
            meanss = np.zeros((data.shape[1]-(i+1),6))           
 
            m2_means = np.zeros((data.shape[1]-(1+i),2))
            m2_stds = np.zeros((data.shape[1]-(i+1),))
            b2_means = np.zeros((data.shape[1]-(1+i),2))
            b2_stds = np.zeros((data.shape[1]-(i+1),))
            rel2_rmss = np.zeros((data.shape[1]-(i+1),))
            rel2_trmss = np.zeros((data.shape[1]-(i+1),))
            labs = np.zeros((data.shape[1]-(i+1),), dtype=object)
            for j in range(len(headings))[i+1:]:
                xlab = headings[i]
                ylab = headings[j]
                labs[j-(i+1)] = ylab
                gname = self.shortname + '_' + xlab + '_' + ylab
                if i == 0:
                    self.make_dir(parent + '/' + self.shortname.encode('string_escape') + '_vs_exp')
                    destination = parent + '/' + self.shortname.encode('string_escape') + '_vs_exp'
                elif i == 1:
                    self.make_dir(parent + '/' + self.shortname.encode('string_escape') + '_vs_funcs')
                    destination = parent + '/' + self.shortname.encode('string_escape') + '_vs_funcs'
                
                x = np.take(data, [i], axis=1)
                x = np.reshape(x,(-1,))
                y = np.take(data, [j], axis=1)
                y = np.reshape(y, (-1,))

                m_mean,m_std, b_mean, b_std, m, b, rel_rms,  rel_trms,means,mat = self.error_test(destination+'/'+gname,x,y, limit=limit)
                #m_mean,m_std, b_mean, b_std, m, b, rel_rms,  mat = self.error_test2(x,y, limit=limit)
                m_means[j-(i+1)][0] = m_mean
                m_means[j-(i+1)][1] = m
                m_stds[j-(i+1)] = m_std
                b_means[j-(i+1)][0] = b_mean
                b_means[j-(i+1)][1] = b
                b_stds[j-(i+1)] = b_std
                rel_rmss[j-(i+1)] = rel_rms
                rel_trmss[j-(i+1)] = rel_trms
                meanss[j-(i+1)] = means
                np.savez(destination + '/' + gname + '_cross_val', Results = mat)
                
                #m_mean,m_std, b_mean, b_std, m, b, rel_rms,  mat = self.error_test(x,y, limit=limit)
                m2_mean,m2_std, b2_mean, b2_std, m2, b2, rel2_rms, rel2_trms, mat2 = self.error_test2(x,y, limit=limit)
                m2_means[j-(i+1)][0] = m2_mean
                m2_means[j-(i+1)][1] = m2
                m2_stds[j-(i+1)] = m2_std
                b2_means[j-(i+1)][0] = b2_mean
                b2_means[j-(i+1)][1] = b2
                b2_stds[j-(i+1)] = b2_std
                rel2_rmss[j-(i+1)] = rel2_rms
                np.savez(destination + '/' + gname + '_cross_val2', Results = mat)
            np.savez(destination + '/' + 'Cross_val_results', Slope_Means = m_means, Slope_Stds = m_stds, Int_Means = b_means, Int_Stds = b_stds, Relative_RMS = rel_rmss, Labels = labs, Errors = meanss)

            cv_mat = np.vstack((np.vstack((np.vstack((m_stds,b_stds)),rel_rmss)),rel_trmss))
            self.cross_val_bar(destination +'/'+'Cross_val_dblbar',destination+'/'+ 'Cross_val_rmsbar', cv_mat, xlabels=['Slope', 'Intercept', 'RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs)
            bs_mat = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((m_stds,b_stds)),np.reshape(np.take(meanss,[4],axis=1),rel_rmss.shape))),np.reshape(np.take(meanss,[5],axis=1),rel_rmss.shape))),np.reshape(np.take(meanss,[2],axis=1),rel_rmss.shape))),np.reshape(np.take(meanss,[3],axis=1),rel_trmss.shape)))
            self.cross_val_bar(destination +'/'+'Cross_val_dblbar',destination+'/'+ 'Cross_val_bsbar', bs_mat, xlabels=['Slope', 'Intercept', 'BS RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs, leg_labels = ['Actual Affine Corrected', 'Actual Bias Corrected', 'Bootstrap Affine Corrected', 'Bootstrap Bias Corrected'])
            cvl_mat = np.vstack((np.reshape(np.take(m_means,[0],axis=1),(-1,)), np.reshape(np.take(b_means,[0],axis=1),(-1,))))
            cvl_std = np.vstack((m_stds,b_stds))
            self.cross_val_line(destination+'/'+'Cross_val_line',cvl_mat,cvl_std, xlabel = 'Functional', ylabels = ['Slope', 'Intercept'], tick_labels=labs)
            
            np.savez(destination + '/' + 'Cross_val2_results', Slope_Means = m2_means, Slope_Stds = m2_stds, Int_Means = b2_means, Int_Stds = b2_stds, Relative_RMS = rel2_rmss, Labels = labs)

            cv_mat = np.vstack((np.vstack((np.vstack((m2_stds,b2_stds)),rel2_rmss)),rel_trmss))
            self.cross_val_bar(destination +'/'+'Cross_val2_dblbar',destination+'/'+ 'Cross_val2_rmsbar', cv_mat, xlabels=['Slope', 'Intercept', 'RMS'], ylabels=['Slope Error', 'Intercept Error', 'Relative RMS'], bar_labels = labs)
            cvl_mat = np.vstack((np.reshape(np.take(m2_means,[0],axis=1),(-1,)), np.reshape(np.take(b2_means,[0],axis=1),(-1,))))
            cvl_std = np.vstack((m2_stds,b2_stds))
            self.cross_val_line(destination+'/'+'Cross_val2_line',cvl_mat,cvl_std, xlabel = 'Functional', ylabels = ['Slope', 'Intercept'], tick_labels=labs)


    def run(self, filename,parent='.',int=True):
        self.init_file(filename)
        headings, data = self.get_data()
        #for i in range(len(headings)-1):
        if int:
            for i in range(1):
                error_mat = np.zeros((data.shape[1]-(1+i),19), dtype=object)
                for j in range(len(headings))[i+1:]:
                    xlab = headings[i]
                    ylab = headings[j]
                    gname = self.shortname + '_' + xlab + '_' + ylab
                    if i == 0:
                        self.make_dir(parent + '/' +self.shortname.encode('string_escape') + '_vs_exp')
                        destination = parent+'/'+self.shortname.encode('string_escape') + '_vs_exp'
                    elif i == 1:
                        self.make_dir(parent+'/'+self.shortname.encode('string_escape') + '_vs_funcs')
                        destination = parent+'/'+self.shortname.encode('string_escape') + '_vs_funcs'

                    x = np.take(data,[i], axis =1)
                    x = np.reshape(x, (-1,))
                
                    y = np.take(data, [j], axis=1)
                    y = np.reshape(y, (-1,))
                    
                    #m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(x,y)
                    #self.reg_plot(destination+'/'+gname,x,y,plty,m,b,mae,nmae,mse,nmse,rms,nrms,std,nstd,xlabel=xlab, ylabel=ylab)
                    
                    m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(y,x)
                    #self.new_reg_plot(destination+'/'+gname+'.png',y,x,xlabel=ylab, ylabel=xlab)
                    self.reg_plot(destination+'/'+gname+'.png',y,x,xlabel=ylab, ylabel=xlab)
                    self.new_reg_plot(destination+'/'+gname+'_lines.png',y,x,xlabel=ylab, ylabel=xlab)
                    self.error_plot(destination+'/'+'error_' +gname+'.png', errs,nerrs,errs.shape[0]/2,'Original Error', 'Fitted Error')
                    res = self.ref_reg(np.reshape(y,(-1,1)),np.reshape(x,(-1,1)),np.reshape(y,(-1,1)),np.reshape(x,(-1,1)))
                    err_list = [mae,nmae,tmae,res[1],mse,nmse,tmse,res[2], rms,nrms,trms,res[3],std,nstd,tstd,res[4],m,b]
                    error_mat[j-(i+1)][0] = ylab
                    for k, err in enumerate(err_list):
                        error_mat[j-(i+1)][k+1] = err
                    
                    #np.savez(destination+'/'+gname,x=x, y=y, new_y=plty, slope=m, intercept=b, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
                    np.savez(destination+'/'+gname,x=y, y=x, new_y=plty, slope=m, intercept=b, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
                np.savez(destination+'/'+'Error_file_for_vs_' + headings[i],Errors = error_mat)
                self.error_bar_plot(destination + '/' + 'Error_plot_rms_for_vs_' + headings[i] +'.png', np.take(error_mat,[0,9,12,11,10],axis=1),legend_labels = ['Standard','Slope Corrected','Translation Corrected', 'Affine Corrected'], ylabel = 'RMSE')
                self.error_bar_plot(destination + '/' + 'Error_plot_mae_for_vs_' + headings[i]+'.png', np.take(error_mat,[0,1,4,3,2],axis=1),legend_labels = ['Standard','Slope Corrected', 'Translation Corrected', 'Affine Corrected'], ylabel = 'MAE')
        else:
            for i in range(1):
                error_mat = np.zeros((data.shape[1]-(1+i),14), dtype=object)
                for j in range(len(headings))[i+1:]:
                    xlab = headings[i]
                    ylab = headings[j]
                    gname = self.shortname + '_' + xlab + '_' + ylab
                    if i == 0:
                        self.make_dir(parent + '/' +self.shortname.encode('string_escape') + '_vs_exp')
                        destination = parent+'/'+self.shortname.encode('string_escape') + '_vs_exp'
                    elif i == 1:
                        self.make_dir(parent+'/'+self.shortname.encode('string_escape') + '_vs_funcs')
                        destination = parent+'/'+self.shortname.encode('string_escape') + '_vs_funcs'

                    x = np.take(data,[i], axis =1)
                    x = np.reshape(x, (-1,))
                
                    y = np.take(data, [j], axis=1)
                    y = np.reshape(y, (-1,))
                    
                    #m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(x,y)
                    #self.reg_plot(destination+'/'+gname,x,y,plty,m,b,mae,nmae,mse,nmse,rms,nrms,std,nstd,xlabel=xlab, ylabel=ylab)
                    
                    m,b,errs,nerrs,plty,mae,nmae,mse,nmse,rms,nrms,std,nstd, tmae,tmse,trms,tstd = self.regression(y,x)
                    co,nmae,nmse,nrms,nstd = self.ref_reg(np.reshape(y,(-1,1)),x,np.reshape(y,(-1,1)),x)
                    m = co[0]
                    self.reg_plot(destination+'/'+gname+'.png',y,x,xlabel=ylab, ylabel=xlab,int='off')
                    self.error_plot(destination+'/'+'error_' +gname+'.png', errs,nerrs,errs.shape[0]/2,'Original Error', 'Fitted Error')
                    err_list = [mae,nmae,tmae,mse,nmse,tmse,rms,nrms,trms,std,nstd,tstd,m]
                    error_mat[j-(i+1)][0] = ylab
                    for k, err in enumerate(err_list):
                        error_mat[j-(i+1)][k+1] = err
                    
                    #np.savez(destination+'/'+gname,x=x, y=y, new_y=plty, slope=m, intercept=b, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
                    np.savez(destination+'/'+gname,x=y, y=x, new_y=plty, slope=m, MAE=mae,MAE_fit=nmae, MSE=mse, MSE_fit=nmse, RMS=rms, RMS_fit=nrms, STD = std, STD_fit=nstd)
                np.savez(destination+'/'+'Error_file_for_vs_' + headings[i],Errors = error_mat)
                self.error_bar_plot(destination + '/' + 'Error_plot_rms_for_vs_' + headings[i] +'.png', np.take(error_mat,[0,7,9,8],axis=1),legend_labels = ['Standard','Translation Corrected', 'Slope Corrected'], ylabel = 'RMSE')
                self.error_bar_plot(destination + '/' + 'Error_plot_mae_for_vs_' + headings[i]+'.png', np.take(error_mat,[0,1,3,2],axis=1),legend_labels = ['Standard','Translation Corrected', 'Slope Corrected'], ylabel = 'MAE')
            

    #Have not used in a while. Gathers across databases?
    def gather_data(self, dir_list):
        third_dim = len(dir_list)
        disps = []
        for i,dir in enumerate(dir_list):
            if 's22' in dir:
                disps.append(i)
            files = os.walk(dir).next()[2]
            for s in files:
                if 'Error_file' in s:
                    fname =s
            if fname:
                f = np.load(dir + '/' + fname)
                errs = f['Errors']
                if i ==0:
                    first_dim = 16
                    second_dim = errs.shape[1]
                    res = np.zeros((third_dim,first_dim,second_dim),dtype=object)
                jcount=0
                for j in range(errs.shape[0]):
                    #print jcount,j
                    #print str(errs[j][0])
                    if not str(errs[j][0]) == 'B3LYP' and not str(errs[j][0]) == 'PBE0':              
                        for k in range(errs.shape[1]):
                            res[i][jcount][k] = errs[j][k]
                        jcount +=1
        return res, disps
    #makes graphs/data using a product cost function across databases
    def make_prod_func(self,dir_list,destination):
        res,disps = self.gather_data(dir_list)
        geo_means = np.ones((res.shape[1],4), dtype=object)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                geo_means[j][0] = res[i][j][0]
                print res.shape[0]
                #print res
                #weight the dispersion stuff weaker
                if i in disps:
                    geo_means[j][1] *= res[i][j][1]**(1./(3*res.shape[0]))/res[i][j][2]**(1./(3*res.shape[0]))
                    geo_means[j][3] *= res[i][j][2]**(1./(3*res.shape[0]))/res[i][j][2]**(1./(3*res.shape[0]))
                    geo_means[j][2] *= res[i][j][3]**(1./(3*res.shape[0]))/res[i][j][2]**(1./(3*res.shape[0]))
                else:    
                    geo_means[j][1] *= res[i][j][1]**(1./res.shape[0])/res[i][j][2]**(1./res.shape[0])
                    geo_means[j][3] *= res[i][j][2]**(1./res.shape[0])/res[i][j][2]**(1./res.shape[0])
                    geo_means[j][2] *= res[i][j][3]**(1./res.shape[0])/res[i][j][2]**(1./res.shape[0])
        self.error_bar_plot(destination+'/rel_mae_geomean_plot.png', geo_means, legend_labels=['Standard','Bias Corrected', 'Affine Corrected'], ylabel = 'Geometric Mean')
        np.savez(destination+'/rel_mae_geomean_file',geo_mean=geo_means)
        
        geo_means = np.ones((res.shape[1],4), dtype=object)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                geo_means[j][0] = res[i][j][0]
                if i in disps:
                    geo_means[j][1] *= res[i][j][7]**(1./(3*res.shape[0]))
                    geo_means[j][3] *= res[i][j][8]**(1./(3*res.shape[0]))
                    geo_means[j][2] *= res[i][j][9]**(1./(3*res.shape[0]))
                else:
                    geo_means[j][1] *= res[i][j][7]**(1./res.shape[0])
                    geo_means[j][3] *= res[i][j][8]**(1./res.shape[0])
                    geo_means[j][2] *= res[i][j][9]**(1./res.shape[0])
        self.error_bar_plot(destination+'/rms_geomean_plot.png', geo_means, legend_labels=['Standard','Bias Corrected', 'Affine Corrected'], ylabel = 'Geometric Mean')
        np.savez(destination+'/rms_geomean_file',geo_mean = geo_means)       
 
        geo_means = np.ones((res.shape[1],4), dtype=object)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                geo_means[j][0] = res[i][j][0]
                if i in disps:
                    geo_means[j][1] *= res[i][j][1]**(1./(3*res.shape[0]))
                    geo_means[j][3] *= res[i][j][2]**(1./(3*res.shape[0]))
                    geo_means[j][2] *= res[i][j][3]**(1./(3*res.shape[0]))
                else:
                    geo_means[j][1] *= res[i][j][1]**(1./res.shape[0])
                    geo_means[j][3] *= res[i][j][2]**(1./res.shape[0])
                    geo_means[j][2] *= res[i][j][3]**(1./res.shape[0])
        self.error_bar_plot(destination+'/mae_geomean_plot.png', geo_means, legend_labels=['Standard','Bias Corrected', 'Affine Corrected'], ylabel = 'Geometric Mean')
        np.savez(destination+'/mae_geomean_file',geo_mean=geo_means)

    def make_line_plots(self, dir_list,destination):
        errs = self.gather_data(dir_list)
        xlabels =[]
        for dir in dir_list:
            xlabels.append(dir.split('_')[0])
        for i in range(errs.shape[1]):
            data = np.reshape(np.take(errs,[i],axis=1),(errs.shape[0],errs.shape[2]))
            mlabel= data[0][0]
            x = np.arange(data.shape[0])
            rms = np.reshape(np.take(data,[7],axis=1),(-1,))
            nrms = np.reshape(np.take(data,[8],axis=1),(-1,))
            trms = np.reshape(np.take(data,[9],axis=1),(-1,))
            mae = np.reshape(np.take(data,[1],axis=1),(-1,))
            nmae = np.reshape(np.take(data,[2],axis=1),(-1,))
            tmae = np.reshape(np.take(data,[3],axis=1),(-1,))

            fig = plt.figure()
            ax = fig.add_subplot(111)
        self.error_bar_plot(destination+'/mae_geomean_plot.png', geo_means, legend_labels=['Standard','Bias Corrected', 'Affine Corrected'], ylabel = 'Geometric Mean')
        np.savez(destination+'/mae_geomean_file',geo_mean=geo_means)
    #Make error vs. functional plots across a DB
    def make_line_plots(self, dir_list,destination):
        errs = self.gather_data(dir_list)
        xlabels =[]
        for dir in dir_list:
            xlabels.append(dir.split('_')[0])
        for i in range(errs.shape[1]):
            data = np.reshape(np.take(errs,[i],axis=1),(errs.shape[0],errs.shape[2]))
            mlabel= data[0][0]
            x = np.arange(data.shape[0])
            rms = np.reshape(np.take(data,[7],axis=1),(-1,))
            nrms = np.reshape(np.take(data,[8],axis=1),(-1,))
            trms = np.reshape(np.take(data,[9],axis=1),(-1,))
            mae = np.reshape(np.take(data,[1],axis=1),(-1,))
            nmae = np.reshape(np.take(data,[2],axis=1),(-1,))
            tmae = np.reshape(np.take(data,[3],axis=1),(-1,))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,rms,'b-',label='RMS')
            ax.plot(x,nrms,'g--',label='RMS_fit')
            ax.plot(x,trms,'m-.',label='RMS_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('RMS')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'rms_db_'+mlabel+'_plot.png')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,mae,'b-',label='MAE')
            ax.plot(x,nmae,'g--',label='MAE_fit')
            ax.plot(x,tmae,'m-.',label='MAE_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('MAE')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'mae_db_'+mlabel+'_plot.png')
                        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,mae-nmae,'g--',label='MAE_fit')
            ax.plot(x,mae-tmae,'m-.',label='MAE_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('Delta MAE')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'del_mae_db_'+mlabel+'_plot.png')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,rms-nrms,'g--',label='RMS_fit')
            ax.plot(x,rms-trms,'m-.',label='RMS_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('Delta RMS')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'del_rms_db_'+mlabel+'_plot.png')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,nmae/mae,'g--',label='MAE_fit')
            ax.plot(x,tmae/mae,'m-.',label='MAE_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('Relative MAE')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'rel_mae_db_'+mlabel+'_plot.png')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,nrms/rms,'g--',label='RMS_fit')
            ax.plot(x,trms/rms,'m-.',label='RMS_trans')
            ax.set_xticklabels(xlabels)
            ax.set_ylabel('Relative RMS')
            ax.set_xlabel('Database')
            ax.legend(loc='best')
            fig.savefig(destination +'/'+'rel_rms_db_'+mlabel+'_plot.png')
