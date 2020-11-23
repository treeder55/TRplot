import numpy as np
import scipy as sp
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lmfit import models
from lmfit import Model
from ipywidgets import interactive

def trsubplot(x,y,nrows=2,ncols=1,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,plotsize=[34,10],markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',label = '',labelcolor = 'dodgerblue',c='gist_rainbow',yfactor=1,yoff=0,markerscale=2):
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    if title == '':
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if len(label) == 0:
        label = np.repeat(label,len(x[0]))
    if len(np.shape(label)) == 1:
        label = np.repeat([label],nrows*ncols,axis=0)
    if len(np.shape(ylim)) == 1:
        ylim = np.repeat([ylim],nrows*ncols,axis=0)
    if len(np.shape(markersize))==0:
        markersize = np.repeat(markersize,len(x[0]))
    if len(np.shape(markersize))==1:
        markersize = np.repeat([markersize],len(x),axis=0)
    if len(np.shape(ylabel))==0:
        ylabel = np.repeat([ylabel],len(x),axis=0)
    if len(np.shape(xlabel))==0:
        xlabel = np.repeat([xlabel],len(x),axis=0)
    for i in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,i+1)
        if c == 'prism':
            colors = cm.prism(np.linspace(0, 1, len(index)+1))
        if c == 'gist_rainbow':
            colors = cm.gist_rainbow(np.linspace(0, 1, len(index)+1))
        if c == 'inferno':
            colors = cm.inferno(np.linspace(0, 1, len(index)+1))
        for l,j in enumerate(index):
            ind = np.min([len(x[i][j]),len(y[i][j])])
            ax.plot(x[i][j][:ind], y[i][j][:ind]*yfactor+yoff*l, color=colors[l],ms=markersize[i][j],marker=marker,label = label[i][j],linestyle=linestyle)
        ax.set_ylabel(ylabel[i], color=labelcolor,fontsize = 30)
        ax.set_xlabel(xlabel[i], color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.set_title(title[i],fontsize=20,color=labelcolor)
        ax.legend(fontsize = 14,markerscale = markerscale)
        if xlim != 'auto':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != 'auto':
            ax.set_ylim([ylim[i][0],ylim[i][1]])
    plt.show()
def trplot(x,y,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',label = ''):
    plt.rcParams['axes.facecolor']='white'
    if len(np.shape(x))==1:
        x = np.array([x])
        y = np.array([y])
        label = np.array([label])
        markersize = np.array([markersize])
        linestyle = np.array([linestyle])
    if len(np.shape(markersize)) == 0:
        markersize = np.repeat([markersize],len(x))
    if len(np.shape(label)) == 0:
        label = np.repeat([label],len(x))
    if len(np.shape(linestyle)) == 0:
        linestyle = np.repeat([linestyle],len(x))
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(x[i],y[i]+i*off,marker=marker,color=colors[i],markersize = markersize[i],linewidth=linewidth,label = label[i],linestyle = linestyle[i])
    ax.set_ylabel(ylabel, color=labelcolor,fontsize = font1)
    ax.set_xlabel(xlabel, color=labelcolor,fontsize = font1)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    if label[0] != '':
        ax.legend(fontsize = font2,markerscale = 5)
    ax.set_title(title,fontsize=font2,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
def errorbar(x,y,err,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',label = ['']):
    plt.rcParams['axes.facecolor']='white'
    if len(np.shape(x))==1:
        x = np.array([x])
        y = np.array([y])
        err = np.array([err])
        label = np.array([label])
        markersize = np.array([markersize])
        linestyle = np.array([linestyle])
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.errorbar(x[i],y[i]+i*off,err[i],marker=marker,color=colors[i],markersize = markersize[i],linewidth=linewidth,label = label[i],linestyle = linestyle[i])
    ax.set_ylabel(ylabel, color=labelcolor,fontsize = font1)
    ax.set_xlabel(xlabel, color=labelcolor,fontsize = font1)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=font2)
    if label[0] != '':
        ax.legend(fontsize = font2,markerscale = 2)
    ax.set_title(title,fontsize=font2,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
def plotvol(M,a,b,c,d,color='inferno',labelcolor='black',labelsize1=20,labelsize2=30,xlabel='',ylabel='',zlabel='',climsliders = [0,3,1,5]):
    color = 'inferno'
    h=M[0];k=M[1];E=M[2];I=M[3];
    def pt(a,b,c,d):
        fig = plt.figure()
        plotsize=[2,2]
        clim = [c,d]
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
        img = ax.scatter(h[(I>clim[0])&(I<clim[1])], k[(I>clim[0])&(I<clim[1])], E[(I>clim[0])&(I<clim[1])], c=I[(I>clim[0])&(I<clim[1])], cmap=color)
        fig.colorbar(img)
        ax.set_xlabel(xlabel, color=labelcolor,fontsize = labelsize2)
        ax.set_ylabel(ylabel, color=labelcolor,fontsize = labelsize2)
        ax.set_zlabel(zlabel, color=labelcolor,fontsize = labelsize2)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.tick_params(axis = 'z',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=labelsize1)
        ax.view_init(a,b)
        #fig.colorbar
        #img.set_clim(0.05,.1)
        #ax.legend(fontsize = 14,markerscale = 5)
        #ax.set_xlim([xlim[0],xlim[1]])
        #ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
        plt.show()
def pt(M,model,a,b,c=1,d=3,color='inferno',plotsize=[34,10],l = 100,markersize=5,marker='o'):
    u=np.array([M[1],model[1]]);v=np.array([M[2],model[2]]);E=np.array([M[3],model[3]]);Ia=np.array([M[4],model[4]]);
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    labelcolor = 'dodgerblue'
    for i in range(2):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        img = ax.scatter(u[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], v[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], E[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], c=Ia[i][(Ia[i]>clim[0])&(Ia[i]<clim[1])], cmap=color,s=markersize,marker=marker)
        fig.colorbar(img)
        ax.set_ylabel('k', color=labelcolor,fontsize = 30)
        ax.set_xlabel('h', color=labelcolor,fontsize = 30)
        ax.set_zlabel('E', color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'z',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.view_init(a,b)
        ax.set_zlim(0,1.3)
        #fig.colorbar
        #img.set_clim(0.05,.1)
        #ax.legend(fontsize = 14,markerscale = 5)
        #ax.set_xlim([xlim[0],xlim[1]])
        #ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    plt.show()
def pt2(M,color='inferno',plotsize=[34,10],l = 100,markersize=5,marker='o'):
    u=M[0];v=M[1];Ia=M[2];
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    labelcolor = 'dodgerblue'
    ax = fig.add_subplot(1,2,1)
    img = ax.scatter(u, v, c=Ia, cmap=color,s=markersize,marker=marker)
    fig.colorbar(img)
    ax.set_ylabel('k', color=labelcolor,fontsize = 30)
    ax.set_xlabel('h', color=labelcolor,fontsize = 30)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    #fig.colorbar
    #img.set_clim(0.05,.1)
    #ax.legend(fontsize = 14,markerscale = 5)
    #ax.set_xlim([xlim[0],xlim[1]])
    #ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    plt.show()
def pt2sub(M,color='',plotsize=[34,10],l = 100,markersize=5,marker='o',nrows=1,ncols=2,clim = '',title='',ylim = '',xlim = '',ylabel = '',xlabel=''): #plot, 2 dimensional, with subtitles
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    labelcolor = 'dodgerblue'
    if title == '':
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if clim == '':
        clim = {}
        for i in range(nrows*ncols):
            clim[i] = [0,np.max(M[i][2])]
    if color == '':
        color = {}
        for i in range(nrows*ncols):
            color[i] = 'inferno'
    if len(np.shape(markersize))==0:
        a = markersize
        markersize = {}
        for i in range(nrows*ncols):
            markersize[i] = a
    for i in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,i+1)
        img = ax.scatter(M[i][0], M[i][1], c=M[i][2], cmap=color[i],s=markersize[i],marker=marker)
        fig.colorbar(img)
        ax.set_ylabel(ylabel, color=labelcolor,fontsize = 30)
        ax.set_xlabel(xlabel, color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        img.set_clim(clim[i][0],clim[i][1])
        ax.set_title(title[i],fontsize=20,color=labelcolor)
    #ax.legend(fontsize = 14,markerscale = 5)
        if xlim != '':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != '':
            ax.set_ylim([ylim[0],ylim[1]])
    plt.show()
def plot2d(x,y,index = [0],xlim = [0,1.4],ylim='auto',off = 0,labelcolor = 'black',plotsize=[2.1,2.5]):
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    for i in index:
        ax.plot(data[i][0],data[i][1]+i*off,'-o',color=colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = str(fields[i])+' T, i = ' + str(i))
    ax.set_ylabel('I', color=labelcolor,fontsize = 30)
    ax.set_xlabel('E (meV)', color=labelcolor,fontsize = 30)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
    ax.legend(fontsize = 14,markerscale = 5)
    ax.set_xlim([xlim[0],xlim[1]])
    ax.set_title('offset = ' + str(off),fontsize=20,color=labelcolor)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])
def plothist(h=[[]],aspect=1,clim='auto',plotsize=[2.1,2.5],nrows=1,ncols=1,index = [0],xlim = 'auto',ylim='auto',xlabel='',ylabel='',markersize=2,linewidth=0.5,marker='o',linestyle='-',font1=30,font2=20,title='',labelcolor='black',c='gist_rainbow',yfactor=1,yoff=0,markerscale=2,numplots = 1):
    fig = plt.figure(figsize = (plotsize[0],plotsize[1]))
    #clim = [c,d]
    #ax = fig.add_axes([0,0,plotsize[0],plotsize[1]],projection='3d')
    if title == '':
        title = {}
        for i in range(nrows*ncols):
            title[i] = ''
    if len(np.shape(markersize))==0:
        markersize = np.repeat(markersize,len(x[0]))
    if len(np.shape(markersize))==1:
        markersize = np.repeat([markersize],len(x),axis=0)
    if len(np.shape(ylabel))==0:
        ylabel = np.repeat([ylabel],len(x),axis=0)
    if len(np.shape(xlabel))==0:
        xlabel = np.repeat([xlabel],len(x),axis=0)
    for i in range(numplots):
        ax = fig.add_subplot(nrows,ncols,i+1)
        im = ax.imshow(h[i],cmap=c,aspect = aspect,interpolation='none',origin='low')
        ax.set_ylabel(ylabel[i], color=labelcolor,fontsize = 30)
        ax.set_xlabel(xlabel[i], color=labelcolor,fontsize = 30)
        ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=20)
        ax.set_title(title[i],fontsize=20,color=labelcolor)
        ax.legend(fontsize = 14,markerscale = markerscale)
        cbar = fig.colorbar(im,shrink = 1)
        if xlim != 'auto':
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim != 'auto':
            ax.set_ylim([ylim[0],ylim[1]])
        if clim != 'auto':
            im.set_clim(clim[0],clim[1])
        plt.show()
        plt.rcParams['axes.facecolor']='white'
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
#    im = ax.imshow(h,cmap='rainbow',aspect = aspect,interpolation='none',origin='low')  
#    ax.set_xlabel(xlabel,fontsize = 20)
#    ax.set_ylabel(ylabel,fontsize = 20)
#    #ax.set_axis(fontsize = 20)
#    ax.set_title(title)
#    cbar = fig.colorbar(im,shrink = 0.8)
#    if ylim != 'auto':
#        ax.set_ylim([ylim[0],ylim[1]])    
#    if xlim != 'auto':
#        ax.set_xlim([xlim[0],xlim[1]])
#    if clim != 'auto':
#        im.set_clim(clim[0],clim[1])
#    plt.show()
def pcolormesh(x = [], y = [], c=[[]],title='',aspect=1,index = [0],clim='auto',xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],clabel = '',cbarshrink = 1,fonts1=30,fonts2=20,cmap = 'gnuplot2'):
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    im = ax.pcolormesh(x,y,c,cmap=cmap)#,aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel(xlabel,fontsize = fonts1,color=labelcolor)
    ax.set_ylabel(ylabel,fontsize = fonts1,color=labelcolor)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    cbar = fig.colorbar(im,shrink = cbarshrink)
    cbar.ax.set_ylabel(clabel,fontsize = fonts1)
    cbar.ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
    if clim != 'auto':
        im.set_clim(clim[0],clim[1])
    plt.show()
def pcolormeshsub(x = [], y = [], c=[[]],title='',aspect=1,index = [0],clim='auto',xlim = 'auto',ylim='auto',xlabel='',ylabel='',off = 0,labelcolor = 'black',plotsize=[2.1,2.5],clabel = '',cbarshrink = 1,fonts1=30,fonts2=20,cmap = 'gnuplot2'):
    plt.rcParams['axes.facecolor']='white'
    fig = plt.figure()
    ax = fig.add_axes([0,0,plotsize[0],plotsize[1]])
    im = ax.pcolormesh(x,y,c,cmap=cmap)#,aspect = aspect,interpolation='none',origin='low')  
    ax.set_xlabel(xlabel,fontsize = fonts1,color=labelcolor)
    ax.set_ylabel(ylabel,fontsize = fonts1,color=labelcolor)
    #ax.set_axis(fontsize = 20)
    ax.set_title(title)
    ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    ax.tick_params(axis = 'x',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    cbar = fig.colorbar(im,shrink = cbarshrink)
    cbar.ax.set_ylabel(clabel,fontsize = fonts1)
    cbar.ax.tick_params(axis = 'y',labelcolor=labelcolor,color=labelcolor,labelsize=fonts2)
    if ylim != 'auto':
        ax.set_ylim([ylim[0],ylim[1]])    
    if xlim != 'auto':
        ax.set_xlim([xlim[0],xlim[1]])
    if clim != 'auto':
        im.set_clim(clim[0],clim[1])
    plt.show()
def trfit2G(x,y,hints):
    model = models.GaussianModel(prefix='one_')+models.GaussianModel(prefix='two_')
    parnames = model.param_names
    pars = model.make_params()
    for j,n in enumerate(parnames):
        pars[n].set(value = hints[j],vary=True)
    result = model.fit(y,pars,x=x)
    print(result.fit_report())
    return result
#def oplotsim(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.sim[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
#def oplotfit(self,index,off):
#    for i in index:
#        self.ax.plot(self.x,self.simfit[i]+i*off,'-',color=self.colors[i],markersize = 2,linewidth=0.5,markeredgewidth = 1,label = 'sim, i = ' + str(i))
